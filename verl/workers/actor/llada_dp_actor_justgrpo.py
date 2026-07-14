# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.mdlm_sp_utils import get_packed_logits


__all__ = ["DLLMDataParallelPPOActor", "build_justgrpo_ar_position_pack"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def build_justgrpo_ar_position_pack(
    input_ids,
    attention_mask,
    response_length,
    position_start,
    position_end,
    mask_token_id,
):
    """Pack JustGRPO AR states for a contiguous range of response positions.

    For response position ``t``, the model sees the prompt, response tokens before
    ``t``, and MASK at ``t`` and every later valid response position. Padding is
    removed before packing, matching the other LLaDA packed actor paths.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have shape [batch, sequence].")

    batch_size, sequence_length = input_ids.shape
    response_start = sequence_length - response_length
    if response_length <= 0 or response_start < 0:
        raise ValueError(f"Invalid response_length={response_length} for sequence_length={sequence_length}.")
    if not (0 <= position_start < position_end <= response_length):
        raise ValueError(
            f"Invalid response position range [{position_start}, {position_end}) for length {response_length}."
        )

    packed_segments = []
    cu_seqlens = [0]
    prompt_lengths = []
    target_offsets = []
    target_ids = []
    batch_indices = []
    relative_positions = []
    max_seqlen = 0

    attention_mask = attention_mask.bool()
    for position in range(position_start, position_end):
        absolute_position = response_start + position
        for batch_index in range(batch_size):
            if not attention_mask[batch_index, absolute_position]:
                continue

            valid_tokens = input_ids[batch_index][attention_mask[batch_index]].clone()
            prompt_length = int(attention_mask[batch_index, :response_start].sum().item())
            observed_response_length = int(
                attention_mask[batch_index, response_start:absolute_position].sum().item()
            )
            target_index = prompt_length + observed_response_length
            if target_index >= valid_tokens.numel():
                raise RuntimeError(
                    f"AR target index {target_index} is outside packed sequence length {valid_tokens.numel()}."
                )

            target_id = input_ids[batch_index, absolute_position]
            if valid_tokens[target_index].item() != target_id.item():
                raise RuntimeError("Packed JustGRPO target no longer matches the response token.")

            valid_tokens[target_index:] = mask_token_id
            packed_segments.append(valid_tokens)
            target_offsets.append(cu_seqlens[-1] + target_index)
            target_ids.append(target_id)
            batch_indices.append(batch_index)
            relative_positions.append(position - position_start)
            prompt_lengths.append(prompt_length)
            cu_seqlens.append(cu_seqlens[-1] + valid_tokens.numel())
            max_seqlen = max(max_seqlen, valid_tokens.numel())

    if not packed_segments:
        return None

    device = input_ids.device
    return {
        "packed_input": torch.cat(packed_segments, dim=0).unsqueeze(0),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        "max_seqlen": max_seqlen,
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        "target_offsets": torch.tensor(target_offsets, dtype=torch.long, device=device),
        "target_ids": torch.stack(target_ids).to(device=device, dtype=torch.long),
        "batch_indices": torch.tensor(batch_indices, dtype=torch.long, device=device),
        "relative_positions": torch.tensor(relative_positions, dtype=torch.long, device=device),
    }


def _aggregate_position_chunk(loss_chunk, mask_chunk, full_mask, loss_agg_mode):
    """Aggregate one position chunk so summing chunks equals ``agg_loss``."""
    mask_chunk = mask_chunk.to(dtype=loss_chunk.dtype)
    full_mask = full_mask.to(dtype=loss_chunk.dtype)

    if loss_agg_mode == "token-mean":
        return (loss_chunk * mask_chunk).sum() / (full_mask.sum() + 1e-8)
    if loss_agg_mode == "seq-mean-token-sum":
        return (loss_chunk * mask_chunk).sum(dim=-1).mean()
    if loss_agg_mode == "seq-mean-token-mean":
        sequence_denominator = full_mask.sum(dim=-1).clamp_min(1.0)
        return ((loss_chunk * mask_chunk).sum(dim=-1) / sequence_denominator).mean()
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return (loss_chunk * mask_chunk).sum() / full_mask.shape[-1]
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def _compute_ppo_token_terms(
    old_log_probs,
    log_probs,
    advantages,
    clip_ratio_low,
    clip_ratio_high,
    clip_ratio_c,
):
    """Unreduced equivalent of ``dllm_core_algos.compute_policy_loss``."""
    ratio = torch.exp(log_probs - old_log_probs)
    ppo_kl = torch.exp(old_log_probs - log_probs) - (old_log_probs - log_probs) - 1

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_losses = torch.maximum(pg_losses1, pg_losses2)
    clip_indicators = torch.gt(pg_losses2, pg_losses1).float()

    pg_losses3 = -advantages * clip_ratio_c
    dual_clipped_losses = torch.min(pg_losses3, clipped_losses)
    lower_clip_indicators = torch.gt(clipped_losses, pg_losses3).float() * (advantages < 0).float()
    pg_losses = torch.where(advantages < 0, dual_clipped_losses, clipped_losses)
    return pg_losses, clip_indicators, ppo_kl, lower_clip_indicators


class DLLMDataParallelPPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config, actor_module, actor_optimizer)

        self.MASK_TOKEN_ID = actor_module.config.mask_token_id
        self.PAD_TOKEN_ID = actor_module.config.pad_token_id
        self.cfg_scale = config["cfg_scale"]  # Whether to use CFG
        self.device_name = get_device_name()
        self.justgrpo_position_chunk_size = max(int(config.get("justgrpo_position_chunk_size", 1)), 1)

    def _position_ranges(self, response_length):
        for start in range(0, response_length, self.justgrpo_position_chunk_size):
            yield start, min(start + self.justgrpo_position_chunk_size, response_length)

    def _dummy_position_pack(self, input_ids, attention_mask):
        """Build a rank-local dummy segment to keep distributed forward counts aligned."""
        valid_tokens = input_ids[0][attention_mask[0].bool()].clone()
        if valid_tokens.numel() == 0:
            valid_tokens = input_ids[0, :1].clone()
        device = input_ids.device
        return {
            "packed_input": valid_tokens.unsqueeze(0),
            "cu_seqlens": torch.tensor([0, valid_tokens.numel()], dtype=torch.int32, device=device),
            "max_seqlen": valid_tokens.numel(),
            "prompt_lengths": torch.tensor([valid_tokens.numel()], dtype=torch.long, device=device),
        }

    def _forward_ar_position_chunk(
        self,
        micro_batch,
        temperature,
        position_start,
        position_end,
        calculate_entropy=False,
    ):
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        response_length = micro_batch["responses"].size(-1)
        batch_size = input_ids.size(0)
        chunk_length = position_end - position_start

        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError(f"JustGRPO learner temperature must be positive, got {temperature}.")

        packed = build_justgrpo_ar_position_pack(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_length=response_length,
            position_start=position_start,
            position_end=position_end,
            mask_token_id=self.MASK_TOKEN_ID,
        )
        has_targets = packed is not None
        if not has_targets:
            packed = self._dummy_position_pack(input_ids, attention_mask)

        autocast_enabled = self.device_name != "cpu"
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = self._get_logits(
                model=self.actor_module,
                packed_input=packed["packed_input"],
                cu_seqlens=packed["cu_seqlens"],
                max_seqlen=packed["max_seqlen"],
                prompt_len=packed["prompt_lengths"],
                cfg_scale=self.cfg_scale,
                MASK_TOKEN_ID=self.MASK_TOKEN_ID,
            )

        if not has_targets:
            graph_zero = logits[0, 0, 0].float() * 0.0
            log_probs = torch.zeros(
                (batch_size, chunk_length), dtype=torch.float32, device=input_ids.device
            ) + graph_zero
            entropy = (
                torch.zeros((batch_size, chunk_length), dtype=torch.float32, device=input_ids.device)
                + graph_zero
                if calculate_entropy
                else None
            )
            return entropy, log_probs, log_probs

        target_logits = logits[0].index_select(0, packed["target_offsets"]).float() / temperature
        log_normalizer = torch.logsumexp(target_logits, dim=-1)
        target_log_probs = target_logits.gather(-1, packed["target_ids"].unsqueeze(-1)).squeeze(-1)
        target_log_probs = target_log_probs - log_normalizer

        log_probs = target_log_probs.new_zeros((batch_size, chunk_length))
        log_probs = log_probs.index_put(
            (packed["batch_indices"], packed["relative_positions"]), target_log_probs
        )

        entropy = None
        if calculate_entropy:
            target_entropy = verl_F.entropy_from_logits(target_logits)
            entropy = target_entropy.new_zeros((batch_size, chunk_length))
            entropy = entropy.index_put(
                (packed["batch_indices"], packed["relative_positions"]), target_entropy
            )

        return entropy, log_probs, log_probs

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, call_fn_name=""
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute exact AR-order JustGRPO log-probs for a micro-batch."""
        response_length = micro_batch["responses"].size(-1)
        entropy_chunks = []
        log_prob_chunks = []
        loss_chunks = []
        for position_start, position_end in self._position_ranges(response_length):
            entropy, log_probs, loss_per_sample = self._forward_ar_position_chunk(
                micro_batch=micro_batch,
                temperature=temperature,
                position_start=position_start,
                position_end=position_end,
                calculate_entropy=calculate_entropy,
            )
            log_prob_chunks.append(log_probs)
            loss_chunks.append(loss_per_sample)
            if calculate_entropy:
                entropy_chunks.append(entropy)

        entropys = torch.cat(entropy_chunks, dim=-1) if calculate_entropy else None
        return entropys, torch.cat(log_prob_chunks, dim=-1), torch.cat(loss_chunks, dim=-1)

    def _get_logits(
        self,
        model,
        packed_input,
        cu_seqlens,
        max_seqlen,
        prompt_len,
        cfg_scale=0.0,
        MASK_TOKEN_ID=126336,
    ):
        """
        packed_input: (1, total_seqlen)
        cu_seqlens: (batch_size+1,)
        max_seqlen: int
        prompt_len: (batch_size,) True prompt length of each sample
        """
        return get_packed_logits(
            actor=self,
            model=model,
            packed_input=packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            prompt_len=prompt_len,
            cfg_scale=cfg_scale,
            mask_token_id=MASK_TOKEN_ID,
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids"""
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_prob_lst = []
        entropy_lst = []
        loss_per_sample_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_prob, loss_per_sample = self._forward_micro_batch(
                    micro_batch=micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    call_fn_name="compute_log_prob",
                )
            log_prob_lst.append(log_prob)
            loss_per_sample_lst.append(loss_per_sample)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_prob_lst, dim=0)
        loss_per_sample = torch.concat(loss_per_sample_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(
                get_reverse_idx(indices), dtype=torch.long, device=log_probs.device
            )
            log_probs = log_probs[revert_indices]
            loss_per_sample = loss_per_sample[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]

        return entropys, log_probs, loss_per_sample

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_probs")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for _, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())

                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_probs = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    clip_ratio_low = clip_ratio if clip_ratio_low is None else clip_ratio_low
                    clip_ratio_high = clip_ratio if clip_ratio_high is None else clip_ratio_high
                    if clip_ratio_c <= 1.0:
                        raise ValueError(f"clip_ratio_c must be greater than 1, got {clip_ratio_c}.")
                    calculate_entropy = entropy_coeff != 0
                    response_token_count = response_mask.sum().clamp_min(1).item()
                    micro_pg_loss = 0.0
                    micro_ppo_kl_sum = 0.0
                    micro_clip_sum = 0.0
                    micro_lower_clip_sum = 0.0
                    micro_kl_loss = 0.0

                    if self.config.use_dynamic_bsz:
                        backward_scale = responses.size(0) / self.config.ppo_mini_batch_size
                    else:
                        backward_scale = 1.0 / self.gradient_accumulation

                    for position_start, position_end in self._position_ranges(response_length):
                        entropy, log_prob, _ = self._forward_ar_position_chunk(
                            micro_batch=data,
                            temperature=temperature,
                            position_start=position_start,
                            position_end=position_end,
                            calculate_entropy=calculate_entropy,
                        )
                        chunk_slice = slice(position_start, position_end)
                        chunk_mask = response_mask[:, chunk_slice]
                        chunk_old_log_probs = old_log_probs[:, chunk_slice]
                        chunk_advantages = advantages[:, chunk_slice]

                        pg_losses, clip_indicators, ppo_kl_values, lower_clip_indicators = _compute_ppo_token_terms(
                            old_log_probs=chunk_old_log_probs,
                            log_probs=log_prob,
                            advantages=chunk_advantages,
                            clip_ratio_low=clip_ratio_low,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                        )
                        pg_loss = _aggregate_position_chunk(
                            pg_losses, chunk_mask, response_mask, loss_agg_mode
                        )
                        policy_loss = pg_loss

                        if entropy_coeff != 0:
                            entropy_loss = _aggregate_position_chunk(
                                entropy, chunk_mask, response_mask, loss_agg_mode
                            )
                            policy_loss = policy_loss - entropy_loss * entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_probs = data["ref_log_probs"][:, chunk_slice]
                            kld = kl_penalty(
                                l_theta=log_prob,
                                ref_l_theta=ref_log_probs,
                                kl_penalty=self.config.kl_loss_type,
                                advantages=chunk_advantages,
                            )
                            kl_loss = _aggregate_position_chunk(
                                kld, chunk_mask, response_mask, loss_agg_mode
                            )
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_kl_loss += kl_loss.detach().item()

                        (policy_loss * backward_scale).backward()

                        mask_float = chunk_mask.to(dtype=ppo_kl_values.dtype)
                        micro_pg_loss += pg_loss.detach().item()
                        micro_ppo_kl_sum += (ppo_kl_values.detach() * mask_float).sum().item()
                        micro_clip_sum += (clip_indicators.detach() * mask_float).sum().item()
                        micro_lower_clip_sum += (
                            lower_clip_indicators.detach() * mask_float
                        ).sum().item()

                    metric_data = {
                        "actor/pg_loss": micro_pg_loss,
                        "actor/pg_clipfrac": micro_clip_sum / response_token_count,
                        "actor/ppo_kl": micro_ppo_kl_sum / response_token_count,
                        "actor/pg_clipfrac_lower": micro_lower_clip_sum / response_token_count,
                    }
                    if self.config.use_kl_loss:
                        metric_data["actor/kl_loss"] = micro_kl_loss
                        metric_data["actor/kl_coef"] = self.config.kl_loss_coef
                    append_to_dict(metrics, metric_data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
