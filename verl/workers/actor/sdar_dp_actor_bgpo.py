# Copyright 2025 Shanghai AI Lab
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import agg_loss, compute_policy_loss_bgpo, kl_penalty  # NOTE: Our core algorithms
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.block_diffusion_utils import (
    BlockDiffusionArtifacts,
    build_block_diffusion_mask,
    build_full_block_diffusion_tensors,
    compact_block_diffusion_artifacts,
    pad_block_diffusion_loss_tensors,
)
from verl.workers.actor.llada_dp_actor_bgpo import DLLMDataParallelPPOActor as BaseDataParallelPPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor", "BaseDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(BaseDataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        model_block_length = int(actor_module.config.block_size)
        self.block_length = int(config.get("block_length", model_block_length))
        if self.block_length != model_block_length:
            raise ValueError(
                "SDAR actor.block_length must match model.config.block_size, "
                f"got {self.block_length} and {model_block_length}"
            )
        if "block_origin" not in config:
            raise ValueError(
                "SDAR actor requires an explicit block_origin resolved from "
                "the rollout backend"
            )
        self.block_origin = config["block_origin"]
        self.model_input_token_cost_multiplier = 2
        self.model_input_uses_padded_batch = True

    def _build_block_diffusion_artifacts(
        self,
        micro_batch,
        noisy_input_ids: torch.Tensor,
        target_mask: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> BlockDiffusionArtifacts:
        response_length = micro_batch["responses"].size(-1)
        prompt_section_length = micro_batch["input_ids"].size(-1) - response_length
        return compact_block_diffusion_artifacts(
            noisy_input_ids=noisy_input_ids,
            clean_input_ids=micro_batch["input_ids"],
            attention_mask=micro_batch["attention_mask"],
            position_ids=micro_batch["position_ids"],
            target_mask=target_mask,
            p_mask=p_mask,
            prompt_section_length=prompt_section_length,
            pad_token_id=self.PAD_TOKEN_ID,
        )

    def _compute_block_diffusion_token_losses(
        self, artifacts: BlockDiffusionArtifacts
    ) -> torch.Tensor:
        """Return globally gathered NLL/p values on compact noisy positions."""

        block_attention_mask = build_block_diffusion_mask(
            artifacts,
            block_size=self.block_length,
            block_origin=self.block_origin,
        )
        (
            full_input_ids,
            full_position_ids,
            full_target_mask,
            full_targets,
            full_p_mask,
        ) = build_full_block_diffusion_tensors(artifacts)

        sp_size = get_ulysses_sequence_parallel_world_size()
        if sp_size > 1:
            local_input_ids, local_position_ids, pad_size = ulysses_pad_and_slice_inputs(
                full_input_ids,
                full_position_ids,
                sp_size=sp_size,
            )
        else:
            local_input_ids = full_input_ids
            local_position_ids = full_position_ids
            pad_size = 0

        full_target_mask, full_targets, full_p_mask = pad_block_diffusion_loss_tensors(
            full_target_mask,
            full_targets,
            full_p_mask,
            pad_size,
        )
        if sp_size > 1:
            local_target_mask = slice_input_tensor(
                full_target_mask, dim=1, padding=False
            )
            local_targets_full = slice_input_tensor(full_targets, dim=1, padding=False)
            local_p_mask_full = slice_input_tensor(full_p_mask, dim=1, padding=False)
        else:
            local_target_mask = full_target_mask
            local_targets_full = full_targets
            local_p_mask_full = full_p_mask

        outputs = self.actor_module(
            input_ids=local_input_ids,
            attention_mask=block_attention_mask,
            position_ids=local_position_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=local_target_mask,
            ulysses_sp_training=True,
            ulysses_sp_targets=local_targets_full[local_target_mask].contiguous(),
            ulysses_sp_p_mask=local_p_mask_full[local_target_mask].contiguous(),
            ulysses_sp_answer_len=artifacts.response_lengths.sum(),
            ulysses_sp_return_token_loss=True,
        )
        local_token_losses = outputs.block_diffusion_token_loss.float()
        if local_token_losses.shape != local_target_mask.shape:
            raise RuntimeError(
                "SDAR token-loss shape mismatch: "
                f"{local_token_losses.shape} vs {local_target_mask.shape}"
            )
        token_losses = gather_outpus_and_unpad(
            local_token_losses,
            gather_dim=1,
            unpad_dim=1,
            padding_size=pad_size,
        )
        noisy_token_losses = token_losses[:, : artifacts.sequence_length]
        return noisy_token_losses * artifacts.target_mask.float()

    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name="") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate log_probs and entropy for micro_batch
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            loss_per_sample: # (bs, mc_num)
        """
        batch_size = micro_batch["input_ids"].size(0)
        response_length = micro_batch["responses"].size(-1)
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]  # (bs, mc_num, seq_len)
            mask_indices = micro_batch["mask_indices"]  # (bs, mc_num, seq_len)
            p_mask = micro_batch["p_mask"]  # (bs, mc_num, seq_len)
            mc_num = perturbed_seq.shape[1]
            loss_per_sample = torch.zeros((batch_size, mc_num), device=device)
            for i in range(mc_num):
                artifacts = self._build_block_diffusion_artifacts(
                    micro_batch=micro_batch,
                    noisy_input_ids=perturbed_seq[:, i, :],
                    target_mask=mask_indices[:, i, :],
                    p_mask=p_mask[:, i, :],
                )
                token_losses = self._compute_block_diffusion_token_losses(artifacts)
                loss_per_sample[:, i] = -token_losses.sum(dim=-1)

            log_likelihood = loss_per_sample.mean(dim=1)
            log_probs = log_likelihood.unsqueeze(-1).expand(-1, response_length)  # (batch_size, response_length)
            response_mask = micro_batch["attention_mask"][:, -response_length:].bool()
            response_count = response_mask.sum(dim=-1).clamp_min(1).to(loss_per_sample.dtype)
            loss_per_sample = loss_per_sample.unsqueeze(-1) / response_count[:, None, None]
            loss_per_sample = loss_per_sample.expand(-1, -1, response_length).contiguous()
            loss_per_sample = loss_per_sample * response_mask[:, None, :]
        
        entropy = None
        if calculate_entropy:
            probs = log_probs.exp()
            entropy = -probs * log_probs  # (bs, response_length) entropy of each token
            
        return entropy, log_probs, loss_per_sample
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "perturbed_seq", "mask_indices", "p_mask"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(
                batch=batch,
                max_token_len=max_token_len,
                token_cost_multiplier=self.model_input_token_cost_multiplier,
                padded_batch=self.model_input_uses_padded_batch,
            )
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        loss_per_sample_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, loss_per_sample = self._forward_micro_batch(micro_batch, temperature=temperature, n_l=self.n_l, mc_num=self.mc_num, calculate_entropy=calculate_entropy, call_fn_name="compute_log_prob")
            log_probs_lst.append(log_probs)
            loss_per_sample_lst.append(loss_per_sample)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
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
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "old_loss_per_sample", "advantages", "perturbed_seq", "mask_indices", "p_mask"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_probs")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch,
                        max_token_len=max_token_len,
                        token_cost_multiplier=self.model_input_token_cost_multiplier,
                        padded_batch=self.model_input_uses_padded_batch,
                    )
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()  # Clear gradient accumulation at the beginning of each micro-batch

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_probs = data["old_log_probs"]  # (bsz, response_length)
                    old_loss_per_sample = data["old_loss_per_sample"]  # (bsz, mc_num, response_length)
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    
                    accumulated_pg_loss = 0.0
                    accumulated_pg_clipfrac = 0.0
                    accumulated_ppo_kl = 0.0
                    accumulated_pg_clipfrac_lower = 0.0
                    
                    perturbed_seq = data["perturbed_seq"]
                    mask_indices = data["mask_indices"]
                    p_mask = data["p_mask"]
                    mc_num = perturbed_seq.shape[1]
                    for i in range(mc_num):
                        cur_data = {
                            **data,
                            "perturbed_seq": perturbed_seq[:, i : i + 1],
                            "mask_indices": mask_indices[:, i : i + 1],
                            "p_mask": p_mask[:, i : i + 1],
                        }
                        entropy, log_prob, loss_per_sample = self._forward_micro_batch(
                            micro_batch=cur_data,
                            temperature=temperature,
                            n_l=1,
                            mc_num=1,
                            calculate_entropy=calculate_entropy,
                            call_fn_name="update_policy",
                        )
                        # Compute policy loss
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_bgpo(
                            old_l_theta=old_loss_per_sample[:, i, :],  # (bsz, response_length)
                            l_theta=loss_per_sample[:, 0, :],  # (bsz, response_length)
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:  # NOTE: Currently not considering KL
                            ref_log_probs = data["ref_log_probs"]
                            # compute kl loss
                            kld = kl_penalty(
                                l_theta=log_prob,
                                ref_l_theta=ref_log_probs,
                                kl_penalty=self.config.kl_loss_type,
                                advantages=advantages,
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        loss /= self.mc_num
                        loss.backward()  # Gradient is accumulated in model parameters, but will not be updated now
                        
                        accumulated_pg_loss += pg_loss.detach().item()
                        accumulated_pg_clipfrac += pg_clipfrac.detach().item()
                        accumulated_ppo_kl += ppo_kl.detach().item()
                        accumulated_pg_clipfrac_lower += pg_clipfrac_lower.detach().item()

                    data = {
                        "actor/pg_loss": accumulated_pg_loss / mc_num,
                        "actor/pg_clipfrac": accumulated_pg_clipfrac / mc_num,
                        "actor/ppo_kl": accumulated_ppo_kl / mc_num,
                        "actor/pg_clipfrac_lower": accumulated_pg_clipfrac_lower / mc_num,
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()  # Update gradients after each mini-batch
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()  # Clear gradient accumulation
        return metrics
