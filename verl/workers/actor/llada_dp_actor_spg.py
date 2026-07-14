# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import agg_loss, compute_policy_loss_spg
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.mdlm_sp_utils import get_packed_logits
import torch.nn.functional as F

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config, actor_module, actor_optimizer)

        # diffusion related parameters
        self.MASK_TOKEN_ID = actor_module.config.mask_token_id
        self.PAD_TOKEN_ID = actor_module.config.pad_token_id
        self.mc_num = config["mc_num"]  # Number of Monte Carlo samples
        self.cfg_scale = config["cfg_scale"]  # Whether to use CFG
        self.logp_estimation = config["logp_estimation"]
        self.eubo_beta = config.get("eubo_beta", 1.5)
        self.num_iterations = config.get("num_iterations", 1)
        if config.get("use_kl_loss", False):
            raise ValueError(
                "SPG does not support actor.use_kl_loss: its reference estimator "
                "depends on the post-reward advantage sign. Set use_kl_loss=False."
            )

    def _forward_micro_batch(self, micro_batch, temperature, iter_idx, mc_num, logp_estimation="eubo", mix_weight=0.5,  calculate_entropy=False, call_fn_name="") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate log_probs and entropy for micro_batch
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            loss_per_sample: # (bs, mc_num)
        """
        batch_size, seq_length = micro_batch["input_ids"].size(0), micro_batch["input_ids"].size(-1)
        response_length = micro_batch["responses"].size(-1)
        prompt_length = seq_length - response_length
        device = micro_batch["input_ids"].device
        multi_modal_inputs = {}
        num_iterations = len(iter_idx)

        if "multi_modal_inputs" in micro_batch:
            # If there are multi-modal inputs, concatenate the content of each key
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)


        # Calculate log_probs
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"][:, iter_idx, :, :].reshape(batch_size * num_iterations, mc_num, seq_length)
            mask_indices = micro_batch["mask_indices"][:, iter_idx, :, :].reshape(batch_size * num_iterations, mc_num, seq_length)
            p_mask = micro_batch["p_mask"][:, iter_idx, :, :].reshape(batch_size * num_iterations, mc_num, seq_length)

            # Flattening [B, I, ...] is batch-major, so repeat each base sample I
            # times to keep targets and masks aligned with the selected perturbations.
            base_seq = micro_batch["input_ids"]
            base_attention_mask = micro_batch["attention_mask"]
            seq = base_seq.repeat_interleave(num_iterations, dim=0)
            attention_mask = base_attention_mask.repeat_interleave(num_iterations, dim=0)

            response_mask = base_attention_mask[:, -response_length:].to(micro_batch["advantages"].dtype)
            sequence_advantages = (micro_batch["advantages"][:, -response_length:] * response_mask).sum(dim=-1)
            reward_mask = sequence_advantages > 0

            loss_per_sample = torch.zeros((batch_size*num_iterations, mc_num), device=device)
            per_token_logps = torch.zeros((batch_size*num_iterations, mc_num, response_length), device=device)
            per_token_probs = torch.zeros((batch_size*num_iterations, mc_num, response_length), device=device)

            for i in range(mc_num):
                cur_perturbed_seq = perturbed_seq[:, i, :]  # (batch_size, seq_len)
                cur_mask_indices = mask_indices[:, i, :]
                cur_p_mask = p_mask[:, i, :]

                packed_perturbed_seq = []
                cu_seqlens = [0]
                max_seqlen = 0
                for b in range(batch_size*num_iterations):
                    valid_tokens = cur_perturbed_seq[b][attention_mask[b] == 1]
                    packed_perturbed_seq.append(valid_tokens)
                    cu_seqlens.append(cu_seqlens[-1] + len(valid_tokens))
                    max_seqlen = max(max_seqlen, len(valid_tokens))
                packed_perturbed_seq = torch.cat(packed_perturbed_seq, dim=0).unsqueeze(0)  # (1, total_seqlen)
                cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

                logits = self._get_logits(model=self.actor_module, packed_input=packed_perturbed_seq, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, prompt_len=attention_mask[:, :prompt_length].sum(dim=1), cfg_scale=0.0, MASK_TOKEN_ID=self.MASK_TOKEN_ID)

                # Restore logits for each sample
                loss = torch.zeros(batch_size * num_iterations, device=device)
                for b in range(batch_size*num_iterations):
                    start, end = cu_seqlens[b], cu_seqlens[b + 1]

                    logits_b = torch.zeros(seq_length, logits.size(-1), device=device, dtype=logits.dtype)
                    logits_b[attention_mask[b] == 1] = logits[0, start:end]

                    completion_logits_b = logits_b[-response_length:,:]  # (response_length, vocab_size)
                    mask = cur_mask_indices[b]  # (seq_len,)

                    per_token_logps[b, i, :] = -1 * F.cross_entropy(completion_logits_b, seq[b][-response_length:], reduction="none")   # (response_length,)
                    per_token_probs[b, i, :] = F.softmax(completion_logits_b, dim=-1).gather(dim=-1, index=seq[b][-response_length:].unsqueeze(-1)).squeeze(-1)  # (response_length,)

                    masked_p = cur_p_mask[b][mask]
                    if not torch.all(torch.isfinite(masked_p) & (masked_p > 0)):
                        raise RuntimeError(
                            "SPG encountered a non-positive or non-finite p_mask on masked tokens"
                        )
                    loss[b] = (F.cross_entropy(logits_b[mask], seq[b][mask], reduction="none") / masked_p).sum()  # cross_entropy returns negative log likelihood
                loss_per_sample[:, i] = -loss  # convert to log likelihood

            log_likelihood = loss_per_sample.sum(dim=1) / mc_num  # (batch_size,)
            log_probs = (log_likelihood / response_length).view(-1, 1).repeat(1, response_length)  # (batch_size, response_length)

            per_token_logps = per_token_logps.view(batch_size, num_iterations, mc_num, response_length).transpose(0,1)
            per_token_probs = per_token_probs.view(batch_size, num_iterations, mc_num, response_length).transpose(0,1)

            # all_ref_per_token_logps: num_iterations, batch_size, logits_to_keep
            # perturb_mask: num_iterations*batch_size, logits_to_keep
            # completion_mask: batch_size, logits_to_keep
            completion_mask_expanded = base_attention_mask[:, -response_length:].unsqueeze(0).unsqueeze(2).expand(num_iterations, -1, mc_num, -1)
            perturb_mask_expanded = mask_indices[:,:, -response_length:].view(batch_size, num_iterations, mc_num, response_length).transpose(0,1)

            assert per_token_logps.shape == (num_iterations, batch_size, mc_num, response_length), f"per_token_logps.shape: {per_token_logps.shape}"
            assert completion_mask_expanded.shape == (num_iterations, batch_size, mc_num, response_length), f"completion_mask_expanded.shape: {completion_mask_expanded.shape}"
            assert perturb_mask_expanded.shape == (num_iterations, batch_size, mc_num, response_length), f"perturb_mask_expanded.shape: {perturb_mask_expanded.shape}"

            # For perturbed tokens, we should weight over instances (sequence-wise)
            # num_iterations, batch_size, num_t

            # Check for zero denominators before division
            denominator = (completion_mask_expanded * perturb_mask_expanded).sum(dim=3)
            if (denominator == 0).any():
                print(f"WARNING: Zero denominator detected in per_seq_logps calculation!")
                print(f"denominator shape: {denominator.shape}")
                print(f"Zero count: {(denominator == 0).sum()}")
                print(f"completion_mask_expanded shape: {completion_mask_expanded.shape}")
                print(f"perturb_mask_expanded shape: {perturb_mask_expanded.shape}")
                print(f"completion_mask_expanded: {completion_mask_expanded}")
                print(f"perturb_mask_expanded: {perturb_mask_expanded}")
                # Add small epsilon to avoid division by zero
                denominator = torch.clamp(denominator, min=1e-8)

            per_seq_logps = (per_token_logps * completion_mask_expanded * perturb_mask_expanded).sum(dim=3) / denominator
            if self.logp_estimation == 'eubo' or self.logp_estimation == 'mix':
                empirical_t = (completion_mask_expanded * perturb_mask_expanded).sum(dim=3) / completion_mask_expanded.sum(dim=3)
                empirical_t_expanded = empirical_t.unsqueeze(3).expand(-1, -1, -1, completion_mask_expanded.size(-1))
                per_token_avg_ps = per_token_probs.pow(self.eubo_beta) * perturb_mask_expanded * completion_mask_expanded / empirical_t_expanded.clamp(min=1e-8)
                per_token_avg_ps = per_token_avg_ps.mean(dim=2) # [num_iterations, batch_size, logits_to_keep]
                # set all zero values to 1
                per_token_avg_ps_dezero = per_token_avg_ps.clone()
                per_token_avg_ps_dezero[per_token_avg_ps_dezero == 0] = 1
                loss_mask = (per_token_avg_ps > 0).bool()

            reward_mask_expanded = reward_mask.unsqueeze(0).expand(num_iterations, -1)
            per_seq_logps_positive = per_seq_logps.mean(dim=2) # [num_iterations, batch_size]
            if self.logp_estimation == 'eubo':
                per_seq_logps_negative = (per_token_avg_ps_dezero.log() * loss_mask).sum(dim=2) / loss_mask.sum(dim=2).clamp(min=1e-8) / self.eubo_beta
            elif self.logp_estimation == 'mix':
                per_seq_logps_negative = mix_weight * (per_token_avg_ps_dezero.log() * loss_mask).sum(dim=2) / loss_mask.sum(dim=2).clamp(min=1e-8) / self.eubo_beta + (1-mix_weight) * per_seq_logps.mean(dim=2)
            elif self.logp_estimation == 'elbo':
                per_seq_logps_negative = per_seq_logps.mean(dim=2)
            elif self.logp_estimation == 'zero':
                per_seq_logps_negative = torch.zeros_like(per_seq_logps_positive)
            else:
                raise ValueError(f"self.logp_estimation: {self.logp_estimation} is not supported")

            per_seq_logps = reward_mask_expanded.float() * per_seq_logps_positive + (~reward_mask_expanded).float() * per_seq_logps_negative
            assert per_seq_logps.shape == (num_iterations, batch_size), f"per_seq_logps.shape: {per_seq_logps.shape}"


        entropy = None
        if calculate_entropy:
            probs = log_probs.exp()
            entropy = -probs * log_probs  # (bs, response_length) entropy of each token

        return entropy, per_seq_logps, loss_per_sample


    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
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

        if "advantages" not in data.batch:
            raise ValueError(
                "SPG compute_log_prob requires advantages to select the positive or "
                "negative estimator. Compute it after reward/advantage generation."
            )
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "advantages", "perturbed_seq", "mask_indices", "p_mask"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        loss_per_sample_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, per_seq_logps, loss_per_sample = self._forward_micro_batch(micro_batch, temperature=temperature, iter_idx=range(self.num_iterations), mc_num=self.mc_num, calculate_entropy=calculate_entropy, call_fn_name="compute_log_prob")

            batch_size = micro_batch["input_ids"].size(0)
            per_seq_logps = per_seq_logps.transpose(0, 1).contiguous()
            loss_per_sample = loss_per_sample.reshape(batch_size, self.num_iterations, self.mc_num)

            log_probs_lst.append(per_seq_logps)
            loss_per_sample_lst.append(loss_per_sample)
            if calculate_entropy:
                entropy_lst.append(entropy.reshape(batch_size, self.num_iterations, -1))

        log_probs = torch.concat(log_probs_lst, dim=0)
        loss_per_sample = torch.concat(loss_per_sample_lst, dim=0)

        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"

            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
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

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "advantages", "perturbed_seq", "mask_indices", "p_mask"]

        if multi_turn:
            select_keys.append("loss_mask")
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
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
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

                    avg_loss = 0.0

                    for i in range(self.num_iterations):
                        entropy, log_prob, _ = self._forward_micro_batch(micro_batch=data, temperature=temperature,
                                                                         mc_num=self.mc_num, iter_idx=[i], logp_estimation=self.logp_estimation,
                                                                         calculate_entropy=calculate_entropy, call_fn_name="update_policy")


                        # use_SPG use RL loss
                        pg_loss, _, _, _ = compute_policy_loss_spg(
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                        print(f"pg_loss: {pg_loss}")


                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation

                        loss.backward()

                        avg_loss += loss.detach().item()

                    data = {
                        "actor/pg_loss": avg_loss / self.num_iterations,
                    }
                    # data = {
                    #     "actor/pg_loss": pg_loss.detach().item(),
                    #     "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    #     "actor/ppo_kl": ppo_kl.detach().item(),
                    #     "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    # }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()  # Update gradients after each mini-batch
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()  # Clear gradient accumulation
        return metrics
