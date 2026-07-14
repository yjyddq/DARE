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
d-TreeRPO Actor for LLaDA (Masked Diffusion LM).
Implements local transition log-prob computation and PPO + self-distillation loss.
"""

import itertools
import logging
import math
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import agg_loss
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.mdlm_sp_utils import get_packed_logits

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

        self.MASK_TOKEN_ID = actor_module.config.mask_token_id
        self.PAD_TOKEN_ID = actor_module.config.pad_token_id

    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
        """
        Get logits from model using packed input format.
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

    def _get_local_transition_logps(self, model, parent_ids, child_ids, attention_mask, prompt_length, cfg_scale=0.0):
        """
        Compute local transition log-probs: only at positions where parent is MASK and child is not MASK.
        Returns per-token log probs of shape (batch_size, completion_length).
        """
        batch_size, seq_len = parent_ids.size()
        device = parent_ids.device
        completion_length = seq_len - prompt_length

        # Pack sequences for efficient model forward
        packed_input = []
        cu_seqlens = [0]
        max_seqlen = 0
        prompt_lens = []
        for b in range(batch_size):
            valid_tokens = parent_ids[b][attention_mask[b] == 1]
            packed_input.append(valid_tokens)
            cu_seqlens.append(cu_seqlens[-1] + len(valid_tokens))
            max_seqlen = max(max_seqlen, len(valid_tokens))
            prompt_lens.append(attention_mask[b, :prompt_length].sum())
        packed_input = torch.cat(packed_input, dim=0).unsqueeze(0)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        prompt_lens = torch.stack(prompt_lens)

        logits = self._get_logits(
            model=model, packed_input=packed_input, cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen, prompt_len=prompt_lens,
            cfg_scale=cfg_scale, MASK_TOKEN_ID=self.MASK_TOKEN_ID
        )

        # Restore logits to padded shape
        all_logps = torch.zeros(batch_size, completion_length, device=device, dtype=torch.float32)
        for b in range(batch_size):
            start, end = cu_seqlens[b], cu_seqlens[b + 1]
            logits_b = torch.zeros(seq_len, logits.size(-1), device=device, dtype=logits.dtype)
            logits_b[attention_mask[b] == 1] = logits[0, start:end]

            comp_logits = logits_b[prompt_length:]  # (completion_length, V)
            comp_child = child_ids[b, prompt_length:]
            comp_parent = parent_ids[b, prompt_length:]

            # Changed mask: parent is MASK, child is not MASK
            changed = (comp_parent == self.MASK_TOKEN_ID) & (comp_child != self.MASK_TOKEN_ID)

            loss_flat = F.cross_entropy(
                comp_logits, comp_child, reduction="none"
            )
            per_token_logps = -loss_flat  # (completion_length,)
            all_logps[b] = per_token_logps * changed.float()

        all_logps = torch.nan_to_num(all_logps, nan=0.0, posinf=0.0, neginf=0.0)
        return all_logps

    def _build_completion_mask(self, completion_ids):
        """Build mask: 1 for unmasked tokens up to (and including) first EOS, 0 elsewhere."""
        device = completion_ids.device
        is_unmasked = (completion_ids != self.MASK_TOKEN_ID).float()

        eos_id = self.PAD_TOKEN_ID  # LLaDA uses pad_token_id as EOS boundary
        is_eos = (completion_ids == eos_id)
        B, L = is_eos.size()
        eos_idx = torch.full((B,), L, dtype=torch.long, device=device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos[has_eos].int().argmax(dim=1)

        seq_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        eos_pos_mask = (seq_idx <= eos_idx.unsqueeze(1)).float()

        return is_unmasked * eos_pos_mask

    def _forward_micro_batch_logps(self, micro_batch, cfg_scale=0.0):
        """Compute local transition log probs for a micro batch."""
        parent_ids = micro_batch["parent_ids"]
        child_ids = micro_batch["child_ids"]
        attention_mask = micro_batch["attention_mask"]
        prompt_length = micro_batch["prompt_length"]

        if isinstance(prompt_length, torch.Tensor):
            prompt_length = prompt_length.item()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            local_logps = self._get_local_transition_logps(
                model=self.actor_module,
                parent_ids=parent_ids,
                child_ids=child_ids,
                attention_mask=attention_mask,
                prompt_length=prompt_length,
                cfg_scale=cfg_scale,
            )
        return local_logps

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute old local transition log probs for d-TreeRPO segments."""
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        cfg_scale = data.meta_info.get("cfg_scale", 0.0)

        select_keys = ["parent_ids", "child_ids", "attention_mask", "prompt_length"]
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)

        logps_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                local_logps = self._forward_micro_batch_logps(micro_batch, cfg_scale=cfg_scale)
            logps_lst.append(local_logps)

        all_logps = torch.concat(logps_lst, dim=0)
        return None, all_logps, None  # (entropy, log_probs, loss_per_sample)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Update policy using PPO loss on local transitions + optional self-distillation."""
        self.actor_module.train()

        select_keys = [
            "parent_ids", "child_ids", "attention_mask", "prompt_length",
            "old_local_logps", "local_advantages", "group_ids",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_local_logps")
        batch = data.select(batch_keys=select_keys).batch

        # d-TreeRPO config
        epsilon = self.config.get("clip_ratio", 0.2)
        kl_coef = self.config.get("kl_loss_coef", 0.0)
        enable_self_distillation = self.config.get("enable_self_distillation", False)
        sd_lambda_max = self.config.get("sd_lambda_max", 3e-3)
        sd_gamma = self.config.get("sd_gamma", 2.0)
        sd_tau_max = self.config.get("sd_tau_max", 2.0)
        sd_beta = self.config.get("sd_beta", 0.7)
        max_steps = self.config.get("max_steps", 30000)
        global_step = data.meta_info.get("global_step", 0)

        cfg_scale = data.meta_info.get("cfg_scale", 0.0)

        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * getattr(self, 'ulysses_sequence_parallel_size', 1)
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_data in micro_batches:
                    micro_data = micro_data.to(get_torch_device().current_device())

                    parent_ids = micro_data["parent_ids"]
                    child_ids = micro_data["child_ids"]
                    attention_mask = micro_data["attention_mask"]
                    prompt_length = micro_data["prompt_length"]
                    if isinstance(prompt_length, torch.Tensor):
                        prompt_length = prompt_length[0].item()

                    old_local_logps = micro_data["old_local_logps"]
                    local_advantages = micro_data["local_advantages"]
                    group_ids = micro_data.get("group_ids", None)

                    # Compute new local transition log probs
                    new_local_logps = self._get_local_transition_logps(
                        model=self.actor_module,
                        parent_ids=parent_ids,
                        child_ids=child_ids,
                        attention_mask=attention_mask,
                        prompt_length=prompt_length,
                        cfg_scale=cfg_scale,
                    )

                    # Build active mask
                    child_completion = child_ids[:, prompt_length:]
                    completion_mask = self._build_completion_mask(child_completion)
                    changed_mask = (new_local_logps != 0).float()
                    active_mask = completion_mask * changed_mask
                    L_eff = active_mask.sum(dim=1).clamp_min(1)

                    # PPO loss
                    ratio = torch.exp(new_local_logps - old_local_logps)
                    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
                    A = local_advantages.unsqueeze(1)  # (B, 1)

                    ppo_obj = torch.min(ratio * A, clipped_ratio * A)
                    ppo_obj_masked = ppo_obj * active_mask
                    policy_loss = -(ppo_obj_masked.sum(dim=1) / L_eff).mean()

                    total_loss = policy_loss

                    # KL penalty
                    if self.config.use_kl_loss and "ref_local_logps" in micro_data.keys():
                        ref_local_logps = micro_data["ref_local_logps"]
                        delta = ref_local_logps - new_local_logps
                        per_token_kl = torch.exp(delta) - delta - 1.0
                        kl_per_sample = (per_token_kl * active_mask).sum(dim=1) / L_eff
                        kl_loss = kl_coef * kl_per_sample.mean()
                        total_loss = total_loss + kl_loss
                        metrics["actor/kl_loss"] = kl_loss.detach().item()

                    # Self-distillation loss
                    sd_loss = torch.tensor(0.0, device=total_loss.device)
                    if enable_self_distillation and group_ids is not None:
                        sd_loss = self._compute_self_distillation_loss(
                            model=self.actor_module,
                            parent_ids=parent_ids,
                            child_ids=child_ids,
                            attention_mask=attention_mask,
                            prompt_length=prompt_length,
                            group_ids=group_ids,
                            local_advantages=local_advantages,
                            active_mask=active_mask,
                            global_step=global_step,
                            max_steps=max_steps,
                            lambda_max=sd_lambda_max,
                            gamma=sd_gamma,
                            tau_max=sd_tau_max,
                            beta=sd_beta,
                            cfg_scale=cfg_scale,
                        )
                        total_loss = total_loss + sd_loss

                    if self.config.use_dynamic_bsz:
                        loss = total_loss * (len(micro_data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = total_loss / self.gradient_accumulation
                    loss.backward()

                    # Metrics
                    with torch.no_grad():
                        clip_hi = verl_F.masked_mean((ratio > (1.0 + epsilon)).float(), active_mask)
                        clip_lo = verl_F.masked_mean((ratio < (1.0 - epsilon)).float(), active_mask)
                    data_metrics = {
                        "actor/pg_loss": policy_loss.detach().item(),
                        "actor/pg_clipfrac_hi": clip_hi.detach().item(),
                        "actor/pg_clipfrac_lo": clip_lo.detach().item(),
                        "actor/sd_loss": sd_loss.detach().item(),
                    }
                    append_to_dict(metrics, data_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

    def _compute_self_distillation_loss(
        self, model, parent_ids, child_ids, attention_mask, prompt_length,
        group_ids, local_advantages, active_mask,
        global_step, max_steps, lambda_max, gamma, tau_max, beta, cfg_scale,
    ):
        """Time-scheduled self-distillation loss that encourages consistency among siblings."""
        device = parent_ids.device

        progress = min(1.0, max(0.0, global_step / max(1, max_steps)))
        lambda_t = lambda_max * (math.exp(gamma * progress) - 1.0) / (math.exp(gamma) - 1.0)
        tau_t = max(tau_max * max(0.0, (1.0 - progress)) ** beta, 1e-6)

        group_ids = group_ids.long()
        uniq_groups = torch.unique(group_ids)
        child_completion = child_ids[:, prompt_length:]
        B_all, Lc = child_completion.size()

        # Find one representative parent per group
        ref_indices = []
        group_to_pos = {}
        for pos, g in enumerate(uniq_groups.tolist()):
            idxs = (group_ids == g).nonzero(as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue
            ref_indices.append(idxs[0])
            group_to_pos[g] = pos

        if len(ref_indices) == 0:
            return torch.tensor(0.0, device=device)

        ref_indices = torch.stack(ref_indices, dim=0)

        # Compute parent logits for reference samples
        ref_parent = parent_ids[ref_indices]
        ref_attn = attention_mask[ref_indices]
        batch_size_ref = ref_parent.size(0)
        seq_len = ref_parent.size(1)

        packed_input = []
        cu_seqlens = [0]
        max_seqlen = 0
        prompt_lens = []
        for b in range(batch_size_ref):
            valid = ref_parent[b][ref_attn[b] == 1]
            packed_input.append(valid)
            cu_seqlens.append(cu_seqlens[-1] + len(valid))
            max_seqlen = max(max_seqlen, len(valid))
            prompt_lens.append(ref_attn[b, :prompt_length].sum())
        packed_input = torch.cat(packed_input, dim=0).unsqueeze(0)
        cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        prompt_lens_t = torch.stack(prompt_lens)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw_logits = self._get_logits(
                model=model, packed_input=packed_input, cu_seqlens=cu_seqlens_t,
                max_seqlen=max_seqlen, prompt_len=prompt_lens_t,
                cfg_scale=cfg_scale, MASK_TOKEN_ID=self.MASK_TOKEN_ID,
            )

        # Restore to padded per-sample logits for completion region
        logp_parent_slice = torch.zeros(batch_size_ref, Lc, raw_logits.size(-1), device=device, dtype=torch.float32)
        for b in range(batch_size_ref):
            start, end = cu_seqlens[b], cu_seqlens[b + 1]
            full_logits = torch.zeros(seq_len, raw_logits.size(-1), device=device, dtype=raw_logits.dtype)
            full_logits[ref_attn[b] == 1] = raw_logits[0, start:end]
            logp_parent_slice[b] = F.log_softmax(full_logits[prompt_length:prompt_length + Lc].float(), dim=-1)

        eps = 1e-8
        consistency_sum = torch.tensor(0.0, device=device)
        consistency_count = 0

        for g in uniq_groups.tolist():
            idxs = (group_ids == g).nonzero(as_tuple=False).squeeze(-1)
            if idxs.numel() <= 1:
                continue

            A_full = local_advantages[idxs].detach()
            A_full = torch.nan_to_num(A_full, nan=0.0, posinf=0.0, neginf=0.0)
            pos_mask = (A_full > 0)
            if pos_mask.sum().item() == 0:
                continue

            A = A_full[pos_mask]
            idxs_pos = idxs[pos_mask]
            w = F.softmax(A / tau_t, dim=0)

            group_active = active_mask[idxs_pos] > 0
            toks_group = child_completion[idxs_pos]
            ref_pos = group_to_pos[g]

            for j in range(Lc):
                active_k = group_active[:, j]
                if not active_k.any():
                    continue
                toks_j = toks_group[active_k, j]
                w_j = w[active_k]

                unique_tokens, inv = torch.unique(toks_j, return_inverse=True)
                q_weights = torch.zeros_like(unique_tokens, dtype=torch.float32)
                q_weights.index_add_(0, inv, w_j.to(q_weights.dtype))
                ws_sum = q_weights.sum()
                if ws_sum <= 0:
                    continue
                q_probs = q_weights / (ws_sum + eps)

                logp_support = logp_parent_slice[ref_pos, j].gather(0, unique_tokens)
                kl_j = torch.sum(q_probs * (torch.log(q_probs + eps) - logp_support))
                consistency_sum = consistency_sum + kl_j
                consistency_count += 1

        if consistency_count > 0:
            return lambda_t * (consistency_sum / float(consistency_count))
        return torch.tensor(0.0, device=device)
