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
LLaDA2.x EBPO actor.

Like the LLaDA2 BGPO path, EBPO uses the official vectorized block-diffusion
forward over ``[noisy_x, clean_x]``. Each forward returns the summed ELBO
contributions from every response-overlapping block.
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.fsdp_utils import FSDPModule
from verl.workers.actor.ebpo_actor_mixin import EBPOActorMixin
from verl.workers.actor.llada2_dp_actor_bgpo import DLLMDataParallelPPOActor as BGPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(EBPOActorMixin, BGPOActor):
    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name=""):
        batch_size, seq_length = micro_batch["input_ids"].size(0), micro_batch["input_ids"].size(-1)
        response_length = micro_batch["responses"].size(-1)
        prompt_section_length = seq_length - response_length
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            seq = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            loss_per_token = torch.zeros((batch_size, mc_num, response_length), device=device)
            for i in range(mc_num):
                cur_perturbed_seq = perturbed_seq[:, i, :]
                cur_mask_indices = mask_indices[:, i, :]
                cur_p_mask = p_mask[:, i, :]

                compact_noisy_seq, compact_clean_seq, compact_valid_mask, compact_target_mask, compact_p_mask, compact_position_ids = self._compact_batch(
                    noisy_seq=cur_perturbed_seq,
                    clean_seq=seq,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cur_mask_indices=cur_mask_indices,
                    cur_p_mask=cur_p_mask,
                )
                block_attention_mask = self._build_block_attention_mask(compact_valid_mask)
                full_input_ids = torch.cat((compact_noisy_seq, compact_clean_seq), dim=1)
                full_position_ids = torch.cat((compact_position_ids, compact_position_ids), dim=1)

                logits = self.actor_module(
                    input_ids=full_input_ids,
                    attention_mask=block_attention_mask,
                    position_ids=full_position_ids,
                    return_dict=True,
                ).logits[:, : compact_noisy_seq.size(1), :]

                for b in range(batch_size):
                    cur_len = int(compact_valid_mask[b].sum().item())
                    valid_logits = logits[b, :cur_len]
                    valid_targets = compact_clean_seq[b, :cur_len]
                    valid_target_mask = compact_target_mask[b, :cur_len]
                    valid_p_mask = compact_p_mask[b, :cur_len]

                    if not valid_target_mask.any():
                        continue

                    compact_prompt_len = int(attention_mask[b, :prompt_section_length].sum().item())
                    compact_response_len = int(attention_mask[b, prompt_section_length:].sum().item())
                    if compact_response_len <= 0:
                        continue

                    response_start = compact_prompt_len
                    response_end = compact_prompt_len + compact_response_len
                    response_target_mask = valid_target_mask[response_start:response_end]
                    if not response_target_mask.any():
                        continue

                    response_logits = valid_logits[response_start:response_end]
                    response_targets = valid_targets[response_start:response_end]
                    response_p_mask = valid_p_mask[response_start:response_end]

                    selected_p_mask = response_p_mask[response_target_mask]
                    if (selected_p_mask <= 0).any():
                        raise ValueError("EBPO masked-token probabilities must be positive")
                    token_losses = -(
                        F.cross_entropy(
                            response_logits[response_target_mask],
                            response_targets[response_target_mask],
                            reduction="none",
                        )
                        / selected_p_mask
                    )
                    target_positions = torch.nonzero(response_target_mask, as_tuple=False).flatten()
                    loss_per_token[b, i, target_positions] = token_losses

            log_likelihood = loss_per_token.mean(dim=1).sum(dim=-1)
            log_prob = log_likelihood.unsqueeze(-1).expand(-1, response_length) # (batch_size, response_length)

        entropy = None
        if calculate_entropy:
            entropy = -log_prob.exp() * log_prob

        return entropy, log_prob, loss_per_token

    def _manual_clip_grad_norm_(self, parameters, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        params = [param for param in parameters if param.grad is not None]
        if len(params) == 0:
            return torch.zeros((), device=self.device_name)

        if norm_type != 2.0:
            raise NotImplementedError("LLaDA2 SFT manual grad clip currently only supports L2 norm.")

        local_sq_norm = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)
        for param in params:
            grad = param.grad.detach()
            local_sq_norm += torch.sum(grad.float() * grad.float())

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_sq_norm, op=torch.distributed.ReduceOp.SUM)

        total_norm = torch.sqrt(local_sq_norm)
        max_norm = float(max_norm)
        if max_norm > 0:
            clip_coef = max_norm / (total_norm.item() + 1e-6)
            if clip_coef < 1.0:
                for param in params:
                    param.grad.mul_(clip_coef)
        return total_norm

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            # llada2 hits the same FSDP2 grad-clip issue we saw in SFT, so keep
            # the manual global-norm path but read the canonical PPO actor key.
            grad_norm = self._manual_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm
