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
LLaDA2.x BGPO actor.

Unlike the original packed LLaDA actor path, LLaDA2 should follow the same
block-diffusion training semantics validated in SFT:
1. compact each sample to its valid tokens,
2. build `[noisy_x, clean_x]`,
3. apply the official block-diffusion 4D mask,
4. score masked positions on the noisy half against clean-token targets.
"""

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor.block_diffusion_utils import (
    BlockDiffusionArtifacts,
    build_block_diffusion_mask,
    build_full_block_diffusion_tensors,
    compact_block_diffusion_artifacts,
    pad_block_diffusion_loss_tensors,
)
from verl.workers.actor.llada_dp_actor_bgpo import DLLMDataParallelPPOActor as BaseDataParallelPPOActor


class DLLMDataParallelPPOActor(BaseDataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        self.block_length = int(config.get("block_length", 32))
        if "block_origin" not in config:
            raise ValueError(
                "LLaDA2 actor requires an explicit block_origin resolved from "
                "the rollout backend"
            )
        self.block_origin = config["block_origin"]
        # A block-diffusion actor model processes [noisy, clean], not L tokens.
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
        """Return positive NLL/p on compact noisy positions, globally over SP."""

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
            local_targets = slice_input_tensor(full_targets, dim=1, padding=False)
            local_p_mask = slice_input_tensor(full_p_mask, dim=1, padding=False)
        else:
            local_targets = full_targets
            local_p_mask = full_p_mask

        outputs = self.actor_module(
            input_ids=local_input_ids,
            attention_mask=block_attention_mask,
            position_ids=local_position_ids,
            use_cache=False,
            return_dict=True,
            block_diffusion_targets=local_targets,
            block_diffusion_p_mask=local_p_mask,
            block_diffusion_answer_len=artifacts.response_lengths.sum(),
            return_block_diffusion_token_loss=True,
        )
        local_token_losses = outputs.block_diffusion_token_loss.float()
        if local_token_losses.shape != local_targets.shape:
            raise RuntimeError(
                "LLaDA2 token-loss shape mismatch: "
                f"{local_token_losses.shape} vs {local_targets.shape}"
            )
        token_losses = gather_outpus_and_unpad(
            local_token_losses,
            gather_dim=1,
            unpad_dim=1,
            padding_size=pad_size,
        )
        noisy_token_losses = token_losses[:, : artifacts.sequence_length]
        return noisy_token_losses * artifacts.target_mask.float()

    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name=""):
        batch_size = micro_batch["input_ids"].size(0)
        response_length = micro_batch["responses"].size(-1)
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            mc_num = perturbed_seq.size(1)
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
            log_prob = log_likelihood.unsqueeze(-1).expand(-1, response_length)
            response_mask = micro_batch["attention_mask"][:, -response_length:].bool()
            response_count = response_mask.sum(dim=-1).clamp_min(1).to(loss_per_sample.dtype)
            loss_per_sample = loss_per_sample.unsqueeze(-1) / response_count[:, None, None]
            loss_per_sample = loss_per_sample.expand(-1, -1, response_length).contiguous()
            loss_per_sample = loss_per_sample * response_mask[:, None, :]

        entropy = None
        if calculate_entropy:
            prob = log_prob.exp()
            entropy = -prob * log_prob

        return entropy, log_prob, loss_per_sample

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
