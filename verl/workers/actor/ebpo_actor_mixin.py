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

"""Shared composite-ratio update path for block-diffusion EBPO actors."""

import logging
import os

import torch

from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import (
    compute_ebpo_composite_elbo,
    compute_ebpo_kl,
    compute_policy_loss_ebpo,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


class EBPOActorMixin:
    """Apply PPO once to the all-block, all-timestep composite ELBO ratio."""

    @staticmethod
    def _composite_elbo(token_elbos: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        if token_elbos.ndim != 3:
            raise ValueError(f"EBPO token ELBOs must have shape [B, N, R], got {token_elbos.shape}")
        if token_elbos.size(0) != response_mask.size(0) or token_elbos.size(2) != response_mask.size(1):
            raise ValueError(
                f"EBPO token ELBOs {token_elbos.shape} do not match response mask {response_mask.shape}"
            )
        return compute_ebpo_composite_elbo(
            token_elbos,
            contribution_mask=response_mask[:, None, :],
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        if self.config.entropy_coeff != 0:
            raise NotImplementedError(
                "EBPO exposes a composite ELBO, not a categorical token entropy; "
                "set actor.entropy_coeff=0."
            )

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_loss_per_sample",
            "advantages",
            "perturbed_seq",
            "mask_indices",
            "p_mask",
        ]
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
        for _ in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_data in micro_batches:
                    if isinstance(micro_data, DataProto):
                        micro_data = {
                            **micro_data.batch.to(get_torch_device().current_device()),
                            **micro_data.non_tensor_batch,
                        }
                    else:
                        micro_data = micro_data.to(get_torch_device().current_device())

                    responses = micro_data["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_data["attention_mask"]
                    if multi_turn:
                        response_mask = micro_data["loss_mask"][:, -response_length:].bool()
                    else:
                        response_mask = attention_mask[:, -response_length:].bool()
                    sample_mask = response_mask.any(dim=-1)

                    perturbed_seq = micro_data["perturbed_seq"]
                    mask_indices = micro_data["mask_indices"]
                    p_mask = micro_data["p_mask"]
                    if perturbed_seq.ndim != 3:
                        raise ValueError(
                            f"EBPO perturbations must have shape [B, N, S], got {perturbed_seq.shape}"
                        )
                    mc_num = perturbed_seq.size(1)
                    if mc_num != self.mc_num:
                        raise ValueError(
                            f"EBPO data has mc_num={mc_num}, actor is configured with mc_num={self.mc_num}"
                        )

                    old_token_elbos = micro_data["old_loss_per_sample"]
                    if old_token_elbos.shape != (responses.size(0), mc_num, response_length):
                        raise ValueError(
                            "EBPO old_loss_per_sample must have shape [B, N, R], "
                            f"got {old_token_elbos.shape}"
                        )
                    old_composite_elbo = self._composite_elbo(old_token_elbos, response_mask)

                    advantages = micro_data["advantages"]
                    if advantages.shape == response_mask.shape:
                        sequence_advantages = _masked_sequence_mean(advantages.float(), response_mask)
                    elif advantages.shape == sample_mask.shape:
                        sequence_advantages = advantages.float()
                    else:
                        raise ValueError(
                            f"EBPO advantages must have shape {response_mask.shape} or {sample_mask.shape}, "
                            f"got {advantages.shape}"
                        )

                    # Evaluate the complete composite objective without retaining
                    # all N model graphs. The graph is replayed one timestep at a
                    # time after differentiating through this scalar proxy.
                    with torch.no_grad():
                        _, _, current_token_elbos = self._forward_micro_batch(
                            micro_batch=micro_data,
                            temperature=temperature,
                            n_l=1,
                            mc_num=mc_num,
                            calculate_entropy=False,
                            call_fn_name="update_policy_value",
                        )
                    current_composite_elbo = self._composite_elbo(current_token_elbos, response_mask)
                    composite_proxy = current_composite_elbo.detach().requires_grad_(True)

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_ebpo(
                        old_l_theta=old_composite_elbo,
                        l_theta=composite_proxy,
                        advantages=sequence_advantages,
                        response_mask=sample_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    policy_loss = pg_loss

                    kl_loss = None
                    if self.config.use_kl_loss:
                        ref_log_probs = micro_data["ref_log_probs"]
                        if ref_log_probs.shape == response_mask.shape:
                            ref_composite_elbo = _masked_sequence_mean(ref_log_probs.float(), response_mask)
                        elif ref_log_probs.shape == sample_mask.shape:
                            ref_composite_elbo = ref_log_probs.float()
                        else:
                            raise ValueError(
                                f"EBPO ref_log_probs must have shape {response_mask.shape} or {sample_mask.shape}, "
                                f"got {ref_log_probs.shape}"
                            )
                        per_sample_kl = compute_ebpo_kl(
                            sequence_elbo=composite_proxy,
                            ref_sequence_elbo=ref_composite_elbo,
                            kl_estimator=self.config.kl_loss_type,
                        )
                        kl_loss = per_sample_kl[sample_mask].mean()
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        backward_scale = responses.size(0) / self.config.ppo_mini_batch_size
                    else:
                        backward_scale = 1.0 / self.gradient_accumulation
                    proxy_gradient = torch.autograd.grad(
                        policy_loss * backward_scale,
                        composite_proxy,
                    )[0]

                    for mc_index in range(mc_num):
                        mc_batch = {
                            **micro_data,
                            "perturbed_seq": perturbed_seq[:, mc_index : mc_index + 1],
                            "mask_indices": mask_indices[:, mc_index : mc_index + 1],
                            "p_mask": p_mask[:, mc_index : mc_index + 1],
                        }
                        _, _, mc_token_elbos = self._forward_micro_batch(
                            micro_batch=mc_batch,
                            temperature=temperature,
                            n_l=1,
                            mc_num=1,
                            calculate_entropy=False,
                            call_fn_name="update_policy_backward",
                        )
                        mc_composite_elbo = self._composite_elbo(mc_token_elbos, response_mask)
                        surrogate_loss = (mc_composite_elbo * proxy_gradient.detach()).sum() / mc_num
                        surrogate_loss.backward()

                    data_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/ebpo_old_elbo": old_composite_elbo[sample_mask].mean().item(),
                        "actor/ebpo_new_elbo": current_composite_elbo[sample_mask].mean().item(),
                        "actor/ebpo_log_ratio": (
                            current_composite_elbo[sample_mask] - old_composite_elbo[sample_mask]
                        ).mean().item(),
                    }
                    if kl_loss is not None:
                        data_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        data_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    append_to_dict(metrics, data_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
