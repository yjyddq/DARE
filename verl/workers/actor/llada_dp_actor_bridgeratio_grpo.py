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
BridgeRatio-GRPO actor for masked diffusion LMs.
"""

import logging
import os

from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import agg_loss, compute_policy_loss_bridgeratio, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.llada_dp_actor_coupled_grpo import DLLMDataParallelPPOActor as CoupledDataParallelPPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(CoupledDataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        self.bridge_ratio_correction = config.get("bridge_ratio_correction", "none")
        self.bridge_ratio_detach_correction = config.get("bridge_ratio_detach_correction", True)
        self.bridge_ratio_score_scale = config.get("bridge_ratio_score_scale", "token")
        self.bridge_ratio_log_clip = config.get("bridge_ratio_log_clip", None)

    def _scale_bridge_path_scores(self, path_scores, response_mask, response_length):
        """Map DARE's normalized per-path scores to the requested ratio scale."""
        score_scale = str(self.bridge_ratio_score_scale).lower()
        if score_scale in ("token", "normalized", "none"):
            return path_scores
        if score_scale in ("sequence", "seq", "response_length"):
            return path_scores * response_length
        if score_scale in ("valid_tokens", "valid_length"):
            valid_lengths = response_mask.sum(dim=-1).clamp(min=1).view(-1, 1, 1)
            return path_scores * valid_lengths
        raise ValueError(f"Unsupported bridge_ratio_score_scale: {self.bridge_ratio_score_scale}")

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

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
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    if isinstance(micro_batch, DataProto):
                        micro_batch = {**micro_batch.batch.to(get_torch_device().current_device()), **micro_batch.non_tensor_batch}
                    else:
                        micro_batch = micro_batch.to(get_torch_device().current_device())

                    responses = micro_batch["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_batch["attention_mask"]
                    if multi_turn:
                        response_mask = micro_batch["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    advantages = micro_batch["advantages"]
                    old_path_scores = micro_batch["old_loss_per_sample"][:, :, -response_length:]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = entropy_coeff != 0

                    entropy, log_probs, path_scores = self._forward_micro_batch(
                        micro_batch=micro_batch,
                        temperature=temperature,
                        n_l=1,
                        mc_num=micro_batch["perturbed_seq"].shape[1],
                        calculate_entropy=calculate_entropy,
                        call_fn_name="update_policy",
                    )
                    log_probs = log_probs[:, -response_length:]
                    path_scores = path_scores[:, :, -response_length:]
                    old_path_scores = self._scale_bridge_path_scores(old_path_scores, response_mask, response_length)
                    path_scores = self._scale_bridge_path_scores(path_scores, response_mask, response_length)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, bridge_metrics = compute_policy_loss_bridgeratio(
                        old_l_theta_paths=old_path_scores,
                        l_theta_paths=path_scores,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                        correction=self.bridge_ratio_correction,
                        detach_correction=self.bridge_ratio_detach_correction,
                        ratio_log_clip=self.bridge_ratio_log_clip,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(
                            loss_mat=entropy[:, -response_length:],
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_probs = micro_batch["ref_log_probs"][:, -response_length:]
                        kld = kl_penalty(
                            l_theta=log_probs,
                            ref_l_theta=ref_log_probs,
                            kl_penalty=self.config.kl_loss_type,
                            advantages=advantages,
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (responses.size(0) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    metric_data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    metric_data.update({f"actor/{key}": value.detach().item() for key, value in bridge_metrics.items()})
                    append_to_dict(metrics, metric_data)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
