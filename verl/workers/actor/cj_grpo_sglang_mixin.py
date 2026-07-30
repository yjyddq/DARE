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
"""Shared CJ-GRPO PPO loop for SGLang-backed dLLM actors."""

import itertools
import logging
import os

import torch

from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor.block_diffusion_utils import (
    build_block_parallel_cj_step_inputs,
    build_cj_replay_model_inputs,
    cj_mass_isclose,
    compute_cj_block_step_token_weights,
    scatter_compact_values_to_response,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CJGRPOActorMixin:
    """Replay block-local CJ trajectories with one actor forward per local step.

    The concrete LLaDA2/SDAR base supplies the block-diffusion artifact builder
    and token-loss forward.  Keeping the CJ state reconstruction here prevents
    the two model families from drifting apart.
    """

    @staticmethod
    def _materialize_cj_actor_batch(batch):
        """Return a private training view containing the complete replay block."""

        required_replay_keys = {
            "cj_replay_responses",
            "cj_replay_attention_mask",
        }
        missing = sorted(required_replay_keys - set(batch.keys()))
        if missing:
            raise KeyError(
                "CJ terminal-block replay requires rollout fields "
                f"{missing}"
            )

        (
            replay_input_ids,
            replay_attention_mask,
            replay_position_ids,
            replay_response_mask,
            prompt_section_length,
        ) = build_cj_replay_model_inputs(
            input_ids=batch["input_ids"],
            responses=batch["responses"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            replay_responses=batch["cj_replay_responses"],
            replay_attention_mask=batch["cj_replay_attention_mask"],
        )
        trajectory = batch["reversed_traj_unmask_positions"]
        expected_trajectory_shape = (
            replay_input_ids.size(0),
            trajectory.size(1),
            replay_input_ids.size(1),
        )
        if trajectory.shape != expected_trajectory_shape:
            raise ValueError(
                "CJ trajectory must align with the complete replay sequence, got "
                f"{tuple(trajectory.shape)} and expected "
                f"{expected_trajectory_shape}"
            )

        # ``TensorDict.clone()`` recursively clones every selected tensor,
        # including the large [B, K, S] trajectory.  Only the container needs
        # to be private because the four model-view fields below are replaced.
        actor_batch = (
            batch.clone(recurse=False) if hasattr(batch, "clone") else dict(batch)
        )
        actor_batch["responses"] = batch["cj_replay_responses"]
        actor_batch["input_ids"] = replay_input_ids
        actor_batch["attention_mask"] = replay_attention_mask
        actor_batch["position_ids"] = replay_position_ids
        if "loss_mask" in actor_batch:
            # Terminal-block suffix tokens are latent actions under the strict
            # on-policy CJ objective even though ordinary decoding stops at EOS.
            actor_batch["loss_mask"] = replay_attention_mask

        if actor_batch["input_ids"].size(1) - actor_batch["responses"].size(
            1
        ) != prompt_section_length:
            raise RuntimeError("CJ replay actor view changed the prompt section")
        if not torch.equal(
            actor_batch["attention_mask"][:, prompt_section_length:].bool(),
            replay_response_mask,
        ):
            raise RuntimeError("CJ replay response mask materialization failed")
        return actor_batch

    def _build_cj_step_inputs(self, micro_batch, step_idx: int):
        response_length = micro_batch["responses"].size(-1)
        noisy_input_ids, target_mask, prompt_length = (
            build_block_parallel_cj_step_inputs(
                clean_input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                trajectory=micro_batch["reversed_traj_unmask_positions"],
                response_length=response_length,
                local_step=step_idx,
                mask_token_id=self.MASK_TOKEN_ID,
            )
        )
        return noisy_input_ids, target_mask, prompt_length, response_length

    def _step_forward_micro_batch(
        self,
        micro_batch,
        step_idx,
        temperature,
        calculate_entropy=False,
        call_fn_name="",
    ):
        del temperature, call_fn_name
        noisy_input_ids, target_mask, _prompt_length, response_length = (
            self._build_cj_step_inputs(micro_batch, step_idx)
        )
        p_mask = torch.ones_like(noisy_input_ids, dtype=torch.float32)

        # Never skip this model call.  FSDP ranks can have different active
        # blocks for the same local step, but their collective schedule must be
        # identical.  The model patches and scatter helper preserve a connected
        # zero graph when this rank has no target.
        autocast_enabled = self.device_name != "cpu"
        with torch.autocast(
            device_type=self.device_name,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            artifacts = self._build_block_diffusion_artifacts(
                micro_batch=micro_batch,
                noisy_input_ids=noisy_input_ids,
                target_mask=target_mask,
                p_mask=p_mask,
            )
            token_losses = self._compute_block_diffusion_token_losses(artifacts)
            log_probs = scatter_compact_values_to_response(
                -token_losses,
                artifacts,
                response_length=response_length,
            )

        entropy = None
        if calculate_entropy:
            probability = log_probs.exp()
            entropy = -probability * log_probs
        return entropy, log_probs, log_probs

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        call_fn_name="",
    ):
        trajectory = micro_batch["reversed_traj_unmask_positions"]
        steps = trajectory.size(1)
        if steps <= 0:
            raise ValueError("CJ trajectory must contain at least one local step")

        response_length = micro_batch["responses"].size(1)
        response_mask = micro_batch["attention_mask"][:, -response_length:].bool()
        compute_cj_block_step_token_weights(
            trajectory,
            response_mask,
            micro_batch["attention_mask"],
            prompt_section_length=micro_batch["input_ids"].size(1) - response_length,
            block_size=self.block_length,
            block_origin=self.block_origin,
        )

        entropy_per_step = []
        log_probs_per_step = []
        loss_per_sample_per_step = []
        for step_idx in range(steps):
            entropy, log_probs, loss_per_sample = self._step_forward_micro_batch(
                micro_batch=micro_batch,
                step_idx=step_idx,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
                call_fn_name=call_fn_name,
            )
            log_probs_per_step.append(log_probs)
            loss_per_sample_per_step.append(loss_per_sample)
            if calculate_entropy:
                entropy_per_step.append(entropy)

        entropy = torch.stack(entropy_per_step, dim=1) if calculate_entropy else None
        return (
            entropy,
            torch.stack(log_probs_per_step, dim=1),
            torch.stack(loss_per_sample_per_step, dim=1),
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self, data: DataProto, calculate_entropy=False
    ) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "cj_replay_responses",
            "cj_replay_attention_mask",
            "reversed_traj_unmask_positions",
        ]
        batch = self._materialize_cj_actor_batch(
            data.select(batch_keys=select_keys).batch
        )
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            selected_data = data.select(select_keys, non_tensor_select_keys)
            selected_data.batch = batch
            micro_batches = selected_data.chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = (
                data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            )
            micro_batches, indices = rearrange_micro_batches(
                batch=batch,
                max_token_len=max_token_len,
                token_cost_multiplier=getattr(
                    self, "model_input_token_cost_multiplier", 1
                ),
                padded_batch=getattr(self, "model_input_uses_padded_batch", False),
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
                entropy, log_probs, loss_per_sample = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    call_fn_name="compute_log_prob",
                )
            log_probs_lst.append(log_probs)
            loss_per_sample_lst.append(loss_per_sample)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        loss_per_sample = torch.concat(loss_per_sample_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(
                0
            ), f"{len(indices)} vs. {log_probs.size()}"
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

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "cj_replay_responses",
            "cj_replay_attention_mask",
            "old_log_probs",
            "advantages",
            "reversed_traj_unmask_positions",
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_probs")
        batch = self._materialize_cj_actor_batch(
            data.select(batch_keys=select_keys).batch
        )
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        loss_agg_mode = self.config.loss_agg_mode
        if loss_agg_mode != "token-mean":
            raise ValueError(
                "CJ-GRPO requires actor.loss_agg_mode='token-mean' so its "
                "block-step objective is invariant to micro-batch partitioning"
            )

        if has_multi_modal_inputs:
            num_mini_batches = (
                data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            )
            non_tensor_select_keys = ["multi_modal_inputs"]
            selected_data = data.select(select_keys, non_tensor_select_keys)
            selected_data.batch = batch
            dataloader = selected_data.chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _epoch in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                mini_batch_tensors = (
                    mini_batch.batch
                    if isinstance(mini_batch, DataProto)
                    else mini_batch
                )
                mini_response_length = mini_batch_tensors["responses"].size(1)
                if multi_turn:
                    mini_response_mask = mini_batch_tensors["loss_mask"][
                        :, -mini_response_length:
                    ].bool()
                else:
                    mini_response_mask = mini_batch_tensors["attention_mask"][
                        :, -mini_response_length:
                    ].bool()
                mini_prompt_section_length = (
                    mini_batch_tensors["input_ids"].size(1) - mini_response_length
                )
                _, mini_batch_step_masses = compute_cj_block_step_token_weights(
                    mini_batch_tensors["reversed_traj_unmask_positions"],
                    mini_response_mask,
                    mini_batch_tensors["attention_mask"],
                    prompt_section_length=mini_prompt_section_length,
                    block_size=self.block_length,
                    block_origin=self.block_origin,
                )
                mini_batch_sample_count = mini_batch_tensors["responses"].size(0)
                mini_batch_mass = float(mini_batch_step_masses.sum().item())
                if not cj_mass_isclose(
                    mini_batch_mass,
                    float(mini_batch_sample_count),
                ):
                    raise RuntimeError(
                        "CJ block-step weights do not reconstruct one objective "
                        f"per sample: {mini_batch_mass} vs. {mini_batch_sample_count}"
                    )

                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = (
                        mini_batch.batch.batch_size[0]
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.select(
                        select_keys, non_tensor_select_keys
                    ).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch,
                        max_token_len=max_token_len,
                        token_cost_multiplier=getattr(
                            self, "model_input_token_cost_multiplier", 1
                        ),
                        padded_batch=getattr(
                            self, "model_input_uses_padded_batch", False
                        ),
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()

                accumulated_pg_loss = 0.0
                accumulated_pg_clipfrac = 0.0
                accumulated_ppo_kl = 0.0
                accumulated_pg_clipfrac_lower = 0.0
                accumulated_kl_loss = 0.0
                accumulated_step_weight = 0.0
                trained_steps = 0

                for micro_batch in micro_batches:
                    if isinstance(micro_batch, DataProto):
                        micro_batch = {
                            **micro_batch.batch.to(get_torch_device().current_device()),
                            **micro_batch.non_tensor_batch,
                        }
                    else:
                        micro_batch = micro_batch.to(
                            get_torch_device().current_device()
                        )

                    responses = micro_batch["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_batch["attention_mask"]
                    if multi_turn:
                        base_response_mask = micro_batch["loss_mask"][
                            :, -response_length:
                        ].bool()
                    else:
                        base_response_mask = attention_mask[:, -response_length:].bool()

                    old_log_probs = micro_batch["old_log_probs"]
                    advantages = micro_batch["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low
                        if self.config.clip_ratio_low is not None
                        else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high
                        if self.config.clip_ratio_high is not None
                        else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    calculate_entropy = entropy_coeff != 0

                    traj_mask = micro_batch["reversed_traj_unmask_positions"].bool()
                    steps = traj_mask.shape[1]
                    if old_log_probs.shape != (
                        responses.size(0),
                        steps,
                        response_length,
                    ):
                        raise ValueError(
                            "CJ old_log_probs must have shape [B, K, R], got "
                            f"{tuple(old_log_probs.shape)}"
                        )
                    if self.config.use_kl_loss:
                        ref_log_probs = micro_batch["ref_log_probs"]
                        if ref_log_probs.shape != old_log_probs.shape:
                            raise ValueError(
                                "CJ ref_log_probs must match old_log_probs with "
                                "shape [B, K, R], got "
                                f"{tuple(ref_log_probs.shape)} and "
                                f"{tuple(old_log_probs.shape)}"
                            )
                    if advantages.shape != (responses.size(0), response_length):
                        raise ValueError(
                            "CJ advantages must have shape [B, R], got "
                            f"{tuple(advantages.shape)}"
                        )

                    prompt_section_length = (
                        micro_batch["input_ids"].size(1) - response_length
                    )
                    step_token_weights, local_step_masses = (
                        compute_cj_block_step_token_weights(
                            traj_mask,
                            base_response_mask,
                            attention_mask,
                            prompt_section_length=prompt_section_length,
                            block_size=self.block_length,
                            block_origin=self.block_origin,
                        )
                    )
                    local_step_mass_values = local_step_masses.tolist()

                    for step_idx in range(steps):
                        entropy, log_probs, _loss_per_sample = (
                            self._step_forward_micro_batch(
                                micro_batch=micro_batch,
                                step_idx=step_idx,
                                temperature=temperature,
                                calculate_entropy=calculate_entropy,
                                call_fn_name="update_policy",
                            )
                        )

                        token_weights = step_token_weights[:, step_idx, :]
                        reduction_step_mass = float(token_weights.sum().item())
                        expected_step_mass = local_step_mass_values[step_idx]
                        if not cj_mass_isclose(
                            reduction_step_mass,
                            expected_step_mass,
                        ):
                            raise RuntimeError(
                                "CJ token weights and step mass disagree: "
                                f"{reduction_step_mass} vs. {expected_step_mass}"
                            )
                        objective_weight = (
                            expected_step_mass / mini_batch_sample_count
                        )
                        if reduction_step_mass > 0:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                                compute_policy_loss(
                                    old_l_theta=old_log_probs[:, step_idx, :],
                                    l_theta=log_probs,
                                    advantages=advantages,
                                    response_mask=token_weights,
                                    cliprange=clip_ratio,
                                    cliprange_low=clip_ratio_low,
                                    cliprange_high=clip_ratio_high,
                                    clip_ratio_c=clip_ratio_c,
                                    loss_agg_mode=loss_agg_mode,
                                )
                            )

                            if entropy_coeff != 0:
                                entropy_loss = agg_loss(
                                    loss_mat=entropy,
                                    loss_mask=token_weights,
                                    loss_agg_mode=loss_agg_mode,
                                )
                                policy_loss = pg_loss - entropy_loss * entropy_coeff
                            else:
                                policy_loss = pg_loss

                            if self.config.use_kl_loss:
                                kld = kl_penalty(
                                    l_theta=log_probs,
                                    ref_l_theta=ref_log_probs[:, step_idx, :],
                                    kl_penalty=self.config.kl_loss_type,
                                    advantages=advantages,
                                )
                                kl_loss = agg_loss(
                                    loss_mat=kld,
                                    loss_mask=token_weights,
                                    loss_agg_mode=loss_agg_mode,
                                )
                                policy_loss = (
                                    policy_loss + kl_loss * self.config.kl_loss_coef
                                )
                            # ``masked_mean`` uses denominator ``mask.sum()+1e-8``.
                            # Include the same guard in the outer coefficient so
                            # the composed reduction is exactly sum(weighted
                            # transition losses) / mini_batch_sample_count.
                            aggregation_weight = (
                                reduction_step_mass + 1e-8
                            ) / mini_batch_sample_count
                        else:
                            # Still traverse the complete model graph and call
                            # backward so every FSDP rank executes one fixed
                            # slot for every local step.
                            policy_loss = log_probs.sum() * 0.0
                            pg_loss = policy_loss.detach()
                            pg_clipfrac = policy_loss.detach()
                            ppo_kl = policy_loss.detach()
                            pg_clipfrac_lower = policy_loss.detach()
                            if self.config.use_kl_loss:
                                kl_loss = policy_loss.detach()
                            aggregation_weight = 0.0

                        loss = policy_loss * aggregation_weight
                        loss.backward()

                        trained_steps += 1
                        accumulated_step_weight += objective_weight
                        accumulated_pg_loss += (
                            pg_loss.detach().item() * aggregation_weight
                        )
                        accumulated_pg_clipfrac += (
                            pg_clipfrac.detach().item() * aggregation_weight
                        )
                        accumulated_ppo_kl += (
                            ppo_kl.detach().item() * aggregation_weight
                        )
                        accumulated_pg_clipfrac_lower += (
                            pg_clipfrac_lower.detach().item() * aggregation_weight
                        )
                        if self.config.use_kl_loss:
                            accumulated_kl_loss += (
                                kl_loss.detach().item() * aggregation_weight
                            )

                        del (
                            loss,
                            policy_loss,
                            pg_loss,
                            pg_clipfrac,
                            ppo_kl,
                            pg_clipfrac_lower,
                        )

                if trained_steps == 0:
                    raise RuntimeError(
                        "CJ mini-batch produced no trainable local steps"
                    )
                if not cj_mass_isclose(accumulated_step_weight, 1.0):
                    raise RuntimeError(
                        "CJ micro-batches did not reconstruct one complete mini-batch "
                        f"objective; total weight={accumulated_step_weight}"
                    )

                metric_data = {
                    "actor/pg_loss": accumulated_pg_loss,
                    "actor/pg_clipfrac": accumulated_pg_clipfrac,
                    "actor/ppo_kl": accumulated_ppo_kl,
                    "actor/pg_clipfrac_lower": accumulated_pg_clipfrac_lower,
                }
                if self.config.use_kl_loss:
                    metric_data["actor/kl_loss"] = accumulated_kl_loss
                    metric_data["actor/kl_coef"] = self.config.kl_loss_coef
                append_to_dict(metrics, metric_data)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
