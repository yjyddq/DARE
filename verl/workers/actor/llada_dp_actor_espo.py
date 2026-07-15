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
Single Process Actor for ESPO on LLaDA-style diffusion language models.
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn

from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import (
    compute_espo_kl,
    compute_espo_sequence_elbo,
    compute_policy_loss_espo,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.mdlm_sp_utils import get_packed_logits

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy."""
        super().__init__(config, actor_module, actor_optimizer)

        self.MASK_TOKEN_ID = actor_module.config.mask_token_id
        self.PAD_TOKEN_ID = actor_module.config.pad_token_id
        self.mc_num = int(config.get("mc_num", 1))
        self.num_iterations = int(config.get("num_iterations", 1))
        self.n_l = int(config.get("n_l", 1))
        self.cfg_scale = config.get("cfg_scale", 0.0)
        self.espo_reduce_var = config.get("espo_reduce_var", config.get("reduce_var", True))

        if self.mc_num < 1:
            raise ValueError(f"ESPO mc_num must be positive, got {self.mc_num}")
        if self.num_iterations < 1:
            raise ValueError(f"ESPO num_iterations must be positive, got {self.num_iterations}")
        if self.n_l != 1:
            raise ValueError(
                "ESPO does not use BGPO's n_l grouping; set actor/ref n_l=1 "
                f"instead of {self.n_l}."
            )
        if config.get("entropy_coeff", 0.0) != 0:
            raise ValueError("ESPO does not define an entropy regularizer; set entropy_coeff=0.")

        if actor_optimizer is not None and self.mc_num > 1:
            dropout_fields = ("attention_dropout", "residual_dropout", "embedding_dropout")
            nonzero_dropout = {
                name: getattr(actor_module.config, name)
                for name in dropout_fields
                if float(getattr(actor_module.config, name, 0.0)) != 0.0
            }
            if nonzero_dropout:
                raise ValueError(
                    "ESPO's memory-bounded MC recomputation requires deterministic learner forwards; "
                    f"disable model dropout first: {nonzero_dropout}"
                )

    def _pack_and_get_logits(self, sequences, attention_mask, prompt_length):
        batch_size = sequences.size(0)
        device = sequences.device
        packed_sequences = []
        cu_seqlens = [0]
        max_seqlen = 0
        for b in range(batch_size):
            valid_tokens = sequences[b][attention_mask[b] == 1]
            packed_sequences.append(valid_tokens)
            cu_seqlens.append(cu_seqlens[-1] + len(valid_tokens))
            max_seqlen = max(max_seqlen, len(valid_tokens))

        packed_sequences = torch.cat(packed_sequences, dim=0).unsqueeze(0)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        prompt_lens = attention_mask[:, :prompt_length].sum(dim=1)
        logits = self._get_logits(
            model=self.actor_module,
            packed_input=packed_sequences,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            prompt_len=prompt_lens,
            cfg_scale=self.cfg_scale,
            MASK_TOKEN_ID=self.MASK_TOKEN_ID,
        )
        return logits, cu_seqlens

    def _restore_logits_for_sample(self, logits, cu_seqlens, attention_mask, sample_idx, seq_length):
        start, end = cu_seqlens[sample_idx], cu_seqlens[sample_idx + 1]
        logits_b = torch.zeros(seq_length, logits.size(-1), device=logits.device, dtype=logits.dtype)
        logits_b[attention_mask[sample_idx] == 1] = logits[0, start:end]
        return logits_b

    def _score_sequence_elbo(
        self,
        seq,
        attention_mask,
        cur_perturbed_seq,
        cur_mask_indices,
        cur_p_mask,
        prompt_length,
        reduce_var,
    ):
        batch_size, seq_length = seq.shape
        valid_response_mask = attention_mask.bool().clone()
        valid_response_mask[:, :prompt_length] = False

        logits, cu_seqlens = self._pack_and_get_logits(cur_perturbed_seq, attention_mask, prompt_length)
        coupled_logits = None
        coupled_cu_seqlens = None
        if reduce_var:
            coupled_mask_indices = valid_response_mask & (~cur_mask_indices)
            if coupled_mask_indices.any():
                coupled_perturbed_seq = seq.clone()
                coupled_perturbed_seq[coupled_mask_indices] = self.MASK_TOKEN_ID
                coupled_logits, coupled_cu_seqlens = self._pack_and_get_logits(
                    coupled_perturbed_seq,
                    attention_mask,
                    prompt_length,
                )

        sequence_elbos = []
        for b in range(batch_size):
            logits_b = self._restore_logits_for_sample(logits, cu_seqlens, attention_mask, b, seq_length)
            coupled_logits_b = None
            if coupled_logits is not None:
                coupled_logits_b = self._restore_logits_for_sample(
                    coupled_logits,
                    coupled_cu_seqlens,
                    attention_mask,
                    b,
                    seq_length,
                )
            sequence_elbos.append(
                compute_espo_sequence_elbo(
                    logits=logits_b,
                    targets=seq[b],
                    mask_indices=cur_mask_indices[b],
                    p_mask=cur_p_mask[b],
                    valid_response_mask=valid_response_mask[b],
                    coupled_logits=coupled_logits_b,
                    reduce_var=reduce_var,
                )
            )

        return torch.stack(sequence_elbos)

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        n_l,
        mc_num,
        calculate_entropy=False,
        call_fn_name="",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate ESPO ELBO estimates for one optimization iteration.

        Returns:
            entropy: (bs, response_len) or None, the official ``-ELBO / R`` metric
            log_probs: (bs, response_len), MC-averaged raw sequence ELBO
            loss_per_sample: (bs, mc_num, response_len), per-MC ``ELBO / R``
        """
        del temperature, n_l, call_fn_name  # Learner ELBO is not temperature-scaled in official ESPO.
        batch_size, seq_length = micro_batch["input_ids"].size(0), micro_batch["input_ids"].size(-1)
        response_length = micro_batch["responses"].size(-1)
        prompt_length = seq_length - response_length

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            seq = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]

            if perturbed_seq.ndim != 3:
                raise ValueError(
                    "_forward_micro_batch expects one ESPO iteration with shape [B, M, S], "
                    f"got {perturbed_seq.shape}"
                )
            if perturbed_seq.shape[:2] != (batch_size, mc_num):
                raise ValueError(
                    f"Expected perturbed_seq [B={batch_size}, M={mc_num}, S], got {perturbed_seq.shape}"
                )

            mc_sequence_elbos = []
            for i in range(mc_num):
                mc_sequence_elbos.append(
                    self._score_sequence_elbo(
                        seq=seq,
                        attention_mask=attention_mask,
                        cur_perturbed_seq=perturbed_seq[:, i, :],
                        cur_mask_indices=mask_indices[:, i, :],
                        cur_p_mask=p_mask[:, i, :],
                        prompt_length=prompt_length,
                        reduce_var=self.espo_reduce_var,
                    )
                )
            mc_sequence_elbos = torch.stack(mc_sequence_elbos, dim=1)

            sequence_elbo = mc_sequence_elbos.mean(dim=1)
            log_prob = sequence_elbo.unsqueeze(-1).expand(-1, response_length)
            loss_per_sample = (mc_sequence_elbos / response_length).unsqueeze(-1).expand(
                -1,
                -1,
                response_length,
            ).contiguous()

        entropy = None
        if calculate_entropy:
            entropy = -log_prob / response_length

        return entropy, log_prob, loss_per_sample

    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
        """
        packed_input: (1, total_seqlen)
        cu_seqlens: (batch_size + 1,)
        max_seqlen: int
        prompt_len: (batch_size,) true prompt length of each sample
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
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "perturbed_seq", "mask_indices", "p_mask"]
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

            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            if perturbed_seq.ndim == 3:
                perturbed_seq = perturbed_seq.unsqueeze(1)
                mask_indices = mask_indices.unsqueeze(1)
                p_mask = p_mask.unsqueeze(1)
            if perturbed_seq.ndim != 4:
                raise ValueError(f"Expected ESPO perturbations [B, I, M, S], got {perturbed_seq.shape}")

            num_iterations = perturbed_seq.size(1)
            mc_num = perturbed_seq.size(2)
            if mc_num != self.mc_num:
                raise ValueError(f"ESPO data has mc_num={mc_num}, actor is configured with mc_num={self.mc_num}")

            iteration_log_probs = []
            iteration_entropys = []
            iteration_losses = []
            with torch.no_grad():
                for iteration in range(num_iterations):
                    iteration_batch = {
                        **micro_batch,
                        "perturbed_seq": perturbed_seq[:, iteration],
                        "mask_indices": mask_indices[:, iteration],
                        "p_mask": p_mask[:, iteration],
                    }
                    entropy, log_prob, loss_per_sample = self._forward_micro_batch(
                        iteration_batch,
                        temperature=temperature,
                        n_l=1,
                        mc_num=mc_num,
                        calculate_entropy=calculate_entropy,
                        call_fn_name="compute_log_prob",
                    )
                    iteration_log_probs.append(log_prob)
                    iteration_losses.append(loss_per_sample)
                    if calculate_entropy:
                        iteration_entropys.append(entropy)

            log_prob_lst.append(torch.stack(iteration_log_probs, dim=1))
            loss_per_sample_lst.append(torch.stack(iteration_losses, dim=1))
            if calculate_entropy:
                entropy_lst.append(torch.stack(iteration_entropys, dim=1))

        log_probs = torch.concat(log_prob_lst, dim=0)
        loss_per_sample = torch.concat(loss_per_sample_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=log_probs.device)
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
            "old_loss_per_sample",
            "advantages",
            "perturbed_seq",
            "mask_indices",
            "p_mask",
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_loss_per_sample")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        perturbed_ndim = batch["perturbed_seq"].ndim
        if perturbed_ndim not in (3, 4):
            raise ValueError(
                "ESPO update expects perturbed_seq [B, I, M, S] (or legacy [B, M, S]), "
                f"got {batch['perturbed_seq'].shape}"
            )
        num_iterations = batch["perturbed_seq"].shape[1] if perturbed_ndim == 4 else 1
        if num_iterations != self.num_iterations:
            raise ValueError(
                f"ESPO data has num_iterations={num_iterations}, actor is configured with "
                f"num_iterations={self.num_iterations}"
            )

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            # Official ESPO performs one optimizer update per iteration while
            # reusing the same rollout and a distinct precomputed mask set.
            for iteration in range(num_iterations):
                for mini_batch in dataloader:
                    if has_multi_modal_inputs:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        num_micro_batches = (
                            mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        )
                        micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(
                            num_micro_batches
                        )
                    elif self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        micro_batches, _ = rearrange_micro_batches(
                            batch=mini_batch,
                            max_token_len=max_token_len,
                        )
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
                            response_mask = micro_data["loss_mask"][:, -response_length:]
                        else:
                            response_mask = attention_mask[:, -response_length:]
                        response_mask_float = response_mask.to(dtype=torch.float32)
                        response_den = response_mask_float.sum(dim=-1).clamp_min(1.0)

                        perturbed_seq = micro_data["perturbed_seq"]
                        mask_indices = micro_data["mask_indices"]
                        p_mask = micro_data["p_mask"]
                        if perturbed_seq.ndim == 3:
                            perturbed_seq = perturbed_seq.unsqueeze(1)
                            mask_indices = mask_indices.unsqueeze(1)
                            p_mask = p_mask.unsqueeze(1)
                        mc_num = perturbed_seq.size(2)
                        if mc_num != self.mc_num:
                            raise ValueError(
                                f"ESPO data has mc_num={mc_num}, actor is configured with mc_num={self.mc_num}"
                            )

                        iteration_batch = {
                            **micro_data,
                            "perturbed_seq": perturbed_seq[:, iteration],
                            "mask_indices": mask_indices[:, iteration],
                            "p_mask": p_mask[:, iteration],
                        }

                        old_loss_per_sample = micro_data["old_loss_per_sample"]
                        if old_loss_per_sample.ndim == 3:
                            old_loss_per_sample = old_loss_per_sample.unsqueeze(1)
                        old_l_theta = old_loss_per_sample[:, iteration].mean(dim=1)
                        advantages = micro_data["advantages"]

                        # First obtain the official MC-averaged objective value
                        # without retaining M full model graphs.
                        with torch.no_grad():
                            _, _, current_mc_l_theta = self._forward_micro_batch(
                                micro_batch=iteration_batch,
                                temperature=temperature,
                                n_l=1,
                                mc_num=mc_num,
                                calculate_entropy=False,
                                call_fn_name="update_policy_value",
                            )
                        current_l_theta = current_mc_l_theta.mean(dim=1)
                        current_seq_logp = (
                            current_l_theta.float() * response_mask_float
                        ).sum(dim=-1) / response_den
                        old_seq_logp = (
                            old_l_theta.float() * response_mask_float
                        ).sum(dim=-1) / response_den

                        # Differentiate the scalar ESPO objective with respect
                        # to its MC-averaged normalized ELBO proxy.
                        sequence_proxy = current_seq_logp.detach().requires_grad_(True)
                        proxy_l_theta = sequence_proxy.unsqueeze(-1).expand_as(old_l_theta)
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
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_espo(
                            old_l_theta=old_l_theta,
                            l_theta=proxy_l_theta,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        policy_loss = pg_loss
                        kl_loss = None
                        if self.config.use_kl_loss:
                            ref_loss_per_sample = micro_data["ref_loss_per_sample"]
                            if ref_loss_per_sample.ndim == 3:
                                ref_loss_per_sample = ref_loss_per_sample.unsqueeze(1)
                            ref_l_theta = ref_loss_per_sample[:, iteration].mean(dim=1)
                            ref_seq_logp = (
                                ref_l_theta.float() * response_mask_float
                            ).sum(dim=-1) / response_den
                            kl_loss = compute_espo_kl(
                                sequence_elbo=sequence_proxy * response_length,
                                ref_sequence_elbo=ref_seq_logp * response_length,
                                normalization_length=response_length,
                                kl_estimator=self.config.kl_loss_type,
                            ).mean()
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            backward_scale = responses.size(0) / self.config.ppo_mini_batch_size
                        else:
                            backward_scale = 1.0 / self.gradient_accumulation
                        proxy_gradient = torch.autograd.grad(
                            policy_loss * backward_scale,
                            sequence_proxy,
                        )[0]

                        # Recompute one MC graph at a time. The resulting model
                        # gradient is exactly dL/d(mean_m ELBO_m), with bounded
                        # activation memory for deterministic LLaDA forwards.
                        for mc_index in range(mc_num):
                            mc_batch = {
                                **iteration_batch,
                                "perturbed_seq": iteration_batch["perturbed_seq"][:, mc_index : mc_index + 1],
                                "mask_indices": iteration_batch["mask_indices"][:, mc_index : mc_index + 1],
                                "p_mask": iteration_batch["p_mask"][:, mc_index : mc_index + 1],
                            }
                            _, _, mc_l_theta = self._forward_micro_batch(
                                micro_batch=mc_batch,
                                temperature=temperature,
                                n_l=1,
                                mc_num=1,
                                calculate_entropy=False,
                                call_fn_name="update_policy_backward",
                            )
                            mc_seq_logp = (
                                mc_l_theta[:, 0].float() * response_mask_float
                            ).sum(dim=-1) / response_den
                            surrogate_loss = (mc_seq_logp * proxy_gradient.detach()).sum() / mc_num
                            surrogate_loss.backward()

                        data_metrics = {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            "actor/espo_ratio": torch.exp(current_seq_logp - old_seq_logp).mean().item(),
                            "actor/espo_old_elbo": old_seq_logp.mean().item(),
                            "actor/espo_new_elbo": current_seq_logp.mean().item(),
                            "actor/espo_iteration": float(iteration),
                        }
                        if kl_loss is not None:
                            data_metrics["actor/kl_loss"] = kl_loss.detach().item()
                            data_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        append_to_dict(metrics, data_metrics)

                    grad_norm = self._optimizer_step()
                    append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
