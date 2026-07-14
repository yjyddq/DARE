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
MDPO Actor for LLaDA model.
Uses the full diffusion trajectory collected during rollout for per-step training
with PPO-clipped loss, lambda_t scaling, and confidence weighting.
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
from verl.trainer.ppo.mdpo_algos import compute_mdpo_policy_loss
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
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
        self.cfg_scale = config["cfg_scale"]

        # MDPO specific parameters
        self.mdpo_epsilon = config.get("mdpo_epsilon", 0.2)
        self.mdpo_beta = config.get("mdpo_beta", 0.02)
        self.sample_train_steps = config.get("sample_train_steps", 16)

        # Keep mc_num/n_l for compatibility with compute_log_prob interface
        self.mc_num = config.get("mc_num", 1)
        self.n_l = config.get("n_l", 1)

    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
        """
        Get logits from model with optional classifier-free guidance.
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

    def _forward_micro_batch_mdpo(self, micro_batch, temperature=1.0):
        """
        Compute per-token log probabilities for MDPO training.

        Input micro_batch contains:
            - input_ids: (bs, seq_len) - prompt + corrupted completion (the diffusion step input)
            - target_ids: (bs, seq_len) - prompt + denoised completion (the predicted tokens)
            - attention_mask: (bs, seq_len) - attention mask
            - completion_mask: (bs, response_len) - mask for valid completion tokens

        Returns:
            per_token_logps: (bs, response_len) log probabilities of target tokens at completion positions
        """
        batch_size = micro_batch["input_ids"].size(0)
        seq_length = micro_batch["input_ids"].size(-1)
        response_length = micro_batch["completion_mask"].size(-1)
        prompt_length = seq_length - response_length
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]  # (bs, seq_len)
            target_ids = micro_batch["target_ids"]  # (bs, seq_len)
            attention_mask = micro_batch["attention_mask"]  # (bs, seq_len)

            # Pack sequences for efficient computation
            packed_input = []
            cu_seqlens = [0]
            max_seqlen = 0
            prompt_lens = []
            for b in range(batch_size):
                valid_tokens = input_ids[b][attention_mask[b] == 1]
                packed_input.append(valid_tokens)
                cu_seqlens.append(cu_seqlens[-1] + len(valid_tokens))
                max_seqlen = max(max_seqlen, len(valid_tokens))
                prompt_lens.append(attention_mask[b, :prompt_length].sum().item())
            packed_input = torch.cat(packed_input, dim=0).unsqueeze(0)
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            prompt_lens_tensor = torch.tensor(prompt_lens, device=device)

            # Get logits
            logits = self._get_logits(
                model=self.actor_module,
                packed_input=packed_input,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                prompt_len=prompt_lens_tensor,
                cfg_scale=0.0,
                MASK_TOKEN_ID=self.MASK_TOKEN_ID,
            )

            # Compute per-token log probs for the completion region
            per_token_logps = torch.zeros(batch_size, response_length, device=device, dtype=torch.float32)
            for b in range(batch_size):
                start, end = cu_seqlens[b], cu_seqlens[b + 1]
                logits_b = torch.zeros(seq_length, logits.size(-1), device=device, dtype=logits.dtype)
                logits_b[attention_mask[b] == 1] = logits[0, start:end]

                # Only compute log probs for completion region
                completion_logits = logits_b[prompt_length:]  # (response_len, vocab_size)
                completion_targets = target_ids[b, prompt_length:]  # (response_len,)
                log_probs = F.log_softmax(completion_logits, dim=-1)
                per_token_logps[b] = log_probs.gather(-1, completion_targets.unsqueeze(-1)).squeeze(-1)

        return per_token_logps

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """
        Compute log probability for MDPO. This is called during the initial old_log_prob computation.
        For MDPO, we compute log probs for each selected diffusion step.

        The data should contain:
            - mdpo_step_input_ids: (batch_size, seq_len) input at the selected step
            - mdpo_step_target_ids: (batch_size, seq_len) target at the selected step
            - completion_mask: (batch_size, response_len) mask for valid completion tokens
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info.get("micro_batch_size", 1)
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)

        select_keys = ["mdpo_step_input_ids", "mdpo_step_target_ids", "attention_mask", "completion_mask"]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_prob_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            # Rename keys for _forward_micro_batch_mdpo
            mb = {
                "input_ids": micro_batch["mdpo_step_input_ids"],
                "target_ids": micro_batch["mdpo_step_target_ids"],
                "attention_mask": micro_batch["attention_mask"],
                "completion_mask": micro_batch["completion_mask"],
            }
            with torch.no_grad():
                per_token_logps = self._forward_micro_batch_mdpo(mb)
            log_prob_lst.append(per_token_logps)

        log_probs = torch.concat(log_prob_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        # Return format compatible with existing interface
        # entropy and loss_per_sample are not needed for MDPO, return placeholders
        entropys = torch.zeros_like(log_probs)
        loss_per_sample = log_probs.unsqueeze(1)  # (batch_size, 1, response_length) for compatibility
        return entropys, log_probs, loss_per_sample

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """
        Update policy using MDPO per-step training.

        data contains:
            - mdpo_step_input_ids: (bs, seq_len) - prompt + corrupted completion at selected step
            - mdpo_step_target_ids: (bs, seq_len) - prompt + denoised completion at selected step
            - attention_mask: (bs, seq_len)
            - completion_mask: (bs, response_len) - mask for valid completion tokens
            - advantages: (bs,) - step-wise advantages
            - confidence: (bs, response_len) - confidence scores
            - old_per_token_logps: (bs, response_len) - old log probs from compute_log_prob
            - ref_per_token_logps: (bs, response_len) - ref log probs (optional)
        """
        self.actor_module.train()

        temperature = data.meta_info.get("temperature", 1.0)

        select_keys = [
            "mdpo_step_input_ids", "mdpo_step_target_ids",
            "attention_mask", "completion_mask",
            "advantages", "confidence", "old_per_token_logps",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_per_token_logps")
        batch = data.select(batch_keys=select_keys).batch

        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for mb_data in micro_batches:
                    if isinstance(mb_data, DataProto):
                        mb_data = {**mb_data.batch.to(get_torch_device().current_device()), **mb_data.non_tensor_batch}
                    else:
                        mb_data = mb_data.to(get_torch_device().current_device())

                    completion_mask = mb_data["completion_mask"]
                    advantages = mb_data["advantages"]
                    confidence = mb_data["confidence"]
                    old_per_token_logps = mb_data["old_per_token_logps"]

                    response_length = completion_mask.size(-1)

                    # Forward pass to get current log probs
                    micro_batch_input = {
                        "input_ids": mb_data["mdpo_step_input_ids"],
                        "target_ids": mb_data["mdpo_step_target_ids"],
                        "attention_mask": mb_data["attention_mask"],
                        "completion_mask": completion_mask,
                    }
                    per_token_logps = self._forward_micro_batch_mdpo(micro_batch_input)

                    # Get ref log probs if available
                    ref_per_token_logps = mb_data.get("ref_per_token_logps", None)

                    # Compute MDPO loss
                    pg_loss, pg_clipfrac, ppo_kl = compute_mdpo_policy_loss(
                        per_token_logps=per_token_logps,
                        old_per_token_logps=old_per_token_logps,
                        advantages=advantages,
                        completion_mask=completion_mask,
                        confidence=confidence,
                        max_completion_length=response_length,
                        epsilon=self.mdpo_epsilon,
                        beta=self.mdpo_beta if self.config.use_kl_loss else 0.0,
                        ref_per_token_logps=ref_per_token_logps,
                    )

                    if self.config.use_dynamic_bsz:
                        loss = pg_loss * (len(mb_data["advantages"]) / self.config.ppo_mini_batch_size)
                    else:
                        loss = pg_loss / self.gradient_accumulation

                    print(f"MDPO loss: {loss.item():.6f}")
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
