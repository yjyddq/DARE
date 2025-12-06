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
from verl.trainer.ppo.dllm_core_algos import agg_loss, compute_policy_loss, kl_penalty  # NOTE: Our core algorithms
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import DataParallelPPOActor
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
        self.n_l = config["n_l"]  # Number of random masks
        self.cfg_scale = config["cfg_scale"]  # Whether to use CFG
        
    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name="") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if "multi_modal_inputs" in micro_batch:
            # If there are multi-modal inputs, concatenate the content of each key
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)
        
        # Calculate log_probs
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]  # (bs, mc_num, seq_len)
            mask_indices = micro_batch["mask_indices"]  # (bs, mc_num, seq_len)
            p_mask = micro_batch["p_mask"]  # (bs, mc_num, seq_len)
            seq = micro_batch["input_ids"]  # (bs, seq_len)
            attention_mask = micro_batch["attention_mask"]  # (bs, mc_num, seq_len)

            loss_per_sample = torch.zeros((batch_size, mc_num), device=device)
            for i in range(mc_num):
                cur_perturbed_seq = perturbed_seq[:, i, :]  # (batch_size, seq_len)
                cur_mask_indices = mask_indices[:, i, :]
                cur_p_mask = p_mask[:, i, :]
                
                packed_perturbed_seq = []
                cu_seqlens = [0]
                max_seqlen = 0
                for b in range(batch_size):
                    valid_tokens = cur_perturbed_seq[b][attention_mask[b] == 1]
                    packed_perturbed_seq.append(valid_tokens)
                    cu_seqlens.append(cu_seqlens[-1] + len(valid_tokens))
                    max_seqlen = max(max_seqlen, len(valid_tokens))
                packed_perturbed_seq = torch.cat(packed_perturbed_seq, dim=0).unsqueeze(0)  # (1, total_seqlen)
                cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
                
                logits = self._get_logits(model=self.actor_module, packed_input=packed_perturbed_seq, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, prompt_len=attention_mask[:, :prompt_length].sum(dim=1), cfg_scale=0.0, MASK_TOKEN_ID=self.MASK_TOKEN_ID)
                
                # Restore logits for each sample
                loss = torch.zeros(batch_size, device=device)
                for b in range(batch_size):
                    start, end = cu_seqlens[b], cu_seqlens[b + 1]
                    
                    logits_b = torch.zeros(seq_length, logits.size(-1), device=device, dtype=logits.dtype)
                    logits_b[attention_mask[b] == 1] = logits[0, start:end]
                    
                    mask = cur_mask_indices[b]  # (seq_len,)
                    loss[b] = (F.cross_entropy(logits_b[mask], seq[b][mask], reduction="none") / cur_p_mask[b][mask]).sum()  # cross_entropy returns negative log likelihood
                loss_per_sample[:, i] = -loss  # convert to log likelihood

            log_likelihood = loss_per_sample.sum(dim=1) / mc_num  # (batch_size,)
            log_probs = (log_likelihood / response_length).view(-1, 1).repeat(1, response_length)  # (batch_size, response_length)
            loss_per_sample = (loss_per_sample / response_length).unsqueeze(-1).expand(-1, -1, response_length).contiguous()
        
        entropy = None
        if calculate_entropy:
            probs = log_probs.exp()
            entropy = -probs * log_probs  # (bs, response_length) entropy of each token
            
        return entropy, log_probs, loss_per_sample
    
    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
        """
        packed_input: (1, total_seqlen)
        cu_seqlens: (batch_size+1,)
        max_seqlen: int
        prompt_len: (batch_size,) True prompt length of each sample
        """
        # If CFG is used, fuse conditional and unconditional logits
        if cfg_scale > 0.:
            un_packed_input = packed_input.clone()
            for i in range(len(cu_seqlens) - 1):
                start = cu_seqlens[i].item()
                un_packed_input[0, start:start + prompt_len[i].item()] = MASK_TOKEN_ID  # mask prompt part and get unconditional input
            packed_input_cat = torch.cat([packed_input, un_packed_input], dim=0)  # concatenate conditional and unconditional input
            cu_seqlens_cat = torch.cat([cu_seqlens, cu_seqlens[1:] + cu_seqlens[-1]], dim=0)
            logits = model(packed_input_cat, cu_seqlens=cu_seqlens_cat, max_seqlen=max_seqlen).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)  # split by batch
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)  # fusion formula
        else:
            logits = model(packed_input, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
        return logits[:, :packed_input.shape[1]]

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
                entropy, log_probs, loss_per_sample = self._forward_micro_batch(micro_batch, temperature=temperature, n_l=self.n_l, mc_num=3, calculate_entropy=calculate_entropy, call_fn_name="compute_log_prob")
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
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            loss_per_sample = loss_per_sample[revert_indices]
        return loss_per_sample, entropys  # loss_per_sample is stored in old_log_probs field

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "perturbed_seq", "mask_indices", "p_mask"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
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

                    old_log_prob = data["old_log_probs"]  # (bsz, mc_num)
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
                    
                    input_ids = data["input_ids"]
                    perturbed_seq = data["perturbed_seq"]  # (1, mc_num, seq_len)
                    mask_indices = data["mask_indices"]
                    p_mask = data["p_mask"]
                    mc_num = perturbed_seq.shape[1]
                    for i in range(mc_num):
                        data = {
                            **data,
                            "perturbed_seq": perturbed_seq[:, i:i+1],
                            "mask_indices": mask_indices[:, i:i+1],
                            "p_mask": p_mask[:, i:i+1],
                        }
                        entropy, log_prob, loss_per_sample = self._forward_micro_batch(micro_batch=data, temperature=temperature, n_l=1, mc_num=1, calculate_entropy=calculate_entropy, call_fn_name="update_policy")
                        print(f"\nBGPO loss_per_sample: {loss_per_sample[0, 0, 0]}")
                        
                        # Compute policy loss
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_l_theta=old_log_prob[:, i, :],  # (bsz, response_length)
                            l_theta=loss_per_sample[:, 0, :],  # (bsz, response_length)
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                        print(f"BGPO pg_loss: {pg_loss}")

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:  # NOTE: Currently not considering KL
                            ref_log_prob = data["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(l_theta=loss_per_sample[:, 0, :], ref_l_theta=ref_log_prob[:, i, :], kl_penalty=self.config.kl_loss_type, advantages=advantages)
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
                        print(f"BGPO loss: {loss}\n")
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
