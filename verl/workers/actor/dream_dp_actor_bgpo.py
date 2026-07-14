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
from verl.trainer.ppo.dllm_core_algos import agg_loss, compute_policy_loss_bgpo, kl_penalty  # NOTE: Our core algorithms
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.llada_dp_actor_bgpo import DLLMDataParallelPPOActor as BaseDataParallelPPOActor
from verl.workers.actor.mdlm_sp_utils import get_packed_logits
import torch.nn.functional as F

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor", "BaseDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(BaseDataParallelPPOActor):
    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=151666):
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
            shift_logits=True,
        )
