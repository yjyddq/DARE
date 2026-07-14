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
d-TreeRPO Actor for Dream (Masked Diffusion LM with shifted logits).
Inherits from the LLaDA d-TreeRPO actor and overrides _get_logits for Dream's shift behavior.
"""

import logging
import os

import torch
from torch import nn

from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.llada_dp_actor_dtreerpo import DLLMDataParallelPPOActor as BaseDataParallelPPOActor
from verl.workers.actor.mdlm_sp_utils import get_packed_logits
from verl.utils.device import is_cuda_available, is_npu_available

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor", "BaseDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(BaseDataParallelPPOActor):
    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
        """
        Dream model requires logits shift: shift_logits = cat([logits[:, 0:1], logits[:, :-1]], dim=1)
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
