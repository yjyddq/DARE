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
BridgeRatio-GRPO actor for Dream models.
"""

import torch

from verl.workers.actor.llada_dp_actor_bridgeratio_grpo import DLLMDataParallelPPOActor as BaseDataParallelPPOActor

__all__ = ["DataParallelPPOActor", "BaseDataParallelPPOActor"]


class DLLMDataParallelPPOActor(BaseDataParallelPPOActor):
    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
        """
        packed_input: (1, total_seqlen)
        cu_seqlens: (batch_size+1,)
        max_seqlen: int
        prompt_len: (batch_size,) True prompt length of each sample
        """
        if cfg_scale > 0.:
            un_packed_input = packed_input.clone()
            for i in range(len(cu_seqlens) - 1):
                start = cu_seqlens[i].item()
                un_packed_input[0, start:start + prompt_len[i].item()] = MASK_TOKEN_ID
            packed_input_cat = torch.cat([packed_input, un_packed_input], dim=0)
            cu_seqlens_cat = torch.cat([cu_seqlens, cu_seqlens[1:] + cu_seqlens[-1]], dim=0)
            logits = model(packed_input_cat, cu_seqlens=cu_seqlens_cat, max_seqlen=max_seqlen).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(packed_input, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
        logits = logits[:, :packed_input.shape[1]]
        shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()
        return shift_logits
