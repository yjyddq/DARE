# Copyright 2026 Shanghai AI Lab
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
"""Shared SP-aware packed logits helper for LLaDA/Dream MDLM actors."""

import torch

from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs


def get_packed_logits(
    actor,
    model,
    packed_input,
    cu_seqlens,
    max_seqlen,
    prompt_len,
    cfg_scale=0.0,
    mask_token_id=126336,
    shift_logits=False,
):
    """Return full packed logits while slicing model input under Ulysses SP.

    LLaDA/Dream RL actors pack valid tokens into shape ``(1, total_seqlen)``.
    Their SP attention patches expect this packed sequence to be sharded before
    model forward and then gathered back before the algorithm-specific loss.
    """

    def forward_one(input_ids):
        if not getattr(actor, "use_ulysses_sp", False):
            logits = model(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
            return logits[:, : input_ids.shape[1]]

        if input_ids.size(0) != 1:
            raise ValueError("MDLM Ulysses sequence parallelism requires packed_input batch size 1.")

        sp_size = actor.ulysses_sequence_parallel_size
        local_input_ids, _, pad_size = ulysses_pad_and_slice_inputs(input_ids, None, sp_size=sp_size)
        local_logits = model(local_input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
        logits = gather_outpus_and_unpad(
            local_logits,
            gather_dim=1,
            unpad_dim=1,
            padding_size=pad_size,
        )
        return logits[:, : input_ids.shape[1]]

    if cfg_scale > 0.0:
        un_packed_input = packed_input.clone()
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            un_packed_input[0, start : start + prompt_len[i].item()] = mask_token_id

        if getattr(actor, "use_ulysses_sp", False):
            logits = forward_one(packed_input)
            un_logits = forward_one(un_packed_input)
        else:
            packed_input_cat = torch.cat([packed_input, un_packed_input], dim=0)
            cu_seqlens_cat = torch.cat([cu_seqlens, cu_seqlens[1:] + cu_seqlens[-1]], dim=0)
            logits = model(packed_input_cat, cu_seqlens=cu_seqlens_cat, max_seqlen=max_seqlen).logits
            logits, un_logits = torch.chunk(logits[:, : packed_input.shape[1]], 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = forward_one(packed_input)

    if shift_logits:
        logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()
    return logits
