# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Ulysses sequence parallelism patch for DREAM models.

DREAM calls flash_attn_varlen_func directly (not via HuggingFace's _flash_attention_forward),
so the generic monkey patch in monkey_patch.py does not apply. This module provides a patched
DreamFlashAttention.forward that inserts all-to-all communication for Ulysses SP.
"""

import sys
from typing import Optional, Tuple

import torch

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)


def validate_dream_sp_attention_backend(model_config, ulysses_sp_size: int, role: str) -> None:
    """Ensure Dream SP uses the attention class patched by this module."""
    if str(getattr(model_config, "model_type", "")).lower() != "dream" or ulysses_sp_size <= 1:
        return

    attn_implementation = getattr(model_config, "_attn_implementation", None)
    if attn_implementation != "flash_attention_2":
        raise ValueError(
            "Dream Ulysses sequence parallelism requires "
            f"attn_implementation='flash_attention_2' for the {role} model, "
            f"but got {attn_implementation!r}."
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to query and key tensors.

    cos/sin shape: (1, seq_len, head_dim) from DreamRotaryEmbedding.
    After unsqueeze(1): (1, 1, seq_len, head_dim), broadcastable to (bsz, n_heads, seq_len, head_dim).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query_len, key_len = q.shape[-2], k.shape[-2]
    q_embed = (q * cos[:, :, key_len - query_len : key_len, :]) + (
        rotate_half(q) * sin[:, :, key_len - query_len : key_len, :]
    )
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads for GQA.
    Input:  (batch, num_kv_heads, seqlen, head_dim)
    Output: (batch, num_kv_heads * n_rep, seqlen, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def apply_dream_ulysses_patch(model):
    """Patch DreamFlashAttention.forward with Ulysses SP-aware version.

    Uses a closure to preserve the original forward for the non-SP fallback path,
    so that all original behavior (output_attentions fallback, dual_cache, etc.) is retained.
    """
    module = sys.modules[model.__module__]
    _original_forward = module.DreamFlashAttention.forward

    from flash_attn import flash_attn_varlen_func

    def dream_ulysses_flash_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dual_cache: Optional[bool] = False,
        replace_position: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Patched DreamFlashAttention.forward with Ulysses SP support.

        Non-SP path (sp_size == 1 or cu_seqlens is None): delegates to original forward.
        SP path:
          1. QKV projection
          2. GQA repeat K/V to n_heads (before all-to-all)
          3. All-to-all #1: gather full sequence, scatter heads
          4. RoPE on full sequence using self.rotary_emb (correct positions for all ranks)
          5. Flash attn varlen with cu_seqlens
          6. All-to-all #2: gather heads, scatter sequence back
          7. Output projection
        """
        sp_size = get_ulysses_sequence_parallel_world_size()

        # Non-SP path: use original implementation
        if sp_size <= 1 or cu_seqlens is None:
            return _original_forward(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                dual_cache=dual_cache,
                replace_position=replace_position,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        # === SP path ===
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "Sequence parallelism requires batch_size == 1 (packed sequences)"

        # QKV projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (bsz, n_heads, q_len, head_dim) / (bsz, n_kv_heads, q_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # GQA repeat to n_heads BEFORE all-to-all (DO NOT apply RoPE yet)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Contiguous for all-to-all
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # All-to-all #1: gather full sequence, scatter heads
        # (1, n_heads, T_local, head_dim) -> (1, n_heads/sp, T_full, head_dim)
        total_nnz = cu_seqlens[-1].item()
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1, unpadded_dim_size=total_nnz)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1, unpadded_dim_size=total_nnz)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1, unpadded_dim_size=total_nnz)

        # Apply RoPE on FULL sequence (positions [0..T_full-1] are correct for all ranks)
        full_position_ids = torch.arange(total_nnz, device=query_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(query_states, full_position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Flash attn varlen
        # (1, n_heads/sp, T_full, head_dim) -> (T_full, n_heads/sp, head_dim)
        q_flat = query_states.squeeze(0).permute(1, 0, 2).contiguous()
        k_flat = key_states.squeeze(0).permute(1, 0, 2).contiguous()
        v_flat = value_states.squeeze(0).permute(1, 0, 2).contiguous()

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout_p,
            causal=False,
        )  # (T_full, n_heads/sp, head_dim)

        # Reshape back: (T_full, n_heads/sp, head_dim) -> (1, n_heads/sp, T_full, head_dim)
        attn_output = attn_output.unsqueeze(0).transpose(1, 2)

        # All-to-all #2: gather heads, scatter sequence
        # (1, n_heads/sp, T_full, head_dim) -> (1, n_heads, T_local, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2)

        # Reshape back to (bsz, q_len, hidden_size) and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None

    module.DreamFlashAttention.forward = dream_ulysses_flash_attention_forward
    print("Monkey patch DreamFlashAttention.forward for DREAM Ulysses SP")
