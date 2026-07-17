"""
Ulysses sequence parallelism patch for SDAR models.

SDAR-8B uses a custom diffusion-training forward with flex attention, while
SDAR-MoE calls scaled_dot_product_attention directly. The generic monkey patch
does not cover either implementation, so this module inserts the necessary
all-to-all communication for Ulysses sequence parallelism.
"""

import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


_compiled_flex_attention = torch.compile(flex_attention)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _repeat_kv_for_ulysses(hidden_states: torch.Tensor, sp_size: int) -> torch.Tensor:
    if hidden_states.size(1) % sp_size == 0:
        return hidden_states
    repeats = max(sp_size // hidden_states.size(1), 1)
    hidden_states = repeat_kv(hidden_states, repeats)
    if hidden_states.size(1) % sp_size != 0:
        raise ValueError(
            f"Expanded KV head count {hidden_states.size(1)} is not divisible by the Ulysses SP size {sp_size}."
        )
    return hidden_states


def _all_gather_sequence_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None

    group = get_ulysses_sequence_parallel_group()
    sp_size = get_ulysses_sequence_parallel_world_size(group)
    gathered = [torch.empty_like(attention_mask) for _ in range(sp_size)]
    torch.distributed.all_gather(gathered, attention_mask.contiguous(), group=group)
    return torch.cat(gathered, dim=-1)


def _build_causal_attention_mask(attention_mask: Optional[torch.Tensor], target_length: int, device: torch.device) -> torch.Tensor:
    causal_mask = torch.tril(torch.ones((target_length, target_length), device=device, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    if attention_mask is None:
        return causal_mask

    key_mask = attention_mask[:, None, None, :].bool()
    return causal_mask & key_mask


def apply_sdar_ulysses_patch(model):
    module = sys.modules[model.__module__]
    sp_size = get_ulysses_sequence_parallel_world_size()

    if model.config.model_type == "sdar":
        if not hasattr(module.SDARAttention, "_original_ulysses_forward"):
            module.SDARAttention._original_ulysses_forward = module.SDARAttention.forward
        if not hasattr(module.SDARForCausalLM, "_original_ulysses_forward"):
            module.SDARForCausalLM._original_ulysses_forward = module.SDARForCausalLM.forward

        def sdar_ulysses_attention_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask,
            past_key_value=None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            current_sp_size = get_ulysses_sequence_parallel_world_size()
            # Validation also uses the BlockMask-based SP path; the upstream SDAR
            # attention fallback only accepts tensor masks and will crash on BlockMask.
            is_block_mask = attention_mask is not None and attention_mask.__class__.__name__ == "BlockMask"
            if not is_block_mask and (current_sp_size <= 1 or not self.training):
                return self.__class__._original_ulysses_forward(
                    self,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kwargs,
                )

            if past_key_value is not None:
                raise NotImplementedError("SDAR Ulysses sequence parallelism does not support KV cache during training.")

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = module.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states = _repeat_kv_for_ulysses(key_states, current_sp_size)
            value_states = _repeat_kv_for_ulysses(value_states, current_sp_size)

            global_sequence_length = attention_mask.shape[-1]
            query_states = gather_seq_scatter_heads(
                query_states,
                seq_dim=2,
                head_dim=1,
                unpadded_dim_size=global_sequence_length,
            )
            key_states = gather_seq_scatter_heads(
                key_states,
                seq_dim=2,
                head_dim=1,
                unpadded_dim_size=global_sequence_length,
            )
            value_states = gather_seq_scatter_heads(
                value_states,
                seq_dim=2,
                head_dim=1,
                unpadded_dim_size=global_sequence_length,
            )

            flex_attention_forward = (
                _compiled_flex_attention if query_states.is_cuda else flex_attention
            )
            attn_output = flex_attention_forward(
                query_states,
                key_states,
                value_states,
                block_mask=attention_mask,
                enable_gqa=True,
                scale=self.scaling,
            )
            attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2)
            attn_weights = None

            attn_output = module.rearrange(attn_output, "b h l d -> b l (h d)")
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        def sdar_ulysses_lm_forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask=None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep=0,
            **kwargs,
        ):
            ulysses_sp_training = kwargs.pop("ulysses_sp_training", False)
            return_token_loss = kwargs.pop("ulysses_sp_return_token_loss", False)
            if "return_dict" in kwargs and return_dict is None:
                return_dict = kwargs.pop("return_dict")
            else:
                kwargs.pop("return_dict", None)
            if not ulysses_sp_training:
                return self.__class__._original_ulysses_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )

            local_targets = kwargs.pop("ulysses_sp_targets")
            local_p_mask = kwargs.pop("ulysses_sp_p_mask")
            answer_len = kwargs.pop("ulysses_sp_answer_len")
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            if not return_dict:
                raise NotImplementedError("SDAR Ulysses SP training requires return_dict=True.")

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = outputs.last_hidden_state[logits_to_keep].contiguous()
            if return_token_loss:
                # Every SP rank must execute lm_head so FSDP collectives stay aligned,
                # including ranks whose local sequence shard has no prediction target.
                diffusion_logits = self.lm_head(hidden_states).float()
                if local_targets.numel() == 0:
                    token_loss = diffusion_logits.new_empty((0,), dtype=torch.float32)
                    loss = diffusion_logits.sum() * 0
                else:
                    token_loss = (
                        F.cross_entropy(
                            diffusion_logits,
                            local_targets.contiguous(),
                            reduction="none",
                        )
                        / local_p_mask.float()
                    )
                    loss = token_loss.sum()
                loss = loss / answer_len.clamp_min(1)

                # Keep both the backbone and lm_head in the graph when this SP
                # rank owns no selected noisy-half target positions.
                token_loss_full = outputs.last_hidden_state.sum(dim=-1).float() * 0
                token_loss_full = token_loss_full + diffusion_logits.sum() * 0
                if token_loss.numel() > 0:
                    token_loss_full[logits_to_keep] = token_loss
                output = module.CausalLMOutputWithPast(
                    loss=loss,
                    logits=token_loss_full,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
                output.block_diffusion_token_loss = token_loss_full
                return output

            if local_targets.numel() == 0:
                loss = hidden_states.sum() * 0
            else:
                loss_fct = module.FusedLinearDiffusionCrossEntropyLoss(reduction="sum")
                loss = loss_fct(
                    x=hidden_states,
                    target=local_targets.contiguous(),
                    weight=self.lm_head.weight,
                    bias=self.lm_head.bias,
                    p_mask=local_p_mask,
                )
            loss = loss / answer_len.clamp_min(1)

            return module.CausalLMOutputWithPast(
                loss=loss,
                logits=None,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        module.SDARAttention.forward = sdar_ulysses_attention_forward
        module.SDARForCausalLM.forward = sdar_ulysses_lm_forward
        print(f"Monkey patch SDAR attention/forward for Ulysses SP (sp={sp_size})")
        return

    if model.config.model_type == "sdar_moe":
        if not hasattr(module.SDARMoeAttention, "_original_ulysses_forward"):
            module.SDARMoeAttention._original_ulysses_forward = module.SDARMoeAttention.forward

        def sdar_moe_ulysses_attention_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value=None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            current_sp_size = get_ulysses_sequence_parallel_world_size()
            if current_sp_size <= 1 or not self.training:
                return self.__class__._original_ulysses_forward(
                    self,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kwargs,
                )

            if past_key_value is not None:
                raise NotImplementedError("SDAR-MoE Ulysses sequence parallelism does not support KV cache during training.")

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = module.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states = _repeat_kv_for_ulysses(key_states, current_sp_size)
            value_states = _repeat_kv_for_ulysses(value_states, current_sp_size)

            query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
            key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
            value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

            local_attention_mask = attention_mask.bool() if attention_mask is not None else None
            full_attention_mask = _all_gather_sequence_mask(local_attention_mask)
            full_causal_mask = _build_causal_attention_mask(
                attention_mask=full_attention_mask,
                target_length=query_states.size(-2),
                device=query_states.device,
            )

            attn_output = F.scaled_dot_product_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                attn_mask=full_causal_mask,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=True,
            )
            if full_attention_mask is not None:
                attn_output = attn_output * full_attention_mask[:, None, :, None]

            attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        module.SDARMoeAttention.forward = sdar_moe_ulysses_attention_forward
        print(f"Monkey patch SDAR-MoE attention for Ulysses SP (sp={sp_size})")
