"""BlockMask and Ulysses support for LLaDA2 block-diffusion training."""

import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)


_compiled_flex_attention = torch.compile(flex_attention)


def _is_block_mask(attention_mask) -> bool:
    return isinstance(attention_mask, BlockMask)


def llada2_block_diffusion_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """Run FlexAttention over a global BlockMask, with optional Ulysses A2A."""

    if not _is_block_mask(attention_mask):
        return self.__class__._original_block_diffusion_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    if past_key_value is not None or use_cache:
        raise NotImplementedError("LLaDA2 block-diffusion training does not support KV cache")
    if output_attentions:
        raise NotImplementedError("FlexAttention does not return LLaDA2 attention weights")
    if position_embeddings is None:
        raise ValueError("position_embeddings is required for LLaDA2 block-diffusion training")
    if self.training and self.attention_dropout:
        raise NotImplementedError("LLaDA2 BlockMask training requires attention_dropout=0")

    input_shape = hidden_states.shape[:-1]
    batch_size, local_sequence_length, _ = hidden_states.shape
    qkv = self.query_key_value(hidden_states).view(
        batch_size,
        local_sequence_length,
        self.num_heads + 2 * self.num_key_value_heads,
        self.head_dim,
    )
    query_states, key_states, value_states = qkv.split(
        [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
    )
    query_states = self.query_layernorm(query_states.transpose(1, 2))
    key_states = self.key_layernorm(key_states.transpose(1, 2))
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    module = sys.modules[self.__module__]
    query_states, key_states = module.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    # Repeating KV before A2A makes the path valid even when num_kv_heads is
    # smaller than the SP degree, and gives FlexAttention equal Q/KV head counts.
    key_states = module.repeat_kv(key_states, self.num_key_value_groups)
    value_states = module.repeat_kv(value_states, self.num_key_value_groups)

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
        scale=self.scaling,
    )
    attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.dense(attn_output), None, None


def llada2_block_diffusion_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask=None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, MoeModelOutputWithPast]:
    """Preserve BlockMask objects instead of materializing a dense 4D mask."""

    if not _is_block_mask(attention_mask):
        return self.__class__._original_block_diffusion_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if not return_dict:
        raise NotImplementedError("LLaDA2 block-diffusion training requires return_dict=True")
    if use_cache or past_key_values is not None:
        raise NotImplementedError("LLaDA2 block-diffusion training does not support KV cache")
    if output_attentions:
        raise NotImplementedError("LLaDA2 block-diffusion training does not return attentions")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Specify exactly one of input_ids and inputs_embeds")
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Specify exactly one of input_ids and inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    if position_ids is None:
        raise ValueError("Sliced position_ids are required for LLaDA2 block-diffusion training")

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    all_hidden_states = () if output_hidden_states else None
    all_router_logits = () if output_router_logits else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                None,
                False,
                output_router_logits,
                False,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                output_router_logits=output_router_logits,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
        hidden_states = layer_outputs[0]
        if output_router_logits and layer_outputs[-1] is not None:
            all_router_logits += (layer_outputs[-1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=all_hidden_states,
        attentions=None,
        router_logits=all_router_logits,
    )


def llada2_block_diffusion_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask=None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
):
    """Project only selected diffusion targets instead of all ``2L`` tokens."""

    targets = kwargs.pop("block_diffusion_targets", None)
    p_mask = kwargs.pop("block_diffusion_p_mask", None)
    answer_len = kwargs.pop("block_diffusion_answer_len", None)
    return_token_loss = kwargs.pop("return_block_diffusion_token_loss", False)
    if targets is None:
        return self.__class__._original_block_diffusion_forward(
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
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

    if not _is_block_mask(attention_mask):
        raise ValueError("block_diffusion_targets require a BlockMask attention mask")
    if p_mask is None or p_mask.shape != targets.shape:
        raise ValueError("block_diffusion_p_mask must match block_diffusion_targets")
    if labels is not None:
        raise ValueError("Use block_diffusion_targets instead of labels on this path")
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if not return_dict:
        raise NotImplementedError("LLaDA2 block-diffusion training requires return_dict=True")

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        return_dict=True,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    selected = targets.ne(-100)
    selected_hidden = hidden_states[selected].contiguous()
    selected_logits = self.lm_head(selected_hidden).float()
    if selected.any():
        selected_p_mask = p_mask[selected].float()
        if (selected_p_mask <= 0).any():
            raise ValueError("Selected block-diffusion probabilities must be positive")
        token_loss = F.cross_entropy(
            selected_logits,
            targets[selected].contiguous(),
            reduction="none",
        ) / selected_p_mask
    else:
        token_loss = selected_logits.new_empty((0,))

    # CopySlices keeps the selected-token graph connected. Exposing this tensor
    # as logits also lets FSDP2 discover the lm_head output during backward.
    token_loss_full = hidden_states.sum(dim=-1).float() * 0
    # Keep lm_head in every rank's returned token-loss graph. A sequence shard
    # can legitimately contain no noisy-half targets under Ulysses SP.
    token_loss_full = token_loss_full + selected_logits.sum() * 0
    if token_loss.numel() > 0:
        token_loss_full[selected] = token_loss
    denominator = (
        torch.as_tensor(answer_len, device=hidden_states.device, dtype=torch.float32)
        if answer_len is not None
        else selected.sum().to(dtype=torch.float32)
    )
    loss = token_loss.sum() / denominator.clamp_min(1)
    if token_loss.numel() == 0:
        loss = loss + selected_logits.sum() * 0

    module = sys.modules[self.__module__]
    output = module.MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=None,
        logits=token_loss_full if return_token_loss else None,
        past_key_values=None,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )
    output.block_diffusion_token_loss = token_loss_full
    return output


def apply_llada2_ulysses_patch(model):
    """Patch one loaded LLaDA2 implementation without changing normal forwards."""

    module = sys.modules[model.__module__]
    attention_classes = [module.LLaDA2MoeAttention]
    for class_name in ("LLaDA2MoeSdpaAttention", "LLaDA2MoeFlexAttention"):
        attention_class = getattr(module, class_name, None)
        if attention_class is not None:
            attention_classes.append(attention_class)

    for attention_class in attention_classes:
        # Subclasses inherit attributes from LLaDA2MoeAttention; inspect their
        # own dictionaries so SDPA/Flex keep their respective original forward.
        if "_original_block_diffusion_forward" not in attention_class.__dict__:
            attention_class._original_block_diffusion_forward = attention_class.forward
        attention_class.forward = llada2_block_diffusion_attention_forward

    if not hasattr(module.LLaDA2MoeModel, "_original_block_diffusion_forward"):
        module.LLaDA2MoeModel._original_block_diffusion_forward = module.LLaDA2MoeModel.forward
    module.LLaDA2MoeModel.forward = llada2_block_diffusion_model_forward

    if not hasattr(module.LLaDA2MoeModelLM, "_original_block_diffusion_forward"):
        module.LLaDA2MoeModelLM._original_block_diffusion_forward = module.LLaDA2MoeModelLM.forward
    module.LLaDA2MoeModelLM.forward = llada2_block_diffusion_lm_forward
    print(
        "Monkey patch LLaDA2 block-diffusion attention/model/LM "
        f"(ulysses_sp={get_ulysses_sequence_parallel_world_size()})"
    )
