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

"""Shared data contract for block-diffusion actor forwards."""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask


BlockOrigin = Literal["global", "response"]


@dataclass(frozen=True)
class BlockDiffusionArtifacts:
    """Compact clean/noisy inputs and their mapping back to rollout tensors."""

    noisy_input_ids: torch.Tensor
    clean_input_ids: torch.Tensor
    valid_mask: torch.Tensor
    target_mask: torch.Tensor
    p_mask: torch.Tensor
    position_ids: torch.Tensor
    prompt_lengths: torch.Tensor
    response_lengths: torch.Tensor
    response_positions: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.clean_input_ids.size(0)

    @property
    def sequence_length(self) -> int:
        return self.clean_input_ids.size(1)


def compact_block_diffusion_artifacts(
    *,
    noisy_input_ids: torch.Tensor,
    clean_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    target_mask: torch.Tensor,
    p_mask: torch.Tensor,
    prompt_section_length: int,
    pad_token_id: int,
) -> BlockDiffusionArtifacts:
    """Remove left/right padding while retaining response-position mappings."""

    expected_shape = clean_input_ids.shape
    tensors = {
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "target_mask": target_mask,
        "p_mask": p_mask,
    }
    if clean_input_ids.dim() != 2:
        raise ValueError(f"Block-diffusion inputs must be rank 2, got {clean_input_ids.shape}")
    for name, tensor in tensors.items():
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tensor.shape}")

    batch_size, padded_sequence_length = expected_shape
    if not 0 <= prompt_section_length <= padded_sequence_length:
        raise ValueError(
            f"prompt_section_length must be in [0, {padded_sequence_length}], "
            f"got {prompt_section_length}"
        )

    valid_mask = attention_mask.bool()
    prompt_lengths = valid_mask[:, :prompt_section_length].sum(dim=1, dtype=torch.long)
    response_lengths = valid_mask[:, prompt_section_length:].sum(dim=1, dtype=torch.long)
    compact_lengths = prompt_lengths + response_lengths
    if (compact_lengths <= 0).any():
        raise ValueError("Every block-diffusion sample must contain at least one valid token")

    compact_sequence_length = int(compact_lengths.max().item())
    device = clean_input_ids.device
    compact_noisy = torch.full(
        (batch_size, compact_sequence_length),
        pad_token_id,
        dtype=noisy_input_ids.dtype,
        device=device,
    )
    compact_clean = torch.full_like(compact_noisy, pad_token_id)
    compact_valid = torch.zeros(
        (batch_size, compact_sequence_length), dtype=torch.bool, device=device
    )
    compact_target = torch.zeros_like(compact_valid)
    compact_p_mask = torch.zeros(
        (batch_size, compact_sequence_length), dtype=p_mask.dtype, device=device
    )
    compact_position_ids = torch.zeros(
        (batch_size, compact_sequence_length), dtype=position_ids.dtype, device=device
    )
    response_positions = torch.full(
        (batch_size, compact_sequence_length), -1, dtype=torch.long, device=device
    )

    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        compact_length = int(compact_lengths[batch_index].item())
        compact_noisy[batch_index, :compact_length] = noisy_input_ids[batch_index, valid_indices]
        compact_clean[batch_index, :compact_length] = clean_input_ids[batch_index, valid_indices]
        compact_valid[batch_index, :compact_length] = True
        compact_position_ids[batch_index, :compact_length] = position_ids[batch_index, valid_indices]

        is_response = valid_indices >= prompt_section_length
        compact_target[batch_index, :compact_length] = (
            target_mask[batch_index, valid_indices].bool() & is_response
        )
        compact_p_mask[batch_index, :compact_length] = p_mask[batch_index, valid_indices]
        response_positions[batch_index, :compact_length] = torch.where(
            is_response,
            valid_indices - prompt_section_length,
            torch.full_like(valid_indices, -1),
        )

    selected_p_mask = compact_p_mask[compact_target]
    if selected_p_mask.numel() > 0 and (selected_p_mask <= 0).any():
        raise ValueError("Every selected block-diffusion target must have positive p_mask")

    return BlockDiffusionArtifacts(
        noisy_input_ids=compact_noisy,
        clean_input_ids=compact_clean,
        valid_mask=compact_valid,
        target_mask=compact_target,
        p_mask=compact_p_mask,
        position_ids=compact_position_ids,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        response_positions=response_positions,
    )


def build_token_block_ids(
    valid_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    block_size: int,
    block_origin: BlockOrigin = "response",
) -> torch.Tensor:
    """Assign blocks, optionally starting a fresh block after each prompt."""

    if valid_mask.dim() != 2:
        raise ValueError(f"valid_mask must be rank 2, got {valid_mask.shape}")
    if prompt_lengths.shape != (valid_mask.size(0),):
        raise ValueError(
            f"prompt_lengths must have shape {(valid_mask.size(0),)}, got {prompt_lengths.shape}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_origin not in ("global", "response"):
        raise ValueError(f"Unsupported block origin: {block_origin}")

    token_indices = torch.arange(valid_mask.size(1), device=valid_mask.device).unsqueeze(0)
    token_indices = token_indices.expand(valid_mask.size(0), -1)
    if block_origin == "global":
        block_ids = torch.div(token_indices, block_size, rounding_mode="floor")
    else:
        prompt_lengths_2d = prompt_lengths.unsqueeze(1)
        prompt_block_count = torch.div(
            prompt_lengths_2d + block_size - 1, block_size, rounding_mode="floor"
        )
        prompt_block_ids = torch.div(token_indices, block_size, rounding_mode="floor")
        response_offsets = (token_indices - prompt_lengths_2d).clamp_min(0)
        response_block_ids = prompt_block_count + torch.div(
            response_offsets, block_size, rounding_mode="floor"
        )
        block_ids = torch.where(
            token_indices < prompt_lengths_2d,
            prompt_block_ids,
            response_block_ids,
        )

    return torch.where(valid_mask.bool(), block_ids, torch.full_like(block_ids, -1))


def build_block_diffusion_mask(
    artifacts: BlockDiffusionArtifacts,
    block_size: int,
    block_origin: BlockOrigin = "response",
) -> BlockMask:
    """Build the BD3LM visibility pattern without materializing a dense 4D mask."""

    sequence_length = artifacts.sequence_length
    full_length = 2 * sequence_length
    block_ids = build_token_block_ids(
        artifacts.valid_mask,
        artifacts.prompt_lengths,
        block_size,
        block_origin,
    )
    valid_mask = artifacts.valid_mask

    def mask_mod(batch_index, head_index, query_index, key_value_index):
        del head_index
        query_in_bounds = query_index < full_length
        key_in_bounds = key_value_index < full_length
        safe_query_index = query_index.clamp(max=full_length - 1)
        safe_key_value_index = key_value_index.clamp(max=full_length - 1)
        query_is_clean = safe_query_index >= sequence_length
        key_is_clean = safe_key_value_index >= sequence_length
        query_token_index = safe_query_index % sequence_length
        key_token_index = safe_key_value_index % sequence_length

        query_block = block_ids[batch_index, query_token_index]
        key_block = block_ids[batch_index, key_token_index]
        query_valid = valid_mask[batch_index, query_token_index]
        key_valid = valid_mask[batch_index, key_token_index]

        block_diagonal = (query_block == key_block) & (query_is_clean == key_is_clean)
        offset_block_causal = (
            (query_block > key_block) & key_is_clean & (~query_is_clean)
        )
        block_causal = (
            (query_block >= key_block) & key_is_clean & query_is_clean
        )
        return query_in_bounds & key_in_bounds & query_valid & key_valid & (
            block_diagonal | offset_block_causal | block_causal
        )

    # ``create_block_mask`` first materializes [B, 1, 2L, 2L] booleans. At
    # L=8192 that transient alone is 256 MiB/sample. Build only the coarse
    # sparse-block occupancy here and retain mask_mod for exact token filtering.
    sparse_block_size = 128
    num_sparse_blocks = (full_length + sparse_block_size - 1) // sparse_block_size
    padded_length = num_sparse_blocks * sparse_block_size
    full_block_ids = torch.cat((block_ids, block_ids), dim=1)
    full_valid = torch.cat((valid_mask, valid_mask), dim=1)
    full_is_clean = (
        torch.arange(full_length, device=full_block_ids.device) >= sequence_length
    ).unsqueeze(0).expand(artifacts.batch_size, -1)
    if padded_length > full_length:
        pad_size = padded_length - full_length
        full_block_ids = F.pad(full_block_ids, (0, pad_size), value=-1)
        full_valid = F.pad(full_valid, (0, pad_size), value=False)
        full_is_clean = F.pad(full_is_clean, (0, pad_size), value=False)

    blocked_ids = full_block_ids.view(
        artifacts.batch_size, num_sparse_blocks, sparse_block_size
    )
    blocked_valid = full_valid.view(
        artifacts.batch_size, num_sparse_blocks, sparse_block_size
    )
    blocked_clean = full_is_clean.view(
        artifacts.batch_size, num_sparse_blocks, sparse_block_size
    )

    def block_ranges(selection: torch.Tensor):
        exists = selection.any(dim=-1)
        minimum = torch.where(
            selection,
            blocked_ids,
            torch.full_like(blocked_ids, torch.iinfo(blocked_ids.dtype).max),
        ).amin(dim=-1)
        maximum = torch.where(
            selection, blocked_ids, torch.full_like(blocked_ids, -1)
        ).amax(dim=-1)
        return exists, minimum, maximum

    noisy_exists, noisy_minimum, noisy_maximum = block_ranges(
        blocked_valid & ~blocked_clean
    )
    clean_exists, clean_minimum, clean_maximum = block_ranges(
        blocked_valid & blocked_clean
    )

    noisy_diagonal = (
        noisy_exists[:, :, None]
        & noisy_exists[:, None, :]
        & (
            torch.maximum(noisy_minimum[:, :, None], noisy_minimum[:, None, :])
            <= torch.minimum(noisy_maximum[:, :, None], noisy_maximum[:, None, :])
        )
    )
    clean_causal = (
        clean_exists[:, :, None]
        & clean_exists[:, None, :]
        & (clean_maximum[:, :, None] >= clean_minimum[:, None, :])
    )
    noisy_to_past_clean = (
        noisy_exists[:, :, None]
        & clean_exists[:, None, :]
        & (noisy_maximum[:, :, None] > clean_minimum[:, None, :])
    )
    coarse_visible = noisy_diagonal | clean_causal | noisy_to_past_clean

    key_value_block = torch.arange(
        num_sparse_blocks, device=coarse_visible.device, dtype=torch.int32
    ).view(1, 1, num_sparse_blocks)
    key_value_block = key_value_block.expand(
        artifacts.batch_size, num_sparse_blocks, -1
    )
    sentinel = torch.full_like(key_value_block, num_sparse_blocks)
    ordered_indices = torch.where(coarse_visible, key_value_block, sentinel).sort(
        dim=-1
    ).values
    ordered_indices = torch.where(
        ordered_indices == num_sparse_blocks,
        torch.zeros_like(ordered_indices),
        ordered_indices,
    )
    key_value_num_blocks = coarse_visible.sum(dim=-1, dtype=torch.int32)

    return BlockMask.from_kv_blocks(
        key_value_num_blocks.unsqueeze(1),
        ordered_indices.unsqueeze(1),
        BLOCK_SIZE=sparse_block_size,
        mask_mod=mask_mod,
        seq_lengths=(full_length, full_length),
    )


def build_full_block_diffusion_tensors(artifacts: BlockDiffusionArtifacts):
    """Return the common ``[noisy, clean]`` model inputs and loss tensors."""

    full_input_ids = torch.cat(
        (artifacts.noisy_input_ids, artifacts.clean_input_ids), dim=1
    )
    full_position_ids = torch.cat(
        (artifacts.position_ids, artifacts.position_ids), dim=1
    )
    full_target_mask = torch.cat(
        (artifacts.target_mask, torch.zeros_like(artifacts.target_mask)), dim=1
    )
    full_targets = torch.cat(
        (
            artifacts.clean_input_ids,
            torch.full_like(artifacts.clean_input_ids, fill_value=-100),
        ),
        dim=1,
    )
    full_targets = torch.where(
        full_target_mask,
        full_targets,
        torch.full_like(full_targets, fill_value=-100),
    )
    full_p_mask = torch.cat(
        (artifacts.p_mask, torch.ones_like(artifacts.p_mask)), dim=1
    )
    return (
        full_input_ids,
        full_position_ids,
        full_target_mask,
        full_targets,
        full_p_mask,
    )


def pad_block_diffusion_loss_tensors(
    full_target_mask: torch.Tensor,
    full_targets: torch.Tensor,
    full_p_mask: torch.Tensor,
    pad_size: int,
):
    if pad_size <= 0:
        return full_target_mask, full_targets, full_p_mask
    return (
        F.pad(full_target_mask, (0, pad_size), value=False),
        F.pad(full_targets, (0, pad_size), value=-100),
        F.pad(full_p_mask, (0, pad_size), value=1),
    )


def scatter_compact_values_to_response(
    compact_values: torch.Tensor,
    artifacts: BlockDiffusionArtifacts,
    response_length: int,
) -> torch.Tensor:
    """Map compact per-token values back to the padded rollout response axis."""

    if compact_values.shape != artifacts.target_mask.shape:
        raise ValueError(
            f"compact_values must have shape {artifacts.target_mask.shape}, "
            f"got {compact_values.shape}"
        )
    output = compact_values.new_zeros((artifacts.batch_size, response_length))
    for batch_index in range(artifacts.batch_size):
        selected = artifacts.target_mask[batch_index]
        positions = artifacts.response_positions[batch_index, selected]
        values = compact_values[batch_index, selected]
        in_range = (positions >= 0) & (positions < response_length)
        output[batch_index, positions[in_range]] = values[in_range]
    return output
