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

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask


BlockOrigin = Literal["global", "response"]

CJ_MASS_REL_TOL = 5e-5
CJ_MASS_ABS_TOL = 1e-6


def cj_mass_isclose(actual: float, expected: float) -> bool:
    """Compare equivalent CJ mass reductions without hiding invalid values."""

    return (
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(
            actual,
            expected,
            rel_tol=CJ_MASS_REL_TOL,
            abs_tol=CJ_MASS_ABS_TOL,
        )
    )


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


def build_cj_replay_model_inputs(
    *,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    replay_responses: torch.Tensor,
    replay_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Materialize the actor-only view of a terminal-block CJ replay.

    The ordinary rollout tensors deliberately end at the stop token so reward
    managers and user-visible decoding retain their usual contract.  Exact CJ
    replay additionally needs every token sampled in the final bidirectional
    diffusion block.  This helper joins those replay response tokens to the
    original prompt without mutating the ordinary rollout view.

    ``replay_attention_mask`` may be response-only ``[B, R_replay]`` (the
    compact transport form) or full-sequence ``[B, P + R_replay]``.
    """

    tensors = {
        "input_ids": input_ids,
        "responses": responses,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "replay_responses": replay_responses,
        "replay_attention_mask": replay_attention_mask,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [B, S], got {tensor.shape}")

    batch_size = input_ids.size(0)
    if any(tensor.size(0) != batch_size for tensor in tensors.values()):
        raise ValueError("All CJ replay tensors must have the same batch dimension")
    if attention_mask.shape != input_ids.shape or position_ids.shape != input_ids.shape:
        raise ValueError(
            "Ordinary attention_mask and position_ids must match input_ids, got "
            f"{attention_mask.shape}, {position_ids.shape}, and {input_ids.shape}"
        )
    if responses.size(1) > input_ids.size(1):
        raise ValueError(
            "responses cannot be longer than input_ids, got "
            f"{responses.size(1)} and {input_ids.size(1)}"
        )

    prompt_section_length = input_ids.size(1) - responses.size(1)
    replay_response_length = replay_responses.size(1)
    replay_sequence_length = prompt_section_length + replay_response_length
    if replay_attention_mask.size(1) == replay_response_length:
        replay_response_mask = replay_attention_mask.bool()
        full_replay_attention_mask = torch.cat(
            (
                attention_mask[:, :prompt_section_length].bool(),
                replay_response_mask,
            ),
            dim=1,
        )
    elif replay_attention_mask.size(1) == replay_sequence_length:
        full_replay_attention_mask = replay_attention_mask.bool()
        replay_response_mask = full_replay_attention_mask[
            :, prompt_section_length:
        ]
        expected_prompt_mask = attention_mask[:, :prompt_section_length].bool()
        if not torch.equal(
            full_replay_attention_mask[:, :prompt_section_length],
            expected_prompt_mask,
        ):
            raise ValueError(
                "Full CJ replay attention mask must preserve the ordinary prompt mask"
            )
    else:
        raise ValueError(
            "cj_replay_attention_mask must be response-only [B, R_replay] or "
            "full-sequence [B, P + R_replay], got "
            f"{replay_attention_mask.shape} for prompt length "
            f"{prompt_section_length} and replay response length "
            f"{replay_response_length}"
        )

    if prompt_section_length <= 0:
        raise ValueError("CJ replay requires at least one prompt position")
    if not replay_response_mask.any(dim=1).all():
        raise ValueError("Every CJ replay sample must contain a response token")

    replay_input_ids = torch.cat(
        (input_ids[:, :prompt_section_length], replay_responses),
        dim=1,
    )
    response_offsets = torch.arange(
        1,
        replay_response_length + 1,
        dtype=position_ids.dtype,
        device=position_ids.device,
    ).unsqueeze(0)
    replay_response_position_ids = (
        position_ids[:, prompt_section_length - 1 : prompt_section_length]
        + response_offsets
    )
    replay_position_ids = torch.cat(
        (
            position_ids[:, :prompt_section_length],
            replay_response_position_ids,
        ),
        dim=1,
    )

    return (
        replay_input_ids,
        full_replay_attention_mask.to(dtype=attention_mask.dtype),
        replay_position_ids,
        replay_response_mask,
        prompt_section_length,
    )


def expand_cj_outcome_values_to_replay(
    values: torch.Tensor,
    source_mask: torch.Tensor,
    replay_mask: torch.Tensor,
) -> torch.Tensor:
    """Extend a per-sample outcome value from the visible prefix to replay.

    GRPO produces the same scalar advantage/return at every valid response
    position.  The final-block suffix is an additional latent action from that
    same sampled outcome, so strict on-policy replay applies the same scalar to
    every replay token.
    """

    if values.ndim != 2 or source_mask.shape != values.shape:
        raise ValueError(
            "CJ outcome values and source mask must have shape [B, R], got "
            f"{values.shape} and {source_mask.shape}"
        )
    if replay_mask.ndim != 2 or replay_mask.size(0) != values.size(0):
        raise ValueError(
            "CJ replay mask must have shape [B, R_replay], got "
            f"{replay_mask.shape}"
        )

    source_mask = source_mask.bool()
    replay_mask = replay_mask.bool()
    source_counts = source_mask.sum(dim=1)
    if (source_counts == 0).any():
        raise ValueError("Every CJ sample needs a visible token for its outcome value")
    if not replay_mask.any(dim=1).all():
        raise ValueError("Every CJ sample needs at least one replay policy token")

    work_values = values.float()
    sample_values = (
        (work_values * source_mask).sum(dim=1)
        / source_counts.to(dtype=work_values.dtype)
    )
    deviations = torch.where(
        source_mask,
        (work_values - sample_values[:, None]).abs(),
        torch.zeros_like(work_values),
    ).amax(dim=1)
    tolerances = 1e-5 * (1.0 + sample_values.abs())
    if (deviations > tolerances).any():
        raise ValueError(
            "CJ terminal-block replay currently requires outcome-style "
            "advantages/returns that are constant across visible tokens"
        )

    return (
        sample_values[:, None].to(dtype=values.dtype)
        * replay_mask.to(dtype=values.dtype)
    )


def build_block_parallel_cj_step_inputs(
    *,
    clean_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    trajectory: torch.Tensor,
    response_length: int,
    local_step: int,
    mask_token_id: int,
):
    """Reconstruct every response block before one block-local CJ step."""

    if clean_input_ids.dim() != 2:
        raise ValueError(
            f"clean_input_ids must be rank 2, got {clean_input_ids.shape}"
        )
    if attention_mask.shape != clean_input_ids.shape:
        raise ValueError(
            "attention_mask must match clean_input_ids, got "
            f"{attention_mask.shape} and {clean_input_ids.shape}"
        )
    if trajectory.dim() != 3:
        raise ValueError(
            "trajectory must have shape [batch, local_steps, sequence], got "
            f"{trajectory.shape}"
        )
    expected_trajectory_shape = (
        clean_input_ids.size(0),
        trajectory.size(1),
        clean_input_ids.size(1),
    )
    if trajectory.shape != expected_trajectory_shape:
        raise ValueError(
            "trajectory must have shape [batch, local_steps, sequence], got "
            f"{trajectory.shape} for input {clean_input_ids.shape}"
        )
    if not 0 < response_length <= clean_input_ids.size(1):
        raise ValueError(
            f"response_length must be in [1, {clean_input_ids.size(1)}], "
            f"got {response_length}"
        )
    if not 0 <= local_step < trajectory.size(1):
        raise IndexError(
            f"local_step must be in [0, {trajectory.size(1)}), got {local_step}"
        )

    prompt_length = clean_input_ids.size(1) - response_length
    initial_noisy = torch.full_like(clean_input_ids, mask_token_id)
    initial_noisy[:, :prompt_length] = clean_input_ids[:, :prompt_length]

    if local_step == 0:
        noisy_input_ids = initial_noisy
    else:
        previously_unmasked = trajectory[:, :local_step, :].bool().any(dim=1)
        noisy_input_ids = torch.where(
            previously_unmasked,
            clean_input_ids,
            initial_noisy,
        )

    target_mask = trajectory[:, local_step, :].bool() & attention_mask.bool()
    target_mask[:, :prompt_length] = False
    return noisy_input_ids, target_mask, prompt_length


def build_padded_token_block_ids(
    attention_mask: torch.Tensor,
    prompt_section_length: int,
    block_size: int,
    block_origin: BlockOrigin = "response",
) -> torch.Tensor:
    """Assign block ids on a padded rollout using compact-token positions.

    Block-diffusion actors remove left prompt padding and right response
    padding before building their ``BlockMask``.  Deriving ids from the padded
    tensor indices would therefore disagree with the attention mask whenever a
    sample is left padded.  This helper mirrors that compaction while keeping
    the result aligned with the original padded tensor.
    """

    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be rank 2, got {attention_mask.shape}"
        )
    if not 0 <= prompt_section_length <= attention_mask.size(1):
        raise ValueError(
            "prompt_section_length must be in "
            f"[0, {attention_mask.size(1)}], got {prompt_section_length}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_origin not in ("global", "response"):
        raise ValueError(f"Unsupported block origin: {block_origin}")

    valid_mask = attention_mask.bool()
    compact_positions = valid_mask.long().cumsum(dim=1) - 1
    if block_origin == "global":
        block_ids = torch.div(
            compact_positions.clamp_min(0),
            block_size,
            rounding_mode="floor",
        )
    else:
        prompt_valid = valid_mask[:, :prompt_section_length]
        prompt_lengths = prompt_valid.sum(dim=1, dtype=torch.long)
        prompt_block_count = torch.div(
            prompt_lengths + block_size - 1,
            block_size,
            rounding_mode="floor",
        )

        response_valid = valid_mask[:, prompt_section_length:]
        response_offsets = response_valid.long().cumsum(dim=1) - 1
        response_block_ids = prompt_block_count[:, None] + torch.div(
            response_offsets.clamp_min(0),
            block_size,
            rounding_mode="floor",
        )
        block_ids = torch.div(
            compact_positions.clamp_min(0),
            block_size,
            rounding_mode="floor",
        )
        block_ids[:, prompt_section_length:] = response_block_ids

    return torch.where(valid_mask, block_ids, torch.full_like(block_ids, -1))


def compute_cj_block_step_token_weights(
    trajectory: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    prompt_section_length: int,
    block_size: int,
    block_origin: BlockOrigin,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paper-exact weights for block-parallel CJ replay.

    Tokens are averaged within each ``(sample, block, local_step)`` transition,
    transitions are averaged within each sample, and samples are averaged by
    the caller.  Token weights stay in FP32 for the policy loss.  The returned
    per-step diagnostic masses use FP64 and sum to one per sample.
    """

    if attention_mask.ndim != 2:
        raise ValueError(
            "CJ attention mask must have shape [B, S], "
            f"got {tuple(attention_mask.shape)}"
        )
    if trajectory.ndim != 3:
        raise ValueError(
            f"CJ trajectory must have shape [B, K, S], got {tuple(trajectory.shape)}"
        )
    if trajectory.size(0) != attention_mask.size(0) or trajectory.size(
        2
    ) != attention_mask.size(1):
        raise ValueError(
            "CJ trajectory must align with attention_mask, got "
            f"{tuple(trajectory.shape)} and {tuple(attention_mask.shape)}"
        )
    if response_mask.ndim != 2 or response_mask.size(0) != trajectory.size(0):
        raise ValueError(
            "CJ response mask must have shape [B, R], got "
            f"{tuple(response_mask.shape)}"
        )

    response_length = response_mask.size(1)
    if response_length <= 0:
        raise ValueError("CJ response must contain at least one position")
    expected_prompt_length = attention_mask.size(1) - response_length
    if prompt_section_length != expected_prompt_length:
        raise ValueError(
            "CJ prompt section and response mask disagree: "
            f"{prompt_section_length} vs. {expected_prompt_length}"
        )

    trajectory = trajectory.bool()
    response_mask = response_mask.bool()
    attention_mask = attention_mask.bool()
    if trajectory[:, :, :prompt_section_length].any():
        raise ValueError("CJ trajectory must not assign prompt tokens")

    response_trajectory = trajectory[:, :, -response_length:]
    attention_response = attention_mask[:, -response_length:]
    if (response_mask & ~attention_response).any():
        raise ValueError("CJ response mask includes an attention-padding token")
    if (response_trajectory & ~response_mask[:, None, :]).any():
        raise ValueError("CJ trajectory assigns a padded or excluded response token")
    assignments = response_trajectory.sum(dim=1)
    if (response_mask & (assignments != 1)).any():
        raise ValueError(
            "Each valid CJ response token must belong to exactly one local step"
        )

    padded_block_ids = build_padded_token_block_ids(
        attention_mask,
        prompt_section_length=prompt_section_length,
        block_size=block_size,
        block_origin=block_origin,
    )
    response_block_ids = padded_block_ids[:, -response_length:]
    selected = response_trajectory & response_mask[:, None, :]
    selected_block_ids = response_block_ids[:, None, :].expand_as(selected)
    if (selected & (selected_block_ids < 0)).any():
        raise ValueError("CJ target does not map to a valid block")
    if not selected.any():
        raise ValueError("Every CJ sample must contain at least one transition")

    token_weights = torch.zeros_like(response_trajectory, dtype=torch.float32)
    num_steps = trajectory.size(1)
    num_blocks = int(response_block_ids.max().item()) + 1
    batch_ids = torch.arange(
        trajectory.size(0), device=trajectory.device, dtype=torch.long
    )[:, None, None]
    step_ids = torch.arange(num_steps, device=trajectory.device, dtype=torch.long)[
        None, :, None
    ]
    transition_ids = (
        batch_ids * num_steps + step_ids
    ) * num_blocks + selected_block_ids.clamp_min(0)

    unique_transition_ids, inverse, token_counts = torch.unique(
        transition_ids[selected],
        sorted=False,
        return_inverse=True,
        return_counts=True,
    )
    unique_batch_ids = torch.div(
        unique_transition_ids,
        num_steps * num_blocks,
        rounding_mode="floor",
    )
    unique_step_ids = torch.div(
        unique_transition_ids,
        num_blocks,
        rounding_mode="floor",
    ).remainder(num_steps)
    unique_block_ids = unique_transition_ids.remainder(num_blocks)
    sample_block_ids = unique_batch_ids * num_blocks + unique_block_ids
    _, sample_block_inverse, sample_block_step_counts = torch.unique(
        sample_block_ids,
        sorted=False,
        return_inverse=True,
        return_counts=True,
    )
    sample_block_max_steps = torch.full(
        (sample_block_step_counts.numel(),),
        -1,
        dtype=torch.long,
        device=trajectory.device,
    )
    sample_block_max_steps.scatter_reduce_(
        0,
        sample_block_inverse,
        unique_step_ids,
        reduce="amax",
        include_self=True,
    )
    if (sample_block_step_counts != sample_block_max_steps + 1).any():
        raise ValueError(
            "Each CJ sample/block trajectory must use contiguous local steps "
            "starting at zero"
        )

    sample_transition_counts = torch.bincount(
        unique_batch_ids,
        minlength=trajectory.size(0),
    )
    if (sample_transition_counts == 0).any():
        raise ValueError("Every CJ sample must contain at least one transition")

    token_weights[selected] = (
        token_counts[inverse].float().reciprocal()
        / sample_transition_counts[unique_batch_ids[inverse]].float()
    )
    sample_masses = token_weights.sum(dim=(1, 2), dtype=torch.float64)
    expected_sample_masses = torch.ones_like(sample_masses)
    valid_sample_masses = torch.isfinite(sample_masses) & torch.isclose(
        sample_masses,
        expected_sample_masses,
        rtol=1e-6,
        atol=1e-8,
    )
    if not bool(valid_sample_masses.all().item()):
        raise RuntimeError(
            "CJ token weights must reconstruct one objective per sample, got "
            f"{sample_masses.tolist()}"
        )

    # Derive diagnostics from the actual FP32 loss weights, but accumulate in
    # FP64 so long responses do not inherit FP32 reduction-order drift.
    step_masses = token_weights.sum(dim=(0, 2), dtype=torch.float64)
    return token_weights, step_masses


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
    # Start from a graph-connected zero.  A CJ local step can legitimately be
    # empty on one FSDP rank while another rank has targets.  The empty rank
    # must still run the same forward/backward schedule, including lm_head.
    output = (
        compact_values.sum()
        .mul(0)
        .expand(artifacts.batch_size, response_length)
        .clone()
    )
    for batch_index in range(artifacts.batch_size):
        selected = artifacts.target_mask[batch_index]
        positions = artifacts.response_positions[batch_index, selected]
        values = compact_values[batch_index, selected]
        in_range = (positions >= 0) & (positions < response_length)
        output[batch_index, positions[in_range]] = values[in_range]
    return output
