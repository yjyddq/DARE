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

"""Convert generic SGLang dLLM replay metadata into CJ-GRPO trajectories.

SGLang keeps its normal response token IDs truncated at the actual stop
position.  For exact replay of a bidirectional terminal diffusion block it
additionally exposes the complete block's token IDs and token-aligned step
maps through ``meta_info``.  DARE validates that generic replay contract here
before constructing any CJ-specific tensors.

SGLang step maps are positive and 1-based.  This module converts them to the
0-based local-step indices consumed by the actor.
"""

from collections.abc import Sequence
from numbers import Integral

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence


CJ_TRAJECTORY_CONTRACT_VERSION = 2
DLLM_REPLAY_CONTRACT_VERSION = 1
DLLM_STEP_MAP_KEY = "step_maps"
DLLM_REPLAY_TOKEN_IDS_KEY = "dllm_replay_token_ids"
DLLM_REPLAY_STEP_MAPS_KEY = "dllm_replay_step_maps"
DLLM_STOP_LENGTH_KEY = "dllm_stop_length"
DLLM_REPLAY_CONTRACT_VERSION_KEY = "dllm_replay_contract_version"


def _require_contract_version(contract_version):
    if isinstance(contract_version, bool) or not isinstance(contract_version, Integral):
        raise TypeError(
            "CJ trajectory contract_version must be an integer, got "
            f"{contract_version!r}"
        )
    contract_version = int(contract_version)
    if contract_version != CJ_TRAJECTORY_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported CJ trajectory contract version "
            f"{contract_version}; this checkout supports only "
            f"version {CJ_TRAJECTORY_CONTRACT_VERSION}."
        )
    return contract_version


def _require_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return converted


def _require_non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return converted


def expand_cj_generation_inputs(input_ids, image_data, num_samples):
    """Expand a CJ batch in prompt-major order and leave engine ``n`` at 1.

    SGLang's batch API preserves request order, but its flattened ``n > 1``
    responses do not carry a stable sample-owner field that DARE can validate.
    Expanding ``[p0, p1]`` to ``[p0, ..., p0, p1, ..., p1]`` before the call
    makes ownership explicit using only the existing batch interface.
    """

    num_samples = _require_positive_integer(num_samples, "num_samples")
    if not isinstance(input_ids, Sequence):
        raise TypeError("input_ids must be a sequence of per-prompt token IDs")
    if not isinstance(image_data, Sequence):
        raise TypeError("image_data must be a sequence aligned with input_ids")
    if len(input_ids) != len(image_data):
        raise ValueError(
            "input_ids and image_data must have the same batch length, got "
            f"{len(input_ids)} and {len(image_data)}"
        )

    expanded_input_ids = [
        prompt_ids for prompt_ids in input_ids for _ in range(num_samples)
    ]
    expanded_image_data = [
        prompt_image for prompt_image in image_data for _ in range(num_samples)
    ]
    return expanded_input_ids, expanded_image_data


def align_cj_max_new_tokens(prompt_lengths, max_new_tokens, block_size):
    """Align each CJ length cap to the global dLLM block grid.

    Length-based termination must not cut a bidirectional diffusion block.
    EOS/stop termination can still occur inside a block and is handled by the
    terminal-block replay payload; this helper only makes the hard generation
    budget safe.
    """

    if not isinstance(prompt_lengths, Sequence):
        raise TypeError("prompt_lengths must be a per-request sequence")
    max_new_tokens = _require_positive_integer(
        max_new_tokens,
        "max_new_tokens",
    )
    block_size = _require_positive_integer(block_size, "block_size")

    aligned_max_new_tokens = []
    for request_index, prompt_length in enumerate(prompt_lengths):
        prompt_length = _require_non_negative_integer(
            prompt_length,
            f"prompt_lengths[{request_index}]",
        )
        aligned = max_new_tokens - (
            (prompt_length + max_new_tokens) % block_size
        )
        if aligned <= 0:
            raise ValueError(
                "No positive global-block-aligned CJ generation budget exists "
                f"for request {request_index}: prompt_length={prompt_length}, "
                f"max_new_tokens={max_new_tokens}, block_size={block_size}."
            )
        aligned_max_new_tokens.append(aligned)
    return aligned_max_new_tokens


def _get_response_token_ids(response):
    if not isinstance(response, dict):
        raise TypeError(
            "Each SGLang response must be a mapping, got " f"{type(response).__name__}"
        )

    meta_info = response.get("meta_info", {})
    if not isinstance(meta_info, dict):
        raise TypeError("SGLang response meta_info must be a mapping")

    # Match ``_post_process_outputs`` exactly: logprob token IDs take
    # precedence when return_logprob=True.
    output_token_logprobs = meta_info.get("output_token_logprobs") or []
    if output_token_logprobs:
        return [token_id for _log_prob, token_id, *_ in output_token_logprobs]

    output_token_ids = response.get("token_ids")
    if output_token_ids is None:
        output_token_ids = response.get("output_ids")
    if output_token_ids is None:
        raise ValueError(
            "SGLang response is missing token_ids, output_ids, and "
            "output_token_logprobs"
        )
    return output_token_ids


def _as_integer_vector(value, *, name):
    try:
        vector = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a 1-D integer sequence") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(vector.shape)}")
    if vector.dtype == torch.bool or vector.dtype.is_floating_point or vector.dtype.is_complex:
        raise TypeError(f"{name} must use an integer dtype, got {vector.dtype}")
    return vector.to(dtype=torch.long)


def post_process_dllm_replay_metadata(
    outputs,
    *,
    pad_token_id,
    prompt_lengths,
    block_size,
    expected_num_responses=None,
    contract_version=CJ_TRAJECTORY_CONTRACT_VERSION,
):
    """Validate and batch SGLang's complete terminal-block replay payload.

    Normal SGLang ``token_ids`` remain the stop prefix.  Replay token IDs and
    replay step maps cover that prefix plus any suffix needed to finish the
    already-denoised terminal block.  The returned tensors are padded
    independently from the normal response tensors:

    ``replay_token_ids``
        Right-padded with ``pad_token_id``.
    ``local_unmask_steps``
        Right-padded with ``-1`` and converted from SGLang's 1-based steps.
    ``stop_lengths``
        The authoritative number of normal response tokens per request.

    Requiring this payload even for boundary-aligned responses is an explicit
    capability handshake.  A legacy SGLang build exposing only ``step_maps``
    cannot prove which stop semantics produced the normal response.
    """

    _require_contract_version(contract_version)
    pad_token_id = _require_non_negative_integer(pad_token_id, "pad_token_id")
    block_size = _require_positive_integer(block_size, "block_size")

    if not isinstance(outputs, (list, tuple)):
        raise TypeError(
            "SGLang batched output must be a list or tuple, got "
            f"{type(outputs).__name__}"
        )
    if expected_num_responses is not None:
        expected_num_responses = _require_positive_integer(
            expected_num_responses,
            "expected_num_responses",
        )
        if len(outputs) != expected_num_responses:
            raise RuntimeError(
                "SGLang CJ response ordering/count contract failed: expected "
                f"{expected_num_responses} prompt-major responses, got "
                f"{len(outputs)}."
            )
    if not outputs:
        raise RuntimeError("SGLang returned an empty batch for CJ rollout")
    if not isinstance(prompt_lengths, Sequence):
        raise TypeError("prompt_lengths must be a per-response sequence")
    if len(prompt_lengths) != len(outputs):
        raise ValueError(
            "prompt_lengths must align with SGLang responses, got "
            f"{len(prompt_lengths)} lengths for {len(outputs)} responses"
        )

    required_keys = (
        DLLM_REPLAY_TOKEN_IDS_KEY,
        DLLM_REPLAY_STEP_MAPS_KEY,
        DLLM_STOP_LENGTH_KEY,
        DLLM_REPLAY_CONTRACT_VERSION_KEY,
    )
    batched_replay_token_ids = []
    batched_local_steps = []
    stop_lengths = []

    for response_index, response in enumerate(outputs):
        if not isinstance(response, dict):
            raise TypeError(
                "Each SGLang response must be a mapping, got "
                f"{type(response).__name__}"
            )
        meta_info = response.get("meta_info", {})
        if not isinstance(meta_info, dict):
            raise TypeError("SGLang response meta_info must be a mapping")

        missing_keys = [key for key in required_keys if key not in meta_info]
        if missing_keys:
            raise RuntimeError(
                "SGLang dLLM terminal-block replay capability check failed for "
                f"response {response_index}: missing meta_info fields "
                f"{missing_keys}. Install the updated editable SGLang PR branch "
                "that returns complete final-block replay metadata; legacy "
                "step_maps cannot recover a suffix omitted after EOS."
            )

        replay_contract_version = meta_info[DLLM_REPLAY_CONTRACT_VERSION_KEY]
        if (
            isinstance(replay_contract_version, bool)
            or not isinstance(replay_contract_version, Integral)
        ):
            raise TypeError(
                f"{DLLM_REPLAY_CONTRACT_VERSION_KEY} for response "
                f"{response_index} must be an integer, got "
                f"{replay_contract_version!r}"
            )
        if int(replay_contract_version) != DLLM_REPLAY_CONTRACT_VERSION:
            raise RuntimeError(
                "Unsupported SGLang dLLM replay contract version "
                f"{replay_contract_version!r} for response {response_index}; "
                f"this checkout supports only version "
                f"{DLLM_REPLAY_CONTRACT_VERSION}."
            )

        stop_length = _require_positive_integer(
            meta_info[DLLM_STOP_LENGTH_KEY],
            f"{DLLM_STOP_LENGTH_KEY} for response {response_index}",
        )
        normal_token_ids = _as_integer_vector(
            _get_response_token_ids(response),
            name=f"normal token IDs for response {response_index}",
        )
        replay_token_ids = _as_integer_vector(
            meta_info[DLLM_REPLAY_TOKEN_IDS_KEY],
            name=f"{DLLM_REPLAY_TOKEN_IDS_KEY} for response {response_index}",
        )
        replay_steps = _as_integer_vector(
            meta_info[DLLM_REPLAY_STEP_MAPS_KEY],
            name=f"{DLLM_REPLAY_STEP_MAPS_KEY} for response {response_index}",
        )
        if replay_steps.numel() != replay_token_ids.numel():
            raise RuntimeError(
                "SGLang dLLM replay metadata/token length mismatch for response "
                f"{response_index}: got {replay_steps.numel()} replay steps for "
                f"{replay_token_ids.numel()} replay tokens."
            )
        if replay_token_ids.numel() == 0:
            raise RuntimeError(
                f"SGLang dLLM replay payload for response {response_index} is empty"
            )
        if (replay_token_ids < 0).any():
            bad_position = int(
                torch.nonzero(replay_token_ids < 0, as_tuple=False)[0].item()
            )
            raise RuntimeError(
                f"SGLang dLLM replay token IDs for response {response_index} "
                f"must be non-negative, but position {bad_position} is invalid."
            )
        if (replay_steps <= 0).any():
            bad_position = int(
                torch.nonzero(replay_steps <= 0, as_tuple=False)[0].item()
            )
            raise RuntimeError(
                f"SGLang dLLM replay steps for response {response_index} must "
                "use positive 1-based values, but found a non-positive value "
                f"at replay-token position {bad_position}."
            )

        if stop_length != normal_token_ids.numel():
            raise RuntimeError(
                f"SGLang dLLM stop-prefix length mismatch for response "
                f"{response_index}: {DLLM_STOP_LENGTH_KEY}={stop_length}, but "
                f"the normal response contains {normal_token_ids.numel()} tokens."
            )
        if stop_length > replay_token_ids.numel():
            raise RuntimeError(
                f"SGLang dLLM stop length {stop_length} exceeds replay length "
                f"{replay_token_ids.numel()} for response {response_index}."
            )
        if not torch.equal(
            replay_token_ids[:stop_length],
            normal_token_ids,
        ):
            mismatch = torch.nonzero(
                replay_token_ids[:stop_length] != normal_token_ids,
                as_tuple=False,
            )
            bad_position = int(mismatch[0].item())
            raise RuntimeError(
                "SGLang dLLM replay prefix differs from the normal stop prefix "
                f"for response {response_index} at token position "
                f"{bad_position}."
            )

        prompt_length = _require_non_negative_integer(
            prompt_lengths[response_index],
            f"prompt_lengths[{response_index}]",
        )
        replay_length = replay_token_ids.numel()
        if (prompt_length + replay_length) % block_size != 0:
            raise RuntimeError(
                "SGLang dLLM replay response ends inside a global block for "
                f"response {response_index}: prompt_length={prompt_length}, "
                f"replay_length={replay_length}, block_size={block_size}."
            )
        suffix_length = replay_length - stop_length
        if suffix_length >= block_size:
            raise RuntimeError(
                "SGLang dLLM replay payload contains more than one terminal "
                f"block suffix for response {response_index}: "
                f"stop_length={stop_length}, replay_length={replay_length}, "
                f"block_size={block_size}."
            )

        batched_replay_token_ids.append(replay_token_ids)
        batched_local_steps.append(replay_steps - 1)
        stop_lengths.append(stop_length)

    return (
        pad_sequence(
            batched_replay_token_ids,
            batch_first=True,
            padding_value=pad_token_id,
        ),
        pad_sequence(
            batched_local_steps,
            batch_first=True,
            padding_value=-1,
        ),
        torch.tensor(stop_lengths, dtype=torch.long),
    )


def post_process_dllm_step_maps(
    outputs,
    *,
    expected_num_responses=None,
    contract_version=CJ_TRAJECTORY_CONTRACT_VERSION,
    prompt_lengths=None,
    block_size=None,
):
    """Validate legacy token-aligned dLLM step maps.

    This helper is retained for direct compatibility tests and non-terminal
    metadata consumers. Active CJ rollout uses
    :func:`post_process_dllm_replay_metadata` and requires the versioned replay
    payload. A positive, 1-based integer step is required for every returned
    token; missing, truncated, zero-based, or padded metadata is rejected
    instead of being repaired silently.

    When ``prompt_lengths`` and ``block_size`` are supplied, every returned
    sequence must end on the global dLLM block grid. SGLang denoises a complete
    block before applying EOS/max-length truncation; a partial returned prefix
    therefore cannot be replayed exactly from token-aligned step maps alone.
    """

    _require_contract_version(contract_version)
    if not isinstance(outputs, (list, tuple)):
        raise TypeError(
            "SGLang batched output must be a list or tuple, got "
            f"{type(outputs).__name__}"
        )
    if expected_num_responses is not None:
        expected_num_responses = _require_positive_integer(
            expected_num_responses,
            "expected_num_responses",
        )
        if len(outputs) != expected_num_responses:
            raise RuntimeError(
                "SGLang CJ response ordering/count contract failed: expected "
                f"{expected_num_responses} prompt-major responses, got "
                f"{len(outputs)}."
            )
    if not outputs:
        raise RuntimeError("SGLang returned an empty batch for CJ rollout")

    if (prompt_lengths is None) != (block_size is None):
        raise ValueError(
            "prompt_lengths and block_size must be provided together for "
            "CJ block-boundary validation"
        )
    if prompt_lengths is not None:
        if not isinstance(prompt_lengths, Sequence):
            raise TypeError("prompt_lengths must be a per-response sequence")
        if len(prompt_lengths) != len(outputs):
            raise ValueError(
                "prompt_lengths must align with SGLang responses, got "
                f"{len(prompt_lengths)} lengths for {len(outputs)} responses"
            )
        block_size = _require_positive_integer(block_size, "block_size")

    batched_steps = []

    for response_index, response in enumerate(outputs):
        output_len = len(_get_response_token_ids(response))
        if prompt_lengths is not None:
            prompt_length = _require_non_negative_integer(
                prompt_lengths[response_index],
                f"prompt_lengths[{response_index}]",
            )
            if (prompt_length + output_len) % block_size != 0:
                raise RuntimeError(
                    "SGLang CJ response ends inside a global dLLM block for "
                    f"response {response_index}: prompt_length={prompt_length}, "
                    f"output_length={output_len}, block_size={block_size}. "
                    "The omitted final-block suffix participated in bidirectional "
                    "denoising, so step_maps alone cannot replay this trajectory "
                    "exactly. Generate to a block boundary or return the complete "
                    "final block."
                )
        meta_info = response.get("meta_info", {})
        if DLLM_STEP_MAP_KEY not in meta_info:
            raise RuntimeError(
                "SGLang dLLM step-map capability check failed for response "
                f"{response_index}: meta_info[{DLLM_STEP_MAP_KEY!r}] is "
                "absent. Use an editable SGLang build that exposes generic "
                "token-aligned dLLM step maps."
            )

        raw_steps = meta_info[DLLM_STEP_MAP_KEY]
        try:
            steps = torch.as_tensor(raw_steps)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"CJ steps for response {response_index} must be a 1-D "
                "integer sequence"
            ) from exc
        if steps.ndim != 1:
            raise ValueError(
                f"CJ steps for response {response_index} must be 1-D, got "
                f"shape {tuple(steps.shape)}"
            )
        if (
            steps.dtype == torch.bool
            or steps.dtype.is_floating_point
            or steps.dtype.is_complex
        ):
            raise TypeError(
                f"CJ steps for response {response_index} must use an integer "
                f"dtype, got {steps.dtype}"
            )
        if steps.numel() != output_len:
            raise RuntimeError(
                f"CJ metadata/token length mismatch for response "
                f"{response_index}: got {steps.numel()} steps for "
                f"{output_len} returned tokens."
            )
        steps = steps.to(dtype=torch.long)
        if (steps <= 0).any():
            bad_position = int(torch.nonzero(steps <= 0, as_tuple=False)[0].item())
            raise RuntimeError(
                f"CJ metadata for response {response_index} must use positive "
                "1-based SGLang steps, but found a non-positive value at "
                f"returned-token position {bad_position}."
            )
        batched_steps.append(steps - 1)

    return pad_sequence(batched_steps, batch_first=True, padding_value=-1)


def _validate_step_tensor(local_unmask_steps, response_attention_mask):
    if not isinstance(local_unmask_steps, torch.Tensor):
        raise TypeError("local_unmask_steps must be a torch.Tensor")
    if not isinstance(response_attention_mask, torch.Tensor):
        raise TypeError("response_attention_mask must be a torch.Tensor")
    if local_unmask_steps.ndim != 2:
        raise ValueError(
            "local_unmask_steps must be rank 2 [batch, response], got "
            f"shape {tuple(local_unmask_steps.shape)}"
        )
    if response_attention_mask.ndim != 2:
        raise ValueError(
            "response_attention_mask must be rank 2 [batch, response], got "
            f"shape {tuple(response_attention_mask.shape)}"
        )
    if local_unmask_steps.shape != response_attention_mask.shape:
        raise ValueError(
            "local_unmask_steps and response_attention_mask must have the same "
            f"shape, got {local_unmask_steps.shape} and "
            f"{response_attention_mask.shape}"
        )
    if local_unmask_steps.device != response_attention_mask.device:
        raise ValueError(
            "local_unmask_steps and response_attention_mask must be on the "
            f"same device, got {local_unmask_steps.device} and "
            f"{response_attention_mask.device}"
        )
    if (
        local_unmask_steps.dtype == torch.bool
        or local_unmask_steps.dtype.is_floating_point
        or local_unmask_steps.dtype.is_complex
    ):
        raise TypeError(
            "local_unmask_steps must use an integer dtype, got "
            f"{local_unmask_steps.dtype}"
        )
    if response_attention_mask.dtype.is_complex:
        raise TypeError("response_attention_mask cannot use a complex dtype")
    if (
        response_attention_mask.dtype.is_floating_point
        and not torch.isfinite(response_attention_mask).all()
    ):
        raise ValueError(
            "response_attention_mask must contain only finite binary values"
        )
    if not ((response_attention_mask == 0) | (response_attention_mask == 1)).all():
        raise ValueError("response_attention_mask must contain only 0/1 values")


def build_block_parallel_cj_trajectory(
    local_unmask_steps,
    response_attention_mask,
    prompt_length,
    num_steps,
):
    """Build ``[batch, max_local_steps, sequence]`` unmask indicators.

    Tokens from different response blocks that were selected at the same
    block-local denoising step share one trajectory slice. BD3LM attention
    isolates each noisy block, so the actor can score that slice in one forward.
    """

    _validate_step_tensor(local_unmask_steps, response_attention_mask)
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, Integral):
        raise TypeError(
            "prompt_length must be a non-negative integer, got " f"{prompt_length!r}"
        )
    prompt_length = int(prompt_length)
    if prompt_length < 0:
        raise ValueError(f"prompt_length must be non-negative, got {prompt_length}")

    num_steps = _require_positive_integer(num_steps, "num_steps")

    response_attention_mask = response_attention_mask.bool()
    missing = response_attention_mask & (local_unmask_steps < 0)
    if missing.any():
        batch_index, response_position = torch.nonzero(missing, as_tuple=False)[
            0
        ].tolist()
        raise RuntimeError(
            "CJ trajectory is missing a step for valid response token at "
            f"batch {batch_index}, response position {response_position}."
        )
    out_of_bounds = response_attention_mask & (local_unmask_steps >= num_steps)
    if out_of_bounds.any():
        batch_index, response_position = torch.nonzero(out_of_bounds, as_tuple=False)[
            0
        ].tolist()
        bad_step = int(local_unmask_steps[batch_index, response_position].item())
        raise RuntimeError(
            f"CJ step {bad_step} at batch {batch_index}, response position "
            f"{response_position} is outside configured num_steps={num_steps}."
        )
    padded_assignment = (~response_attention_mask) & (local_unmask_steps != -1)
    if padded_assignment.any():
        batch_index, response_position = torch.nonzero(
            padded_assignment, as_tuple=False
        )[0].tolist()
        raise RuntimeError(
            "CJ trajectory assigns a step to padding/post-EOS token at "
            f"batch {batch_index}, response position {response_position}."
        )

    valid = response_attention_mask

    batch_size, response_length = local_unmask_steps.shape
    response_trajectory = torch.zeros(
        (batch_size, num_steps, response_length),
        dtype=torch.bool,
        device=local_unmask_steps.device,
    )
    batch_indices, response_positions = torch.nonzero(valid, as_tuple=True)
    step_indices = local_unmask_steps[batch_indices, response_positions]
    response_trajectory[batch_indices, step_indices, response_positions] = True

    assignment_count = response_trajectory.sum(dim=1)
    if not torch.equal(
        assignment_count, response_attention_mask.to(assignment_count.dtype)
    ):
        raise RuntimeError(
            "CJ trajectory contract violation: every valid response token must "
            "be assigned to exactly one local step."
        )

    prompt_trajectory = torch.zeros(
        (batch_size, num_steps, prompt_length),
        dtype=torch.bool,
        device=local_unmask_steps.device,
    )
    return torch.cat((prompt_trajectory, response_trajectory), dim=-1)


def infer_max_local_cj_steps(local_unmask_steps):
    """Synchronize the maximum block-local step count across rollout ranks."""

    if not isinstance(local_unmask_steps, torch.Tensor):
        raise TypeError("local_unmask_steps must be a torch.Tensor")
    if local_unmask_steps.ndim != 2:
        raise ValueError(
            "local_unmask_steps must be rank 2 [batch, response], got "
            f"shape {tuple(local_unmask_steps.shape)}"
        )
    if (
        local_unmask_steps.dtype == torch.bool
        or local_unmask_steps.dtype.is_floating_point
        or local_unmask_steps.dtype.is_complex
    ):
        raise TypeError(
            "local_unmask_steps must use an integer dtype, got "
            f"{local_unmask_steps.dtype}"
        )
    if (local_unmask_steps < -1).any():
        raise ValueError("local_unmask_steps may use only -1 as its padding sentinel")

    valid_steps = local_unmask_steps[local_unmask_steps >= 0]
    num_steps = int(valid_steps.max().item()) + 1 if valid_steps.numel() > 0 else 0
    if dist.is_available() and dist.is_initialized():
        num_steps_tensor = torch.tensor(
            num_steps,
            dtype=torch.long,
            device=local_unmask_steps.device,
        )
        dist.all_reduce(num_steps_tensor, op=dist.ReduceOp.MAX)
        num_steps = int(num_steps_tensor.item())
    return max(num_steps, 1)
