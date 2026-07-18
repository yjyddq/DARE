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

"""CPU tests for paper-exact block-parallel CJ actor utilities."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


block_utils = _load_module(
    "dare_test_cj_block_utils",
    "verl/workers/actor/block_diffusion_utils.py",
)
trajectory_utils = _load_module(
    "dare_test_cj_trajectory_for_actor",
    "verl/workers/rollout/cj_trajectory.py",
)


def _make_case():
    batch_size, num_steps, prompt_length, response_length = 4, 3, 4, 12
    local_steps = torch.tensor(
        [0, 0, 1, 1, 0, 1, 2, 2, 0, 0, 0, 0],
        dtype=torch.long,
    ).repeat(batch_size, 1)
    response_mask = torch.ones_like(local_steps, dtype=torch.bool)
    response_mask[1, 9:] = False
    response_mask[3, 6:] = False
    local_steps = local_steps.masked_fill(~response_mask, -1)
    trajectory = trajectory_utils.build_block_parallel_cj_trajectory(
        local_unmask_steps=local_steps,
        response_attention_mask=response_mask,
        prompt_length=prompt_length,
        num_steps=num_steps,
    )
    attention_mask = torch.cat(
        (
            torch.ones((batch_size, prompt_length), dtype=torch.bool),
            response_mask,
        ),
        dim=1,
    )
    values = torch.arange(
        batch_size * num_steps * response_length,
        dtype=torch.float32,
    ).reshape(batch_size, num_steps, response_length)
    values = values.square() / 97.0 - 3.0
    return values, trajectory, response_mask, attention_mask, prompt_length


def _paper_reference(
    values: torch.Tensor,
    trajectory: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    response_length = response_mask.size(1)
    block_ids = block_utils.build_padded_token_block_ids(
        attention_mask,
        prompt_section_length=prompt_length,
        block_size=4,
        block_origin="global",
    )[:, -response_length:]
    step_mask = trajectory[:, :, -response_length:].bool()
    sample_losses = []
    for batch_index in range(values.size(0)):
        transition_losses = []
        for block_id in torch.unique(
            block_ids[batch_index, response_mask[batch_index]]
        ):
            for local_step in range(values.size(1)):
                selected = (
                    step_mask[batch_index, local_step]
                    & response_mask[batch_index]
                    & (block_ids[batch_index] == block_id)
                )
                if selected.any():
                    transition_losses.append(
                        values[batch_index, local_step, selected].mean()
                    )
        sample_losses.append(torch.stack(transition_losses).mean())
    return torch.stack(sample_losses).mean()


def _production_reduction(
    values: torch.Tensor,
    trajectory: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
    batch_denominator: int,
) -> torch.Tensor:
    token_weights, step_masses = block_utils.compute_cj_block_step_token_weights(
        trajectory,
        response_mask,
        attention_mask,
        prompt_section_length=prompt_length,
        block_size=4,
        block_origin="global",
    )
    result = values.new_zeros(())
    for local_step in range(values.size(1)):
        weights = token_weights[:, local_step]
        mass = weights.sum()
        torch.testing.assert_close(mass, step_masses[local_step])
        if mass.item() == 0:
            continue
        # Mirrors agg_loss(token-mean) and the actor's outer coefficient.
        local_mean = (values[:, local_step] * weights).sum() / (mass + 1e-8)
        result = result + local_mean * (mass + 1e-8) / batch_denominator
    return result


class TestCJBlockParallelUtilities(unittest.TestCase):
    def test_weighted_aggregation_matches_paper_exact_reference(self):
        values, trajectory, response_mask, attention_mask, prompt_length = _make_case()
        expected = _paper_reference(
            values,
            trajectory,
            response_mask,
            attention_mask,
            prompt_length,
        )
        actual = _production_reduction(
            values,
            trajectory,
            response_mask,
            attention_mask,
            prompt_length,
            batch_denominator=values.size(0),
        )
        torch.testing.assert_close(actual, expected)

    def test_weighted_aggregation_is_microbatch_invariant(self):
        values, trajectory, response_mask, attention_mask, prompt_length = _make_case()
        full = _production_reduction(
            values,
            trajectory,
            response_mask,
            attention_mask,
            prompt_length,
            batch_denominator=values.size(0),
        )
        split = values.new_zeros(())
        for start, end in ((0, 1), (1, 3), (3, 4)):
            split = split + _production_reduction(
                values[start:end],
                trajectory[start:end],
                response_mask[start:end],
                attention_mask[start:end],
                prompt_length,
                batch_denominator=values.size(0),
            )
        torch.testing.assert_close(split, full)

    def test_padded_block_ids_match_compact_block_mask_ids(self):
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]],
            dtype=torch.bool,
        )
        prompt_section_length = 5  # three valid prompt tokens, four response
        prompt_lengths = attention_mask[:, :prompt_section_length].sum(dim=1)
        compact_valid = torch.ones(
            (1, int(attention_mask.sum().item())),
            dtype=torch.bool,
        )
        for block_origin in ("global", "response"):
            with self.subTest(block_origin=block_origin):
                padded_ids = block_utils.build_padded_token_block_ids(
                    attention_mask,
                    prompt_section_length=prompt_section_length,
                    block_size=4,
                    block_origin=block_origin,
                )
                compact_ids = block_utils.build_token_block_ids(
                    compact_valid,
                    prompt_lengths,
                    block_size=4,
                    block_origin=block_origin,
                )
                torch.testing.assert_close(
                    padded_ids[attention_mask].view(1, -1),
                    compact_ids,
                )

    def test_empty_scatter_keeps_connected_zero_gradient(self):
        compact_values = torch.randn((1, 3), requires_grad=True)
        artifacts = block_utils.BlockDiffusionArtifacts(
            noisy_input_ids=torch.ones((1, 3), dtype=torch.long),
            clean_input_ids=torch.ones((1, 3), dtype=torch.long),
            valid_mask=torch.ones((1, 3), dtype=torch.bool),
            target_mask=torch.zeros((1, 3), dtype=torch.bool),
            p_mask=torch.ones((1, 3)),
            position_ids=torch.arange(3).unsqueeze(0),
            prompt_lengths=torch.tensor([1]),
            response_lengths=torch.tensor([2]),
            response_positions=torch.full((1, 3), -1, dtype=torch.long),
        )
        output = block_utils.scatter_compact_values_to_response(
            compact_values,
            artifacts,
            response_length=2,
        )
        self.assertTrue(output.requires_grad)
        output.sum().backward()
        self.assertIsNotNone(compact_values.grad)
        torch.testing.assert_close(
            compact_values.grad, torch.zeros_like(compact_values)
        )

    def test_noisy_half_can_only_see_prior_clean_blocks(self):
        sequence_length = 12
        clean = torch.arange(sequence_length).unsqueeze(0)
        noisy = clean.clone()
        artifacts = block_utils.BlockDiffusionArtifacts(
            noisy_input_ids=noisy,
            clean_input_ids=clean,
            valid_mask=torch.ones_like(clean, dtype=torch.bool),
            target_mask=torch.zeros_like(clean, dtype=torch.bool),
            p_mask=torch.ones_like(clean, dtype=torch.float32),
            position_ids=clean.clone(),
            prompt_lengths=torch.tensor([4]),
            response_lengths=torch.tensor([8]),
            response_positions=torch.arange(sequence_length).unsqueeze(0) - 4,
        )
        full_input_ids, *_ = block_utils.build_full_block_diffusion_tensors(artifacts)
        torch.testing.assert_close(full_input_ids[:, :sequence_length], noisy)
        torch.testing.assert_close(full_input_ids[:, sequence_length:], clean)

        class CapturedBlockMask:
            @classmethod
            def from_kv_blocks(cls, *args, mask_mod, **kwargs):
                del args, kwargs
                return types.SimpleNamespace(mask_mod=mask_mod)

        original_block_mask = block_utils.BlockMask
        try:
            block_utils.BlockMask = CapturedBlockMask
            mask = block_utils.build_block_diffusion_mask(
                artifacts,
                block_size=4,
                block_origin="global",
            )
        finally:
            block_utils.BlockMask = original_block_mask

        scalar = lambda value: torch.tensor(value, dtype=torch.long)
        # Noisy block 1 query: noisy same-block and clean prior-block visible.
        self.assertTrue(bool(mask.mask_mod(scalar(0), scalar(0), scalar(4), scalar(4))))
        self.assertTrue(
            bool(
                mask.mask_mod(scalar(0), scalar(0), scalar(4), scalar(sequence_length))
            )
        )
        # Clean same/future blocks must not leak into the noisy query.
        self.assertFalse(
            bool(
                mask.mask_mod(
                    scalar(0), scalar(0), scalar(4), scalar(sequence_length + 4)
                )
            )
        )
        self.assertFalse(
            bool(
                mask.mask_mod(
                    scalar(0), scalar(0), scalar(4), scalar(sequence_length + 8)
                )
            )
        )

    def test_noncontiguous_block_local_steps_fail_fast(self):
        response_mask = torch.ones((1, 4), dtype=torch.bool)
        attention_mask = torch.ones((1, 6), dtype=torch.bool)
        trajectory = torch.zeros((1, 3, 6), dtype=torch.bool)
        trajectory[0, 0, 2:4] = True
        trajectory[0, 2, 4:6] = True
        with self.assertRaisesRegex(ValueError, "contiguous local steps"):
            block_utils.compute_cj_block_step_token_weights(
                trajectory,
                response_mask,
                attention_mask,
                prompt_section_length=2,
                block_size=4,
                block_origin="response",
            )


if __name__ == "__main__":
    unittest.main()
