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

"""CPU references for paper-exact block-parallel CJ-GRPO.

The CJ-GRPO paper averages adjacent denoising-transition losses within each
sample, then averages the sample losses across the batch (Eq. 11).  Combining
different blocks that share a block-local step may reduce model forwards, but
must not merge those distinct transitions in the loss reduction.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_TOKEN_ID = 63


def _load_repo_module(name: str, relative_path: str):
    """Load a leaf module without importing verl's optional runtime stack."""

    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cj_trajectory = _load_repo_module(
    "dare_test_cj_trajectory",
    "verl/workers/rollout/cj_trajectory.py",
)
block_diffusion_utils = _load_repo_module(
    "dare_test_block_diffusion_utils",
    "verl/workers/actor/block_diffusion_utils.py",
)


def _validate_cj_tensors(
    step_values: torch.Tensor,
    trajectory: torch.Tensor,
    response_mask: torch.Tensor,
    response_block_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if step_values.ndim != 3:
        raise ValueError(
            f"step_values must have shape [B, K, R], got {step_values.shape}"
        )
    if trajectory.ndim != 3:
        raise ValueError(
            f"trajectory must have shape [B, K, S], got {trajectory.shape}"
        )

    batch_size, num_steps, response_length = step_values.shape
    if trajectory.shape[:2] != (batch_size, num_steps):
        raise ValueError("trajectory batch/local-step dimensions do not match")
    if trajectory.size(-1) < response_length:
        raise ValueError("trajectory is shorter than the response")
    if response_mask.shape != (batch_size, response_length):
        raise ValueError("response_mask must have shape [B, R]")
    if response_block_ids.shape != (batch_size, response_length):
        raise ValueError("response_block_ids must have shape [B, R]")

    response_mask = response_mask.bool()
    step_mask = trajectory[:, :, -response_length:].bool()
    assignment_count = step_mask.sum(dim=1)
    if (response_mask & (assignment_count != 1)).any():
        raise ValueError("Each valid response token must be assigned exactly once")
    if (step_mask & ~response_mask[:, None, :]).any():
        raise ValueError("CJ trajectory assigns an invalid response token")
    if (response_mask & (response_block_ids < 0)).any():
        raise ValueError("Each valid response token must have a block id")
    return step_mask, response_mask


def paper_exact_cj_loss(
    step_values: torch.Tensor,
    trajectory: torch.Tensor,
    response_mask: torch.Tensor,
    response_block_ids: torch.Tensor,
) -> torch.Tensor:
    """Eq. 11 reference: token mean per transition, then sample/batch means."""

    step_mask, response_mask = _validate_cj_tensors(
        step_values,
        trajectory,
        response_mask,
        response_block_ids,
    )
    sample_losses = []
    for batch_index in range(step_values.size(0)):
        transition_losses = []
        valid_blocks = torch.unique(
            response_block_ids[batch_index, response_mask[batch_index]],
            sorted=True,
        )
        for block_id in valid_blocks.tolist():
            block_mask = response_block_ids[batch_index] == block_id
            for local_step in range(step_values.size(1)):
                transition_mask = (
                    step_mask[batch_index, local_step]
                    & response_mask[batch_index]
                    & block_mask
                )
                if transition_mask.any():
                    transition_losses.append(
                        step_values[batch_index, local_step, transition_mask].mean()
                    )
        if not transition_losses:
            raise ValueError("Every CJ sample must contain at least one transition")
        sample_losses.append(torch.stack(transition_losses).mean())
    return torch.stack(sample_losses).mean()


def local_step_mean_loss(
    step_values: torch.Tensor,
    trajectory: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """The non-paper objective used to expose accidental local-step merging."""

    response_length = response_mask.size(1)
    step_mask = trajectory[:, :, -response_length:].bool()
    active_step_losses = []
    for local_step in range(step_values.size(1)):
        selected = step_mask[:, local_step] & response_mask.bool()
        if selected.any():
            active_step_losses.append(step_values[:, local_step][selected].mean())
    if not active_step_losses:
        raise ValueError("CJ batch contains no active local steps")
    return torch.stack(active_step_losses).mean()


def ppo_token_losses(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> torch.Tensor:
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    objective = torch.minimum(
        ratio * advantages[:, None, :],
        clipped_ratio * advantages[:, None, :],
    )
    return -objective


def make_231_case(batch_size: int = 1):
    """Return three response blocks with respectively 2, 3 and 1 steps."""

    # Four tokens per block.  The largest local step in each block is 1/2/0.
    local_steps = torch.tensor(
        [0, 0, 1, 1, 0, 1, 2, 2, 0, 0, 0, 0],
        dtype=torch.long,
    ).repeat(batch_size, 1)
    response_mask = torch.ones_like(local_steps, dtype=torch.bool)
    trajectory = cj_trajectory.build_block_parallel_cj_trajectory(
        local_unmask_steps=local_steps,
        response_attention_mask=response_mask,
        prompt_length=4,
        num_steps=3,
    )
    response_block_ids = (
        torch.arange(local_steps.size(1), dtype=torch.long) // 4
    ).repeat(batch_size, 1)
    return local_steps, response_mask, trajectory, response_block_ids


class ToyBD3LM(nn.Module):
    """A differentiable block-local scorer with BD3LM visibility semantics."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 7):
        super().__init__()
        generator = torch.Generator().manual_seed(20260717)
        embedding = torch.randn(
            vocab_size,
            hidden_size,
            generator=generator,
            dtype=torch.float64,
        )
        self.embedding = nn.Parameter(embedding / hidden_size**0.5)
        self.projection = nn.Parameter(
            torch.randn(hidden_size, generator=generator, dtype=torch.float64)
        )
        self.bias = nn.Parameter(torch.tensor(0.13, dtype=torch.float64))
        self.forward_calls = 0

    def forward(
        self,
        clean_input_ids: torch.Tensor,
        noisy_input_ids: torch.Tensor,
        target_mask: torch.Tensor,
        block_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        rows = []
        for batch_index in range(clean_input_ids.size(0)):
            values = []
            for token_position in range(clean_input_ids.size(1)):
                if not bool(target_mask[batch_index, token_position]):
                    values.append(self.bias * 0.0)
                    continue

                target_block = block_ids[batch_index, token_position]
                prior_clean = (block_ids[batch_index] >= 0) & (
                    block_ids[batch_index] < target_block
                )
                current_visible = (block_ids[batch_index] == target_block) & (
                    noisy_input_ids[batch_index] != MASK_TOKEN_ID
                )
                visible_ids = torch.cat(
                    (
                        clean_input_ids[batch_index, prior_clean],
                        noisy_input_ids[batch_index, current_visible],
                    )
                )
                context = self.embedding[visible_ids].mean(dim=0)
                target = self.embedding[clean_input_ids[batch_index, token_position]]
                score = (context * self.projection * target).sum() + self.bias
                values.append(F.logsigmoid(score))
            rows.append(torch.stack(values))
        return torch.stack(rows)


def score_231_trajectory(model: ToyBD3LM, *, group_blocks: bool):
    _, response_mask, trajectory, response_block_ids = make_231_case()
    prompt_ids = torch.tensor([[2, 3, 5, 7]], dtype=torch.long)
    response_ids = torch.arange(11, 23, dtype=torch.long).unsqueeze(0)
    clean_input_ids = torch.cat((prompt_ids, response_ids), dim=1)
    attention_mask = torch.ones_like(clean_input_ids, dtype=torch.bool)
    full_block_ids = (
        torch.arange(clean_input_ids.size(1), dtype=torch.long) // 4
    ).unsqueeze(0)

    step_outputs = [
        model.bias * torch.zeros_like(clean_input_ids, dtype=torch.float64)
        for _ in range(3)
    ]
    for local_step in range(3):
        noisy_input_ids, local_targets, _ = (
            block_diffusion_utils.build_block_parallel_cj_step_inputs(
                clean_input_ids=clean_input_ids,
                attention_mask=attention_mask,
                trajectory=trajectory,
                response_length=response_ids.size(1),
                local_step=local_step,
                mask_token_id=MASK_TOKEN_ID,
            )
        )
        if group_blocks:
            step_outputs[local_step] = model(
                clean_input_ids,
                noisy_input_ids,
                local_targets,
                full_block_ids,
            )
            continue

        # Serial reference: one call for each real (block, local-step)
        # transition.  There are 2 + 3 + 1 = 6 calls in this fixture.
        for response_block in range(3):
            block_targets = local_targets & (full_block_ids == response_block + 1)
            if block_targets.any():
                step_outputs[local_step] = step_outputs[local_step] + model(
                    clean_input_ids,
                    noisy_input_ids,
                    block_targets,
                    full_block_ids,
                )

    step_log_probs = torch.stack(
        [output[:, -response_ids.size(1) :] for output in step_outputs],
        dim=1,
    )
    return step_log_probs, trajectory, response_mask, response_block_ids


class TestPaperExactCJGRPO(unittest.TestCase):
    def test_231_serial_six_transitions_matches_three_grouped_forwards(self):
        serial_model = ToyBD3LM()
        grouped_model = copy.deepcopy(serial_model)

        serial_log_probs, trajectory, response_mask, block_ids = score_231_trajectory(
            serial_model,
            group_blocks=False,
        )
        grouped_log_probs, _, _, _ = score_231_trajectory(
            grouped_model,
            group_blocks=True,
        )
        self.assertEqual(serial_model.forward_calls, 6)
        self.assertEqual(grouped_model.forward_calls, 3)

        selected = trajectory[:, :, -response_mask.size(1) :]
        torch.testing.assert_close(
            serial_log_probs[selected],
            grouped_log_probs[selected],
            rtol=0.0,
            atol=1e-12,
        )

        offsets = torch.linspace(
            -0.35,
            0.35,
            response_mask.size(1),
            dtype=torch.float64,
        )[None, None, :]
        old_log_probs = serial_log_probs.detach() - offsets
        advantages = torch.tensor(
            [[0.9, -0.4, 0.3, -0.8, 1.1, -0.6, 0.7, -0.2, 0.5, -1.0, 0.4, -0.3]],
            dtype=torch.float64,
        )
        serial_token_losses = ppo_token_losses(
            serial_log_probs,
            old_log_probs,
            advantages,
        )
        grouped_token_losses = ppo_token_losses(
            grouped_log_probs,
            old_log_probs,
            advantages,
        )
        serial_loss = paper_exact_cj_loss(
            serial_token_losses,
            trajectory,
            response_mask,
            block_ids,
        )
        grouped_loss = paper_exact_cj_loss(
            grouped_token_losses,
            trajectory,
            response_mask,
            block_ids,
        )
        torch.testing.assert_close(serial_loss, grouped_loss, rtol=0.0, atol=1e-12)
        self.assertGreater(abs(float(serial_loss)), 1e-6)

        serial_loss.backward()
        grouped_loss.backward()
        nonzero_gradients = 0
        for (serial_name, serial_parameter), (grouped_name, grouped_parameter) in zip(
            serial_model.named_parameters(),
            grouped_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(serial_name, grouped_name)
            self.assertIsNotNone(serial_parameter.grad)
            self.assertIsNotNone(grouped_parameter.grad)
            torch.testing.assert_close(
                serial_parameter.grad,
                grouped_parameter.grad,
                rtol=1e-11,
                atol=1e-12,
            )
            nonzero_gradients += int(torch.count_nonzero(serial_parameter.grad))
        self.assertGreater(nonzero_gradients, 0)

    def test_paper_exact_reduction_does_not_merge_distinct_block_transitions(self):
        _, response_mask, trajectory, block_ids = make_231_case()
        transition_values = {
            (0, 0): 1.0,
            (0, 1): 2.0,
            (1, 0): 10.0,
            (1, 1): 20.0,
            (1, 2): 30.0,
            (2, 0): 100.0,
        }
        values = torch.zeros((1, 3, 12), dtype=torch.float64)
        step_mask = trajectory[:, :, -12:]
        for (block_id, local_step), value in transition_values.items():
            selected = step_mask[0, local_step] & (block_ids[0] == block_id)
            values[0, local_step, selected] = value

        paper_loss = paper_exact_cj_loss(
            values,
            trajectory,
            response_mask,
            block_ids,
        )
        merged_loss = local_step_mean_loss(values, trajectory, response_mask)
        expected = torch.tensor(
            sum(transition_values.values()) / len(transition_values),
            dtype=torch.float64,
        )
        torch.testing.assert_close(paper_loss, expected)
        self.assertFalse(torch.isclose(paper_loss, merged_loss))

    def test_paper_exact_reduction_is_microbatch_partition_invariant(self):
        local_steps, response_mask, _, block_ids = make_231_case(batch_size=4)
        # EOS is valid; only positions after EOS are removed from the objective.
        response_mask[1, 9:] = False
        response_mask[3, 6:] = False
        local_steps = local_steps.masked_fill(~response_mask, -1)
        trajectory = cj_trajectory.build_block_parallel_cj_trajectory(
            local_unmask_steps=local_steps,
            response_attention_mask=response_mask,
            prompt_length=4,
            num_steps=3,
        )
        values = torch.arange(4 * 3 * 12, dtype=torch.float64).reshape(4, 3, 12)
        values = values.square() / 97.0 - 3.0

        full_loss = paper_exact_cj_loss(
            values,
            trajectory,
            response_mask,
            block_ids,
        )
        split_loss = values.new_zeros(())
        for start, end in ((0, 1), (1, 3), (3, 4)):
            local_loss = paper_exact_cj_loss(
                values[start:end],
                trajectory[start:end],
                response_mask[start:end],
                block_ids[start:end],
            )
            split_loss = split_loss + local_loss * ((end - start) / values.size(0))
        torch.testing.assert_close(full_loss, split_loss, rtol=0.0, atol=1e-12)

    def test_sglang_metadata_to_old_logprob_to_optimizer_step(self):
        """Smoke the complete CJ hand-off without requiring SGLang or CUDA."""

        local_steps = [0, 0, 1, 1, 0, 1, 2, 2, 0, 0, 0, 0]
        outputs = [
            {
                "token_ids": list(range(11, 23)),
                "meta_info": {
                    cj_trajectory.DLLM_STEP_MAP_KEY: [
                        step + 1 for step in local_steps
                    ],
                },
            }
        ]
        collected_steps = cj_trajectory.post_process_dllm_step_maps(
            outputs,
            expected_num_responses=1,
        )
        response_mask = torch.ones_like(collected_steps, dtype=torch.bool)
        rollout_trajectory = cj_trajectory.build_block_parallel_cj_trajectory(
            local_unmask_steps=collected_steps,
            response_attention_mask=response_mask,
            prompt_length=4,
            num_steps=3,
        )
        _, _, expected_trajectory, response_block_ids = make_231_case()
        torch.testing.assert_close(rollout_trajectory, expected_trajectory)

        model = ToyBD3LM()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        new_log_probs, _, _, _ = score_231_trajectory(
            model,
            group_blocks=True,
        )
        offsets = torch.linspace(
            -0.1,
            0.1,
            new_log_probs.size(-1),
            dtype=new_log_probs.dtype,
        )[None, None, :]
        old_log_probs = new_log_probs.detach() - offsets
        advantages = torch.tensor(
            [
                [
                    0.8,
                    -0.4,
                    0.6,
                    -0.2,
                    1.0,
                    -0.7,
                    0.3,
                    -0.5,
                    0.9,
                    -0.1,
                    0.4,
                    -0.6,
                ]
            ],
            dtype=new_log_probs.dtype,
        )
        loss = paper_exact_cj_loss(
            ppo_token_losses(new_log_probs, old_log_probs, advantages),
            rollout_trajectory,
            response_mask,
            response_block_ids,
        )

        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        loss.backward()
        nonzero_gradient_count = sum(
            int(torch.count_nonzero(parameter.grad).item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(nonzero_gradient_count, 0)
        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(parameters_before[name], parameter.detach())
                for name, parameter in model.named_parameters()
            )
        )

    def test_reference_rejects_missing_duplicate_and_padded_assignments(self):
        _, response_mask, trajectory, block_ids = make_231_case()
        values = torch.ones((1, 3, 12), dtype=torch.float64)

        missing = trajectory.clone()
        missing[0, :, 4] = False
        duplicate = trajectory.clone()
        duplicate[0, 1, 4] = True
        padded = trajectory.clone()
        padded_response_mask = response_mask.clone()
        padded_response_mask[0, -1] = False

        for invalid_trajectory, invalid_mask in (
            (missing, response_mask),
            (duplicate, response_mask),
            (padded, padded_response_mask),
        ):
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "assigned|invalid"):
                    paper_exact_cj_loss(
                        values,
                        invalid_trajectory,
                        invalid_mask,
                        block_ids,
                    )


class TestCJTrajectoryContract(unittest.TestCase):
    def test_build_trajectory_assigns_valid_tokens_once_and_excludes_post_eos(self):
        local_steps = torch.tensor(
            [[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]],
            dtype=torch.long,
        )
        # EOS positions (3 and 4) remain valid.  Tokens after EOS are padding.
        response_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]],
            dtype=torch.bool,
        )
        local_steps = local_steps.masked_fill(~response_mask, -1)
        trajectory = cj_trajectory.build_block_parallel_cj_trajectory(
            local_unmask_steps=local_steps,
            response_attention_mask=response_mask,
            prompt_length=3,
            num_steps=3,
        )
        self.assertEqual(tuple(trajectory.shape), (2, 3, 9))
        self.assertFalse(bool(trajectory[:, :, :3].any()))
        torch.testing.assert_close(
            trajectory[:, :, 3:].sum(dim=1),
            response_mask.to(torch.long),
        )
        assigned_to_padding = trajectory[:, :, 3:] & ~response_mask[:, None, :]
        self.assertFalse(bool(assigned_to_padding.any()))

    def test_build_trajectory_rejects_valid_out_of_range_steps(self):
        response_mask = torch.ones((1, 3), dtype=torch.bool)
        for invalid_steps in (
            torch.tensor([[0, 3, 1]], dtype=torch.long),
            torch.tensor([[0, -1, 1]], dtype=torch.long),
        ):
            with self.subTest(steps=invalid_steps.tolist()):
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError),
                    "step|range|assigned|missing",
                ):
                    cj_trajectory.build_block_parallel_cj_trajectory(
                        local_unmask_steps=invalid_steps,
                        response_attention_mask=response_mask,
                        prompt_length=2,
                        num_steps=3,
                    )

    def test_build_trajectory_rejects_assignment_after_eos(self):
        with self.assertRaisesRegex(RuntimeError, "padding|post-EOS"):
            cj_trajectory.build_block_parallel_cj_trajectory(
                local_unmask_steps=torch.tensor([[0, 1, 2]], dtype=torch.long),
                response_attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
                prompt_length=2,
                num_steps=3,
            )

    def test_post_process_preserves_n_generations_and_token_id_fallbacks(self):
        outputs = [
            {
                "token_ids": [11, 12, 13],
                "meta_info": {"step_maps": [1, 2, 3]},
            },
            {
                "output_ids": [21, 22],
                "meta_info": {"step_maps": [2, 1]},
            },
            {
                "meta_info": {
                    "output_token_logprobs": [(-0.1, 31), (-0.2, 32), (-0.3, 33)],
                    "step_maps": [3, 2, 1],
                }
            },
            {
                "token_ids": [41],
                "meta_info": {"step_maps": [1]},
            },
        ]
        steps = cj_trajectory.post_process_dllm_step_maps(
            outputs,
            expected_num_responses=4,
        )
        expected = torch.tensor(
            [[0, 1, 2], [1, 0, -1], [2, 1, 0], [0, -1, -1]],
            dtype=torch.long,
        )
        torch.testing.assert_close(steps, expected)
        self.assertEqual(steps.size(0), 4)

    def test_n_generation_expansion_is_prompt_major(self):
        prompt_ids = [[11, 12], [21, 22, 23]]
        image_data = [{"image": "p0"}, {"image": "p1"}]
        expanded_ids, expanded_images = cj_trajectory.expand_cj_generation_inputs(
            prompt_ids,
            image_data,
            num_samples=3,
        )
        self.assertEqual(
            expanded_ids,
            [
                prompt_ids[0],
                prompt_ids[0],
                prompt_ids[0],
                prompt_ids[1],
                prompt_ids[1],
                prompt_ids[1],
            ],
        )
        self.assertEqual(
            expanded_images,
            [
                image_data[0],
                image_data[0],
                image_data[0],
                image_data[1],
                image_data[1],
                image_data[1],
            ],
        )


if __name__ == "__main__":
    unittest.main()
