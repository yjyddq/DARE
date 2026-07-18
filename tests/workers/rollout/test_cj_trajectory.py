import importlib.util
import unittest
from pathlib import Path

import torch


# Load the pure tensor helper without importing ``verl.__init__``.  The latter
# pulls in Ray, which is intentionally unnecessary for this CPU-only unit test.
_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "verl"
    / "workers"
    / "rollout"
    / "cj_trajectory.py"
)
_SPEC = importlib.util.spec_from_file_location("cj_trajectory_under_test", _MODULE_PATH)
_CJ_TRAJECTORY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CJ_TRAJECTORY)

DLLM_STEP_MAP_KEY = _CJ_TRAJECTORY.DLLM_STEP_MAP_KEY
build_block_parallel_cj_trajectory = _CJ_TRAJECTORY.build_block_parallel_cj_trajectory
expand_cj_generation_inputs = _CJ_TRAJECTORY.expand_cj_generation_inputs
infer_max_local_cj_steps = _CJ_TRAJECTORY.infer_max_local_cj_steps
post_process_dllm_step_maps = _CJ_TRAJECTORY.post_process_dllm_step_maps


def _response(token_ids, steps):
    return {
        "token_ids": token_ids,
        "meta_info": {DLLM_STEP_MAP_KEY: steps},
    }


class CJTrajectoryMetadataTest(unittest.TestCase):
    def test_collects_and_pads_exact_metadata(self):
        steps = post_process_dllm_step_maps(
            [
                _response([10, 11, 12], [1, 2, 1]),
                _response([20, 21], [3, 1]),
            ],
            expected_num_responses=2,
        )

        torch.testing.assert_close(
            steps,
            torch.tensor([[0, 1, 0], [2, 0, -1]]),
        )

    def test_requires_capability_on_every_response(self):
        outputs = [
            _response([10], [1]),
            {"token_ids": [20], "meta_info": {}},
        ]
        with self.assertRaisesRegex(RuntimeError, "capability check failed"):
            post_process_dllm_step_maps(
                outputs,
                expected_num_responses=2,
            )

    def test_rejects_response_count_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "expected 4"):
            post_process_dllm_step_maps(
                [_response([10], [1])],
                expected_num_responses=4,
            )

    def test_rejects_metadata_length_repair(self):
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            post_process_dllm_step_maps(
                [_response([10, 11], [1])],
                expected_num_responses=1,
            )

    def test_rejects_non_integer_and_non_positive_steps(self):
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            post_process_dllm_step_maps(
                [_response([10], [1.0])],
                expected_num_responses=1,
            )
        with self.assertRaisesRegex(RuntimeError, "positive 1-based"):
            post_process_dllm_step_maps(
                [_response([10], [0])],
                expected_num_responses=1,
            )

    def test_rejects_unsupported_contract_version(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CJ trajectory"):
            post_process_dllm_step_maps(
                [_response([10], [1])],
                expected_num_responses=1,
                contract_version=2,
            )

    def test_accepts_response_on_global_block_boundary(self):
        steps = post_process_dllm_step_maps(
            [_response([10, 11], [1, 2])],
            expected_num_responses=1,
            prompt_lengths=[2],
            block_size=4,
        )
        torch.testing.assert_close(steps, torch.tensor([[0, 1]]))

    def test_rejects_partial_final_block(self):
        with self.assertRaisesRegex(RuntimeError, "ends inside a global dLLM block"):
            post_process_dllm_step_maps(
                [_response([10, 11], [1, 2])],
                expected_num_responses=1,
                prompt_lengths=[1],
                block_size=4,
            )


class CJPromptMajorExpansionTest(unittest.TestCase):
    def test_expands_prompt_major_for_n_samples(self):
        prompts, images = expand_cj_generation_inputs(
            [[10], [20]],
            ["image-a", "image-b"],
            3,
        )
        self.assertEqual(prompts, [[10], [10], [10], [20], [20], [20]])
        self.assertEqual(
            images,
            ["image-a", "image-a", "image-a", "image-b", "image-b", "image-b"],
        )

    def test_rejects_unaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "same batch length"):
            expand_cj_generation_inputs([[10]], [], 2)

    def test_rejects_non_integer_sample_count(self):
        with self.assertRaisesRegex(TypeError, "positive integer"):
            expand_cj_generation_inputs([[10]], [None], 2.0)


class CJTrajectoryBuildTest(unittest.TestCase):
    def test_builds_exactly_once_assignment(self):
        local_steps = torch.tensor(
            [
                [0, 1, 0, -1],
                [2, 0, -1, -1],
            ]
        )
        response_mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ]
        )

        trajectory = build_block_parallel_cj_trajectory(
            local_steps,
            response_mask,
            prompt_length=2,
            num_steps=3,
        )

        self.assertEqual(tuple(trajectory.shape), (2, 3, 6))
        self.assertFalse(trajectory[:, :, :2].any())
        torch.testing.assert_close(
            trajectory[:, :, 2:].sum(dim=1),
            response_mask,
        )

    def test_rejects_missing_valid_assignment(self):
        with self.assertRaisesRegex(RuntimeError, "missing a step"):
            build_block_parallel_cj_trajectory(
                torch.tensor([[0, -1]]),
                torch.tensor([[1, 1]]),
                prompt_length=1,
                num_steps=1,
            )

    def test_rejects_step_outside_explicit_bound(self):
        with self.assertRaisesRegex(RuntimeError, "outside configured"):
            build_block_parallel_cj_trajectory(
                torch.tensor([[0, 2]]),
                torch.tensor([[1, 1]]),
                prompt_length=1,
                num_steps=2,
            )

    def test_rejects_assignment_after_eos(self):
        with self.assertRaisesRegex(RuntimeError, "padding/post-EOS"):
            build_block_parallel_cj_trajectory(
                torch.tensor([[0, 1]]),
                torch.tensor([[1, 0]]),
                prompt_length=1,
                num_steps=2,
            )

    def test_rejects_non_binary_mask_and_non_integer_prompt_length(self):
        with self.assertRaisesRegex(ValueError, "only 0/1"):
            build_block_parallel_cj_trajectory(
                torch.tensor([[0, 1]]),
                torch.tensor([[1, 2]]),
                prompt_length=1,
                num_steps=2,
            )
        with self.assertRaisesRegex(TypeError, "non-negative integer"):
            build_block_parallel_cj_trajectory(
                torch.tensor([[0]]),
                torch.tensor([[1]]),
                prompt_length=1.0,
                num_steps=1,
            )

    def test_infers_max_step_with_padding(self):
        self.assertEqual(
            infer_max_local_cj_steps(torch.tensor([[0, 2, -1], [1, -1, -1]])),
            3,
        )


if __name__ == "__main__":
    unittest.main()
