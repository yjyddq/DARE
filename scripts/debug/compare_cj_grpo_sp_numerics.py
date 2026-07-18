#!/usr/bin/env python3
"""Compare paper-exact CJ-GRPO replay with SP=1 and SP=2 on two GPUs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from cj_grpo_gpu_test_utils import (
    averaged_gradients,
    build_cj_actor,
    build_cj_micro_batch,
    build_old_log_probs,
    build_tiny_model,
    cj_weights,
    compare_gradients,
    comparison_metrics,
    metrics_allclose,
    paper_exact_policy_loss,
    response_block_ids,
)
from verl.workers.actor.block_diffusion_utils import (
    scatter_compact_values_to_response,
)
from verl.utils.ulysses import set_ulysses_sequence_parallel_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("sdar", "llada2"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa", "flex_attention"),
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collective-timeout-seconds", type=int, default=180)
    parser.add_argument("--value-atol", type=float, default=5e-2)
    parser.add_argument("--value-rtol", type=float, default=2e-2)
    parser.add_argument("--rank-atol", type=float, default=1e-6)
    parser.add_argument("--gradient-relative-l2", type=float, default=1e-1)
    parser.add_argument("--gradient-cosine", type=float, default=0.995)
    return parser.parse_args()


def distributed_max(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def rank_spread(value: torch.Tensor) -> float:
    rank_zero_value = value.detach().clone()
    dist.broadcast(rank_zero_value, src=0)
    return distributed_max(
        (value.detach().float() - rank_zero_value.float()).abs().max()
    ).item()


def grouped_forward(actor, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return actor._forward_micro_batch(
        micro_batch=micro_batch,
        temperature=1.0,
        calculate_entropy=False,
        call_fn_name="cj_sp_grouped",
    )[1]


def serial_forward(actor, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Six-call oracle for the 2/3/1 fixture, one call per transition."""

    response_length = micro_batch["responses"].size(1)
    padded_response_block_ids = response_block_ids(actor, micro_batch)
    trajectory = micro_batch["reversed_traj_unmask_positions"]
    outputs = []
    for step_idx in range(trajectory.size(1)):
        noisy_input_ids, target_mask, _prompt_length, _response_length = (
            actor._build_cj_step_inputs(micro_batch, step_idx)
        )
        step_output = noisy_input_ids.sum().float() * 0.0
        selected_response = target_mask[:, -response_length:]
        active_blocks = torch.unique(
            padded_response_block_ids[selected_response], sorted=True
        )
        for block_id in active_blocks.tolist():
            block_target_mask = target_mask.clone()
            block_target_mask[:, -response_length:] &= (
                padded_response_block_ids == block_id
            )
            artifacts = actor._build_block_diffusion_artifacts(
                micro_batch=micro_batch,
                noisy_input_ids=noisy_input_ids,
                target_mask=block_target_mask,
                p_mask=torch.ones_like(noisy_input_ids, dtype=torch.float32),
            )
            token_losses = actor._compute_block_diffusion_token_losses(artifacts)
            step_output = step_output + scatter_compact_values_to_response(
                -token_losses,
                artifacts,
                response_length=response_length,
            )
        outputs.append(step_output)
    return torch.stack(outputs, dim=1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CJ SP numerical validation requires CUDA")
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.collective_timeout_seconds),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"Expected exactly two ranks, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260717)
    torch.cuda.manual_seed_all(20260717)
    torch.backends.cuda.matmul.allow_tf32 = False

    model = build_tiny_model(
        model_kind=args.model_kind,
        model_path=args.model_path,
        attention_backend=args.attention_backend,
        device=device,
        ulysses_sp_size=world_size,
    )
    actor = build_cj_actor(
        model_kind=args.model_kind,
        model=model,
        ulysses_sp_size=world_size,
    )
    micro_batch, advantages, response_mask = build_cj_micro_batch(device)
    token_weights, step_masses = cj_weights(actor, micro_batch, response_mask)

    forward_calls = 0

    def count_forward(_module, _inputs):
        nonlocal forward_calls
        forward_calls += 1

    hook = model.register_forward_pre_hook(count_forward)
    set_ulysses_sequence_parallel_group(None)

    model.zero_grad(set_to_none=True)
    forward_calls = 0
    serial_log_probs = serial_forward(actor, micro_batch)
    serial_forward_calls = forward_calls
    old_log_probs = build_old_log_probs(serial_log_probs)
    serial_loss = paper_exact_policy_loss(
        log_probs=serial_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        token_weights=token_weights,
        step_masses=step_masses,
    )
    serial_loss.backward()
    serial_gradients = averaged_gradients(model)

    model.zero_grad(set_to_none=True)
    forward_calls = 0
    grouped_log_probs = grouped_forward(actor, micro_batch)
    grouped_forward_calls = forward_calls
    grouped_loss = paper_exact_policy_loss(
        log_probs=grouped_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        token_weights=token_weights,
        step_masses=step_masses,
    )
    grouped_loss.backward()
    grouped_gradients = averaged_gradients(model)

    model.zero_grad(set_to_none=True)
    dist.barrier()
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    forward_calls = 0
    sp2_log_probs = grouped_forward(actor, micro_batch)
    sp2_forward_calls = forward_calls
    sp2_loss = paper_exact_policy_loss(
        log_probs=sp2_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        token_weights=token_weights,
        step_masses=step_masses,
    )
    sp2_loss.backward()
    sp2_gradients = averaged_gradients(model)
    hook.remove()

    serial_grouped_values = {
        "log_probs": comparison_metrics(serial_log_probs, grouped_log_probs),
        "policy_loss": comparison_metrics(
            serial_loss.reshape(1), grouped_loss.reshape(1)
        ),
    }
    sp_values = {
        "log_probs": comparison_metrics(grouped_log_probs, sp2_log_probs),
        "policy_loss": comparison_metrics(grouped_loss.reshape(1), sp2_loss.reshape(1)),
    }
    serial_grouped_gradients = compare_gradients(serial_gradients, grouped_gradients)
    sp_gradients = compare_gradients(grouped_gradients, sp2_gradients)
    rank_spreads = {
        "serial_log_probs": rank_spread(serial_log_probs),
        "serial_loss": rank_spread(serial_loss.reshape(1)),
        "grouped_log_probs": rank_spread(grouped_log_probs),
        "grouped_loss": rank_spread(grouped_loss.reshape(1)),
        "sp2_log_probs": rank_spread(sp2_log_probs),
        "sp2_loss": rank_spread(sp2_loss.reshape(1)),
    }
    checks = {
        "serial_forward_calls_are_six": serial_forward_calls == 6,
        "grouped_forward_calls_are_three": grouped_forward_calls == 3,
        "sp2_forward_calls_are_three": sp2_forward_calls == 3,
        "rank_consistent": max(rank_spreads.values()) <= args.rank_atol,
        "serial_grouped_values_close": all(
            metrics_allclose(metrics, args.value_atol, args.value_rtol)
            for metrics in serial_grouped_values.values()
        ),
        "serial_grouped_gradient_relative_l2": (
            serial_grouped_gradients["relative_l2"] <= args.gradient_relative_l2
        ),
        "serial_grouped_gradient_cosine": (
            serial_grouped_gradients["cosine"] >= args.gradient_cosine
        ),
        "sp_values_close": all(
            metrics_allclose(metrics, args.value_atol, args.value_rtol)
            for metrics in sp_values.values()
        ),
        "sp_gradient_relative_l2": (
            sp_gradients["relative_l2"] <= args.gradient_relative_l2
        ),
        "sp_gradient_cosine": sp_gradients["cosine"] >= args.gradient_cosine,
        "serial_loss_nonzero": abs(serial_loss.detach().float().item()) > 1e-8,
        "serial_gradients_nonzero": serial_grouped_gradients["reference_nonzero"] > 0,
        "grouped_gradients_nonzero": serial_grouped_gradients["candidate_nonzero"] > 0,
        "sp2_gradients_nonzero": sp_gradients["candidate_nonzero"] > 0,
        "transition_weights_sum_to_batch": abs(step_masses.sum().item() - 1.0) < 1e-6,
    }
    result = {
        "passed": all(checks.values()),
        "model_kind": args.model_kind,
        "model_path": args.model_path,
        "attention_backend": model.config._attn_implementation,
        "world_size": world_size,
        "dtype": "bfloat16",
        "fixture": "three response blocks with transition depths 2/3/1",
        "forward_calls": {
            "serial_sp1": serial_forward_calls,
            "grouped_sp1": grouped_forward_calls,
            "grouped_sp2": sp2_forward_calls,
        },
        "losses": {
            "serial_sp1": serial_loss.detach().float().item(),
            "grouped_sp1": grouped_loss.detach().float().item(),
            "grouped_sp2": sp2_loss.detach().float().item(),
        },
        "checks": checks,
        "metrics": {
            "serial_grouped_values": serial_grouped_values,
            "serial_grouped_gradients": serial_grouped_gradients,
            "sp_values": sp_values,
            "sp_gradients": sp_gradients,
            "rank_spreads": rank_spreads,
            "step_masses": step_masses.tolist(),
        },
        "tolerances": {
            "value_atol": args.value_atol,
            "value_rtol": args.value_rtol,
            "rank_atol": args.rank_atol,
            "gradient_relative_l2": args.gradient_relative_l2,
            "gradient_cosine": args.gradient_cosine,
        },
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "CJ_GRPO_SP_NUMERICAL_RESULT=" + json.dumps(result, sort_keys=True),
            flush=True,
        )
    dist.barrier()
    set_ulysses_sequence_parallel_group(None)
    return result


def main() -> None:
    args = parse_args()
    result = None
    try:
        result = run(args)
    finally:
        set_ulysses_sequence_parallel_group(None)
        if dist.is_initialized():
            dist.destroy_process_group()
    if result is None or not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
