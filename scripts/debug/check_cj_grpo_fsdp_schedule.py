#!/usr/bin/env python3
"""Exercise rank-asymmetric CJ local steps under two-rank FSDP FULL_SHARD."""

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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

from cj_grpo_gpu_test_utils import (
    build_cj_actor,
    build_cj_micro_batch,
    build_tiny_model,
    cj_weights,
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
    return parser.parse_args()


def gather_scalars(value: float, device: torch.device) -> list[float]:
    local = torch.tensor([value], dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [tensor.item() for tensor in gathered]


def global_gradient_stats(model: FSDP) -> tuple[float, bool, bool]:
    device = torch.device("cuda", torch.cuda.current_device())
    local_squared_norm = torch.zeros((), dtype=torch.float32, device=device)
    local_has_gradient = False
    local_finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        local_has_gradient = True
        gradient = parameter.grad.detach().float()
        local_squared_norm += gradient.square().sum()
        local_finite = local_finite and bool(torch.isfinite(gradient).all().item())
    dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
    has_gradient = torch.tensor(
        int(local_has_gradient), dtype=torch.int32, device=local_squared_norm.device
    )
    finite = torch.tensor(
        int(local_finite), dtype=torch.int32, device=local_squared_norm.device
    )
    dist.all_reduce(has_gradient, op=dist.ReduceOp.MIN)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return (
        local_squared_norm.sqrt().item(),
        bool(has_gradient.item()),
        bool(finite.item()),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CJ FSDP schedule validation requires CUDA")
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

    # FSDP and Ulysses intentionally do not share this two-rank group.  This
    # entrypoint isolates the data-parallel collective schedule from SP.
    set_ulysses_sequence_parallel_group(None)
    model = build_tiny_model(
        model_kind=args.model_kind,
        model_path=args.model_path,
        attention_backend=args.attention_backend,
        device=device,
        ulysses_sp_size=1,
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        sync_module_states=True,
        device_id=device,
        use_orig_params=False,
        forward_prefetch=False,
    )
    actor = build_cj_actor(
        model_kind=args.model_kind,
        model=fsdp_model,
        ulysses_sp_size=1,
    )

    # Rank 0 owns the 2/3/1 sample; rank 1 retains K=3 but has no step-2
    # transition.  Every rank must nevertheless execute three forwards and
    # three backwards in the same order.
    micro_batch, _advantages, response_mask = build_cj_micro_batch(
        device, empty_last_step=(rank == 1)
    )
    token_weights, step_masses = cj_weights(actor, micro_batch, response_mask)
    trajectory = micro_batch["reversed_traj_unmask_positions"]
    local_target_counts = trajectory[:, :, -response_mask.size(1) :].sum(dim=(0, 2))

    forward_calls = 0

    def count_forward(_module, _inputs):
        nonlocal forward_calls
        forward_calls += 1

    hook = fsdp_model.register_forward_pre_hook(count_forward)
    per_step = []
    for step_idx in range(trajectory.size(1)):
        fsdp_model.zero_grad(set_to_none=True)
        _entropy, log_probs, _loss_per_sample = actor._step_forward_micro_batch(
            micro_batch=micro_batch,
            step_idx=step_idx,
            temperature=1.0,
            calculate_entropy=False,
            call_fn_name="cj_fsdp_schedule",
        )
        step_mass = float(step_masses[step_idx].item())
        if step_mass > 0.0:
            # A simple differentiable token objective is enough to validate
            # FSDP scheduling; paper-exact PPO numerics are covered separately.
            loss = (
                -(log_probs * token_weights[:, step_idx, :]).sum()
                / token_weights[:, step_idx, :].sum()
            )
        else:
            loss = log_probs.sum() * 0.0

        connected_before_backward = bool(
            loss.requires_grad and loss.grad_fn is not None
        )
        local_output_is_zero = bool((log_probs.detach() == 0).all().item())
        loss.backward()
        gradient_norm, all_ranks_have_grad, gradients_finite = global_gradient_stats(
            fsdp_model
        )
        per_step.append(
            {
                "step": step_idx,
                "target_counts_by_rank": gather_scalars(
                    float(local_target_counts[step_idx].item()), device
                ),
                "step_masses_by_rank": gather_scalars(step_mass, device),
                "losses_by_rank": gather_scalars(loss.detach().float().item(), device),
                "connected_by_rank": gather_scalars(
                    float(connected_before_backward), device
                ),
                "outputs_all_zero_by_rank": gather_scalars(
                    float(local_output_is_zero), device
                ),
                "global_gradient_l2": gradient_norm,
                "all_ranks_have_gradient": all_ranks_have_grad,
                "gradients_finite": gradients_finite,
            }
        )
    hook.remove()

    forward_calls_by_rank = gather_scalars(float(forward_calls), device)
    rank_asymmetric_step = per_step[2]
    checks = {
        "full_shard_enabled": fsdp_model.sharding_strategy
        == ShardingStrategy.FULL_SHARD,
        "fixed_three_forwards_per_rank": forward_calls_by_rank == [3.0, 3.0],
        "step_two_is_rank_asymmetric": (
            rank_asymmetric_step["target_counts_by_rank"][0] > 0
            and rank_asymmetric_step["target_counts_by_rank"][1] == 0
        ),
        "empty_rank_output_is_zero": (
            rank_asymmetric_step["outputs_all_zero_by_rank"][1] == 1.0
        ),
        "empty_rank_loss_is_graph_connected": (
            rank_asymmetric_step["connected_by_rank"][1] == 1.0
        ),
        "every_step_has_gradients_on_all_ranks": all(
            item["all_ranks_have_gradient"] for item in per_step
        ),
        "every_step_gradient_is_finite": all(
            item["gradients_finite"] for item in per_step
        ),
        "active_step_two_gradient_nonzero": (
            rank_asymmetric_step["global_gradient_l2"] > 0.0
        ),
    }
    result = {
        "passed": all(checks.values()),
        "model_kind": args.model_kind,
        "model_path": args.model_path,
        "attention_backend": model.config._attn_implementation,
        "world_size": world_size,
        "dtype": "bfloat16",
        "sharding_strategy": str(fsdp_model.sharding_strategy),
        "forward_calls_by_rank": forward_calls_by_rank,
        "checks": checks,
        "steps": per_step,
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "CJ_GRPO_FSDP_SCHEDULE_RESULT=" + json.dumps(result, sort_keys=True),
            flush=True,
        )
    dist.barrier()
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
