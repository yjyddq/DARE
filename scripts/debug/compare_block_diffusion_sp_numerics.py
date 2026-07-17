#!/usr/bin/env python3
"""Compare dense block-diffusion BGPO/EBPO numerics with SP=1 and SP=2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.ppo.dllm_core_algos import (
    compute_ebpo_composite_elbo,
    compute_policy_loss_bgpo,
    compute_policy_loss_ebpo,
)
from verl.utils.ulysses import set_ulysses_sequence_parallel_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("sdar", "llada2"), required=True)
    parser.add_argument("--algorithm", choices=("bgpo", "ebpo"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa", "flex_attention"),
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-atol", type=float, default=5e-2)
    parser.add_argument("--value-rtol", type=float, default=2e-2)
    parser.add_argument("--rank-atol", type=float, default=1e-6)
    parser.add_argument("--gradient-relative-l2", type=float, default=1e-1)
    parser.add_argument("--gradient-cosine", type=float, default=0.995)
    parser.add_argument("--parameter-gradient-relative-l2", type=float, default=3e-1)
    parser.add_argument("--parameter-gradient-cosine", type=float, default=0.95)
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


def comparison_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    difference = candidate - reference
    return {
        "max_abs_diff": difference.abs().max().item(),
        "reference_max_abs": reference.abs().max().item(),
        "relative_l2": (
            difference.norm() / reference.norm().clamp_min(1e-20)
        ).item(),
    }


def metrics_allclose(metrics: dict[str, float], atol: float, rtol: float) -> bool:
    return metrics["max_abs_diff"] <= atol + rtol * metrics["reference_max_abs"]


def _ensure_sdar_remote_code_compatibility() -> None:
    # SDAR's remote code only uses LossKwargs as a TypedDict base. Transformers
    # 4.57 removed that public alias, so recreate the type-only compatibility
    # shim without changing any model computation.
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: torch.Tensor

        transformers_utils.LossKwargs = LossKwargs


def build_tiny_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model_kind == "sdar":
        _ensure_sdar_remote_code_compatibility()
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    expected_model_types = {
        "sdar": {"sdar"},
        "llada2": {"llada2", "llada2_moe"},
    }
    if config.model_type not in expected_model_types[args.model_kind]:
        raise ValueError(
            f"--model-kind={args.model_kind} does not match model_type={config.model_type}"
        )

    attention_backend = args.attention_backend
    if attention_backend is None:
        attention_backend = "flex_attention" if args.model_kind == "sdar" else "sdpa"
    if args.model_kind == "sdar" and attention_backend != "flex_attention":
        raise ValueError("Dense SDAR numerical validation requires flex_attention")

    overrides: dict[str, Any] = {
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        # KV heads below SP degree exercises the repeat-before-A2A branch.
        "num_key_value_heads": 1,
        "head_dim": 32,
        "max_position_embeddings": 128,
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "mask_token_id": 1,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "use_cache": False,
        "tie_word_embeddings": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
    }
    if args.model_kind == "sdar":
        overrides.update(
            {
                "block_size": 4,
                "fuse_cross_entropy": False,
                "use_sliding_window": False,
                "sliding_window": None,
            }
        )
        config._attn_implementation = attention_backend
    else:
        overrides.update(
            {
                "first_k_dense_replace": 1000,
                "moe_intermediate_size": 64,
                "partial_rotary_factor": 0.5,
            }
        )
        config._attn_implementation = attention_backend
    for key, value in overrides.items():
        setattr(config, key, value)

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    # Plain modules do not get FSDP's construction-time parameter sync. Make
    # both reference ranks start from the exact same state before comparison.
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)
    model.train()
    apply_monkey_patch(
        model=model,
        use_remove_padding=False,
        ulysses_sp_size=dist.get_world_size(),
        use_fused_kernels=False,
    )
    return model


def build_actor(args: argparse.Namespace, model: torch.nn.Module):
    mc_num = 1 if args.algorithm == "bgpo" else 2
    actor_config = OmegaConf.create(
        {
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "use_torch_compile": False,
            "ulysses_sequence_parallel_size": dist.get_world_size(),
            "mc_num": mc_num,
            "n_l": 1,
            "cfg_scale": 0.0,
            "block_length": 4,
            "block_origin": "global",
        }
    )
    if args.model_kind == "sdar":
        if args.algorithm == "bgpo":
            from verl.workers.actor.sdar_dp_actor_bgpo import (
                DLLMDataParallelPPOActor,
            )
        else:
            from verl.workers.actor.sdar_dp_actor_ebpo import (
                DLLMDataParallelPPOActor,
            )
    else:
        if args.algorithm == "bgpo":
            from verl.workers.actor.llada2_dp_actor_bgpo import (
                DLLMDataParallelPPOActor,
            )
        else:
            from verl.workers.actor.llada2_dp_actor_ebpo import (
                DLLMDataParallelPPOActor,
            )
    return DLLMDataParallelPPOActor(actor_config, model, actor_optimizer=None)


def build_micro_batch(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    # Prompt occupies five padded columns; response occupies seven. The two
    # rows compact to lengths 8 and 9, exercising B>1 and unequal lengths.
    input_ids = torch.tensor(
        [
            [0, 0, 11, 12, 13, 21, 22, 23, 24, 25, 0, 0],
            [0, 0, 31, 32, 33, 41, 42, 43, 44, 45, 46, 0],
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = input_ids.ne(0).long()
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    responses = input_ids[:, -7:].clone()

    mc_num = 1 if args.algorithm == "bgpo" else 2
    target_mask = torch.zeros(
        (input_ids.size(0), mc_num, input_ids.size(1)),
        dtype=torch.bool,
        device=device,
    )
    target_mask[0, 0, [5, 7, 9]] = True
    target_mask[1, 0, [6, 8, 10]] = True
    if mc_num == 2:
        target_mask[0, 1, [6, 8]] = True
        target_mask[1, 1, [5, 7, 9]] = True

    perturbed_seq = input_ids[:, None, :].expand(-1, mc_num, -1).clone()
    perturbed_seq[target_mask] = 1
    p_mask = torch.zeros_like(perturbed_seq, dtype=torch.float32)
    p_mask[0, 0, [5, 7, 9]] = torch.tensor([0.35, 0.60, 0.80], device=device)
    p_mask[1, 0, [6, 8, 10]] = torch.tensor([0.45, 0.70, 0.55], device=device)
    if mc_num == 2:
        p_mask[0, 1, [6, 8]] = torch.tensor([0.50, 0.75], device=device)
        p_mask[1, 1, [5, 7, 9]] = torch.tensor([0.40, 0.65, 0.85], device=device)

    response_mask = attention_mask[:, -7:].float()
    advantages = torch.tensor(
        [
            [1.20, -0.35, 0.70, -0.20, 0.45, 0.0, 0.0],
            [-0.45, 0.95, 0.30, -0.25, 1.10, 0.55, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    micro_batch = {
        "input_ids": input_ids,
        "responses": responses,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "perturbed_seq": perturbed_seq,
        "mask_indices": target_mask,
        "p_mask": p_mask,
    }
    return micro_batch, advantages, response_mask


def actor_forward(actor, micro_batch: dict[str, torch.Tensor]):
    _, log_probs, token_elbos = actor._forward_micro_batch(
        micro_batch=micro_batch,
        temperature=1.0,
        n_l=1,
        mc_num=micro_batch["perturbed_seq"].size(1),
        calculate_entropy=False,
        call_fn_name="block_diffusion_sp_numerics",
    )
    return log_probs, token_elbos


def raw_token_losses(actor, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for mc_index in range(micro_batch["perturbed_seq"].size(1)):
            artifacts = actor._build_block_diffusion_artifacts(
                micro_batch=micro_batch,
                noisy_input_ids=micro_batch["perturbed_seq"][:, mc_index, :],
                target_mask=micro_batch["mask_indices"][:, mc_index, :],
                p_mask=micro_batch["p_mask"][:, mc_index, :],
            )
            losses.append(actor._compute_block_diffusion_token_losses(artifacts))
    return torch.stack(losses, dim=1)


def build_old_objective(
    args: argparse.Namespace,
    token_elbos: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if args.algorithm == "bgpo":
        if token_elbos.size(1) != 1:
            raise ValueError(f"BGPO test expects one MC sample, got {token_elbos.shape}")
        offset = torch.tensor(
            [0.03, -0.02, 0.01, 0.04, -0.03, 0.02, 0.0],
            dtype=torch.float32,
            device=token_elbos.device,
        ).expand_as(token_elbos[:, 0, :])
        return token_elbos[:, 0, :].detach() - offset

    composite_elbo = compute_ebpo_composite_elbo(
        token_elbos, contribution_mask=response_mask[:, None, :]
    )
    offset = torch.tensor([0.03, -0.02], dtype=torch.float32, device=token_elbos.device)
    return composite_elbo.detach() - offset


def policy_loss(
    args: argparse.Namespace,
    token_elbos: torch.Tensor,
    old_objective: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if args.algorithm == "ebpo":
        composite_elbo = compute_ebpo_composite_elbo(
            token_elbos, contribution_mask=response_mask[:, None, :]
        )
        sequence_advantages = (advantages * response_mask).sum(dim=-1) / response_mask.sum(
            dim=-1
        ).clamp_min(1)
        sample_mask = response_mask.bool().any(dim=-1)
        return compute_policy_loss_ebpo(
            old_l_theta=old_objective,
            l_theta=composite_elbo,
            advantages=sequence_advantages,
            response_mask=sample_mask,
            cliprange=0.2,
            cliprange_low=0.2,
            cliprange_high=0.2,
            loss_agg_mode="token-mean",
        )[0]

    return compute_policy_loss_bgpo(
        old_l_theta=old_objective,
        l_theta=token_elbos[:, 0, :],
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
    )[0]


def averaged_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float().clone()
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient /= dist.get_world_size()
            gradients[name] = gradient
    if not gradients:
        raise RuntimeError("Model produced no gradients")
    return gradients


def compare_gradients(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if reference.keys() != candidate.keys():
        raise RuntimeError(
            "SP=1 and SP=2 produced gradients for different parameters: "
            f"{reference.keys() ^ candidate.keys()}"
        )

    reference_vector = torch.cat([gradient.reshape(-1) for gradient in reference.values()])
    candidate_vector = torch.cat([gradient.reshape(-1) for gradient in candidate.values()])
    difference_vector = candidate_vector - reference_vector
    aggregate = {
        "reference_l2": reference_vector.norm().item(),
        "sp2_l2": candidate_vector.norm().item(),
        "max_abs_diff": difference_vector.abs().max().item(),
        "relative_l2": (
            difference_vector.norm() / reference_vector.norm().clamp_min(1e-20)
        ).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            reference_vector, candidate_vector, dim=0, eps=1e-20
        ).item(),
        "reference_nonzero": int(torch.count_nonzero(reference_vector).item()),
        "sp2_nonzero": int(torch.count_nonzero(candidate_vector).item()),
        "sample_count": int(reference_vector.numel()),
    }

    per_parameter = {}
    for name, reference_gradient in reference.items():
        candidate_gradient = candidate[name]
        reference_norm = reference_gradient.norm()
        candidate_norm = candidate_gradient.norm()
        difference = candidate_gradient - reference_gradient
        if reference_norm.item() == 0.0:
            relative_l2 = 0.0 if candidate_norm.item() == 0.0 else float("inf")
            cosine = 1.0 if candidate_norm.item() == 0.0 else 0.0
        else:
            relative_l2 = (difference.norm() / reference_norm).item()
            cosine = torch.nn.functional.cosine_similarity(
                reference_gradient.reshape(-1),
                candidate_gradient.reshape(-1),
                dim=0,
                eps=1e-20,
            ).item()
        per_parameter[name] = {
            "reference_l2": reference_norm.item(),
            "sp2_l2": candidate_norm.item(),
            "max_abs_diff": difference.abs().max().item(),
            "relative_l2": relative_l2,
            "cosine": cosine,
            "reference_nonzero": int(torch.count_nonzero(reference_gradient).item()),
            "sp2_nonzero": int(torch.count_nonzero(candidate_gradient).item()),
        }
    return aggregate, per_parameter


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This numerical comparison requires CUDA")
    dist.init_process_group(backend="nccl")
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

    model = build_tiny_model(args, device)
    actor = build_actor(args, model)
    micro_batch, advantages, response_mask = build_micro_batch(args, device)

    set_ulysses_sequence_parallel_group(None)
    model.zero_grad(set_to_none=True)
    reference_raw_token_losses = raw_token_losses(actor, micro_batch)
    reference_log_probs, reference_token_elbos = actor_forward(actor, micro_batch)
    old_objective = build_old_objective(args, reference_token_elbos, response_mask)
    reference_loss = policy_loss(
        args, reference_token_elbos, old_objective, advantages, response_mask
    )
    reference_spreads = {
        "raw_token_losses": rank_spread(reference_raw_token_losses),
        "log_probs": rank_spread(reference_log_probs),
        "token_elbos": rank_spread(reference_token_elbos),
        "policy_loss": rank_spread(reference_loss.reshape(1)),
    }
    reference_loss.backward()
    reference_gradients = averaged_gradients(model)

    model.zero_grad(set_to_none=True)
    dist.barrier()
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    sp2_raw_token_losses = raw_token_losses(actor, micro_batch)
    sp2_log_probs, sp2_token_elbos = actor_forward(actor, micro_batch)
    sp2_loss = policy_loss(
        args, sp2_token_elbos, old_objective, advantages, response_mask
    )
    sp2_spreads = {
        "raw_token_losses": rank_spread(sp2_raw_token_losses),
        "log_probs": rank_spread(sp2_log_probs),
        "token_elbos": rank_spread(sp2_token_elbos),
        "policy_loss": rank_spread(sp2_loss.reshape(1)),
    }
    sp2_loss.backward()
    sp2_gradients = averaged_gradients(model)

    value_metrics = {
        "raw_token_losses": comparison_metrics(
            reference_raw_token_losses, sp2_raw_token_losses
        ),
        "log_probs": comparison_metrics(reference_log_probs, sp2_log_probs),
        "token_elbos": comparison_metrics(reference_token_elbos, sp2_token_elbos),
        "policy_loss": comparison_metrics(reference_loss.reshape(1), sp2_loss.reshape(1)),
    }
    gradient_metrics, parameter_gradient_metrics = compare_gradients(
        reference_gradients, sp2_gradients
    )
    nonzero_parameter_metrics = [
        metrics
        for metrics in parameter_gradient_metrics.values()
        if metrics["reference_nonzero"] > 0
    ]
    checks = {
        "reference_rank_consistent": max(reference_spreads.values()) <= args.rank_atol,
        "sp2_rank_consistent": max(sp2_spreads.values()) <= args.rank_atol,
        "values_close": all(
            metrics_allclose(metrics, args.value_atol, args.value_rtol)
            for metrics in value_metrics.values()
        ),
        "reference_loss_nonzero": abs(reference_loss.detach().float().item()) > 1e-8,
        "reference_gradients_nonzero": gradient_metrics["reference_nonzero"] > 0,
        "sp2_gradients_nonzero": gradient_metrics["sp2_nonzero"] > 0,
        "gradient_relative_l2": gradient_metrics["relative_l2"] <= args.gradient_relative_l2,
        "gradient_cosine": gradient_metrics["cosine"] >= args.gradient_cosine,
        "parameter_gradients_present": bool(nonzero_parameter_metrics),
        "parameter_gradient_relative_l2": all(
            metrics["relative_l2"] <= args.parameter_gradient_relative_l2
            for metrics in nonzero_parameter_metrics
        ),
        "parameter_gradient_cosine": all(
            metrics["cosine"] >= args.parameter_gradient_cosine
            for metrics in nonzero_parameter_metrics
        ),
    }
    result = {
        "passed": all(checks.values()),
        "model_kind": args.model_kind,
        "algorithm": args.algorithm,
        "model_path": args.model_path,
        "attention_backend": model.config._attn_implementation,
        "world_size": world_size,
        "dtype": "bfloat16",
        "reference_loss": reference_loss.detach().float().item(),
        "sp2_loss": sp2_loss.detach().float().item(),
        "checks": checks,
        "metrics": {
            "values": value_metrics,
            "gradient_aggregate": gradient_metrics,
            "gradient_parameters": parameter_gradient_metrics,
            "reference_rank_spread": reference_spreads,
            "sp2_rank_spread": sp2_spreads,
        },
        "tolerances": {
            "value_atol": args.value_atol,
            "value_rtol": args.value_rtol,
            "rank_atol": args.rank_atol,
            "gradient_relative_l2": args.gradient_relative_l2,
            "gradient_cosine": args.gradient_cosine,
            "parameter_gradient_relative_l2": args.parameter_gradient_relative_l2,
            "parameter_gradient_cosine": args.parameter_gradient_cosine,
        },
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "BLOCK_DIFFUSION_SP_NUMERICAL_RESULT="
            + json.dumps(result, sort_keys=True),
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
        if dist.is_initialized():
            dist.destroy_process_group()
    if result is None or not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
