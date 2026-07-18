#!/usr/bin/env python3
"""Shared helpers for the CJ-GRPO multi-GPU debug entrypoints."""

from __future__ import annotations

from typing import Any, TypedDict

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.ppo.dllm_core_algos import compute_policy_loss
from verl.workers.actor.block_diffusion_utils import (
    build_padded_token_block_ids,
    compute_cj_block_step_token_weights,
)


def _ensure_sdar_remote_code_compatibility() -> None:
    """Restore the type-only alias used by older SDAR remote code."""

    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "LossKwargs"):

        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: torch.Tensor

        transformers_utils.LossKwargs = LossKwargs


def build_tiny_model(
    *,
    model_kind: str,
    model_path: str,
    attention_backend: str | None,
    device: torch.device,
    ulysses_sp_size: int,
) -> torch.nn.Module:
    """Build the same small remote-code model used by the SP smoke tests."""

    if model_kind == "sdar":
        _ensure_sdar_remote_code_compatibility()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    expected_model_types = {
        "sdar": {"sdar"},
        "llada2": {"llada2", "llada2_moe"},
    }
    if model_kind not in expected_model_types:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    if config.model_type not in expected_model_types[model_kind]:
        raise ValueError(
            f"--model-kind={model_kind} does not match model_type={config.model_type}"
        )

    if attention_backend is None:
        attention_backend = "flex_attention" if model_kind == "sdar" else "sdpa"
    if model_kind == "sdar" and attention_backend != "flex_attention":
        raise ValueError("Dense SDAR CJ validation requires flex_attention")

    overrides: dict[str, Any] = {
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        # Keep KV heads below SP degree to exercise repeat-before-A2A.
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
    if model_kind == "sdar":
        overrides.update(
            {
                "block_size": 4,
                "fuse_cross_entropy": False,
                "use_sliding_window": False,
                "sliding_window": None,
            }
        )
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
    # Plain modules do not receive FSDP's construction-time state sync.
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)
    model.train()
    apply_monkey_patch(
        model=model,
        use_remove_padding=False,
        ulysses_sp_size=ulysses_sp_size,
        use_fused_kernels=False,
    )
    return model


def build_cj_actor(
    *,
    model_kind: str,
    model: torch.nn.Module,
    ulysses_sp_size: int,
):
    actor_config = OmegaConf.create(
        {
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "use_torch_compile": False,
            "ulysses_sequence_parallel_size": ulysses_sp_size,
            "mc_num": 1,
            "n_l": 1,
            "cfg_scale": 0.0,
            "block_length": 4,
            "block_origin": "global",
        }
    )
    if model_kind == "sdar":
        from verl.workers.actor.sdar_dp_actor_cj_grpo import (
            DLLMDataParallelPPOActor,
        )
    elif model_kind == "llada2":
        from verl.workers.actor.llada2_dp_actor_cj_grpo import (
            DLLMDataParallelPPOActor,
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    return DLLMDataParallelPPOActor(actor_config, model, actor_optimizer=None)


def build_cj_micro_batch(
    device: torch.device,
    *,
    empty_last_step: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build three response blocks with 2/3/1 transition depths.

    The padded prompt section is five columns but contains four real tokens.
    With global block origin and block size four, the twelve response tokens
    therefore form exactly three response blocks.  ``empty_last_step`` changes
    the middle block from three transitions to two while retaining K=3, which
    is the rank-asymmetric FSDP fixture.
    """

    input_ids = torch.tensor(
        [[0, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]],
        dtype=torch.long,
        device=device,
    )
    attention_mask = input_ids.ne(0).long()
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    response_length = 12
    responses = input_ids[:, -response_length:].clone()

    if empty_last_step:
        # Block depths 2/2/1; the padded K=3 slot is intentionally empty.
        local_steps = (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0)
    else:
        # Paper-exact fixture: block depths 2/3/1, hence 6 transitions.
        local_steps = (0, 0, 1, 1, 0, 1, 2, 2, 0, 0, 0, 0)
    trajectory = torch.zeros((1, 3, input_ids.size(1)), dtype=torch.bool, device=device)
    response_offset = input_ids.size(1) - response_length
    for response_index, local_step in enumerate(local_steps):
        trajectory[0, local_step, response_offset + response_index] = True

    advantages = torch.tensor(
        [[1.20, -0.35, 0.70, -0.20, 0.45, -0.80, 1.05, -0.15, 0.60, -0.55, 0.25, 0.90]],
        dtype=torch.float32,
        device=device,
    )
    response_mask = attention_mask[:, -response_length:].bool()
    micro_batch = {
        "input_ids": input_ids,
        "responses": responses,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "reversed_traj_unmask_positions": trajectory,
    }
    return micro_batch, advantages, response_mask


def cj_weights(
    actor,
    micro_batch: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    response_length = micro_batch["responses"].size(1)
    return compute_cj_block_step_token_weights(
        micro_batch["reversed_traj_unmask_positions"],
        response_mask,
        micro_batch["attention_mask"],
        prompt_section_length=micro_batch["input_ids"].size(1) - response_length,
        block_size=actor.block_length,
        block_origin=actor.block_origin,
    )


def build_old_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    offsets = torch.linspace(
        -0.08,
        0.08,
        log_probs.size(-1),
        dtype=torch.float32,
        device=log_probs.device,
    )
    step_offsets = torch.tensor(
        [-0.015, 0.025, -0.035], dtype=torch.float32, device=log_probs.device
    )
    return log_probs.detach() - offsets[None, None, :] - step_offsets[None, :, None]


def paper_exact_policy_loss(
    *,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    token_weights: torch.Tensor,
    step_masses: torch.Tensor,
) -> torch.Tensor:
    """Transition mean inside each sample, followed by the batch mean."""

    if log_probs.shape != old_log_probs.shape or log_probs.shape != token_weights.shape:
        raise ValueError(
            "CJ log-probability and weight shapes disagree: "
            f"{log_probs.shape}, {old_log_probs.shape}, {token_weights.shape}"
        )
    batch_size = log_probs.size(0)
    loss = log_probs.sum() * 0.0
    for step_idx in range(log_probs.size(1)):
        step_mass = float(step_masses[step_idx].item())
        if step_mass == 0.0:
            continue
        step_loss = compute_policy_loss(
            old_l_theta=old_log_probs[:, step_idx, :],
            l_theta=log_probs[:, step_idx, :],
            advantages=advantages,
            response_mask=token_weights[:, step_idx, :],
            cliprange=0.2,
            cliprange_low=0.2,
            cliprange_high=0.2,
            clip_ratio_c=3.0,
            loss_agg_mode="token-mean",
        )[0]
        # ``masked_mean`` divides by ``mask.sum() + 1e-8``.  Match the actor's
        # outer coefficient so the composed reduction is exactly the desired
        # weighted sum divided by batch size.
        loss = loss + step_loss * ((step_mass + 1e-8) / batch_size)
    return loss


def response_block_ids(actor, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    response_length = micro_batch["responses"].size(1)
    prompt_section_length = micro_batch["input_ids"].size(1) - response_length
    return build_padded_token_block_ids(
        micro_batch["attention_mask"],
        prompt_section_length=prompt_section_length,
        block_size=actor.block_length,
        block_origin=actor.block_origin,
    )[:, -response_length:]


def comparison_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    difference = candidate - reference
    return {
        "max_abs_diff": difference.abs().max().item(),
        "reference_max_abs": reference.abs().max().item(),
        "relative_l2": (difference.norm() / reference.norm().clamp_min(1e-20)).item(),
    }


def metrics_allclose(metrics: dict[str, float], atol: float, rtol: float) -> bool:
    return metrics["max_abs_diff"] <= atol + rtol * metrics["reference_max_abs"]


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
) -> dict[str, float]:
    if reference.keys() != candidate.keys():
        raise RuntimeError(
            "Compared runs produced gradients for different parameters: "
            f"{reference.keys() ^ candidate.keys()}"
        )
    reference_vector = torch.cat(
        [gradient.reshape(-1) for gradient in reference.values()]
    )
    candidate_vector = torch.cat(
        [gradient.reshape(-1) for gradient in candidate.values()]
    )
    difference = candidate_vector - reference_vector
    return {
        "reference_l2": reference_vector.norm().item(),
        "candidate_l2": candidate_vector.norm().item(),
        "max_abs_diff": difference.abs().max().item(),
        "relative_l2": (
            difference.norm() / reference_vector.norm().clamp_min(1e-20)
        ).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            reference_vector, candidate_vector, dim=0, eps=1e-20
        ).item(),
        "reference_nonzero": int(torch.count_nonzero(reference_vector).item()),
        "candidate_nonzero": int(torch.count_nonzero(candidate_vector).item()),
        "sample_count": int(reference_vector.numel()),
    }
