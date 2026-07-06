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

from verl.trainer.ppo.core_algos import *
import random
from accelerate.utils import set_seed

def _cv_squared_from_log_weights(log_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    shifted_weights = torch.exp(log_weights - log_weights.max(dim=1, keepdim=True).values)
    mean = shifted_weights.mean(dim=1)
    var = (shifted_weights - mean.unsqueeze(1)).square().mean(dim=1)
    return var / mean.square().clamp_min(eps)


def estimate_bridge_log_ratio(
    l_theta_paths: torch.Tensor,
    old_l_theta_paths: torch.Tensor,
    correction: str = "none",
    detach_correction: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    Estimate log p_theta(y|c) / p_old(y|c) from shared denoising paths.

    The estimator uses the coupled samples already produced by the dLLM forward
    corruption process:
        logsumexp_k l_theta(tau_k) - logsumexp_k l_old(tau_k).
    This directly targets the policy ratio needed by PPO/GRPO instead of
    subtracting two independently estimated sequence log-likelihoods.
    """
    assert isinstance(l_theta_paths, torch.Tensor), f"l_theta_paths must be a tensor, got {type(l_theta_paths)}"
    assert isinstance(old_l_theta_paths, torch.Tensor), f"old_l_theta_paths must be a tensor, got {type(old_l_theta_paths)}"
    assert l_theta_paths.shape == old_l_theta_paths.shape, (
        "l_theta_paths and old_l_theta_paths must have the same shape, "
        f"got {l_theta_paths.shape} and {old_l_theta_paths.shape}"
    )
    assert l_theta_paths.dim() in (2, 3), f"expected (batch, paths) or (batch, paths, length), got {l_theta_paths.shape}"

    num_paths = l_theta_paths.size(1)
    correction = correction.lower()
    log_ratio = torch.logsumexp(l_theta_paths, dim=1) - torch.logsumexp(old_l_theta_paths, dim=1)
    correction_term = torch.zeros_like(log_ratio)

    if correction in ("none", "off", ""):
        pass
    elif correction in ("delta", "bias", "bias_correction"):
        correction_term = 0.5 / max(num_paths, 1) * (
            _cv_squared_from_log_weights(l_theta_paths, eps=eps)
            - _cv_squared_from_log_weights(old_l_theta_paths, eps=eps)
        )
        if detach_correction:
            correction_term = correction_term.detach()
        log_ratio = log_ratio + correction_term
    elif correction == "jackknife":
        if num_paths > 1:
            loo_estimates = []
            for idx in range(num_paths):
                keep = [path_idx for path_idx in range(num_paths) if path_idx != idx]
                loo_estimates.append(
                    torch.logsumexp(l_theta_paths[:, keep, ...], dim=1)
                    - torch.logsumexp(old_l_theta_paths[:, keep, ...], dim=1)
                )
            loo_mean = torch.stack(loo_estimates, dim=1).mean(dim=1)
            log_ratio = num_paths * log_ratio - (num_paths - 1) * loo_mean
            correction_term = log_ratio - (
                torch.logsumexp(l_theta_paths, dim=1) - torch.logsumexp(old_l_theta_paths, dim=1)
            )
    elif correction == "cumulant":
        path_delta = l_theta_paths - old_l_theta_paths
        old_bridge_weights = torch.softmax(old_l_theta_paths, dim=1)
        if detach_correction:
            old_bridge_weights = old_bridge_weights.detach()
        mean_delta = (old_bridge_weights * path_delta).sum(dim=1)
        var_delta = (old_bridge_weights * (path_delta - mean_delta.unsqueeze(1)).square()).sum(dim=1)
        log_ratio = mean_delta + 0.5 * var_delta
        correction_term = log_ratio - (
            torch.logsumexp(l_theta_paths, dim=1) - torch.logsumexp(old_l_theta_paths, dim=1)
        )
    else:
        raise ValueError(f"Unsupported bridge-ratio correction: {correction}")

    with torch.no_grad():
        ratio = torch.exp(log_ratio.clamp(min=-30, max=30))
        metrics = {
            "bridge_ratio/log_ratio_mean": log_ratio.mean(),
            "bridge_ratio/log_ratio_abs_mean": log_ratio.abs().mean(),
            "bridge_ratio/ratio_mean": ratio.mean(),
            "bridge_ratio/ratio_max": ratio.max(),
            "bridge_ratio/correction_mean": correction_term.mean(),
            "bridge_ratio/num_paths": torch.tensor(float(num_paths), device=log_ratio.device),
        }

    return log_ratio, metrics


def compute_policy_loss_bridgeratio(
    old_l_theta_paths,
    l_theta_paths,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    correction: str = "none",
    detach_correction: bool = True,
    ratio_log_clip: float | None = None,
):
    """
    Compute PPO/GRPO loss from a coupled BridgeRatio path estimator.
    """
    assert isinstance(advantages, torch.Tensor), f"advantages must be a tensor, got {type(advantages)}"
    assert isinstance(response_mask, torch.Tensor), f"response_mask must be a tensor, got {type(response_mask)}"
    assert clip_ratio_c > 1.0, "The lower bound of clip_ratio_c for dual-clip PPO should be greater than 1.0"

    log_ratio, bridge_metrics = estimate_bridge_log_ratio(
        l_theta_paths=l_theta_paths,
        old_l_theta_paths=old_l_theta_paths,
        correction=correction,
        detach_correction=detach_correction,
    )
    if log_ratio.dim() == 1:
        log_ratio = log_ratio.unsqueeze(-1).expand_as(advantages)
    assert log_ratio.shape == advantages.shape, (
        f"estimated log_ratio and advantages must have the same shape, got {log_ratio.shape} and {advantages.shape}"
    )

    ratio_log = log_ratio if ratio_log_clip is None else log_ratio.clamp(min=-ratio_log_clip, max=ratio_log_clip)
    ratio = torch.exp(ratio_log)
    ppo_kl = verl_F.masked_mean(torch.exp(-ratio_log) + ratio_log - 1, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, bridge_metrics


def compute_policy_loss_bgpo(
    old_l_theta,
    l_theta,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO (token-level cross-entropy losses).

    Args:
        old_l_theta (Tensor): (batch_size, response_length)
        l_theta (Tensor): (batch_size, response_length)
        advantages (Tensor): (batch_size, response_length)
        response_mask (Tensor): (batch_size, response_length) 1/0 mask for valid tokens
        cliprange (float, optional): Clipping parameter ε for standard PPO.
        cliprange_low (float, optional): Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional): Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional): Lower bound of the ratio for dual-clip PPO. Defaults to 3.0.
        loss_agg_mode (str, optional): Aggregation mode: 'token-mean', 'sentence-mean', etc.
    """
    # Ensure all inputs are tensor format
    assert isinstance(old_l_theta, torch.Tensor), f"old_l_theta must be a tensor, got {type(old_l_theta)}"
    assert isinstance(l_theta, torch.Tensor), f"l_theta must be a tensor, got {type(l_theta)}"
    assert isinstance(advantages, torch.Tensor), f"advantages must be a tensor, got {type(advantages)}"
    assert old_l_theta.shape == l_theta.shape == advantages.shape, f"old_l_theta, l_theta and advantages must have the same shape, but got {old_l_theta.shape}, {l_theta.shape} and {advantages.shape}"
    
    # Check if the lower bound parameter of dual-clip PPO ratio is reasonable
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    # Use different approximation based on the sign of advantages
    # When advantages > 0, use first-order Taylor expansion to approximate lower bound: ratio ≈ (1 + l_theta - old_l_theta)
    # When advantages < 0, use Jensen inequality to approximate upper bound: ratio ≈ exp(l_theta - old_l_theta)
    negative_approx_kl = torch.where(advantages > 0, torch.log(1 + l_theta - old_l_theta), l_theta - old_l_theta)  # (batch_size, response_length)
    # negative_approx_kl = l_theta - old_l_theta  # TODO
    
    # Policy ratio r(θ)
    ratio = torch.exp(negative_approx_kl)
    
    # KL divergence
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio  # Unclipped policy gradient loss: -A(s,a) * r(θ)
    
    # Set clip range, if not specified, use standard clip range
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    # Calculate clipped policy gradient loss: -A(s,a) * clip(r(θ), 1-ε, 1+ε)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    
    # Take the maximum of the two, achieve PPO's minimax objective: max(-A*r, -A*clip(r))
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)  # Compute the clipping ratio, count how many samples are clipped

    # Dual-clip PPO: When advantages are negative, set a stricter lower bound for clipping
    # Compute lower bound loss: -A(s,a) * c, where c is the lower bound of the clipping ratio
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)  # Take the minimum of the standard clipping loss and the lower bound loss
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    # According to the sign of advantages, choose different loss calculation methods
    # When advantages >= 0, use standard PPO clipping; when advantages < 0, use dual-clip PPO
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # Return the policy loss, clipping ratio, KL divergence and lower bound clipping ratio
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_ebpo(
    old_l_theta,
    l_theta,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped PPO objective for EBPO on block-level ELBO tensors.
    """
    return compute_policy_loss_bgpo(
        old_l_theta=old_l_theta,
        l_theta=l_theta,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=cliprange,
        cliprange_low=cliprange_low,
        cliprange_high=cliprange_high,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode=loss_agg_mode,
    )


def compute_policy_loss(
    old_l_theta,
    l_theta,
    advantages,
    response_mask,
    ref_l_theta=None,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO (token-level cross-entropy losses).

    Args:
        old_l_theta (Tensor): (batch_size, response_length)
        l_theta (Tensor): (batch_size, response_length)
        advantages (Tensor): (batch_size, response_length)
        response_mask (Tensor): (batch_size, response_length) 1/0 mask for valid tokens
        cliprange (float, optional): Clipping parameter ε for standard PPO.
        cliprange_low (float, optional): Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional): Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional): Lower bound of the ratio for dual-clip PPO. Defaults to 3.0.
        loss_agg_mode (str, optional): Aggregation mode: 'token-mean', 'sentence-mean', etc.
    """
    # Ensure all inputs are tensor format
    assert isinstance(old_l_theta, torch.Tensor), f"old_l_theta must be a tensor, got {type(old_l_theta)}"
    assert isinstance(l_theta, torch.Tensor), f"l_theta must be a tensor, got {type(l_theta)}"
    assert isinstance(advantages, torch.Tensor), f"advantages must be a tensor, got {type(advantages)}"
    assert old_l_theta.shape == l_theta.shape == advantages.shape, f"old_l_theta, l_theta and advantages must have the same shape, but got {old_l_theta.shape}, {l_theta.shape} and {advantages.shape}"
    
    # Check if the lower bound parameter of dual-clip PPO ratio is reasonable
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."
    
    # Policy ratio r(θ)
    ratio = torch.exp(l_theta - old_l_theta)

    # KL divergence
    if ref_l_theta is not None:
        ppo_kl = verl_F.masked_mean(torch.exp(ref_l_theta - l_theta) - (ref_l_theta - l_theta) - 1, response_mask)
    else:
        ppo_kl = verl_F.masked_mean(torch.exp(old_l_theta - l_theta) - (old_l_theta - l_theta) - 1, response_mask)

    pg_losses1 = -advantages * ratio  # Unclipped policy gradient loss: -A(s,a) * r(θ)
    
    # Set clip range, if not specified, use standard clip range
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    # Calculate clipped policy gradient loss: -A(s,a) * clip(r(θ), 1-ε, 1+ε)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    
    # Take the maximum of the two, achieve PPO's minimax objective: max(-A*r, -A*clip(r))
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)  # Compute the clipping ratio, count how many samples are clipped

    # Dual-clip PPO: When advantages are negative, set a stricter lower bound for clipping
    # Compute lower bound loss: -A(s,a) * c, where c is the lower bound of the clipping ratio
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)  # Take the minimum of the standard clipping loss and the lower bound loss
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    # According to the sign of advantages, choose different loss calculation methods
    # When advantages >= 0, use standard PPO clipping; when advantages < 0, use dual-clip PPO
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # Return the policy loss, clipping ratio, KL divergence and lower bound clipping ratio
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_spg(
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the policy objective and related metrics for RL (token-level cross-entropy losses).

    Args:
        l_theta (Tensor): (batch_size, response_length)
        advantages (Tensor): (batch_size, response_length)
        response_mask (Tensor): (batch_size, response_length) 1/0 mask for valid tokens
        cliprange (float, optional): Clipping parameter ε for standard PPO.
        cliprange_low (float, optional): Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional): Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional): Lower bound of the ratio for dual-clip PPO. Defaults to 3.0.
        loss_agg_mode (str, optional): Aggregation mode: 'token-mean', 'sentence-mean', etc.
    """
    # Ensure all inputs are tensor format
    assert isinstance(log_prob, torch.Tensor), f"log_prob_positive must be a tensor, got {type(log_prob)}"
    assert isinstance(advantages, torch.Tensor), f"advantages must be a tensor, got {type(advantages)}"
    
    per_seq_loss = - advantages[:,-1].unsqueeze(0) * log_prob
    
    completion_length = response_mask.sum(dim=1).unsqueeze(0) # [1, batch_size]
    pg_loss = (per_seq_loss * completion_length).sum() / completion_length.sum() # 

    # Return the policy loss, clipping ratio, KL divergence and lower bound clipping ratio
    return pg_loss, None, None, None

def compute_dpo_loss(
    chosen_log_prob: torch.Tensor,
    rejected_log_prob: torch.Tensor,
    chosen_ref_log_prob: torch.Tensor,
    rejected_ref_log_prob: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    import torch.nn.functional as F

    pi_logratios = chosen_log_prob - rejected_log_prob
    ref_logratios = chosen_ref_log_prob - rejected_ref_log_prob

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)

    return losses.mean()

def _forward_process_vrpo(batch, attention_mask, prompt_len, t=None, eps=1e-3, MASK_TOKEN_ID=126336):
    """
    Forward process: add noise to the batch
    Only mask the part where attention_mask == 1, padding part and prompt part are not masked
    batch: (batch_size, seq_len) Each data should be the same
    attention_mask: (seq_len,)
    prompt_len: int
    t: (batch_size) Diffusion time step (float between 0 and 1), if not passed, automatically sample
    eps: float, small constant to prevent mask probability to be 0
    """
    b, seq_len = batch.shape  # (batch_size, seq_len)
    
    # Valid token region (excluding prompt/padding)
    response_mask = attention_mask.bool().clone()  # (seq_len,)
    response_mask[:prompt_len] = False
    # print(f"pad prompt_len: {prompt_len}")
    response_indices = torch.where(response_mask)[0]  # Valid token indices (target_len,)
    target_len = response_mask.sum().item()  # Valid token count in response region
    # print(f"true target_len: {target_len}")

    # NOTE: discrete version (refer to https://github.com/ML-GSAI/LLaDA/blob/main/eval_llada.py):
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    # x is a integer vector of shape [b]. x[i] represents the number of tokens to be masked in the target region of the i-th sample, ensuring uniform distribution and in the range of [1, target_len].
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert 1 <= x.min() and x.max() <= target_len

    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    for i in range(b):
        perm = torch.randperm(target_len, device=batch.device)
        mask_pos = response_indices[perm[:x[i]]]
        mask_indices[i, mask_pos] = True  # [False, False, ..., True, True, ..., False, False]

    noisy_batch = torch.where(mask_indices, MASK_TOKEN_ID, batch)  # mask tokens and get noisy batch
    p_mask = (x / target_len).unsqueeze(1).repeat(1, seq_len)  # Normalized weight for each sample's mask ratio (mask probability)
    # print(f"noisy_batch[0] sum: {noisy_batch[0][attention_mask == 1].sum()}")
    return noisy_batch, mask_indices, p_mask


def kl_penalty(l_theta: torch.FloatTensor, ref_l_theta: torch.FloatTensor, kl_penalty, advantages: torch.FloatTensor) -> torch.FloatTensor:
    """Compute KL divergence given l_theta and ref_l_theta.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    If advantages is provided, use the same approximation as compute_policy_loss:
      - adv > 0: log(1 + l_theta - ref_l_theta)
      - adv <= 0: l_theta - ref_l_theta

    Args:
        l_theta:
        ref_l_theta:
        kl_penalty:
        advantages:

    Returns:

    """
    diff = l_theta - ref_l_theta  # Based on the new distribution
    kl = torch.where(advantages > 0, torch.log(1 + diff), diff)
    # kl = diff  # TODO

    if kl_penalty in ("kl", "k1"):
        return kl

    if kl_penalty == "abs":
        return kl.abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * kl.square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = -kl  # Based on the old distribution
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here l_theta and ref_l_theta should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def _forward_process_bgpo(batch, attention_mask, prompt_len, t=None, eps=1e-3, MASK_TOKEN_ID=126336):
    """
    Forward process: add noise to the batch
    Only mask the part where attention_mask == 1, padding part and prompt part are not masked
    batch: (batch_size, seq_len) Each data should be the same
    attention_mask: (seq_len,)
    prompt_len: int
    t: (batch_size) Diffusion time step (float between 0 and 1), if not passed, automatically sample
    eps: float, small constant to prevent mask probability to be 0
    """
    b, seq_len = batch.shape  # (batch_size, seq_len)
    
    # Valid token region (excluding prompt/padding)
    response_mask = attention_mask.bool().clone()  # (seq_len,)
    response_mask[:prompt_len] = False
    # print(f"pad prompt_len: {prompt_len}")
    response_indices = torch.where(response_mask)[0]  # Valid token indices (target_len,)
    target_len = response_mask.sum().item()  # Valid token count in response region
    # assert target_len == seq_len - prompt_len
    # print(f"true target_len: {target_len}")

    # NOTE: discrete version (refer to https://github.com/ML-GSAI/LLaDA/blob/main/eval_llada.py):
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    # x is a integer vector of shape [b]. x[i] represents the number of tokens to be masked in the target region of the i-th sample, ensuring uniform distribution and in the range of [1, target_len].
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert 1 <= x.min() and x.max() <= target_len

    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    for i in range(b):
        perm = torch.randperm(target_len, device=batch.device)
        mask_pos = response_indices[perm[:x[i]]]
        mask_indices[i, mask_pos] = True  # [False, False, ..., True, True, ..., False, False]

    noisy_batch = torch.where(mask_indices, MASK_TOKEN_ID, batch)  # mask tokens and get noisy batch
    p_mask = (x / target_len).unsqueeze(1).repeat(1, seq_len)  # Normalized weight for each sample's mask ratio (mask probability)
    # print(f"noisy_batch[0] sum: {noisy_batch[0][attention_mask == 1].sum()}")
    return noisy_batch, mask_indices, p_mask


def _forward_process_ebpo(batch, attention_mask, prompt_len, block_length=32, t=None, eps=1e-3, MASK_TOKEN_ID=126336):
    """
    Forward process for EBPO.

    Unlike BGPO, EBPO samples a single response block per trajectory and masks
    only tokens inside that block. This matches the paper's block-level ELBO
    approximation under block-causal attention.
    """
    b, seq_len = batch.shape

    response_mask = attention_mask.bool().clone()
    response_mask[:prompt_len] = False
    response_indices = torch.where(response_mask)[0]
    target_len = response_mask.sum().item()

    num_blocks = max((target_len + block_length - 1) // block_length, 1)
    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    p_mask = torch.zeros((b, seq_len), dtype=torch.float32, device=batch.device)

    sampled_blocks = torch.randint(0, num_blocks, (b,), device=batch.device)
    for i in range(b):
        block_id = int(sampled_blocks[i].item())
        block_start = block_id * block_length
        block_end = min(block_start + block_length, target_len)
        block_token_indices = response_indices[block_start:block_end]
        cur_block_len = block_token_indices.numel()
        if cur_block_len == 0:
            continue

        mask_num = torch.randint(1, cur_block_len + 1, (), device=batch.device).item()
        perm = torch.randperm(cur_block_len, device=batch.device)
        selected = block_token_indices[perm[:mask_num]]

        mask_indices[i, selected] = True
        p_mask[i, selected] = float(mask_num) / float(cur_block_len)

    noisy_batch = torch.where(mask_indices, MASK_TOKEN_ID, batch)
    return noisy_batch, mask_indices, p_mask


def _forward_process_d1(batch, attention_mask, prompt_len, p=0.15, MASK_TOKEN_ID=126336):
    """
    batch: (batch_size, seq_len) Each data should be the same
    attention_mask: (seq_len,)
    prompt_len: int
    """
    b, seq_len = batch.shape  # (batch_size, seq_len)

    # mask prompt part with probability p
    prompt_mask = attention_mask[:prompt_len].bool()  # (prompt_len,)
    random_mask = torch.rand((b, prompt_len), device=batch.device) < p  # (batch_size, prompt_len)
    prompt_mask_indices = prompt_mask & random_mask

    # mask all response part
    response_mask = attention_mask[prompt_len:].bool()  # (seq_len - prompt_len,)
    response_mask_indices = response_mask

    # Merge masks
    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    mask_indices[:, :prompt_len] = prompt_mask_indices
    mask_indices[:, prompt_len:] = response_mask_indices

    noisy_batch = torch.where(mask_indices, MASK_TOKEN_ID, batch)
    
    # p_mask: The probability of each token being masked: prompt part is p, response part is 1, other parts are 0
    p_mask = torch.zeros((b, seq_len), device=batch.device)
    p_mask[:, :prompt_len] = p * prompt_mask.float()
    p_mask[:, prompt_len:] = response_mask.float()
    return noisy_batch, mask_indices, p_mask


def _forward_process_coupled_grpo(batch, attention_mask, prompt_len, seed=42, MASK_TOKEN_ID=126336):
    """
    batch: (batch_size, seq_len) Each data should be the same
    attention_mask: (seq_len,)
    prompt_len: int
    """
    set_seed(seed)
    b, l = batch.shape  # (batch_size, seq_len)
    prompt_index = attention_mask.clone().bool()  # (seq_len,)
    prompt_index[prompt_len:] = False
    prompt_index = prompt_index.unsqueeze(0).repeat(b, 1)  # (batch_size, prompt_len)
    noisy_batch = []
    mask_indices = []
    p_mask = []
    mask_ratio = random.uniform(0.2, 0.8)
    t_p = torch.ones((b, l), device=batch.device) * mask_ratio
    # Create a random matrix to decide whether each prompt token is masked
    random_matrix = torch.rand((b, l), device=batch.device)

    # 1. always mask completion tokens
    mask_indices_full = ~prompt_index
    noisy_batch.append(torch.where(mask_indices_full, MASK_TOKEN_ID, batch))
    mask_indices.append(mask_indices_full)
    p_mask.append(mask_indices_full.float())

    # 2. mask completion tokens with probability t_p
    mask_indices_tp = ~prompt_index & (random_matrix < t_p)
    noisy_batch.append(torch.where(mask_indices_tp, MASK_TOKEN_ID, batch))
    mask_indices.append(mask_indices_tp)
    p_mask.append(mask_indices_tp.float() * t_p)

    # 3. mask completion tokens reversely
    mask_indices_comp_tp = ~prompt_index & (random_matrix > t_p)
    noisy_batch.append(torch.where(mask_indices_comp_tp, MASK_TOKEN_ID, batch))
    mask_indices.append(mask_indices_comp_tp)
    p_mask.append(mask_indices_comp_tp.float() * (1. - t_p))

    noisy_batch = torch.cat(noisy_batch, dim=0)
    mask_indices = torch.cat(mask_indices, dim=0)
    p_mask = torch.cat(p_mask, dim=0)
    return noisy_batch, mask_indices, p_mask


def _forward_process_spg(batch, attention_mask, prompt_len, seed=42, block_length=32, num_t=1, min_t=0, max_t=1, use_mask_prompt=True, p_mask_prompt=0.15, MASK_TOKEN_ID=126336):
    """
    batch: (batch_size, seq_len) Each data should be the same
    attention_mask: (seq_len,)
    prompt_len: int
    """
    
    set_seed(seed)
    
    b, l = batch.shape
    gen_length = l - prompt_len
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    completion_mask = attention_mask[prompt_len:].unsqueeze(0)
    p_mask = torch.zeros((b, num_t, l), device=batch.device)

    
    completion_num_blocks = (completion_mask.sum(-1)-1)//block_length+1
    assert num_t <= num_blocks
    indices = torch.arange(num_blocks, device=batch.device).repeat(b, 1) # [b, num_blocks]
    for i in range(b):
        indices[i] = indices[i][torch.randperm(num_blocks)] % completion_num_blocks[i]
    mask_block_idx = indices[:, :num_t]
    is_mask = torch.zeros((b, num_t, l), dtype=torch.bool, device=batch.device)
    block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=batch.device)
    for i in range(b):
        for j in range(num_t):
            is_mask[i, j, -(num_blocks - mask_block_idx[i, j]) * block_length:] = True
            if mask_block_idx[i, j] < num_blocks - 1:
                block_mask[i, j, -(num_blocks - mask_block_idx[i, j] - 1) * block_length:] = False
    completion_length = completion_mask.sum(-1)
    is_mask_following = torch.ones((b, num_t, l), dtype=torch.bool, device=batch.device)
    for i in range(b):
        for j in range(num_t):
            mask_length = min(block_length, completion_length[i] - block_length * mask_block_idx[i, j])
            assert mask_length > 0
            start_mask_num = max(int(mask_length * min_t), 1)
            end_mask_num = min(int(mask_length * max_t), mask_length)
            assert start_mask_num <= end_mask_num
            mask_num = torch.randint(start_mask_num, end_mask_num + 1, (1, 1), device=batch.device) # [1, 1]
            
            # randomly select mask_num tokens to mask for each sequence
            indices = torch.arange(block_length, device=batch.device).repeat(1, 1, 1) # [1, 1, block_length]
            is_mask_next = indices < mask_num.unsqueeze(2) # [1, 1, block_length]
            if mask_block_idx[i, j] == num_blocks - 1 and mask_length == block_length:
                is_mask_following[i, j, -(num_blocks - mask_block_idx[i, j]) * block_length:] = is_mask_next[0, 0][torch.randperm(block_length)]
            else:
                is_mask_following[i, j, -(num_blocks - mask_block_idx[i, j]) * block_length: -(num_blocks - mask_block_idx[i, j]) * block_length + mask_length] = is_mask_next[0, 0, :mask_length][torch.randperm(mask_length)]
                p_mask[i, j, -(num_blocks - mask_block_idx[i, j]) * block_length: -(num_blocks - mask_block_idx[i, j]) * block_length + mask_length] = float(mask_num) / mask_length
                p_mask[i, j,-(num_blocks - mask_block_idx[i, j]) * block_length + mask_length: ] = 1
                
    completion_mask_append = torch.cat((torch.ones(b, num_t, prompt_len, dtype=torch.bool, device=batch.device), completion_mask.unsqueeze(1).repeat(1, num_t, 1)), dim=2).to(torch.bool)
    if use_mask_prompt:
        p_mask = torch.where(~is_mask, p_mask_prompt, p_mask)
        
        t_p = torch.ones(b, num_t, device=batch.device) * p_mask_prompt
        random_matrix = torch.rand((b, num_t, l), device=batch.device)
        is_mask_prompt = ~is_mask & (random_matrix < t_p.unsqueeze(2))
        
        is_mask = is_mask_prompt | (is_mask & is_mask_following) | ~completion_mask_append
    else:
        is_mask = (is_mask & is_mask_following) | ~completion_mask_append
    noisy_batch = torch.where(is_mask, MASK_TOKEN_ID, batch.unsqueeze(1).repeat(1, num_t, 1)) # [b, num_t, l]
    # noisy_batch, mask_indices, p_mask


    return noisy_batch.view(-1, l), is_mask.view(-1, l), p_mask.view(-1, l)


# Re-export MDPO loss function for convenience
from verl.trainer.ppo.mdpo_algos import compute_mdpo_policy_loss
