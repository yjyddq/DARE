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
import torch.nn.functional as F

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

def _masked_sequence_mean(values, response_mask):
    response_mask = response_mask.to(dtype=values.dtype)
    return (values * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1.0)


def compute_policy_loss_espo(
    old_l_theta,
    l_theta,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode: str = "token-mean",
):
    """
    ESPO sequence-level clipped PPO objective.

    old_l_theta and l_theta are expected to be per-token-normalized sequence
    ELBOs expanded over the response axis, matching the tensors produced by the
    dLLM actors' compute_log_prob path.
    """
    assert isinstance(old_l_theta, torch.Tensor), f"old_l_theta must be a tensor, got {type(old_l_theta)}"
    assert isinstance(l_theta, torch.Tensor), f"l_theta must be a tensor, got {type(l_theta)}"
    assert isinstance(advantages, torch.Tensor), f"advantages must be a tensor, got {type(advantages)}"
    assert old_l_theta.shape == l_theta.shape == advantages.shape, (
        f"old_l_theta, l_theta and advantages must have the same shape, "
        f"but got {old_l_theta.shape}, {l_theta.shape} and {advantages.shape}"
    )

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    old_seq_logp = _masked_sequence_mean(old_l_theta, response_mask)
    seq_logp = _masked_sequence_mean(l_theta, response_mask)
    seq_advantages = _masked_sequence_mean(advantages, response_mask)

    log_ratio = seq_logp - old_seq_logp
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)

    pg_losses1 = -seq_advantages * ratio
    pg_losses2 = -seq_advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    pg_loss = pg_losses.mean()

    is_low_clipped = (ratio < 1 - cliprange_low) & (seq_advantages < 0)
    is_high_clipped = (ratio > 1 + cliprange_high) & (seq_advantages > 0)
    pg_clipfrac = (is_low_clipped | is_high_clipped).float().mean()
    ppo_kl = (-log_ratio).mean()
    pg_clipfrac_lower = is_low_clipped.float().mean()

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_espo_kl(
    sequence_elbo,
    ref_sequence_elbo,
    normalization_length,
    kl_estimator: str = "k2",
):
    """Compute the official ESPO sequence-level KL surrogate.

    ESPO applies the KL estimator to the raw (unnormalized) sequence ELBO and
    divides the resulting batch loss by the fixed response length. Applying
    the estimator after length normalization changes k2 by that same length.
    """
    assert sequence_elbo.shape == ref_sequence_elbo.shape, (
        "sequence_elbo and ref_sequence_elbo must have the same shape, "
        f"got {sequence_elbo.shape} and {ref_sequence_elbo.shape}"
    )
    if normalization_length <= 0:
        raise ValueError(f"normalization_length must be positive, got {normalization_length}")

    log_ratio = sequence_elbo - ref_sequence_elbo
    if kl_estimator in ("kl", "k1"):
        # This is the gradient correction used by the official ESPO code.
        kl = log_ratio + (sequence_elbo - sequence_elbo.detach()) * (
            sequence_elbo.detach() - ref_sequence_elbo
        )
    elif kl_estimator == "abs":
        kl = log_ratio.abs()
    elif kl_estimator in ("mse", "k2"):
        kl = 0.5 * log_ratio.square()
    elif kl_estimator in ("low_var_kl", "k3"):
        kl = torch.exp(-log_ratio) + log_ratio - 1
    else:
        raise NotImplementedError(f"Unsupported ESPO KL estimator: {kl_estimator}")

    return kl / normalization_length


def compute_espo_sequence_elbo(
    logits,
    targets,
    mask_indices,
    p_mask,
    valid_response_mask,
    coupled_logits=None,
    reduce_var=True,
):
    """Compute one ESPO sequence ELBO from precomputed model logits.

    This is the padding-aware equivalent of ``ESPOTrainer._get_elbo``. The
    masked and complementary estimates use the same corruption probability,
    and variance reduction averages the two estimates before summing tokens.
    """
    if logits.ndim != 2 or targets.shape != logits.shape[:-1]:
        raise ValueError(f"Expected logits [S, V] and targets [S], got {logits.shape} and {targets.shape}")
    if not (mask_indices.shape == p_mask.shape == valid_response_mask.shape == targets.shape):
        raise ValueError(
            "mask_indices, p_mask, valid_response_mask and targets must have the same shape, "
            f"got {mask_indices.shape}, {p_mask.shape}, {valid_response_mask.shape}, and {targets.shape}"
        )

    valid_response_mask = valid_response_mask.bool()
    masked_positions = mask_indices.bool() & valid_response_mask
    sequence_elbo = logits.sum() * 0.0
    if masked_positions.any():
        denominator = p_mask[masked_positions]
        if (denominator <= 0).any():
            raise ValueError("ESPO masked-token denominator must be positive")
        sequence_elbo = sequence_elbo - (
            F.cross_entropy(logits[masked_positions], targets[masked_positions], reduction="none") / denominator
        ).sum()

    if reduce_var:
        complementary_positions = valid_response_mask & (~mask_indices.bool())
        if complementary_positions.any():
            if coupled_logits is None:
                raise ValueError("coupled_logits is required when ESPO variance reduction has unmasked tokens")
            if coupled_logits.shape != logits.shape:
                raise ValueError(
                    f"coupled_logits must match logits shape, got {coupled_logits.shape} and {logits.shape}"
                )
            target_length = valid_response_mask.sum().to(dtype=p_mask.dtype)
            unmask_probability = target_length / (target_length + 1.0)
            denominator = unmask_probability - p_mask[complementary_positions]
            if (denominator <= 0).any():
                raise ValueError("ESPO complementary-token denominator must be positive")
            sequence_elbo = sequence_elbo - (
                F.cross_entropy(
                    coupled_logits[complementary_positions],
                    targets[complementary_positions],
                    reduction="none",
                )
                / denominator
            ).sum()
        sequence_elbo = sequence_elbo / 2.0

    return sequence_elbo

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
    
    completion_length = response_mask.sum(dim=1)
    sequence_advantages = (advantages * response_mask).sum(dim=1) / completion_length.clamp_min(1)
    per_seq_loss = -sequence_advantages.unsqueeze(0) * log_prob

    pg_loss = (per_seq_loss * completion_length.unsqueeze(0)).sum() / completion_length.sum().clamp_min(1)

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


def _forward_process_espo(batch, attention_mask, prompt_len, t=None, eps=1e-3, MASK_TOKEN_ID=126336):
    """
    ESPO forward process for sequence-level ELBO estimation.

    This matches ESPO's discrete corruption distribution while keeping DARE's
    padding-aware response mask: k is sampled from [0, target_len], and p_mask
    uses target_len + 1 as denominator.
    """
    b, seq_len = batch.shape

    if attention_mask.dim() == 1:
        response_mask = attention_mask.bool().unsqueeze(0).expand(b, -1).clone()
    elif attention_mask.dim() == 2:
        assert attention_mask.shape == batch.shape, (
            f"2D attention_mask must match batch shape, got {attention_mask.shape} and {batch.shape}"
        )
        response_mask = attention_mask.bool().clone()
    else:
        raise ValueError(f"attention_mask must be 1D or 2D, got {attention_mask.dim()}D")
    response_mask[:, :prompt_len] = False
    target_lens = response_mask.sum(dim=1)

    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    p_mask = torch.zeros((b, seq_len), dtype=torch.float32, device=batch.device)
    if target_lens.max().item() == 0:
        return batch.clone(), mask_indices, p_mask

    equal_target_lengths = bool(torch.all(target_lens == target_lens[0]))
    if equal_target_lengths:
        target_len = int(target_lens[0].item())
        k = torch.randint(0, target_len + 1, (), device=batch.device)
        k_value = float(k)
        x = torch.round(
            torch.linspace(
                k_value,
                k_value + (b - 1) * ((target_len + 1) / b),
                steps=b,
                device=batch.device,
            )
        ).long()
        x = x % (target_len + 1)
        assert x.min() >= 0 and x.max() <= target_len
    else:
        x = torch.zeros((b,), dtype=torch.long, device=batch.device)
        for i in range(b):
            target_len = int(target_lens[i].item())
            if target_len > 0:
                x[i] = torch.randint(0, target_len + 1, (), device=batch.device)

    for i in range(b):
        target_len = int(target_lens[i].item())
        mask_num = int(x[i].item())
        if target_len == 0:
            continue
        response_indices = torch.where(response_mask[i])[0]
        if equal_target_lengths:
            # The official code samples a permutation for every row, including
            # rows with k=0; this RNG consumption affects subsequent rows.
            perm = torch.randperm(target_len, device=batch.device)
            local_mask = torch.arange(target_len, device=batch.device) < mask_num
            mask_indices[i, response_indices] = local_mask[perm]
        elif mask_num > 0:
            perm = torch.randperm(target_len, device=batch.device)
            mask_pos = response_indices[perm[:mask_num]]
            mask_indices[i, mask_pos] = True

    noisy_batch = torch.where(mask_indices, MASK_TOKEN_ID, batch)
    for i in range(b):
        target_len = int(target_lens[i].item())
        if target_len > 0:
            p_mask[i] = x[i].float() / float(target_len + 1)
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


def _forward_process_spg(batch, attention_mask, prompt_len, seed=None, block_length=32, num_t=1, min_t=0, max_t=1, use_mask_prompt=True, p_mask_prompt=0.15, MASK_TOKEN_ID=126336):
    """
    batch: (batch_size, seq_len) Each data should be the same
    attention_mask: (seq_len,)
    prompt_len: int
    """
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=batch.device)
        generator.manual_seed(seed)
    
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
        indices[i] = indices[i][torch.randperm(num_blocks, device=batch.device, generator=generator)] % completion_num_blocks[i]
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
            block_idx = int(mask_block_idx[i, j].item())
            mask_length = min(block_length, int(completion_length[i].item()) - block_length * block_idx)
            assert mask_length > 0
            start_mask_num = max(int(mask_length * min_t), 1)
            end_mask_num = min(int(mask_length * max_t), mask_length)
            assert start_mask_num <= end_mask_num
            mask_num = torch.randint(start_mask_num, end_mask_num + 1, (1, 1), device=batch.device, generator=generator) # [1, 1]
            mask_probability = mask_num.item() / mask_length
            
            # randomly select mask_num tokens to mask for each sequence
            indices = torch.arange(block_length, device=batch.device).repeat(1, 1, 1) # [1, 1, block_length]
            is_mask_next = indices < mask_num.unsqueeze(2) # [1, 1, block_length]
            if block_idx == num_blocks - 1 and mask_length == block_length:
                is_mask_following[i, j, -block_length:] = is_mask_next[0, 0][torch.randperm(block_length, device=batch.device, generator=generator)]
                p_mask[i, j, -block_length:] = mask_probability
            else:
                block_start = -(num_blocks - block_idx) * block_length
                block_end = block_start + mask_length
                is_mask_following[i, j, block_start:block_end] = is_mask_next[0, 0, :mask_length][torch.randperm(mask_length, device=batch.device, generator=generator)]
                p_mask[i, j, block_start:block_end] = mask_probability
                p_mask[i, j, block_end:] = 1
                
    completion_mask_append = torch.cat((torch.ones(b, num_t, prompt_len, dtype=torch.bool, device=batch.device), completion_mask.unsqueeze(1).repeat(1, num_t, 1)), dim=2).to(torch.bool)
    if use_mask_prompt:
        p_mask = torch.where(~is_mask, p_mask_prompt, p_mask)
        
        t_p = torch.ones(b, num_t, device=batch.device) * p_mask_prompt
        random_matrix = torch.rand((b, num_t, l), device=batch.device, generator=generator)
        is_mask_prompt = ~is_mask & (random_matrix < t_p.unsqueeze(2))
        
        is_mask = is_mask_prompt | (is_mask & is_mask_following) | ~completion_mask_append
    else:
        is_mask = (is_mask & is_mask_following) | ~completion_mask_append
    noisy_batch = torch.where(is_mask, MASK_TOKEN_ID, batch.unsqueeze(1).repeat(1, num_t, 1)) # [b, num_t, l]
    # noisy_batch, mask_indices, p_mask


    return noisy_batch.view(-1, l), is_mask.view(-1, l), p_mask.view(-1, l)


# Re-export MDPO loss function for convenience
from verl.trainer.ppo.mdpo_algos import compute_mdpo_policy_loss
