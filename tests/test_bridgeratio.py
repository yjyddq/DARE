import torch

from verl.trainer.ppo.dllm_core_algos import compute_policy_loss_bridgeratio, estimate_bridge_log_ratio


def test_bridge_log_ratio_matches_coupled_logsumexp():
    current_paths = torch.tensor(
        [
            [[-1.0, -0.8], [-1.2, -0.6], [-0.9, -0.7]],
            [[-0.4, -1.1], [-0.3, -1.0], [-0.5, -0.9]],
        ]
    )
    old_paths = torch.tensor(
        [
            [[-1.1, -0.9], [-1.3, -0.5], [-1.0, -0.8]],
            [[-0.6, -1.0], [-0.4, -1.2], [-0.7, -0.8]],
        ]
    )

    log_ratio, metrics = estimate_bridge_log_ratio(current_paths, old_paths)
    expected = torch.logsumexp(current_paths, dim=1) - torch.logsumexp(old_paths, dim=1)

    assert torch.allclose(log_ratio, expected)
    assert metrics["bridge_ratio/num_paths"].item() == 3


def test_bridge_policy_loss_is_zero_for_zero_advantage():
    current_paths = torch.tensor([[[-1.0, -1.0], [-1.0, -1.0]]])
    old_paths = current_paths.clone()
    advantages = torch.zeros(1, 2)
    response_mask = torch.ones(1, 2)

    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, _ = compute_policy_loss_bridgeratio(
        old_l_theta_paths=old_paths,
        l_theta_paths=current_paths,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
    )

    assert torch.allclose(pg_loss, torch.tensor(0.0))
    assert torch.allclose(pg_clipfrac, torch.tensor(0.0))
    assert torch.allclose(ppo_kl, torch.tensor(0.0))
    assert torch.allclose(pg_clipfrac_lower, torch.tensor(0.0))
