from typing import List

import torch


def compute_advantages(
    on_policy_rewards: List[float], off_policy_rewards: List[float], eps: float = 1e-8
) -> torch.Tensor:
    rewards = torch.tensor(on_policy_rewards + off_policy_rewards)
    mean_rewards = torch.mean(rewards)
    std_rewards = torch.std(rewards) + eps

    return (rewards - mean_rewards) / std_rewards
