from typing import List

import torch

from data import ON_POLICY_RATIO
from model import compute_log_probs

GAMMA = 0.1
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.01


def policy_shaping(r_hat: torch.Tensor, gamma: float = 0.1):
    return r_hat / (r_hat + gamma)


def compute_loss(
    model,
    tokenizer,
    batch,
    on_policy_completions,
    old_log_probs,
    advantages,
    eps=0.2,
):
    batch_size = len(on_policy_completions)
    flat_prompts = [p for p in batch["prompts"] for _ in range(ON_POLICY_RATIO)]
    flat_completions = [c for comps in on_policy_completions for c in comps]
    flat_sequences = [p + c for p, c in zip(flat_prompts, flat_completions)]

    on_lp, on_mask = compute_log_probs(model, tokenizer, flat_sequences, grad=True)
    r_t = torch.exp(on_lp - old_log_probs) * on_mask
    adv_on = advantages[: batch_size * ON_POLICY_RATIO].unsqueeze(-1)
    on_policy_loss = -torch.sum(torch.clamp(r_t, 1 - eps, 1 + eps) * adv_on)
    entropy = -(on_lp * on_mask).sum() / on_mask.sum()
    entropy_loss = ENTROPY_COEFF * entropy
    on_token_count = on_mask.sum().detach()

    del on_lp, r_t

    off_sequences = [
        p + c for p, c in zip(batch["prompts"], batch["off_policy_traces"])
    ]
    off_lp, off_mask = compute_log_probs(model, tokenizer, off_sequences, grad=True)
    r_hat = torch.exp(off_lp)
    shaped = policy_shaping(r_hat, GAMMA) * off_mask
    adv_off = advantages[batch_size * ON_POLICY_RATIO :].unsqueeze(-1)
    off_policy_loss = -torch.sum(shaped * adv_off * off_mask)
    off_token_count = off_mask.sum().detach()

    del r_hat, shaped, off_lp
    torch.cuda.empty_cache()

    Z = on_token_count + off_token_count
    total_loss = (on_policy_loss + off_policy_loss) / Z + entropy_loss
    return total_loss
