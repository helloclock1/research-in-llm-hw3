import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import wandb
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from advantage import compute_advantages
from data import ON_POLICY_RATIO, build_dataloader
from loss import compute_loss
from model import (
    MODEL_NAME,
    compute_entropy,
    compute_log_probs,
    load_model,
    save_for_vllm,
)
from reward import compute_rewards, extract_boxed
from rollout import DirectRollout, VLLMRollout

SYNC_EVERY = 20
UPDATE_EPOCHS = 1
LR = 2e-5
MAX_STEPS = 1500
BATCH_SIZE = 8
VLLM_MAX_TOKENS = 2048
VLLM_GPU_MEM = 0.35
VLLM_SYNC_PATH = "/tmp/luffy_sync"
EVAL_EVERY = 200
SEED = 1489
MAX_SEQ_LENGTH = 4096

TRAIN_PATH = "./baseline_results/hard_train.jsonl"
VAL_PATH = "./baseline_results/hard_val.jsonl"

SYSTEM_PROMPT = "You are helpful assistant. Please reason step by step, and put your final answer within \\boxed{}."


def format_chatml_prompt(problem: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def get_datasets():
    data_files = {"train": TRAIN_PATH, "test": VAL_PATH}
    dataset = load_dataset("json", data_files=data_files)
    train_ds = dataset["train"].filter(lambda x: x["is_hard"] == True)
    val_ds = dataset["test"].filter(lambda x: x["is_hard"] == True)

    def format_example(example):
        return {
            "prompt": format_chatml_prompt(example["problem"]),
            "answer": example["answer"],
            "off_policy_trace": example["gold_trace"],
        }

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names)

    print(f"Loaded {len(train_ds)} hard train, {len(val_ds)} hard val examples")
    return train_ds, val_ds


def evaluate(model, tokenizer, rollout, val_loader):
    total_reward = 0.0
    total_samples = 0

    for batch in val_loader:
        # completions = rollout.generate(batch["prompts"])
        completions = rollout.generate(model, tokenizer, batch["prompts"])
        flat_completions = [c for comps in completions for c in comps]
        repeated_answers = [a for a in batch["answers"] for _ in range(ON_POLICY_RATIO)]
        rewards = compute_rewards(flat_completions, repeated_answers)
        total_reward += sum(rewards)
        total_samples += len(rewards)

    return total_reward / total_samples


def train(model, tokenizer, rollout, train_loader, val_loader, optimizer):
    train_iter = iter(train_loader)
    pbar = tqdm(range(MAX_STEPS), desc="Training", ncols=140)

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # on_policy_completions = rollout.generate(batch["prompts"])
        on_policy_completions = rollout.generate(model, tokenizer, batch["prompts"])
        flat_completions = [c for comps in on_policy_completions for c in comps]
        repeated_answers = [a for a in batch["answers"] for _ in range(ON_POLICY_RATIO)]

        on_policy_rewards = compute_rewards(flat_completions, repeated_answers)
        off_policy_rewards = compute_rewards(
            batch["off_policy_traces"], batch["answers"]
        )

        for trace, answer, reward in zip(
            batch["off_policy_traces"], batch["answers"], off_policy_rewards
        ):
            if reward < 1.0:
                boxed = extract_boxed(trace)
                print(f"OFF-POLICY FAIL: boxed={boxed}, answer={answer}")

        advantages = compute_advantages(on_policy_rewards, off_policy_rewards).to(
            model.device
        )

        flat_prompts = [p for p in batch["prompts"] for _ in range(ON_POLICY_RATIO)]
        flat_sequences = [p + c for p, c in zip(flat_prompts, flat_completions)]
        old_log_probs, _ = compute_log_probs(
            model, tokenizer, flat_sequences, grad=False
        )
        old_log_probs = old_log_probs.detach()

        entropy = compute_entropy(model, tokenizer, flat_sequences)
        torch.cuda.empty_cache()

        completion_tok_lens = [
            len(tokenizer.encode(c, add_special_tokens=False)) for c in flat_completions
        ]
        mean_comp_toks = sum(completion_tok_lens) / len(completion_tok_lens)

        for _ in range(UPDATE_EPOCHS):
            optimizer.zero_grad()
            loss = compute_loss(
                model,
                tokenizer,
                batch,
                on_policy_completions,
                old_log_probs,
                advantages,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()

        if step % SYNC_EVERY == 0:
            rollout.sync_weights(model, tokenizer, VLLM_SYNC_PATH)

        mean_on_reward = sum(on_policy_rewards) / len(on_policy_rewards)
        mean_off_reward = sum(off_policy_rewards) / len(off_policy_rewards)
        frac_on_correct = sum(1 for r in on_policy_rewards if r >= 1.0) / len(
            on_policy_rewards
        )
        frac_off_correct = sum(1 for r in off_policy_rewards if r >= 1.0) / len(
            off_policy_rewards
        )

        wandb.log(
            {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm,
                "train/entropy": entropy,
                "train/on_policy_reward": mean_on_reward,
                "train/off_policy_reward": mean_off_reward,
                "train/on_policy_acc": frac_on_correct,
                "train/off_policy_acc": frac_off_correct,
                "train/advantage_mean": advantages.mean().item(),
                "train/advantage_std": advantages.std().item(),
                "train/mean_completion_tokens": mean_comp_toks,
                "train/lr": optimizer.param_groups[0]["lr"],
            },
            step=step,
        )

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            on_r=f"{mean_on_reward:.3f}",
            off_r=f"{mean_off_reward:.3f}",
            ent=f"{entropy:.2f}",
            toks=f"{mean_comp_toks:.0f}",
        )

        if step > 0 and step % EVAL_EVERY == 0:
            val_reward = evaluate(model, tokenizer, rollout, val_loader)
            wandb.log({"val/reward": val_reward}, step=step)
            pbar.write(f"step {step} | val_reward {val_reward:.4f}")


def main():
    train_ds, val_ds = get_datasets()

    wandb.init(
        project="qwen-luffy-hard",
        config={
            "model": MODEL_NAME,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_steps": MAX_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "sync_every": SYNC_EVERY,
            "update_epochs": UPDATE_EPOCHS,
            "on_policy_ratio": ON_POLICY_RATIO,
            "vllm_max_tokens": VLLM_MAX_TOKENS,
            "vllm_gpu_mem": VLLM_GPU_MEM,
            "seed": SEED,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "system_prompt": SYSTEM_PROMPT,
        },
    )

    model, tokenizer = load_model()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_loader = build_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    rollout = DirectRollout(max_tokens=VLLM_MAX_TOKENS)
    optimizer = AdamW(model.parameters(), lr=LR)

    train(model, tokenizer, rollout, train_loader, val_loader, optimizer)
    wandb.finish()


if __name__ == "__main__":
    main()
