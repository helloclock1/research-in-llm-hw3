import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from reward import compute_reward

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N_SAMPLES = 64
MAX_TOKENS = 2048
TEMPERATURE = 1.0
DATASET_NAME = "open-r1/OpenR1-Math-220k"
OUTPUT_DIR = "./baseline_results"
SEED = 1489

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(
        np.arange(n - c - k + 1, n - c + 1) / np.arange(n - k + 1, n + 1)
    )


def get_correct_trace(example):
    for trace, is_correct in zip(
        example["generations"], example["correctness_math_verify"]
    ):
        if is_correct:
            return trace
    return None


def save_jsonl(data, filename):
    with open(f"{OUTPUT_DIR}/{filename}", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def main():
    ds = load_dataset(DATASET_NAME, "default", split="train")
    ds = ds.filter(lambda x: any(x["correctness_math_verify"]), num_proc=8)
    ds = ds.shuffle(seed=SEED).select(range(min(len(ds), 10000)))

    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    all_results = []

    batch_size = 100
    SYSTEM_PROMPT = "You are helpful assistant. Please reason step by step, and put your final answer within \\boxed{}."
    live_log_path = f"{OUTPUT_DIR}/live_results.jsonl"
    for i in tqdm(range(0, len(ds), batch_size), desc="Evaluating baseline"):
        batch = ds.select(range(i, min(i + batch_size, len(ds))))
        batch_prompts = []
        for x in batch:
            full_prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{x['problem']}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
            batch_prompts.append(full_prompt)

        outputs = llm.generate(batch_prompts, sampling_params)

        with open(live_log_path, "a") as f_live:
            for idx, output in enumerate(outputs):
                completions = [o.text for o in output.outputs]
                ground_truth = batch[idx]["answer"]

                correct_count = sum(
                    [
                        1
                        for att in completions
                        if compute_reward(att, ground_truth, verbose=False) == 1.0
                    ]
                )

                res = {
                    "problem": batch[idx]["problem"],
                    "answer": batch[idx]["answer"],
                    "gold_trace": get_correct_trace(batch[idx]),
                    "correct_count": correct_count,
                    "pass_at_1": calculate_pass_at_k(N_SAMPLES, correct_count, 1),
                    "pass_at_64": calculate_pass_at_k(N_SAMPLES, correct_count, 64),
                    "is_hard": (correct_count == 0),
                }
                all_results.append(res)
                f_live.write(json.dumps(res) + "\n")

        curr_hard = sum(1 for r in all_results if r["is_hard"])
        print(
            f"Step {i // batch_size} | Hard Found: {curr_hard} | Global Pass@1: {np.mean([r['pass_at_1'] for r in all_results]):.4f}",
            flush=True,
        )

    hard_subset = [r for r in all_results if r["is_hard"]]
    easy_subset = [r for r in all_results if not r["is_hard"]]

    avg_pass_1 = np.mean([r["pass_at_1"] for r in all_results])
    avg_pass_64 = np.mean([r["pass_at_64"] for r in all_results])

    stats = {
        "total_evaluated": len(all_results),
        "total_hard": len(hard_subset),
        "total_easy": len(easy_subset),
        "global_pass_at_1": avg_pass_1,
        "global_pass_at_64": avg_pass_64,
        "hard_subset_pass_at_64": 0.0,
    }

    with open(f"{OUTPUT_DIR}/baseline_stats.json", "json") as f:
        json.dump(stats, f, indent=4)

    random.shuffle(all_results)
    split_pt = int(len(all_results) * 0.95)
    save_jsonl(all_results[:split_pt], "full_train.jsonl")
    save_jsonl(all_results[split_pt:], "full_val.jsonl")

    random.shuffle(hard_subset)
    h_split_pt = int(len(hard_subset) * 0.95)
    save_jsonl(hard_subset[:h_split_pt], "hard_train.jsonl")
    save_jsonl(hard_subset[h_split_pt:], "hard_val.jsonl")


if __name__ == "__main__":
    main()
