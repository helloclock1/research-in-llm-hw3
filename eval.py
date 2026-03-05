import json
import math
import numpy as np
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch

from reward import compute_reward

MODEL_PATH = "./outputs/sft/merged"
VAL_DATA_PATH = "./baseline_results/hard_val.jsonl"
N_SAMPLES = 64
MAX_TOKENS = 2048
TEMPERATURE = 1.0
SYSTEM_PROMPT = "You are helpful assistant. Please reason step by step, and put your final answer within \\boxed{}."


def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    full_val = []
    with open(VAL_DATA_PATH, "r") as f:
        for line in f:
            full_val.append(json.loads(line))

    hard_val = [x for x in full_val if x.get("is_hard", False)]
    print(f"Loaded {len(full_val)} total tasks. ({len(hard_val)} are Hard-only)")

    llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.9, max_model_len=4096)

    sampling_params = SamplingParams(
        n=N_SAMPLES, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
    )

    k_values = [1, 2, 4, 8, 16, 32, 64]

    def evaluate_subset(dataset, name):
        print(f"Evaluating {name}")

        prompts = [
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{x['problem']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for x in dataset
        ]

        outputs = llm.generate(prompts, sampling_params)

        correct_counts = []
        token_lengths = []

        for i, output in enumerate(outputs):
            correct_in_problem = 0
            ground_truth = dataset[i]["answer"]

            for sample in output.outputs:
                token_lengths.append(len(sample.token_ids))
                if compute_reward(sample.text, ground_truth, verbose=False) == 1.0:
                    correct_in_problem += 1
            correct_counts.append(correct_in_problem)

        stats = {}
        for k in k_values:
            pass_k = np.mean([calculate_pass_at_k(N_SAMPLES, c, k) for c in correct_counts])
            stats[f"pass@{k}"] = pass_k

        mean_len = np.mean(token_lengths)
        stats["mean_token_length"] = mean_len

        return stats

    full_stats = evaluate_subset(full_val, "Full val")
    hard_stats = evaluate_subset(hard_val, "Hard val")

    with open("sft_final_results.json", "w") as f:
        json.dump({"full_val": full_stats, "hard_val": hard_stats}, f, indent=4)


if __name__ == "__main__":
    main()
