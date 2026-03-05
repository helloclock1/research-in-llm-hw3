import random
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset
from math_verify import parse, verify
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler

from reward import compute_reward

MAX_TRACE_WORDS = 1000
OFF_POLICY_RATIO = 1
ON_POLICY_RATIO = 7


"""
example = {
    "problem": Problem description,
    "generations": Reasoning traces,
    "answer": Ground truth answer,
}
"""


def get_correct_trace(example: Dict[str, Any]) -> Optional[str]:
    for trace, is_correct in zip(
        example["generations"], example["correctness_math_verify"]
    ):
        if is_correct:
            return trace
    return None


def is_answer_parseable(example):
    try:
        result = parse(example["answer"])
        return result is not None and result != []
    except Exception:
        return False


def is_trace_suitable_length(example: Dict[str, Any]) -> bool:
    trace = get_correct_trace(example)
    if trace is None:
        return False
    return len(trace.split()) <= MAX_TRACE_WORDS


def has_boxed_answer(example):
    trace = get_correct_trace(example)
    return trace is not None and "\\boxed" in trace


def off_policy_reward_valid(example):
    trace = get_correct_trace(example)
    if trace is None:
        return False
    reward = compute_reward(trace, example["answer"], verbose=False)
    return reward == 1.0


class LUFFYDataset(Dataset):
    def __init__(self, dataset):
        self.ds = dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return self.ds[idx]

def collate_fn(examples):
    return {
        "prompts": [ex["prompt"] for ex in examples],
        "answers": [ex["answer"] for ex in examples],
        "off_policy_traces": [ex["off_policy_trace"] for ex in examples],
    }

class MixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def _shuffle_indices(self):
        random.shuffle(self.indices)

    def __iter__(self):
        if self.shuffle:
            self._shuffle_indices()
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            yield batch_indices

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def build_dataloader(dataset, batch_size: int, shuffle: bool = True):
    luffy_dataset = LUFFYDataset(dataset)
    sampler = MixedBatchSampler(luffy_dataset, batch_size, shuffle)
    dataloader = DataLoader(luffy_dataset, batch_sampler=sampler, collate_fn=collate_fn)
    return dataloader
