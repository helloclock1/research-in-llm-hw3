import re
from typing import List

from math_verify import parse, verify


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return f"\\boxed{{{matches[-1]}}}"
    return None


def compute_reward(trace: str, ground_truth: str, verbose: bool = True) -> float:
    try:
        boxed = extract_boxed(trace)
        if boxed is None:
            return 0.0
        answer = parse(boxed)
        golden = parse(ground_truth)
        correct = verify(golden, answer)
        if verbose:
            if not correct:
                print(f"MISMATCH: parsed={answer}, golden={golden}")
            else:
                print(f"SUCCESS: parsed={answer}, golden={golden}")
        return 1.0 if correct else 0.0
    except Exception as e:
        if verbose:
            print(f"PARSE FAILED: {e}, trace_tail=...{trace[-200:]}")
        return 0.0


def compute_rewards(
    traces: List[str], ground_truths: List[str], verbose: bool = False
) -> List[float]:
    return [
        compute_reward(trace, ground_truth, verbose)
        for trace, ground_truth in zip(traces, ground_truths)
    ]
