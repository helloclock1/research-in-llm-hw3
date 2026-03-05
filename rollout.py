from unsloth import FastLanguageModel  # isort: skip

from typing import List

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from data import ON_POLICY_RATIO
from model import save_for_vllm


class VLLMRollout:
    def __init__(self, model_path: str, gpu_memory_utilization: float, max_tokens: int):
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=max_tokens,
            n=ON_POLICY_RATIO,
        )
        self.gpu_memory_utilization = gpu_memory_utilization

    def generate(self, prompts: List[str]) -> List[List[str]]:
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
        )
        return [[o.text for o in output.outputs] for output in outputs]

    def sync_weights(self, model, tokenizer, path: str):
        save_for_vllm(model, tokenizer, path)
        self.llm = LLM(
            model=path,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )


class DirectRollout:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    @torch.no_grad()
    def generate(self, model, tokenizer, prompts, gen_batch_size=4):
        FastLanguageModel.for_inference(model)
        results = []

        for i in range(0, len(prompts), gen_batch_size):
            chunk = prompts[i : i + gen_batch_size]
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=ON_POLICY_RATIO,
            )

            prompt_len = inputs["input_ids"].shape[1]
            texts = [
                tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in out
            ]
            for j in range(len(chunk)):
                results.append(texts[j * ON_POLICY_RATIO : (j + 1) * ON_POLICY_RATIO])

        FastLanguageModel.for_training(model)
        return results

    def sync_weights(self, *args):
        pass
