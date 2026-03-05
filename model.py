from unsloth import FastLanguageModel  # isort: skip

from typing import List

import torch
from torch.nn.functional import log_softmax

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 16
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

SEED = 1489


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=LORA_TARGET_MODULES,
        random_state=SEED,
    )

    model.gradient_checkpointing_enable()

    return model, tokenizer


def compute_log_probs(
    model,
    tokenizer,
    sequences,
    grad=False,
    chunk_size=4,
):
    results = []
    masks = []
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i : i + chunk_size]
        grad_ctx = torch.enable_grad() if grad else torch.no_grad()
        with grad_ctx:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            log_probs = log_softmax(logits[:, :-1, :], dim=-1)
            del logits
            target_ids = inputs["input_ids"][:, 1:]
            token_log_probs = log_probs.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)
            del log_probs
            mask = inputs["attention_mask"][:, 1:]
            results.append(token_log_probs)
            masks.append(mask)
    max_len = max(r.shape[1] for r in results)
    padded_results = []
    padded_masks = []
    for r, m in zip(results, masks):
        pad_len = max_len - r.shape[1]
        if pad_len > 0:
            padded_results.append(torch.nn.functional.pad(r, (0, pad_len), value=0.0))
            padded_masks.append(torch.nn.functional.pad(m, (0, pad_len), value=0))
        else:
            padded_results.append(r)
            padded_masks.append(m)
    return torch.cat(padded_results, dim=0), torch.cat(padded_masks, dim=0)


def compute_entropy(model, tokenizer, sequences, chunk_size=1):
    total_entropy = 0.0
    total_tokens = 0

    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i : i + chunk_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        logits = logits[:, :-1, :]
        mask = attention_mask[:, 1:].bool()

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)

        total_entropy += (token_entropy * mask).sum().item()
        total_tokens += mask.sum().item()

        del logits, log_probs, probs, token_entropy

    return total_entropy / total_tokens if total_tokens > 0 else 0.0


def save_for_vllm(model, tokenizer, path: str):
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
