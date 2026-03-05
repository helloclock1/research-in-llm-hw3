from unsloth import FastLanguageModel
import os
import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from reward import compute_reward

MODEL_PATH = "./outputs/sft/final"
SYSTEM_PROMPT = "You are helpful assistant. Please reason step by step, and put your final answer within \\boxed{}."
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "./outputs/sft_grpo"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


def reward_fn(completions, **kwargs):
    answers = kwargs.get("answer", [])
    return [compute_reward(c[0]["content"], a) for c, a in zip(completions, answers)]


def get_datasets():
    data_files = {"train": "./baseline_results/hard_train.jsonl"}
    dataset = load_dataset("json", data_files=data_files)
    return dataset["train"]


train_ds = get_datasets()


def format_grpo(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "answer": example["answer"],
    }


train_ds = train_ds.map(format_grpo)

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name="qwen-grpo-sft-refined",
    num_generations=8,
    max_completion_length=2048,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.7,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    max_steps=1000,
    bf16=True,
    logging_steps=5,
    save_steps=200,
    gradient_checkpointing=True,
    report_to="wandb",
)

FastLanguageModel.for_training(model)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_fn,
    train_dataset=train_ds,
    args=training_args,
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
