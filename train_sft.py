from unsloth import FastLanguageModel  # isort: skip
from trl import SFTConfig, SFTTrainer  # isort: skip

import weave

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from data import get_correct_trace
from datasets import load_dataset

SEED = 1489
MAX_SEQ_LENGTH = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
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

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def get_datasets():
    data_files = {
        "train": "./baseline_results/hard_train.jsonl",
        "test": "./baseline_results/hard_val.jsonl",
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset["train"], dataset["test"]


train_ds, val_ds = get_datasets()


SYSTEM_PROMPT = "You are helpful assistant. Please reason step by step, and put your final answer within \\boxed{}."


def format_sft(example):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['problem']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    completion = f"{example['gold_trace']}{tokenizer.eos_token}"
    return {"text": prompt + completion}


train_ds = train_ds.map(format_sft)
val_ds = val_ds.map(format_sft)
train_ds = train_ds.filter(lambda x: x["is_hard"] == True)
val_ds = val_ds.filter(lambda x: x["is_hard"] == True)


config = SFTConfig(
    output_dir="./outputs/sft",
    dataset_text_field="text",
    eos_token=tokenizer.eos_token,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    learning_rate=5e-5,
    logging_steps=1,
    eval_strategy="no",
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    save_strategy="steps",
    save_steps=100,
    max_seq_length=MAX_SEQ_LENGTH,
    bf16=True,
    report_to="wandb",
    run_name="qwen-sft-hard",
    optim="adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=config,
)
trainer.train()
model.save_pretrained("./outputs/sft/final")
tokenizer.save_pretrained("./outputs/sft/final")
