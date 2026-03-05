from unsloth import FastLanguageModel  # isort: skip
from trl import GRPOConfig, GRPOTrainer  # isort: skip

from data import get_datasets
from reward import compute_reward


def reward_fn(completions, **kwargs):
    answers = kwargs.get("answer", [])
    return [compute_reward(c[0]["content"], a) for c, a in zip(completions, answers)]


model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

train_ds, val_ds = get_datasets(test_size=0.05, seed=1489)


def format_grpo(example):
    return {
        "prompt": [{"role": "user", "content": example["problem"]}],
        "answer": example["answer"],
    }


train_ds = train_ds.map(format_grpo)

training_args = GRPOConfig(
    output_dir="./outputs/grpo",
    per_device_train_batch_size=4,
    num_generations=8,
    max_completion_length=4096,
    learning_rate=1e-5,
    logging_steps=10,
    gradient_checkpointing=True,
    save_steps=500,
    max_steps=1000,
    use_vllm=True,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_fn,
    train_dataset=train_ds,
    args=training_args,
)
trainer.train()
model.save_pretrained("./outputs/grpo/final")
tokenizer.save_pretrained("./outputs/grpo/final")
