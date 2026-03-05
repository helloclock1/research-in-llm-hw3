from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs/sft/final",
    max_seq_length=4096,
    load_in_4bit=False,
)

model.save_pretrained_merged(
    "./outputs/sft/merged", tokenizer, save_method="merged_16bit"
)
