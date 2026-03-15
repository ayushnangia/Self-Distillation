"""DEPRECATED: Use train.py --method sft --task tooluse instead.

SFT baseline for tool-use task.

Matches the paper's hyperparameter sweep:
- Learning rates: {5e-6, 1e-5, 5e-5}
- Batch sizes: {16, 32, 64}
- Epochs: {1, 2}
- Optimizer: AdamW, cosine schedule, 10 warmup steps
"""

import argparse
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Baseline Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_tooluse_dataset(seed=42):
    """Load tool-use dataset formatted for SFT (prompt -> golden_response)."""
    train_data = json.load(open("data/tooluse_data/train_data.json"))
    eval_data = json.load(open("data/tooluse_data/eval_data.json"))

    def format_for_sft(examples):
        formatted = []
        for ex in examples:
            golden_resp = ex.get("golden_response", [])
            if isinstance(golden_resp, list):
                response_text = "\n".join(golden_resp)
            else:
                response_text = str(golden_resp)

            formatted.append({
                "messages": [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": response_text},
                ]
            })
        return formatted

    train_formatted = format_for_sft(train_data)
    eval_formatted = format_for_sft(eval_data)

    train_dataset = Dataset.from_list(train_formatted).shuffle(seed=seed)
    eval_dataset = Dataset.from_list(eval_formatted)
    return train_dataset, eval_dataset


if __name__ == "__main__":
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset, eval_dataset = load_tooluse_dataset(args.seed)

    # Match paper's hyperparameters exactly
    config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        max_grad_norm=1.0,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        seed=args.seed,
        max_length=4096,
        report_to="wandb",
        eval_strategy="steps",
        eval_steps=50,
        weight_decay=0.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()
