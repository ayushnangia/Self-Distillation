"""DEPRECATED: Use train.py --method sdft --task science instead."""

from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--num_prompts_per_batch", type=int, default=32)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser.parse_args()


def load_science_dataset(seed=42):
    """Load SciKnowEval Chemistry L-3 for SDFT."""
    ds = load_dataset("hicai-zju/SciKnowEval", split="test")
    df = ds.to_pandas()

    # Filter Chemistry L3
    df["level"] = df["details"].apply(
        lambda d: d.get("level", "") if isinstance(d, dict) else ""
    )
    chem_l3 = df[(df["domain"] == "Chemistry") & (df["level"] == "L3")]

    formatted = []
    for _, row in chem_l3.iterrows():
        prompt_template = row["prompt"]
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("default", "")
        question = row["question"]
        answer = row.get("answerKey", "")

        choices = row.get("choices", {})
        if isinstance(choices, dict):
            choice_text = choices.get("text", [])
            choice_labels = choices.get("label", [])
            choices_str = "\n".join(f"{l}. {t}" for l, t in zip(choice_labels, choice_text))
        else:
            choices_str = ""

        full_prompt = f"{prompt_template}\n\nQuestion: {question}\n{choices_str}"

        # Teacher gets the answer as demonstration
        teacher_content = f"""{full_prompt}

This is an example for a response to the question:
The correct answer is {answer}.

Now answer with a response of your own.
"""
        formatted.append({
            "prompt": [{"role": "user", "content": full_prompt}],
            "teacher_prompt": [{"role": "user", "content": teacher_content}],
        })

    dataset = Dataset.from_list(formatted).shuffle(seed=seed)
    # 75% train, 5% val, 20% test
    n = len(dataset)
    train_n = int(n * 0.75)
    train = dataset.select(range(train_n))
    return train, None


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, _ = load_science_dataset(args.seed)

    config = DistilConfig(
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.5,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=args.num_train_epochs,
        save_steps=50,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=args.output_dir,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
