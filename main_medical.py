"""DEPRECATED: Use train.py --method sdft --task medical instead."""

from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset
import argparse


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


def load_medical_dataset(seed=42):
    """Load HuatuoGPT-o1 English for SDFT."""
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    ds = ds.shuffle(seed=seed)

    formatted = []
    for ex in ds:
        question = ex["Question"]
        response = ex["Response"]

        teacher_content = f"""{question}

This is an example for a response to the question:
{response}

Now answer with a response of your own, including the thinking process.
"""
        formatted.append({
            "prompt": [{"role": "user", "content": question}],
            "teacher_prompt": [{"role": "user", "content": teacher_content}],
        })

    dataset = Dataset.from_list(formatted)
    # Use ~19000 for training (paper says ~20K)
    train = dataset.select(range(min(19000, len(dataset))))
    return train, None


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, _ = load_medical_dataset(args.seed)

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
        save_steps=100,
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
