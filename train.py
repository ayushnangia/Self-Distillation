"""Unified training entry point for SDFT, SFT, and DFT.

Usage:
    # SDFT (self-distillation fine-tuning)
    accelerate launch --config_file configs/accelerate/sdft.yaml \
        train.py --method sdft --task tooluse --output_dir output/sdft_tooluse

    # SFT (supervised fine-tuning baseline)
    accelerate launch --config_file configs/accelerate/sft.yaml \
        train.py --method sft --task science --output_dir output/sft_science

    # DFT (distillation fine-tuning baseline)
    accelerate launch --config_file configs/accelerate/sdft.yaml \
        train.py --method dft --task medical --output_dir output/dft_medical
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import TASK_LOADERS


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SDFT/SFT/DFT Training")
    parser.add_argument("--method", type=str, choices=["sdft", "sft", "dft"], required=True)
    parser.add_argument("--task", type=str, choices=["tooluse", "science", "medical"], required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser.parse_args()


def train_sdft(args, train_dataset, tokenizer):
    """Train with SDFT: DistilTrainer + EMA teacher (sync_ref_model=True)."""
    from distil_trainer import DistilTrainer
    from distil_config import DistilConfig

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

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
        gradient_accumulation_steps=args.batch_size,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()


def train_dft(args, train_dataset, tokenizer):
    """Train with DFT: DistilTrainer + fixed teacher (generate_from_teacher=True, sync_ref_model=False)."""
    from distil_trainer import DistilTrainer
    from distil_config import DistilConfig

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

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
        gradient_accumulation_steps=args.batch_size,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=False,
        generate_from_teacher=True,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()


def train_sft(args, train_dataset, tokenizer):
    """Train with SFT: TRL SFTTrainer + DeepSpeed ZeRO + gradient checkpointing."""
    from trl import SFTConfig, SFTTrainer

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

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
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        max_length=4096,
        report_to="wandb",
        weight_decay=0.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()


if __name__ == "__main__":
    args = parse_args()

    print(f"Method: {args.method} | Task: {args.task}")
    print(f"LR: {args.learning_rate} | BS: {args.batch_size} | Epochs: {args.num_train_epochs}")
    print(f"Output: {args.output_dir}")

    # Load dataset
    loader = TASK_LOADERS[args.task]
    train_dataset, val_dataset, test_dataset = loader(args.seed)
    print(f"Train: {len(train_dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Route to appropriate trainer
    if args.method == "sdft":
        train_sdft(args, train_dataset, tokenizer)
    elif args.method == "dft":
        train_dft(args, train_dataset, tokenizer)
    elif args.method == "sft":
        # SFT uses 'messages' column, not 'prompt'/'teacher_prompt'
        train_sft(args, train_dataset, tokenizer)
