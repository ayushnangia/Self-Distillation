"""Sequential continual learning experiment (Figure 4).

Trains on 3 tasks sequentially: Science Q&A -> Tool Use -> Medical.
After each task, saves checkpoint for evaluation.
Supports both SDFT and SFT modes.

Usage:
    python train_sequential.py --method sdft --output_dir output/seq_sdft
    python train_sequential.py --method sft --output_dir output/seq_sft
"""

import argparse
import copy
import glob
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import TASK_LOADERS


def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint-N directory in output_dir, if any."""
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if checkpoints:
        return checkpoints[-1]
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["sdft", "sft", "dft"], required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_order", type=str, default="science,tooluse,medical",
                        help="Comma-separated task order")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a checkpoint directory")
    return parser.parse_args()


def train_sdft_on_task(model, tokenizer, train_dataset, output_dir, args):
    """Train using SDFT on a single task."""
    from distil_trainer import DistilTrainer
    from distil_config import DistilConfig

    # EMA teacher starts as a copy of the current student (not base model).
    # Critical for sequential CL: after task 1, the teacher for task 2
    # must be the task-1-trained model, not the original base model.
    ref_model = copy.deepcopy(model)

    config = DistilConfig(
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.5,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        num_generations=1,  # Paper: "single on-policy rollout per training example"
        max_prompt_length=1024,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=50,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=output_dir,
        log_completions=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    resume_ckpt = get_latest_checkpoint(output_dir)
    if resume_ckpt:
        print(f"  Resuming SDFT from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(output_dir)
    # Reload clean model from saved checkpoint to avoid DeepSpeed state issues
    # when passing to the next task in sequential training
    del trainer, ref_model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
    return model


def train_dft_on_task(model, tokenizer, train_dataset, output_dir, args):
    """Train using DFT on a single task (fixed teacher, generate_from_teacher=True)."""
    from distil_trainer import DistilTrainer
    from distil_config import DistilConfig

    # DFT teacher is a frozen copy of the current student at task start.
    ref_model = copy.deepcopy(model)

    config = DistilConfig(
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.5,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        num_generations=1,  # Paper: "single on-policy rollout per training example"
        max_prompt_length=1024,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=50,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=output_dir,
        log_completions=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    resume_ckpt = get_latest_checkpoint(output_dir)
    if resume_ckpt:
        print(f"  Resuming DFT from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(output_dir)
    del trainer, ref_model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
    return model


def train_sft_on_task(model, tokenizer, train_dataset, output_dir, args):
    """Train using SFT on a single task."""
    from trl import SFTConfig, SFTTrainer

    config = SFTConfig(
        output_dir=output_dir,
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
        seed=args.seed,
        max_length=4096,
        report_to="wandb",
        weight_decay=0.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # SFTTrainer expects only 'messages' column — remove extras
    sft_dataset = train_dataset.select_columns(["messages"])

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )
    resume_ckpt = get_latest_checkpoint(output_dir)
    if resume_ckpt:
        print(f"  Resuming SFT from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(output_dir)
    del trainer
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
    return model


if __name__ == "__main__":
    args = parse_args()
    tasks = args.task_order.split(",")

    print(f"Sequential training: {' -> '.join(tasks)}")
    print(f"Method: {args.method}")
    print(f"Output: {args.output_dir}")

    start_model = args.resume_from or args.model_name
    print(f"\nLoading model from {start_model}...")
    model = AutoModelForCausalLM.from_pretrained(start_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    for i, task_name in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {i+1}/{len(tasks)}: {task_name}")
        print(f"{'='*60}")

        loader = TASK_LOADERS[task_name]
        train_ds, val_ds, test_ds = loader(args.seed)
        print(f"  Train: {len(train_ds)}, Test: {len(test_ds) if test_ds else 'N/A'}")

        task_output = f"{args.output_dir}/task{i+1}_{task_name}"

        # Skip if this task was already completed (resume support)
        if os.path.exists(os.path.join(task_output, "config.json")):
            print(f"  Already completed — loading from {task_output}")
            del model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(task_output, torch_dtype=torch.bfloat16)
            continue

        if args.method == "sdft":
            model = train_sdft_on_task(model, tokenizer, train_ds, task_output, args)
        elif args.method == "dft":
            model = train_dft_on_task(model, tokenizer, train_ds, task_output, args)
        else:
            model = train_sft_on_task(model, tokenizer, train_ds, task_output, args)

        print(f"  Saved to {task_output}")

    print(f"\n{'='*60}")
    print("Sequential training complete!")
    print(f"{'='*60}")
