#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=0-03:00
#SBATCH --output=logs/sdft-%j.out
#SBATCH --error=logs/sdft-%j.err
#SBATCH --job-name=sdft

cd ~/Self-Distillation
source setup_env.sh --job

# Single H100 — no DeepSpeed needed
python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir $SCRATCH/sdft_output \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --num_prompts_per_batch 32 \
    --ref_model_mixup_alpha 0.01
