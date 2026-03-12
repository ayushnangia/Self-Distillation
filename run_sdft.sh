#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128000M
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out
#SBATCH --job-name=sdft

module load StdEnv/2023 gcc/12.3 arrow/18.1.0 opencv/4.11.0 python/3.11 cuda/12.6
source ~/sdft_env/bin/activate

# Use scratch for HF model cache (HOME is only 50GB)
export HF_HOME=/scratch/anangia/hf_cache

# Compute nodes may not have internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Triton cache on local SSD to avoid NFS slowdown
export TRITON_CACHE_DIR=$SLURM_TMPDIR/.triton

cd /home/anangia/Self-Distillation

# 2x L40S + DeepSpeed ZeRO-3
# - Model params sharded across 2 GPUs (~7GB each)
# - Optimizer states offloaded to CPU (~56GB)
# - vLLM colocate with sleep mode, gpu_memory_utilization=0.5
accelerate launch --config_file accelerate_config.yaml main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir /scratch/anangia/sdft_output \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --num_prompts_per_batch 32 \
    --ref_model_mixup_alpha 0.01 \
    --num_gpus 2
