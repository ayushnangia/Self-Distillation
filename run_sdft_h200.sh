#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out
#SBATCH --job-name=sdft

module load python/3.11 cuda/12.6
source ~/sdft_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ~/Self-Distillation

# Single H200 — no DeepSpeed, no FSDP, no offloading. Just works.
python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir $SCRATCH/sdft_output \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --num_prompts_per_batch 32 \
    --ref_model_mixup_alpha 0.01
