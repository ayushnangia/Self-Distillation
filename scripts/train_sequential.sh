#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=0-08:00
#SBATCH --output=logs/seq-%x-%j.out
#SBATCH --error=logs/seq-%x-%j.err

# Usage:
#   sbatch --job-name=seq-sdft scripts/train_sequential.sh sdft
#   sbatch --job-name=seq-sft  scripts/train_sequential.sh sft
#   sbatch --job-name=seq-dft  scripts/train_sequential.sh dft

METHOD=${1:?Usage: $0 METHOD}

cd ~/Self-Distillation
source setup_env.sh --job

OUTPUT_DIR="$SCRATCH/seq_${METHOD}_output"
mkdir -p "$OUTPUT_DIR"

echo "Sequential CL: method=$METHOD"
echo "Task order: Science -> Tool Use -> Medical"
echo "Output: $OUTPUT_DIR"

python train_sequential.py \
    --method "$METHOD" \
    --output_dir "$OUTPUT_DIR"
