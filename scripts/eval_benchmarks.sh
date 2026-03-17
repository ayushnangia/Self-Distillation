#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --time=0-01:30
#SBATCH --output=logs/eval-benchmarks-%j.out
#SBATCH --error=logs/eval-benchmarks-%j.err
#SBATCH --job-name=eval-bench

cd ~/Self-Distillation
source setup_env.sh --job

# lm_eval datasets pre-cached on login node — keep offline mode
export HF_ALLOW_CODE_EVAL=1

RESULTS_BASE="$SCRATCH/benchmark_results"
CACHE="$SCRATCH/hf_cache"

# Resolve snapshot paths
BASE_MODEL=$(ls -d "$CACHE"/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ | head -1)
SDFT_1000=$(ls -d "$CACHE"/models--ayushnangia-sdft--qwen2.5-7b-instruct-sdft-tooluse-step-1000/snapshots/*/ | head -1)

# Base model
echo "=== Evaluating base model benchmarks ==="
echo "Model path: $BASE_MODEL"
mkdir -p "$RESULTS_BASE/base"
python evaluate.py \
    --model_path "$BASE_MODEL" \
    --task benchmarks \
    --output_dir "$RESULTS_BASE/base"

# SDFT best (step-1000, 64.7% greedy acc)
echo "=== Evaluating SDFT step-1000 benchmarks ==="
echo "Model path: $SDFT_1000"
mkdir -p "$RESULTS_BASE/sdft-step-1000"
python evaluate.py \
    --model_path "$SDFT_1000" \
    --task benchmarks \
    --output_dir "$RESULTS_BASE/sdft-step-1000"

echo "=== BENCHMARK EVALUATIONS COMPLETE ==="
