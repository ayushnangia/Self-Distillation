#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --time=0-02:59
#SBATCH --output=logs/eval-benchmarks-%j.out
#SBATCH --error=logs/eval-benchmarks-%j.err
#SBATCH --job-name=eval-bench

# Usage:
#   sbatch scripts/eval_benchmarks.sh <model_path> <output_name>

MODEL_PATH=${1:?Usage: $0 MODEL_PATH OUTPUT_NAME}
OUTPUT_NAME=${2:?Usage: $0 MODEL_PATH OUTPUT_NAME}

cd ~/Self-Distillation
source setup_env.sh --job

export HF_ALLOW_CODE_EVAL=1

RESULTS_DIR="$SCRATCH/benchmark_results/$OUTPUT_NAME"
mkdir -p "$RESULTS_DIR"

echo "=== Evaluating benchmarks: $OUTPUT_NAME ==="
echo "Model: $MODEL_PATH"
echo "Output: $RESULTS_DIR"

# Call lm_eval directly (not through evaluate.py) to avoid CUDA init issues
lm_eval --model vllm \
    --model_args "pretrained=$MODEL_PATH,gpu_memory_utilization=0.9,max_model_len=2048,trust_remote_code=True" \
    --tasks hellaswag,truthfulqa_mc2,mmlu,winogrande,ifeval \
    --batch_size auto \
    --output_path "$RESULTS_DIR/lm_eval" \
    --log_samples \
    --confirm_run_unsafe_code

echo "=== Done. Run 'seff $SLURM_JOB_ID' for resource usage ==="
