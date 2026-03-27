#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --time=0-04:00
#SBATCH --output=logs/eval-bench-%j.out
#SBATCH --error=logs/eval-bench-%j.err
#SBATCH --job-name=eval-bench

# Run all 6 paper benchmarks (no task evals, no medical judge).
# Usage: sbatch scripts/eval_benchmarks_only.sh <model_path> <output_name>

MODEL_PATH=${1:?Usage: $0 MODEL_PATH OUTPUT_NAME}
OUTPUT_NAME=${2:?Usage: $0 MODEL_PATH OUTPUT_NAME}

cd ~/Self-Distillation
source setup_env.sh --job
export HF_ALLOW_CODE_EVAL=1

RESULTS_DIR="$SCRATCH/corrected_results/$OUTPUT_NAME"
mkdir -p "$RESULTS_DIR"

echo "=== Benchmarks: $OUTPUT_NAME ==="
echo "Model: $MODEL_PATH"

# Benchmarks WITHOUT chat template (HellaSwag, TruthfulQA mc1, MMLU, Winogrande, HumanEval)
echo "--- Benchmarks (no chat template) ---"
lm_eval --model vllm \
    --model_args "pretrained=$MODEL_PATH,gpu_memory_utilization=0.9,max_model_len=2048,trust_remote_code=True" \
    --tasks hellaswag,truthfulqa_mc1,mmlu,winogrande,humaneval \
    --batch_size auto \
    --output_path "$RESULTS_DIR/lm_eval" \
    --log_samples \
    --confirm_run_unsafe_code

# IFEval WITH chat template
echo "--- IFEval (with chat template) ---"
lm_eval --model vllm \
    --model_args "pretrained=$MODEL_PATH,gpu_memory_utilization=0.9,max_model_len=2048,trust_remote_code=True" \
    --tasks ifeval \
    --apply_chat_template \
    --batch_size auto \
    --output_path "$RESULTS_DIR/lm_eval" \
    --log_samples \
    --confirm_run_unsafe_code

echo "=== Done: $OUTPUT_NAME ==="
