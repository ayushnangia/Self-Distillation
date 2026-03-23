#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --time=0-06:00
#SBATCH --output=logs/eval-corrected-%j.out
#SBATCH --error=logs/eval-corrected-%j.err
#SBATCH --job-name=eval-fix

# Full evaluation matching paper protocol:
# - Tooluse: 68-example paper split, greedy + pass@128
# - Science: exact match on answer letter
# - Medical: generate only (GPT judge runs separately on login node)
# - Benchmarks: HellaSwag, TruthfulQA mc1, MMLU, Winogrande, HumanEval (no chat template)
# - IFEval: with chat template
#
# Usage: sbatch scripts/eval_corrected.sh <model_path> <output_name>

MODEL_PATH=${1:?Usage: $0 MODEL_PATH OUTPUT_NAME}
OUTPUT_NAME=${2:?Usage: $0 MODEL_PATH OUTPUT_NAME}

cd ~/Self-Distillation
source setup_env.sh --job
export HF_ALLOW_CODE_EVAL=1

RESULTS_DIR="$SCRATCH/corrected_results/$OUTPUT_NAME"
mkdir -p "$RESULTS_DIR"

echo "=== Corrected eval: $OUTPUT_NAME ==="
echo "Model: $MODEL_PATH"
echo "Output: $RESULTS_DIR"

# 1. Tooluse eval on original 68 examples
echo ""
echo "--- Tooluse (original 68 examples) ---"
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --task tooluse \
    --output_dir "$RESULTS_DIR" \
    --eval_data data/tooluse_data/eval_data_paper_68.json \
    --n_samples 128 \
    --k_values 1,5,10,20,50,128

# 2. Science eval (exact match)
echo ""
echo "--- Science Q&A ---"
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --task science \
    --output_dir "$RESULTS_DIR"

# 3. Medical eval (generate only — judge runs on login node later)
echo ""
echo "--- Medical (generate only, no judge) ---"
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --task medical \
    --generate_only \
    --output_dir "$RESULTS_DIR"

# 5. Benchmarks WITHOUT chat template (HellaSwag, TruthfulQA, MMLU, Winogrande, HumanEval)
echo ""
echo "--- Benchmarks (mc1, no chat template) ---"
lm_eval --model vllm \
    --model_args "pretrained=$MODEL_PATH,gpu_memory_utilization=0.9,max_model_len=2048,trust_remote_code=True" \
    --tasks hellaswag,truthfulqa_mc1,mmlu,winogrande,humaneval \
    --batch_size auto \
    --output_path "$RESULTS_DIR/lm_eval" \
    --log_samples

# 6. IFEval WITH chat template
echo ""
echo "--- IFEval (with chat template) ---"
lm_eval --model vllm \
    --model_args "pretrained=$MODEL_PATH,gpu_memory_utilization=0.9,max_model_len=2048,trust_remote_code=True" \
    --tasks ifeval \
    --apply_chat_template \
    --batch_size auto \
    --output_path "$RESULTS_DIR/lm_eval" \
    --log_samples

echo "=== Done: $OUTPUT_NAME ==="
