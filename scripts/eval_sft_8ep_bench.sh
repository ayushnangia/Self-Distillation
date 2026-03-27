#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --time=0-02:59
#SBATCH --output=logs/eval-sft8ep-bench-%j.out
#SBATCH --error=logs/eval-sft8ep-bench-%j.err
#SBATCH --job-name=eval-sft8ep-b

cd ~/Self-Distillation
source setup_env.sh --job
export HF_ALLOW_CODE_EVAL=1

MODEL="$SCRATCH/sft_tooluse_output/lr2e-5_bs32_ep8"
RESULTS="$SCRATCH/benchmark_results/sft-8ep-lr2e-5"
mkdir -p "$RESULTS"

echo "=== Benchmarks: SFT 8-epoch ==="
lm_eval --model vllm \
    --model_args "pretrained=$MODEL,gpu_memory_utilization=0.9,max_model_len=2048,trust_remote_code=True" \
    --tasks hellaswag,truthfulqa_mc2,mmlu,winogrande,ifeval \
    --batch_size auto \
    --output_path "$RESULTS/lm_eval" \
    --log_samples \
    --confirm_run_unsafe_code

echo "=== Done ==="
