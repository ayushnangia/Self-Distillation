#!/bin/bash
# Environment setup for Self-Distillation on Alliance Canada clusters (Rorqual)
#
# Two modes:
#   source setup_env.sh          — activate env (interactive / login node)
#   source setup_env.sh --job    — activate env inside Slurm job (sets offline mode)
#
# First-time setup (run once):
#   bash setup_env.sh --install
#
# IMPORTANT: modules must be loaded BEFORE activating virtualenv
# (arrow and opencv provide pyarrow and cv2 as system modules)

set -e

# Load required modules (must come before venv activation)
module load gcc/12.3 arrow python/3.11 cuda/12.6 opencv

if [[ "$1" == "--install" ]]; then
    # === ONE-TIME INSTALLATION ===
    virtualenv --no-download ~/sdft_env
    source ~/sdft_env/bin/activate
    pip install --no-index --upgrade pip

    # Core ML packages (all from Alliance pre-built wheels)
    pip install --no-index torch numpy scipy matplotlib pandas
    pip install --no-index transformers accelerate datasets deepspeed peft trl
    pip install --no-index vllm==0.12.0 wandb openai rich

    echo ""
    echo "Environment setup complete at ~/sdft_env"
    echo "Activate with: source setup_env.sh"
    exit 0
fi

# === ACTIVATION ===
source ~/sdft_env/bin/activate

# HuggingFace cache
export HF_HOME=$SCRATCH/hf_cache

# Offline mode for job scripts (no internet on compute nodes)
if [[ "$1" == "--job" ]] || [[ -n "$SLURM_JOB_ID" ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export TRITON_CACHE_DIR=$SLURM_TMPDIR/.triton
    export WANDB_MODE=offline
fi
