#!/bin/bash
# Environment setup for Self-Distillation on Slurm HPC clusters
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
    # Run on login node (has internet). Takes ~5 min.
    virtualenv --no-download ~/sdft_env
    source ~/sdft_env/bin/activate
    pip install --no-index --upgrade pip

    # Core ML packages (from pre-built wheels)
    pip install --no-index torch numpy scipy matplotlib pandas
    pip install --no-index transformers accelerate datasets deepspeed peft trl
    pip install --no-index vllm==0.12.0 wandb openai rich

    # lm-eval for benchmarks + its missing deps
    pip install --no-index lm-eval langdetect immutabledict

    echo ""
    echo "Environment setup complete at ~/sdft_env"
    echo ""
    echo "=== NEXT: Run pre-caching (required — compute nodes have no internet) ==="
    echo "  source setup_env.sh --precache"
    exit 0
fi

if [[ "$1" == "--precache" ]]; then
    # === PRE-CACHE MODELS & DATASETS (run on login node) ===
    # Compute nodes have no internet — everything must be cached here first.
    source ~/sdft_env/bin/activate
    export HF_HOME=$SCRATCH/hf_cache
    mkdir -p $HF_HOME

    echo "1/3 Pre-caching base model (Qwen2.5-7B-Instruct)..."
    python3 -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='$HF_HOME')
print('  Base model cached')
"

    echo "2/3 Pre-caching lm-eval benchmark datasets..."
    python3 -c "
import os
os.environ['HF_ALLOW_CODE_EVAL'] = '1'
from lm_eval.tasks import TaskManager
tm = TaskManager()
for t in ['hellaswag', 'truthfulqa_mc2', 'mmlu', 'winogrande', 'humaneval', 'ifeval']:
    print(f'  Caching {t}...')
    try:
        tm.load_task_or_group(t)
    except Exception as e:
        print(f'    Warning: {e}')
print('  Benchmark datasets cached')
"

    echo "3/3 Pre-caching training datasets..."
    python3 -c "
from src.data import load_tooluse, load_science, load_medical
load_tooluse(); print('  tooluse cached')
load_science(); print('  science cached')
load_medical(); print('  medical cached')
" 2>/dev/null || echo "  (some datasets already in data/ dir, skipping)"

    echo ""
    echo "=== Pre-caching complete. Ready to submit jobs. ==="
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
    export HF_HUB_DISABLE_XET=1
    export TOKENIZERS_PARALLELISM=false
    export TRITON_CACHE_DIR=$SLURM_TMPDIR/.triton
    # W&B offline — sync later from login node
    # Sync later from login node: wandb sync $SCRATCH/wandb/offline-run-*
    export WANDB_MODE=offline
fi

# Copy data to $SLURM_TMPDIR for fast local I/O (if in a job)
if [[ -n "$SLURM_TMPDIR" ]] && [[ -d "data" ]] && [[ ! -d "$SLURM_TMPDIR/data" ]]; then
    echo "Copying data to \$SLURM_TMPDIR for fast I/O..."
    cp -r data "$SLURM_TMPDIR/"
    export SDFT_DATA_DIR="$SLURM_TMPDIR/data"
else
    export SDFT_DATA_DIR="data"
fi
