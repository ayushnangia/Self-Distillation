# SDFT Reproduction — Full Run Plan

## Overview

Reproduce the core results of "Self-Distillation Enables Continual Learning" (arXiv:2601.19897).
Goal: 3 methods (SDFT, SFT, DFT) × 3 tasks (Tool Use, Science Q&A, Medical) + sequential CL + benchmarks.

**Base model:** Qwen/Qwen2.5-7B-Instruct
**Cluster:** Any with L40S 48GB or better (H100 preferred for speed)

---

## HuggingFace Assets (all public)

Everything trained so far is on HuggingFace. Pull these on any new cluster instead of transferring files.

### SDFT Checkpoints (11 models, all evaluated)

| Repo | Step | Greedy Acc |
|------|------|-----------|
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-100` | 100 | 55.9% |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-200` | 200 | 48.5% |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-300` | 300 | 44.1% |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-400` | 400 | 47.1% |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-500` | 500 | 57.4% |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-600` | 600 | 47.1% |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-700` | 700 | 54.4% |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-800` | 800 | 52.9% |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-900` | 900 | 57.4% |
| **`ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-1000`** | **1000** | **64.7%** |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-1011` | 1011 | 57.4% |

### SFT Checkpoints (6 models, NOT yet evaluated)

**1-epoch (completed training, final model saved):**

| Repo | LR | BS | Epochs |
|------|-----|-----|--------|
| `ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr5e-6-bs32-ep1` | 5e-6 | 32 | 1 |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr1e-5-bs32-ep1` | 1e-5 | 32 | 1 |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr5e-5-bs32-ep1` | 5e-5 | 32 | 1 |

**2-epoch (timed out at ~98%, best checkpoint saved):**

| Repo | LR | BS | Best Step |
|------|-----|-----|-----------|
| `ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr5e-6-bs32-ep2-step250` | 5e-6 | 32 | 250 |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr1e-5-bs32-ep2-step250` | 1e-5 | 32 | 250 |
| `ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr5e-5-bs32-ep2-step200` | 5e-5 | 32 | 200 |

### Eval Results Dataset

| Repo | Contents |
|------|----------|
| `ayushnangia-sdft/sdft-reproduction-eval-results` | SDFT greedy + pass@k JSON for all 11 steps + base |

### Base Model

| Repo | What |
|------|------|
| `Qwen/Qwen2.5-7B-Instruct` | Base model (from Qwen, not ours) |

---

## Setup on a New Cluster

### 1. Clone code

```bash
git clone https://github.com/ayushnangia/Self-Distillation.git
cd Self-Distillation
```

### 2. Python environment

```bash
# Slurm HPC clusters (with pre-built wheels):
bash setup_env.sh --install
source setup_env.sh --precache

# Other environments:
pip install -r requirements.txt
```

### 3. Download all models from HuggingFace

```bash
export HF_HOME=/scratch/$USER/hf_cache
export HF_TOKEN=hf_...  # your token
mkdir -p $HF_HOME

python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

token = os.environ["HF_TOKEN"]
cache = os.environ["HF_HOME"]

# Base model
print("Downloading base model...")
snapshot_download("Qwen/Qwen2.5-7B-Instruct", cache_dir=cache, token=token)

# SDFT checkpoints (all 11)
for step in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1011]:
    repo = f"ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-{step}"
    print(f"Downloading {repo}...")
    snapshot_download(repo, cache_dir=cache, token=token)

# SFT 1-epoch models (3)
for lr in ["5e-6", "1e-5", "5e-5"]:
    repo = f"ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr{lr}-bs32-ep1"
    print(f"Downloading {repo}...")
    snapshot_download(repo, cache_dir=cache, token=token)

# SFT 2-epoch best checkpoints (3)
for lr, step in [("5e-6", 250), ("1e-5", 250), ("5e-5", 200)]:
    repo = f"ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr{lr}-bs32-ep2-step{step}"
    print(f"Downloading {repo}...")
    snapshot_download(repo, cache_dir=cache, token=token)

# Eval results dataset
from huggingface_hub import snapshot_download
snapshot_download("ayushnangia-sdft/sdft-reproduction-eval-results", repo_type="dataset",
                  cache_dir=cache, token=token)

print("\nAll downloads complete!")
EOF
```

### 4. Pre-cache datasets (needs internet)

```bash
python -c "from src.data import load_tooluse; load_tooluse()"
python -c "from src.data import load_science; load_science()"
python -c "from src.data import load_medical; load_medical()"
```

---

## Current State (2026-03-17)

### Done

| What | Status | HuggingFace |
|------|--------|-------------|
| SDFT Tool Use (train + eval) | Complete | 11 model repos + eval dataset |
| SFT Tool Use (6 configs trained) | Trained, **not evaluated** | 6 model repos |
| Codebase | Complete | github.com/ayushnangia/Self-Distillation |
| Replication report | Complete | `REPLICATION_REPORT.md` |

### Key Results So Far

| Method | Tool Use Acc | Notes |
|--------|-------------|-------|
| Base (ours) | 54.4% | Paper reports 42.9% |
| SDFT best (ours, step-1000) | 64.7% | Paper reports 70.6% |
| SFT (ours) | **unevaluated** | Paper reports 63.2% |

### Blocked

- Primary cluster fairshare tanked — jobs stuck on Priority
- Using alternative cluster with H100 GPUs

---

## Phase 1: Evaluate What We Have (no training needed)

**Priority: HIGHEST — unlocks SDFT vs SFT comparison immediately**

### 1a. Eval all 6 SFT checkpoints on Tool Use

```bash
# Using HF model IDs (works on any cluster after download):
for REPO in \
    ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr5e-6-bs32-ep1 \
    ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr1e-5-bs32-ep1 \
    ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr5e-5-bs32-ep1 \
    ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr5e-6-bs32-ep2-step250 \
    ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr1e-5-bs32-ep2-step250 \
    ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-lr5e-5-bs32-ep2-step200 \
; do
    NAME=$(echo $REPO | sed 's|.*/||')
    echo "Evaluating: $NAME"
    python evaluate.py \
        --model_path "$REPO" \
        --task tooluse \
        --output_dir results/sft_eval/$NAME \
        --n_samples 50 \
        --k_values 1,5,10,20,50
done
```

Or via Slurm:
```bash
sbatch --time=1:00:00 --mem=48G scripts/eval_all_sft.sh
```

**Resource:** 1 GPU, 48G RAM, ~15 min per model, ~1.5h total
**Output:** 6 JSON files with greedy accuracy + pass@k

### 1b. Run prior capability benchmarks on SDFT best + base

```bash
# Base model
python evaluate.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --task benchmarks \
    --output_dir results/benchmarks/base

# SDFT best
python evaluate.py \
    --model_path ayushnangia-sdft/qwen2.5-7b-instruct-sdft-tooluse-step-1000 \
    --task benchmarks \
    --output_dir results/benchmarks/sdft-step-1000
```

**Output:** HellaSwag, HumanEval, IFEval, MMLU, TruthfulQA, Winogrande scores

### 1c. Benchmarks on best SFT checkpoint (after 1a identifies it)

```bash
python evaluate.py \
    --model_path ayushnangia-sdft/qwen2.5-7b-instruct-sft-tooluse-BEST \
    --task benchmarks \
    --output_dir results/benchmarks/sft-best
```

---

## Phase 2: Train Missing Models

**Priority: HIGH — completes the 3-method × 3-task matrix**

```bash
# SDFT on Science + Medical
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method sdft --task science --output_dir output/sdft_science
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method sdft --task medical --output_dir output/sdft_medical

# DFT on all 3 tasks
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method dft --task tooluse --output_dir output/dft_tooluse
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method dft --task science --output_dir output/dft_science
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method dft --task medical --output_dir output/dft_medical

# SFT on Science + Medical
accelerate launch --config_file configs/accelerate/sft.yaml \
    train.py --method sft --task science --output_dir output/sft_science
accelerate launch --config_file configs/accelerate/sft.yaml \
    train.py --method sft --task medical --output_dir output/sft_medical
```

**Resource requirements:**
- SDFT/DFT: 375G RAM (DeepSpeed ZeRO-3 + vLLM colocate), 1 GPU, ~6-8h
- SFT: 64G RAM (DeepSpeed ZeRO-2), 1 GPU, ~3-6h

On Slurm, use `scripts/run_single_task.sh METHOD TASK LR BS EPOCHS SEED`.
Jobs support `--resume` for checkpoint recovery.

---

## Phase 3: Sequential Continual Learning (Figure 4)

**Priority: HIGH — most visually compelling result**

```bash
python train_sequential.py --method sdft --output_dir output/seq_sdft
python train_sequential.py --method sft  --output_dir output/seq_sft
python train_sequential.py --method dft  --output_dir output/seq_dft
```

Task order: Science → Tool Use → Medical (per paper).
After each task, checkpoint saved for evaluation.

---

## Phase 4: Evaluate Everything

After training, evaluate all checkpoints on task-specific metrics + 6 benchmarks:

```bash
# For each trained model:
python evaluate.py --model_path output/MODEL --task TASK --output_dir results/MODEL
python evaluate.py --model_path output/MODEL --task benchmarks --output_dir results/MODEL
```

---

## Phase 5: Generate Figures + Tables

```bash
python analysis/collect_results.py --results_dir results/ --output results.csv
python analysis/plot_pareto.py --csv results.csv --output figures/pareto.pdf          # Figure 2
python analysis/plot_pass_at_k.py --results_dir results/ --output figures/pass_at_k.pdf  # Figure 3 right
python analysis/plot_sequential.py --sdft_dir results/seq_sdft --sft_dir results/seq_sft --output figures/seq.pdf  # Figure 4
python analysis/generate_table.py --csv results.csv --output figures/table1.tex       # Table 1
```

---

## Cluster Options

Any Slurm cluster with H100 80GB or L40S 48GB GPUs works. Use `setup_env.sh` for environment setup. Adjust `--account` and `--gpus-per-node` in scripts as needed.

---

## Summary: What's Needed

**Minimum viable (Phase 1 only, ~2h GPU):**
- Eval 6 SFT models on tool-use → SDFT vs SFT comparison
- Benchmarks on base + SDFT best + SFT best → Pareto plot

**Full reproduction (Phases 1-5, ~80-100h GPU):**
- 3 methods × 3 tasks + sequential CL + all figures/tables
