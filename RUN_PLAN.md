# SDFT Reproduction — Full Run Plan

## Overview

Reproduce the core results of "Self-Distillation Enables Continual Learning" (arXiv:2601.19897).
Goal: 3 methods (SDFT, SFT, DFT) × 3 tasks (Tool Use, Science Q&A, Medical) + sequential CL + benchmarks.

**Base model:** Qwen/Qwen2.5-7B-Instruct
**Cluster:** Any with L40S 48GB or better (H100 preferred for speed)

---

## Current State (2026-03-17)

### Done

| What | Status | Location |
|------|--------|----------|
| Codebase (train/eval/analysis) | Complete | `~/Self-Distillation/` |
| SDFT Tool Use (train + eval) | Complete | `/scratch/anangia/sdft_output/` |
| SDFT Tool Use eval results | Complete | `/scratch/anangia/sdft_eval_results/` |
| SFT Tool Use (12 configs trained) | Trained, **not evaluated** | `/scratch/anangia/sft_output/` |
| Replication report | Complete | `REPLICATION_REPORT.md` |

### Key Results So Far

| Method | Tool Use Acc | Notes |
|--------|-------------|-------|
| Base (ours) | 54.4% | Paper reports 42.9% |
| SDFT best (ours, step-1000) | 64.7% | Paper reports 70.6% |
| SFT (ours) | **unevaluated** | Paper reports 63.2% |

### Blocked

- Vulcan fairshare tanked (LevelFS=0.07 for `aip-rgrosse`)
- Jobs stuck on Priority even at 1h/48G
- Need alternative cluster or wait for fairshare recovery (~1 week half-life)

---

## Phase 1: Evaluate What We Have (no GPU training needed)

**Priority: HIGHEST — unlocks SDFT vs SFT comparison immediately**

### 1a. Eval all SFT checkpoints on Tool Use

12 trained models, zero evaluated. Run on any machine with 1 GPU.

```bash
# On cluster with GPU:
cd Self-Distillation
for CKPT in /scratch/anangia/sft_output/lr*; do
    NAME=$(basename $CKPT)
    python evaluate.py \
        --model_path "$CKPT" \
        --task tooluse \
        --output_dir /scratch/anangia/sft_eval_results/$NAME \
        --n_samples 50 \
        --k_values 1,5,10,20,50
done
```

Or use the Slurm script:
```bash
sbatch --time=1:00:00 --mem=48G scripts/eval_all_sft.sh
```

**Expected output:** 12 JSON files with greedy accuracy + pass@k for each SFT config.

### 1b. Run prior capability benchmarks on SDFT best + base

```bash
# Base model
python evaluate.py --model_path Qwen/Qwen2.5-7B-Instruct --task benchmarks \
    --output_dir /scratch/anangia/sdft_eval_results/base

# SDFT best checkpoint
python evaluate.py --model_path /scratch/anangia/sdft_output/checkpoint-1000 --task benchmarks \
    --output_dir /scratch/anangia/sdft_eval_results/step-1000
```

**Expected output:** HellaSwag, HumanEval, IFEval, MMLU, TruthfulQA, Winogrande scores.

### 1c. Run benchmarks on best SFT checkpoint (after 1a identifies it)

```bash
# After finding best SFT config from 1a:
python evaluate.py --model_path /scratch/anangia/sft_output/BEST_CONFIG \
    --task benchmarks --output_dir /scratch/anangia/sft_eval_results/BEST_CONFIG
```

---

## Phase 2: Train Missing Models

**Priority: HIGH — completes the 3-method × 3-task matrix**

### Training commands

Each uses `scripts/run_single_task.sh METHOD TASK LR BS EPOCHS SEED`:

```bash
# SDFT on Science + Medical
sbatch --time=8:00:00 --job-name=sdft-science scripts/run_single_task.sh sdft science
sbatch --time=8:00:00 --job-name=sdft-medical scripts/run_single_task.sh sdft medical

# DFT on all 3 tasks
sbatch --time=8:00:00 --job-name=dft-tooluse scripts/run_single_task.sh dft tooluse
sbatch --time=8:00:00 --job-name=dft-science scripts/run_single_task.sh dft science
sbatch --time=8:00:00 --job-name=dft-medical scripts/run_single_task.sh dft medical

# SFT on Science + Medical
sbatch --time=8:00:00 --job-name=sft-science scripts/run_single_task.sh sft science
sbatch --time=8:00:00 --job-name=sft-medical scripts/run_single_task.sh sft medical
```

**Resource requirements:**
- SDFT/DFT: 375G RAM (DeepSpeed ZeRO-3 + vLLM colocate), 1 GPU, ~6-8h
- SFT: 64G RAM (DeepSpeed ZeRO-2), 1 GPU, ~3-6h

### If using shorter time limits (backfill friendly)

Jobs support resuming. Submit with `--time=2:59:00` and resubmit if they timeout:
- `train.py` has `--resume` flag
- `train_sequential.py` has `--resume_from` flag
- Checkpoints auto-save every 50 steps

---

## Phase 3: Sequential Continual Learning (Figure 4)

**Priority: HIGH — most visually compelling result**

```bash
sbatch --time=8:00:00 --job-name=seq-sdft scripts/train_sequential.sh sdft
sbatch --time=8:00:00 --job-name=seq-sft scripts/train_sequential.sh sft
sbatch --time=8:00:00 --job-name=seq-dft scripts/train_sequential.sh dft
```

Task order: Science → Tool Use → Medical (per paper).
After each task, checkpoint is saved for evaluation.

---

## Phase 4: Evaluate Everything

After training completes, evaluate all checkpoints:

```bash
# Single-task models: task-specific eval + benchmarks
sbatch scripts/eval_checkpoint.sh /scratch/anangia/sdft_science_output/... tooluse sdft-science
sbatch scripts/eval_checkpoint.sh /scratch/anangia/dft_tooluse_output/... tooluse dft-tooluse
# ... repeat for all method×task combos

# Sequential CL: evaluate each stage checkpoint on all 3 tasks
# (custom script needed)
```

---

## Phase 5: Generate Figures + Tables

```bash
python analysis/collect_results.py --results_dir /scratch/anangia/ --output results.csv
python analysis/plot_pareto.py --csv results.csv --output figures/pareto.pdf          # Figure 2
python analysis/plot_pass_at_k.py --results_dir results/ --output figures/pass_at_k.pdf  # Figure 3 right
python analysis/plot_sequential.py --sdft_dir ... --sft_dir ... --output figures/seq.pdf  # Figure 4
python analysis/generate_table.py --csv results.csv --output figures/table1.tex       # Table 1
```

---

## Cluster Options

### Vulcan (current)
- L40S 48GB, Slurm, `aip-rgrosse` account
- **Problem:** fairshare tanked (LevelFS=0.07), jobs stuck on Priority
- **Mitigation:** Use <3h jobs for backfill, reduce --mem for eval jobs (48G not 375G)
- Fairshare recovers with 1-week half-life

### Killarney (Alliance cloud)
- L40S, OpenStack VMs (no Slurm, no fairshare queue)
- Need to set up environment from scratch on a VM
- No scheduling competition — run immediately
- Good for: eval jobs, short training runs

### Narval / Cedar / Trillium
- Would need to check if `aip-rgrosse` has allocation there
- Different fairshare pools — might not be tanked
- Trillium has H100s (24h max)

---

## Portable Setup (for any new cluster)

```bash
git clone https://github.com/ayushnangia/Self-Distillation.git
cd Self-Distillation

# Python env
module load python/3.11 cuda/12.6  # adjust per cluster
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# HuggingFace
export HF_HOME=/scratch/$USER/hf_cache  # or wherever
export HF_TOKEN=hf_...  # your token

# Pre-cache datasets (needs internet)
python -c "from src.data import load_tooluse; load_tooluse()"
python -c "from src.data import load_science; load_science()"
python -c "from src.data import load_medical; load_medical()"

# Pre-cache model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

---

## Summary: What's Needed for Advisor

**Minimum viable demo (Phase 1 only):**
- SDFT vs SFT on tool-use (accuracy + pass@k) — just need to run eval on existing SFT models
- Prior capability benchmarks for both — shows the Pareto tradeoff

**Full reproduction (Phases 1-5):**
- 3 methods × 3 tasks + sequential CL + all figures/tables
- Estimated GPU hours: ~80-100h total (training + eval)
