# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch Purpose

This is the **run-plan** branch — experiment orchestration and execution tracking for reproducing the SDFT paper. The `main` branch has the core code; this branch adds run plans, TODO tracking, and Slurm scripts on top.

Key files unique to this branch:
- `RUN_PLAN.md` — phased reproduction plan (5 phases, cluster guidance, sbatch commands)
- `TODO.md` — task checklist with ready-to-submit sbatch commands
- `scripts/eval_all_sft.sh` — batch eval of all 12 SFT checkpoints

## Commands

### Training

```bash
# SDFT (self-distillation fine-tuning)
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method sdft --task tooluse --output_dir output/sdft_tooluse

# SFT (supervised fine-tuning)
accelerate launch --config_file configs/accelerate/sft.yaml \
    train.py --method sft --task science --output_dir output/sft_science

# DFT (distillation fine-tuning)
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method dft --task medical --output_dir output/dft_medical

# Sequential continual learning (Science → Tool Use → Medical)
python train_sequential.py --method sdft --output_dir output/seq_sdft
```

Training supports `--resume` for checkpoint resumption. Checkpoints auto-save every 100 steps.

### Evaluation

```bash
# Single task
python evaluate.py --model_path output/sdft_tooluse --task tooluse --output_dir results/

# Prior capability benchmarks (HellaSwag, TruthfulQA, MMLU, Winogrande, HumanEval, IFEval)
python evaluate.py --model_path output/sdft_tooluse --task benchmarks --output_dir results/

# All at once
python evaluate.py --model_path output/sdft_tooluse --task all --output_dir results/

# Medical two-stage (generate on GPU, then GPT judge — requires OPENAI_API_KEY)
python evaluate.py --model_path output/sdft_medical --task medical --generate_only --output_dir results/
python evaluate.py --model_path output/sdft_medical --task medical --judge_only --output_dir results/

# Batch eval all SFT checkpoints (Slurm)
sbatch --time=1:00:00 --mem=48G scripts/eval_all_sft.sh
```

### Analysis

```bash
python analysis/collect_results.py --results_dir results --output results.csv
python analysis/plot_pareto.py --csv results.csv --output figures/pareto.pdf
python analysis/generate_table.py --csv results.csv --output figures/table1.tex
```

### Install

```bash
pip install -r requirements.txt
```

### Cluster Setup (Alliance Canada / any Slurm)

```bash
module load python/3.11 cuda/12.6
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/scratch/$USER/hf_cache
```

## Architecture

### Training Flow

`train.py` is the unified entry point. Routes by `--method` (sdft/sft/dft) and `--task` (tooluse/science/medical):

- **SDFT/DFT** → `DistilTrainer` (distil_trainer.py) with `DistilConfig` (distil_config.py)
- **SFT** → TRL's `SFTTrainer`

**DistilTrainer** (extends TRL BaseTrainer):
- Generates completions via vLLM in colocate mode (same GPU as training)
- Computes KL divergence between student and teacher (demonstration-conditioned) logprobs
- EMA teacher sync for SDFT (`sync_ref_model=True`, `ref_model_mixup_alpha=0.01`)
- Fixed teacher for DFT (`sync_ref_model=False`, `generate_from_teacher=True`)

**DistilConfig** extends HuggingFace `TrainingArguments` with vLLM settings, generation params, loss type, distillation sync params.

### Data Format

`src/data.py` provides loaders (`load_tooluse`, `load_science`, `load_medical`) returning datasets with columns:
- `prompt`: user question (chat message format)
- `teacher_prompt`: question + in-context demonstration from reference answer
- `messages`: user + assistant pairs (used by SFT only)

Data sources:
- **tooluse**: `data/tooluse_data/` — Arrow format in `eval_data/` and `train_data/` dirs
- **science**: `data/science_data/` — Arrow format, also available from HuggingFace `hicai-zju/SciKnowEval` Chemistry L-3
- **medical**: HuggingFace `FreedomIntelligence/medical-o1-reasoning-SFT` English (19000 train, 500 test)

### Evaluation Flow

`evaluate.py` dispatches to modules in `src/`:
- `src/eval_tooluse.py` — regex matching on API calls + pass@k
- `src/eval_science.py` — exact match on answer letter (A/B/C/D)
- `src/eval_medical.py` — two-stage: vLLM generation then GPT judge (OpenAI API)
- `src/eval_benchmarks.py` — lm-eval-harness wrapper (6 benchmarks)

There is also a standalone `eval_tooluse.py` at root level (upstream's version using Arrow data via `load_from_disk`).

### DeepSpeed Configs

- `configs/deepspeed/zero3_cpu_offload.json` — ZeRO-3, optimizer CPU offload (for SDFT/DFT)
- `configs/deepspeed/zero2_cpu_offload.json` — ZeRO-2, optimizer CPU offload (for SFT)

Referenced by accelerate configs in `configs/accelerate/`.

## Reproduction Status

See `RUN_PLAN.md` for full details. Current state:
- SDFT Tool Use: trained + evaluated (best 64.7% vs paper 70.6%)
- SFT Tool Use: 12 configs trained, **not yet evaluated**
- Remaining: SDFT/DFT/SFT on Science + Medical, sequential CL, benchmarks

Output paths on cluster: `/scratch/anangia/sdft_output/`, `/scratch/anangia/sft_output/`, `/scratch/anangia/sdft_eval_results/`

## Resource Requirements

- SDFT/DFT: 1 GPU (L40S 48GB+), 375G RAM (ZeRO-3 + vLLM colocate), ~6-8h
- SFT: 1 GPU, 64G RAM (ZeRO-2), ~3-6h
- Eval: 1 GPU, 48-64G RAM, ~30min-1h per checkpoint
- Use `--time=2:59:00` for Slurm backfill-friendly jobs; training supports `--resume`

## Key Defaults

- Model: `Qwen/Qwen2.5-7B-Instruct`
- vLLM colocate mode, bf16, cosine LR schedule, wandb logging
- Gradient accumulation = batch_size (default 32)
- `max_prompt_length=1024`, `max_completion_length=1024`
- `save_steps=100`
- No formal test suite — validation via wandb metrics and eval scripts
