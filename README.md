# Self-Distillation Fine-Tuning

Reproduction codebase for **"Self-Distillation Enables Continual Learning"** ([arXiv:2601.19897](https://arxiv.org/abs/2601.19897)).

Supports three training methods (SDFT, SFT, DFT) across three tasks (Tool Use, Science Q&A, Medical), with unified evaluation, benchmark tracking, and figure/table generation.

## Setup

```bash
git clone https://github.com/Continual-Intelligence/Self-Distillation.git
cd Self-Distillation
pip install -r requirements.txt
```

## Training

All training goes through `train.py` with `--method` and `--task` flags:

```bash
# SDFT (self-distillation fine-tuning)
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method sdft --task tooluse --output_dir output/sdft_tooluse

# SFT (supervised fine-tuning baseline)
accelerate launch --config_file configs/accelerate/sft.yaml \
    train.py --method sft --task science --output_dir output/sft_science

# DFT (distillation fine-tuning baseline)
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method dft --task medical --output_dir output/dft_medical
```

### Sequential Continual Learning (Figure 4)

Trains on three tasks sequentially (Science -> Tool Use -> Medical):

```bash
python train_sequential.py --method sdft --output_dir output/seq_sdft
python train_sequential.py --method sft  --output_dir output/seq_sft
python train_sequential.py --method dft  --output_dir output/seq_dft
```

## Evaluation

All evaluation goes through `evaluate.py`:

```bash
# Task-specific evaluation
python evaluate.py --model_path output/sdft_tooluse --task tooluse --output_dir results/sdft_tooluse

# Prior capabilities benchmarks (HellaSwag, TruthfulQA, MMLU, Winogrande, HumanEval, IFEval)
python evaluate.py --model_path output/sdft_tooluse --task benchmarks --output_dir results/sdft_tooluse

# All evaluations at once
python evaluate.py --model_path output/sdft_tooluse --task all --output_dir results/sdft_tooluse
```

### Medical Evaluation

Medical uses a two-stage evaluation (generate on GPU, then judge via OpenAI API):

```bash
# Stage 1: Generate responses (GPU)
python evaluate.py --model_path output/sdft_medical --task medical --generate_only --output_dir results/

# Stage 2: GPT judge (CPU, requires OPENAI_API_KEY)
python evaluate.py --model_path output/sdft_medical --task medical --judge_only --output_dir results/
```

## Analysis

Generate figures and tables from evaluation results:

```bash
# Collect all results into CSV
python analysis/collect_results.py --results_dir results --output results.csv

# Figure 2: Pareto frontiers (new task accuracy vs prior capabilities)
python analysis/plot_pareto.py --csv results.csv --output figures/pareto.pdf

# Figure 3 right: pass@k curves
python analysis/plot_pass_at_k.py --results_dir results --output figures/pass_at_k.pdf

# Figure 3 left: scaling across model sizes
python analysis/plot_scaling.py --csv results.csv --output figures/scaling.pdf

# Figure 4: sequential continual learning
python analysis/plot_sequential.py \
    --sdft_dir results/seq_sdft --sft_dir results/seq_sft \
    --output figures/sequential.pdf

# Table 1: full results breakdown (LaTeX)
python analysis/generate_table.py --csv results.csv --output figures/table1.tex
```

## Project Structure

```
Self-Distillation/
├── train.py                  # Unified training: --method {sdft,sft,dft} --task {tooluse,science,medical}
├── train_sequential.py       # Sequential continual learning (Figure 4)
├── evaluate.py               # Unified evaluation: --task {tooluse,science,medical,benchmarks,all}
├── distil_trainer.py         # Core DistilTrainer (upstream, untouched)
├── distil_config.py          # DistilConfig (upstream, untouched)
│
├── src/
│   ├── data.py               # Dataset loaders: load_tooluse(), load_science(), load_medical()
│   ├── eval_tooluse.py       # Tool-use eval: regex matching + pass@k
│   ├── eval_science.py       # Science eval: exact match on answer letter
│   ├── eval_medical.py       # Medical eval: GPT judge (generate then judge)
│   └── eval_benchmarks.py    # lm-eval-harness wrapper (6 benchmarks)
│
├── configs/
│   ├── deepspeed/
│   │   ├── zero3_cpu_offload.json   # For SDFT/DFT (with vLLM colocate)
│   │   └── zero2_cpu_offload.json   # For SFT
│   └── accelerate/
│       ├── sdft.yaml                # Use for SDFT and DFT
│       └── sft.yaml                 # Use for SFT
│
├── analysis/
│   ├── collect_results.py    # Aggregate results into CSV
│   ├── plot_pareto.py        # Figure 2: Pareto frontiers
│   ├── plot_pass_at_k.py     # Figure 3 right: pass@k curves
│   ├── plot_scaling.py       # Figure 3 left: scaling experiment
│   ├── plot_sequential.py    # Figure 4: sequential CL
│   └── generate_table.py     # Table 1: LaTeX output
│
├── data/tooluse_data/        # ToolAlpaca dataset (train + eval)
└── paper/                    # LaTeX source
```

## Methods

| Method | Flag | Description |
|--------|------|-------------|
| **SDFT** | `--method sdft` | On-policy self-distillation with EMA teacher. Student generates rollouts; demonstration-conditioned teacher provides KL target. |
| **SFT** | `--method sft` | Standard supervised fine-tuning on expert demonstrations (TRL SFTTrainer). |
| **DFT** | `--method dft` | Distillation from fixed teacher. Teacher generates rollouts instead of student (`generate_from_teacher=True`). |

## Tasks

| Task | Flag | Dataset | Train / Test | Evaluation |
|------|------|---------|--------------|------------|
| Tool Use | `--task tooluse` | ToolAlpaca | 4046 / 68 | Regex match on API call + pass@k |
| Science Q&A | `--task science` | SciKnowEval Chemistry L3 | ~1800 / ~480 | Exact match on answer letter (A/B/C/D) |
| Medical | `--task medical` | HuatuoGPT-o1 English | 19000 / 500 | GPT judge (requires OpenAI API) |

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | required | Training method: `sdft`, `sft`, `dft` |
| `--task` | required | Task: `tooluse`, `science`, `medical` |
| `--model_name` | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `--output_dir` | required | Output directory for checkpoints |
| `--learning_rate` | `1e-5` | Learning rate |
| `--num_train_epochs` | `1` | Number of training epochs |
| `--batch_size` | `32` | Effective batch size (via gradient accumulation) |
| `--ref_model_mixup_alpha` | `0.01` | EMA rate for teacher updates (SDFT only) |
| `--seed` | `42` | Random seed |
| `--resume` | `False` | Resume from checkpoint |

## Hardware Requirements

All experiments in the paper were conducted on a single NVIDIA H200 GPU. The codebase also supports L40S 48GB GPUs using DeepSpeed ZeRO-3 with CPU offloading (configured in `configs/deepspeed/`).

For HPC/Slurm environments, see the example job scripts in `scripts/`.
