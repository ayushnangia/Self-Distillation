# CLAUDE.md — Self-Distillation

## Project Overview

TRL-based reproduction of **On-Policy Self-Distillation Fine-Tuning (SDFT)** from [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897). Supports SDFT, SFT, and DFT across three tasks (Tool Use, Science Q&A, Medical), with unified evaluation, benchmark tracking, and figure/table generation for full paper reproduction.

## Project Structure

```
Self-Distillation/
├── train.py                  # Unified: --method {sdft,sft,dft} --task {tooluse,science,medical}
├── train_sequential.py       # Figure 4: sequential continual learning
├── evaluate.py               # Unified: --task {tooluse,science,medical,benchmarks,all}
├── distil_trainer.py         # Core DistilTrainer (upstream, untouched)
├── distil_config.py          # DistilConfig (upstream, untouched)
│
├── src/
│   ├── data.py               # All 3 dataset loaders (load_tooluse, load_science, load_medical)
│   ├── eval_tooluse.py       # Regex matching + pass@k
│   ├── eval_science.py       # Exact match on answer letter
│   ├── eval_medical.py       # GPT judge (2-stage: generate then judge)
│   └── eval_benchmarks.py    # lm-eval-harness wrapper
│
├── configs/
│   ├── deepspeed/            # zero3_cpu_offload.json, zero2_cpu_offload.json
│   └── accelerate/           # sdft.yaml, sft.yaml
│
├── scripts/                  # Slurm job scripts (sweep, sequential, scaling, eval)
├── analysis/                 # collect_results.py, plot_*.py, generate_table.py
├── data/tooluse_data/        # ToolAlpaca dataset (train + eval)
└── paper/                    # LaTeX source
```

## How to Run

### Training

```bash
# SDFT
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method sdft --task tooluse --output_dir output/sdft_tooluse

# SFT baseline
accelerate launch --config_file configs/accelerate/sft.yaml \
    train.py --method sft --task science --output_dir output/sft_science

# DFT baseline
accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py --method dft --task medical --output_dir output/dft_medical
```

### Evaluation

```bash
python evaluate.py --model_path output/sdft_tooluse --task tooluse --output_dir results/
python evaluate.py --model_path output/sdft_tooluse --task benchmarks --output_dir results/
python evaluate.py --model_path output/sdft_tooluse --task all --output_dir results/
```

## Key Technical Notes

- **DFT baseline**: Uses `generate_from_teacher=True` + `sync_ref_model=False` in DistilConfig (line 492)
- **SFT needs DeepSpeed**: 7B model OOMs on L40S 48GB without ZeRO + CPU offload
- **Medical eval**: Two-stage (GPU generate → CPU judge via OpenAI API). Set `OPENAI_API_KEY`.
- **Configs**: Use `configs/accelerate/sdft.yaml` for SDFT/DFT, `configs/accelerate/sft.yaml` for SFT
- Old entry points (`main.py`, `main_science.py`, `main_medical.py`, `train_sft.py`) are deprecated

## Dependencies

Core: `torch`, `transformers`, `trl`, `vllm`, `accelerate`, `deepspeed`, `wandb`, `lm-eval`

## HPC Notes (Vulcan)

- L40S 48GB GPUs, Slurm account `aip-rgrosse`
- Set `HF_HOME` to `$SCRATCH` or `$PROJECT`
- Use `--no-index` pip installs for cluster-optimized wheels
- Pre-cache datasets on login node: `bash scripts/precache_datasets.sh`
