# CLAUDE.md — Self-Distillation

## Project Overview

TRL-based implementation of **On-Policy Self-Distillation Fine-Tuning (SDFT)** from [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897). Trains a student model using its own demonstration-conditioned version as a teacher, reducing catastrophic forgetting while learning new skills.

## Project Structure

```
Self-Distillation/
├── main.py              # Entry point — parses args, loads model/data, runs training
├── distil_trainer.py    # Core DistilTrainer class (extends TRL BaseTrainer)
├── distil_config.py     # DistilConfig class (extends TrainingArguments)
├── requirements.txt     # Dependencies (torch, transformers, trl, vllm, etc.)
└── data/tooluse_data/
    ├── train_data.json  # 4046 tool-use training examples
    └── eval_data.json   # 68 evaluation examples
```

## How to Run

```bash
python main.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./output \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --num_prompts_per_batch 32 \
  --ref_model_mixup_alpha 0.01
```

Key defaults: vLLM colocated mode, bf16, cosine LR schedule, wandb logging, gradient accumulation of 32.

## Key Components

- **DistilTrainer** (`distil_trainer.py`): Generates completions via vLLM, computes KL divergence loss between student and teacher (demonstration-conditioned) log probabilities. Supports forward/reverse KL and Jensen-Shannon divergence.
- **DistilConfig** (`distil_config.py`): Extends HuggingFace `TrainingArguments` with SDFT-specific params (entropy filtering, importance sampling, ref model sync, vLLM settings).
- **main.py**: Loads model + tokenizer, formats dataset with prompt/teacher_prompt columns, instantiates trainer.

## Dependencies

Core: `torch`, `transformers`, `trl`, `vllm`, `accelerate`, `peft`, `deepspeed`, `wandb`

Install with: `pip install -r requirements.txt`

## Testing

No formal test suite. Validation via wandb metrics and eval dataset during training.

## HPC Notes (Vulcan)

- Designed for single GPU (H200 originally; L40S 48GB on Vulcan)
- Set `HF_HOME` to `$SCRATCH` or `$PROJECT` for model caching
- Use `--no-index` pip installs for cluster-optimized wheels where possible
- vLLM requires sufficient GPU memory — adjust `vllm_gpu_memory_utilization` if needed
