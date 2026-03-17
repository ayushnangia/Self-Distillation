# SDFT Paper Reproduction — TODO

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

---

## Infrastructure (done)

- [x] Unified `train.py` — `--method {sdft,sft,dft} --task {tooluse,science,medical}`
- [x] Unified `evaluate.py` — `--task {tooluse,science,medical,benchmarks,all}`
- [x] Shared dataset loaders (`src/data.py`)
- [x] Eval modules (`src/eval_tooluse.py`, `eval_science.py`, `eval_medical.py`, `eval_benchmarks.py`)
- [x] Configs moved to `configs/deepspeed/` and `configs/accelerate/`
- [x] Slurm scripts (`scripts/sweep.sh`, `train_sequential.sh`, `train_scaling.sh`, `eval_checkpoint.sh`, `eval_sweep.sh`, `precache_datasets.sh`)
- [x] Analysis scripts (`analysis/collect_results.py`, `plot_pareto.py`, `plot_pass_at_k.py`, `plot_scaling.py`, `plot_sequential.py`, `generate_table.py`)
- [x] Refactored `train_sequential.py` to use `src/data.py`
- [x] Updated README.md, CLAUDE.md, .gitignore
- [x] Deprecated old entry points (`main.py`, `main_science.py`, `main_medical.py`, `train_sft.py`)

---

## Completed Training

- [x] SDFT Tool Use — checkpoints 100–1011 at `$SCRATCH/sdft_output/`
- [x] SDFT Tool Use evals — `$SCRATCH/sdft_eval_results/{base,step-100,...,step-1011}/`
- [x] SFT Tool Use (1-epoch, 3 LRs) — completed: sft-5e-6-32-1, sft-1e-5-32-1, sft-5e-5-32-1
- [x] SFT Tool Use (2-epoch, 3 LRs) — timed out at 98%, epoch 1 checkpoints saved

---

## Currently Running

| Job ID | Name | Status |
|--------|------|--------|
| 4345993 | sdft-science | Pending (Priority) |

---

## Queued — Submit After sdft-science Completes

Submit these one at a time (375G RAM → hard to schedule multiple):

```bash
# Single-task training (8h each)
sbatch --time=8:00:00 --job-name=sdft-medical scripts/run_single_task.sh sdft medical
sbatch --time=8:00:00 --job-name=dft-tooluse scripts/run_single_task.sh dft tooluse
sbatch --time=8:00:00 --job-name=dft-science scripts/run_single_task.sh dft science
sbatch --time=8:00:00 --job-name=dft-medical scripts/run_single_task.sh dft medical
sbatch --time=8:00:00 --job-name=sft-science scripts/run_single_task.sh sft science
sbatch --time=8:00:00 --job-name=sft-medical scripts/run_single_task.sh sft medical

# Sequential CL (8h each, may need resume)
sbatch --time=8:00:00 --job-name=seq-sdft scripts/train_sequential.sh sdft
sbatch --time=8:00:00 --job-name=seq-sft scripts/train_sequential.sh sft
sbatch --time=8:00:00 --job-name=seq-dft scripts/train_sequential.sh dft
```

---

## After All Training — Evaluate

- [ ] Eval all single-task checkpoints: task-specific + 6 benchmarks
- [ ] Eval sequential CL checkpoints: each task after each stage
- [ ] Medical eval needs OpenAI API key for GPT judge (stage 2)
- [ ] Generate figures + Table 1

---

## Full Sweep (later, not needed for advisor demo)

- [ ] Expand to full HP grid: 3 LRs × 3 BSs × 2 epochs × 3 EMA alphas × 3 seeds
- [ ] Re-invoke baseline (SFT + on-policy distillation to restore capabilities)
- [ ] Knowledge Acquisition task (Wikipedia 2025 disasters, CPT, RAG — Table 2)
- [ ] Reasoning model experiment (Olmo-3-7B-Think — Table 3)
- [ ] Scaling: 3B/14B Qwen on Science Q&A (Figure 3 left)
- [ ] EMA alpha ablation
- [ ] On-policy vs offline distillation ablation
- [ ] KL gradient estimator ablation
- [ ] Knowledge acquisition context ablation
