# SDFT Paper Replication — Progress Report

**Paper:** "Self-Distillation Enables Continual Learning" (arXiv:2601.19897)
**Base Model:** Qwen/Qwen2.5-7B-Instruct
**Hardware:** L40S 48GB (Vulcan, Alliance Canada)
**Date:** 2026-03-16

---

## 1. What We've Done

Rebuilt the paper's codebase from scratch — the open-source repo only had the algorithm + one dataset (tool-use), no evaluation scripts, no baselines, no multi-task setup.

**Built:**
- Unified training script supporting all 3 methods (SDFT, SFT, DFT) and all 3 tasks (Tool Use, Science Q&A, Medical)
- Evaluation pipeline: task-specific evals + 6 prior-capability benchmarks
- Sequential continual learning pipeline (Figure 4)
- Analysis/plotting scripts for all paper figures and tables

**Trained so far:**
- SDFT on Tool Use — full run, 11 checkpoints evaluated
- SFT on Tool Use — 12 configs trained (3 LRs x 2 BSs x 2 epochs), **not yet evaluated**
- SDFT on Science Q&A — submitted, pending in queue

---

## 2. Our Results — SDFT on Tool Use

*All numbers in this section are from our reproduction.*

### Greedy Accuracy Across Training

| Checkpoint | Accuracy |
|------------|----------|
| Base (no training) | 54.4% |
| Step-100 | 55.9% |
| Step-200 | 48.5% |
| Step-300 | 44.1% |
| Step-400 | 47.1% |
| Step-500 | 57.4% |
| Step-600 | 47.1% |
| Step-700 | 54.4% |
| Step-800 | 52.9% |
| Step-900 | 57.4% |
| **Step-1000** | **64.7%** |
| Step-1011 | 57.4% |

Best checkpoint: **step-1000 at 64.7%** (+10.3pp over base).

### Our Pass@K Results (Best Checkpoint vs Base)

| K | Base | SDFT (Step-1000) | Improvement |
|---|------|------------------|-------------|
| 1 | 52.6% | 56.2% | +3.6pp |
| 5 | 61.5% | 70.1% | +8.5pp |
| 10 | 64.4% | 74.4% | +10.0pp |
| 20 | 67.3% | 77.5% | +10.2pp |
| 50 | 70.6% | 79.4% | +8.8pp |

SDFT consistently improves over base across all K values. The improvement is largest at K=10-20.

### Our SFT Training (Tool Use) — Trained, Not Yet Evaluated

12 SFT configs completed training (checkpoints saved). Evaluation on tool-use test set is pending.

| LR | BS | Epochs | Training Status |
|----|-----|--------|----------------|
| 5e-6 | 16, 32 | 1, 2 | Trained |
| 1e-5 | 16, 32 | 1, 2 | Trained |
| 5e-5 | 16, 32 | 1, 2 | Trained |

Note: 2-epoch bs=32 runs timed out at ~98% completion, but epoch-1 checkpoints were saved.

---

## 3. Paper's Reported Numbers

*All numbers in this section are from the published paper, not our reproduction.*

### Tool Use Task — Method Comparison (Paper Table 1)

| Method | Tool Use Acc | Prior Capabilities Avg |
|--------|-------------|----------------------|
| Base | 42.9% | 65.5% |
| SFT | 63.2% | 56.0% |
| DFT | 64.2% | 60.8% |
| SFT + re-invoke | 63.1% | 63.7% |
| **SDFT** | **70.6%** | **65.4%** |

The key claim: SDFT gets the highest task accuracy (70.6%) while preserving prior capabilities (65.4%, nearly matching base's 65.5%). SFT trades prior capabilities heavily (56.0%) for modest task gains (63.2%).

### Prior Capabilities Breakdown (Paper Table 1)

| Method | HellaSwag | HumanEval | IFEval | MMLU | TruthfulQA | Winogrande |
|--------|-----------|-----------|--------|------|------------|------------|
| Base | 62.0 | 65.8 | 74.3 | 71.7 | 47.9 | 71.1 |
| SFT | 57.3 | 50.0 | 49.8 | 70.2 | 37.5 | 73.1 |
| SDFT | 61.6 | 68.3 | 71.9 | 71.5 | 47.3 | 71.7 |

SFT tanks IFEval (74.3 -> 49.8) and HumanEval (65.8 -> 50.0). SDFT preserves all of them.

---

## 4. Our Results vs Paper

| Metric | Ours | Paper | Notes |
|--------|------|-------|-------|
| Base accuracy | 54.4% | 42.9% | Ours is 11.5pp higher |
| Best SDFT accuracy | 64.7% | 70.6% | Gap of 5.9pp |
| SDFT improvement over base | +10.3pp | +27.7pp | Paper shows larger gain |
| SFT accuracy | **not yet evaluated** | 63.2% | Our SFT is trained but unevaluated |
| DFT accuracy | **not yet trained** | 64.2% | — |
| Prior capabilities | **not yet run** | 65.4% (SDFT) | Benchmarks not run yet |

### Why the gap?

1. **Higher baseline.** Our base model scores 54.4% vs paper's 42.9%. Different model checkpoint or eval split could explain this. A higher baseline may mean less room to improve.

2. **Hyperparameters.** We ran one config (lr=1e-5, bs=32, 2 epochs, alpha=0.01). The paper sweeps over 54 configs (3 LRs x 3 BSs x 2 epochs x 3 EMA alphas). Their best likely comes from a different HP combo.

3. **Eval protocol.** Regex matching for tool-use calls may differ in argument ordering tolerance.

---

## 5. What's Missing

| Experiment | Status | Priority |
|-----------|--------|----------|
| **Eval SFT checkpoints on tool-use** | Trained, needs eval | High |
| SDFT Science Q&A | Pending in queue | High |
| SDFT Medical | Not submitted | High |
| DFT all 3 tasks | Not submitted | High |
| SFT Science + Medical | Not submitted | High |
| Sequential CL (Fig 4) | Not submitted | High |
| Prior capability benchmarks (6 benchmarks) | Not run | High |
| HP sweep (full grid) | Not started | Medium |
| Scaling (3B/14B) | Not started | Low |
| Knowledge acquisition (Table 2) | Not started | Low |
| Reasoning model (Table 3) | Not started | Low |
| All ablations | Not started | Low |

---

## 6. Immediate Next Steps

1. **Evaluate SFT checkpoints on tool-use** — we have 12 trained models sitting unevaluated, this gives us the SDFT vs SFT comparison immediately
2. **Run SDFT on Science + Medical** — confirms the method generalizes
3. **Run DFT baselines** — completes the 3-method comparison
4. **Run prior capability benchmarks** — the other axis of the Pareto plot
5. **Sequential CL** — the most visually compelling result (Figure 4)

Once we have 3 methods x 3 tasks + benchmarks, we can generate the Pareto plot (Figure 2) and Table 1.

---

## 7. File Locations

| What | Where |
|------|-------|
| SDFT checkpoints | `/scratch/anangia/sdft_output/checkpoint-{100..1011}/` |
| SDFT eval results | `/scratch/anangia/sdft_eval_results/{base,step-100,...}/` |
| SFT checkpoints (unevaluated) | `/scratch/anangia/sft_output/lr*_bs*_ep*_seed42/` |
| Training logs | `/home/anangia/Self-Distillation/logs/` |
| Codebase | `/home/anangia/Self-Distillation/` |
| Paper source | `/home/anangia/Self-Distillation/paper/arxiv.tex` |
