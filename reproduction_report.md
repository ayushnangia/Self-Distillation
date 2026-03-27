# SDFT reproduction report

**Paper:** *Self-Distillation Enables Continual Learning* (arXiv:2601.19897)
**Date:** March 27, 2026
**Base model:** Qwen2.5-7B-Instruct
**Setup:** Single H100 80GB, TRL DistilTrainer, DeepSpeed ZeRO-2 (SFT) / ZeRO-3 (SDFT)

---

## 1. What we ran

We're reproducing two experiments from the paper:

1. **Single-task learning** -- train on one task, measure task accuracy and forgetting on 6 benchmarks.
2. **Sequential continual learning (Figure 4)** -- train on science, then tooluse, then medical, evaluating after each phase.

Hyperparameters: LR=1e-5, batch size=32, 2 epochs/task, EMA alpha=0.01, seed=42. These match what the paper reports in Tables 5-6.

Benchmarks are evaluated with lm-evaluation-harness (HellaSwag, TruthfulQA MC1, MMLU, Winogrande, HumanEval, IFEval). Task evals use our custom scripts (tooluse: regex + pass@k, science: exact match, medical: GPT judge -- not run yet).

---

## 2. Single-task results vs paper

### Science Q&A

|                  | Science | HellaSwag | HumanEval | IFEval | MMLU  | TruthfulQA | Winogrande | Bench Avg |
|------------------|:-------:|:---------:|:---------:|:------:|:-----:|:----------:|:----------:|:---------:|
| Paper Base       |  32.1   |   62.0    |   65.8    |  74.3  | 71.7  |    47.9    |    71.1    |   65.5    |
| Ours Base        |  41.5   |   65.3    |   62.2    |  72.8  | 68.7  |    47.4    |    60.3    |   62.8    |
| Paper SFT        |  66.2   |   55.0    |   54.8    |  35.3  | 64.6  |    36.8    |    73.7    |   53.4    |
| Ours SFT         |  50.6   |   79.4    |   50.6    |  40.9  | 72.0  |    41.4    |    72.2    |   59.4    |
| Paper SDFT       |  70.2   |   60.9    |   68.9    |  66.8  | 70.7  |    46.5    |    73.1    |   64.5    |
| Ours SDFT        |  48.3   |   80.2    |   71.3    |  48.6  | 71.4  |    46.5    |    71.8    |   65.0    |

### Tool use

|                  | Tooluse | HellaSwag | HumanEval | IFEval | MMLU  | TruthfulQA | Winogrande | Bench Avg |
|------------------|:-------:|:---------:|:---------:|:------:|:-----:|:----------:|:----------:|:---------:|
| Paper Base       |  42.9   |   62.0    |   65.8    |  74.3  | 71.7  |    47.9    |    71.1    |   65.5    |
| Ours Base        |  52.9   |   65.3    |   62.2    |  72.8  | 68.7  |    47.4    |    60.3    |   62.8    |
| Paper SFT        |  63.2   |   57.3    |   50.0    |  49.8  | 70.2  |    37.5    |    73.1    |   56.0    |
| Ours SFT         |  63.2   |   79.1    |   55.5    |  39.6  | 70.2  |    40.0    |    73.6    |   59.7    |
| Paper SDFT       |  70.6   |   61.6    |   68.3    |  71.9  | 71.5  |    47.3    |    71.7    |   65.4    |
| Ours SDFT        |  63.2   |   79.9    |   68.9    |  43.3  | 70.9  |    43.6    |    72.9    |   63.2    |

### Medical (incomplete)

Paper reports SDFT 40.2% vs SFT 35.5% vs Base 30.1%. Our SDFT medical is still training (epoch 0.91/2). SFT medical is stuck at checkpoint 950/1138 -- keeps hitting the walltime.

---

## 3. Sequential continual learning

Training order: science -> tooluse -> medical

|                        | Science | Tooluse | HellaSwag | HumanEval | IFEval | MMLU  | TruthfulQA | Winogrande | Bench Avg |
|------------------------|:-------:|:-------:|:---------:|:---------:|:------:|:-----:|:----------:|:----------:|:---------:|
| Base                   |  41.5   |  52.9   |   65.3    |   62.2    |  72.8  | 68.7  |    47.4    |    60.3    |   62.8    |
| SDFT after science     |  48.3   |  58.8   |   80.2    |   71.3    |  48.6  | 71.4  |    46.5    |    71.8    |   65.0    |
| SFT after science      |  50.6   |  50.0   |   79.4    |   50.6    |  40.9  | 72.0  |    41.4    |    72.2    |   59.4    |
| SDFT after tooluse     |  51.0   |  63.2   |   79.9    |   68.9    |  43.3  | 70.9  |    43.6    |    72.9    |   63.2    |
| SFT after tooluse      |  51.7   |  63.2   |   79.1    |   55.5    |  39.6  | 70.2  |    40.0    |    73.6    |   59.7    |
| SFT after medical      |  43.1   |  55.9   |   78.7    |   53.7    |  40.9  | 66.3  |    37.0    |    71.0    |   57.9    |
| SDFT after medical     |   --    |   --    |    --     |    --     |   --   |  --   |     --     |     --     |    --     |

---

## 4. What matches

**SDFT does preserve benchmarks better than SFT.** The paper's main claim holds in our reproduction. After science training, SDFT is at 65.0 bench avg vs SFT's 59.4. After tooluse, 63.2 vs 59.7. SFT keeps sliding with each task (57.9 after medical). The direction is consistent with the paper, though the gap is smaller for us (3-5 points vs their ~10 points).

**Both methods learn tooluse equally well.** Our SFT and SDFT both reach 63.2% on tooluse, which matches the paper's SFT number exactly.

**SFT forgetting compounds over sequential tasks.** After all three tasks, SFT science drops from 50.6% back to 43.1% and tooluse from 63.2% to 55.9%. This matches the "severe interference" the paper describes.

---

## 5. What doesn't match

### SDFT task performance is way below what the paper reports

This is the biggest problem.

| Task     | Paper SDFT | Ours SDFT | Gap    |
|----------|:----------:|:---------:|:------:|
| Science  |   70.2%    |   48.3%   | -21.9  |
| Tooluse  |   70.6%    |   63.2%   |  -7.4  |

The paper's headline result is that SDFT outperforms SFT on the new task while also forgetting less. We don't see this at all. Our SDFT is slightly worse than SFT on science (48.3 vs 50.6) and tied on tooluse. The forgetting part works, but the "learn better" part doesn't.

### Base model numbers are off

| Metric       | Paper | Ours  | Delta  |
|--------------|:-----:|:-----:|:------:|
| Science      | 32.1  | 41.5  |  +9.4  |
| Tooluse      | 42.9  | 52.9  | +10.0  |
| HellaSwag    | 62.0  | 65.3  |  +3.3  |
| HumanEval    | 65.8  | 62.2  |  -3.6  |
| IFEval       | 74.3  | 72.8  |  -1.5  |
| MMLU         | 71.7  | 68.7  |  -3.0  |
| TruthfulQA   | 47.9  | 47.4  |  -0.5  |
| Winogrande   | 71.1  | 60.3  | -10.8  |

Our base model does better on the tasks (+9-10 points) but worse on Winogrande (-10.8). If we're evaluating the same model on the same benchmarks, something in the eval pipeline is different. This is probably where most of the downstream discrepancies come from.

### IFEval collapses for us, not for them

Paper's SDFT keeps IFEval near base (74.3 -> 66.8-71.9). Ours drops off a cliff (72.8 -> 43-49 for SDFT, 39-41 for SFT). I'm not sure what's going on here. We run IFEval with `--apply_chat_template` as a separate lm_eval call, which is what the paper says to do. But something is clearly different.

### HellaSwag goes up after fine-tuning (suspicious)

Our HellaSwag jumps from 65.3 to ~80 after training. The paper's stays flat around 60-62. A 15-point HellaSwag improvement from task-specific fine-tuning makes no sense. This almost certainly means our eval config differs from theirs, probably something about chat template application. If HellaSwag is wrong, the other benchmark numbers are probably affected too.

---

## 6. What I think is going on

The most likely explanation is an eval configuration mismatch. The HellaSwag anomaly is the smoking gun: fine-tuning on science or tooluse data should not improve HellaSwag by 15 points. The paper says they run benchmarks "without chat template" except IFEval, but Qwen2.5-Instruct models behave differently with and without chat templates, and lm-eval-harness has several ways to handle this. A small difference in prompt formatting can swing multiple-choice benchmarks by 10+ points.

The base model divergence (Winogrande off by 10.8, task evals off by ~10) reinforces this. We're almost certainly not running the exact same eval commands.

Training config differences could explain the lower SDFT task performance. The paper doesn't specify all generation parameters for the on-policy rollouts (temperature? top-p? max tokens?), and those matter a lot for distillation quality.

---

## 7. Status

| What                         | Where we are                                         |
|------------------------------|------------------------------------------------------|
| SFT sequential (3 tasks)     | Tasks 1-2 done. Task 3 at step 950/1138, keeps timing out |
| SDFT sequential (3 tasks)    | Tasks 1-2 done. Task 3 training, epoch 0.91/2        |
| Benchmark evals              | Done for all available checkpoints                    |
| Task evals (tooluse, science)| Done for all checkpoints                              |
| Medical GPT judge            | Not run yet, waiting on training                      |

---

## 8. Questions for the authors

1. **Eval commands.** Could you share the exact lm-eval-harness commands? Our base HellaSwag is 65.3% vs your 62.0%, and our Winogrande is 60.3% vs your 71.1%. That's a big gap for the same model on the same benchmark, so something in the eval setup must differ. Specifically: which benchmarks get `--apply_chat_template`? What `--model_args` do you pass? Any `num_fewshot` settings?

2. **Science Q&A eval.** Our base model scores 41.5% vs your 32.1%. How many test examples are in your split? What prompt format do you use? We're using exact match on the answer letter from the Chemistry L-3 subset of SciKnowEval with a 75/5/20 random split.

3. **SDFT task accuracy.** This is the biggest gap -- our SDFT science is 48.3% vs your 70.2%. What vLLM generation parameters do you use during on-policy rollouts (temperature, top-p, max_completion_length)? How many training epochs for single-task experiments? We use `num_loss_tokens_to_skip=3` and `num_generations=1` -- are those consistent with your setup?

4. **IFEval.** Our IFEval drops to 43-49% after SDFT training while yours stays at 67-72%. Are there specific generation parameters for IFEval that we might be missing?

5. **HellaSwag.** Our HellaSwag goes from 65.3 to ~80 after fine-tuning, which seems wrong. Yours stays flat at ~60-62. Does this suggest we have a chat template or formatting difference in our eval?

6. **Code release.** Are the training and eval scripts available or planned for release? That would probably resolve most of these discrepancies faster than anything else.

---

*March 27, 2026. SDFT medical and SFT medical training still in progress.*
