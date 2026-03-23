"""Wrapper for lm-evaluation-harness benchmarks.

Runs: HellaSwag, TruthfulQA (mc1), MMLU, Winogrande, HumanEval, IFEval

Note: IFEval requires --apply_chat_template; the other benchmarks do NOT.
This matches the paper's evaluation protocol for Qwen2.5-7B-Instruct.
"""

import json
import subprocess
from pathlib import Path


# Benchmarks that run WITHOUT chat template
NO_CHAT_TASKS = "hellaswag,truthfulqa_mc1,mmlu,winogrande,humaneval"
# Benchmarks that require chat template
CHAT_TASKS = "ifeval"


def _run_lm_eval_cmd(model_args, tasks, output_dir, batch_size="auto",
                      apply_chat_template=False):
    """Run a single lm_eval CLI invocation."""
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", str(batch_size),
        "--output_path", output_dir,
        "--log_samples",
    ]
    if apply_chat_template:
        cmd.append("--apply_chat_template")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"lm_eval exited with code {result.returncode}")
    return result.returncode


def run_lm_eval(model_path, output_dir, tasks=None, gpu_memory_utilization=0.9,
                max_model_len=2048, batch_size="auto"):
    """Run lm-evaluation-harness via CLI.

    Splits benchmarks into two groups:
    - HellaSwag, TruthfulQA mc1, MMLU, Winogrande, HumanEval: no chat template
    - IFEval: with chat template

    Args:
        model_path: Path to model checkpoint or HF model name.
        output_dir: Directory to save results.
        tasks: Comma-separated task string. If None, runs all 6 benchmarks
               with correct chat template settings.
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: Max model sequence length.
        batch_size: Batch size for lm-eval.

    Returns:
        Path to output directory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_args = (
        f"pretrained={model_path},"
        f"gpu_memory_utilization={gpu_memory_utilization},"
        f"max_model_len={max_model_len},"
        f"trust_remote_code=True"
    )

    if tasks is not None:
        # Custom task list — run as-is (caller controls chat template)
        _run_lm_eval_cmd(model_args, tasks, output_dir, batch_size)
    else:
        # Default: run both groups with correct settings
        print("\n--- Benchmarks (no chat template) ---")
        _run_lm_eval_cmd(model_args, NO_CHAT_TASKS, output_dir, batch_size,
                         apply_chat_template=False)

        print("\n--- IFEval (with chat template) ---")
        _run_lm_eval_cmd(model_args, CHAT_TASKS, output_dir, batch_size,
                         apply_chat_template=True)

    print(f"Results saved to {output_dir}")
    return output_dir


def collect_lm_eval_results(output_dir):
    """Parse lm-eval results directory and return metrics dict."""
    results = {}
    results_dir = Path(output_dir)

    # lm-eval saves results in a nested structure
    for results_file in results_dir.rglob("results.json"):
        with open(results_file) as f:
            data = json.load(f)
        if "results" in data:
            for task_name, task_results in data["results"].items():
                # Extract the primary metric for each task
                for metric_name, value in task_results.items():
                    if metric_name.endswith(",none"):
                        clean_name = metric_name.replace(",none", "")
                        results[f"{task_name}/{clean_name}"] = value

    return results
