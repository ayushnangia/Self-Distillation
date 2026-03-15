"""Collect all evaluation results into a single CSV.

Scans sweep/eval output directories, extracts metrics from JSON files,
and produces results.csv.

Usage:
    python analysis/collect_results.py --results_dir results/
    python analysis/collect_results.py --results_dir results/ --output results.csv
"""

import argparse
import csv
import json
from pathlib import Path


def extract_tooluse_metrics(results_dir):
    """Extract tool-use greedy accuracy and pass@k from results JSON."""
    tooluse_file = results_dir / "tooluse_results.json"
    if not tooluse_file.exists():
        return {}

    with open(tooluse_file) as f:
        data = json.load(f)

    metrics = {"tooluse_accuracy": data.get("greedy_accuracy", None)}
    for k, v in data.get("pass_at_k", {}).items():
        metrics[f"tooluse_pass@{k}"] = v
    return metrics


def extract_science_metrics(results_dir):
    """Extract science accuracy from results JSON."""
    science_file = results_dir / "science_results.json"
    if not science_file.exists():
        return {}

    with open(science_file) as f:
        data = json.load(f)
    return {"science_accuracy": data.get("accuracy", None)}


def extract_medical_metrics(results_dir):
    """Extract medical accuracy from judge results JSON."""
    judge_file = results_dir / "medical_judge_results.json"
    if not judge_file.exists():
        return {}

    with open(judge_file) as f:
        data = json.load(f)
    return {"medical_accuracy": data.get("accuracy", None)}


def extract_lm_eval_metrics(results_dir):
    """Extract lm-eval-harness benchmark metrics."""
    lm_eval_dir = results_dir / "lm_eval"
    if not lm_eval_dir.exists():
        return {}

    metrics = {}
    for results_file in lm_eval_dir.rglob("results.json"):
        with open(results_file) as f:
            data = json.load(f)
        if "results" not in data:
            continue
        for task_name, task_results in data["results"].items():
            for metric_name, value in task_results.items():
                if isinstance(value, (int, float)) and metric_name.endswith(",none"):
                    clean_name = metric_name.replace(",none", "")
                    metrics[f"{task_name}/{clean_name}"] = value
    return metrics


def parse_run_name(name):
    """Parse method, task, hyperparams from directory name.

    Expected format: {method}_{task}_lr{lr}_bs{bs}_ep{epochs}_seed{seed}
    """
    info = {"run_name": name}
    parts = name.split("_")

    if len(parts) >= 2:
        info["method"] = parts[0]
        info["task"] = parts[1]

    for part in parts:
        if part.startswith("lr"):
            info["lr"] = part[2:]
        elif part.startswith("bs"):
            info["bs"] = part[2:]
        elif part.startswith("ep"):
            info["epochs"] = part[2:]
        elif part.startswith("seed"):
            info["seed"] = part[4:]

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    results_base = Path(args.results_dir)
    all_rows = []

    for run_dir in sorted(results_base.iterdir()):
        if not run_dir.is_dir():
            continue

        run_info = parse_run_name(run_dir.name)
        metrics = {}
        metrics.update(extract_tooluse_metrics(run_dir))
        metrics.update(extract_science_metrics(run_dir))
        metrics.update(extract_medical_metrics(run_dir))
        metrics.update(extract_lm_eval_metrics(run_dir))

        if not metrics:
            continue

        row = {**run_info, **metrics}
        all_rows.append(row)
        print(f"  {run_dir.name}: {len(metrics)} metrics")

    if not all_rows:
        print("No results found.")
        return

    # Collect all column names
    all_keys = []
    for row in all_rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
