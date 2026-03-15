"""Figure 4: Sequential continual learning — 2-panel SDFT vs SFT.

Shows task performance after each sequential training stage.
Left panel: SDFT retains prior task performance.
Right panel: SFT catastrophically forgets.

Usage:
    python analysis/plot_sequential.py --sdft_dir results/seq_sdft --sft_dir results/seq_sft --output figures/sequential.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASK_COLORS = {"science": "#2196F3", "tooluse": "#FF9800", "medical": "#4CAF50"}
TASK_LABELS = {"science": "Science Q&A", "tooluse": "Tool Use", "medical": "Medical"}


def load_sequential_results(eval_dir):
    """Load results from sequential eval directory.

    Expects structure:
        eval_dir/
            after_task1_science/
                science_results.json
                tooluse_results.json
                ...
            after_task2_tooluse/
                ...
            after_task3_medical/
                ...
    """
    eval_dir = Path(eval_dir)
    stages = sorted(eval_dir.iterdir())

    results = []
    for stage_dir in stages:
        if not stage_dir.is_dir():
            continue
        stage_results = {"stage": stage_dir.name}

        # Check for each task's results
        for task in ["science", "tooluse", "medical"]:
            result_file = stage_dir / f"{task}_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                stage_results[task] = data.get("accuracy", data.get("greedy_accuracy"))

        results.append(stage_results)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdft_dir", type=str, required=True)
    parser.add_argument("--sft_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/sequential.pdf")
    args = parser.parse_args()

    sdft_results = load_sequential_results(args.sdft_dir)
    sft_results = load_sequential_results(args.sft_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, results, method_name in zip(axes, [sdft_results, sft_results], ["SDFT", "SFT"]):
        stages = list(range(len(results)))

        for task in ["science", "tooluse", "medical"]:
            values = [r.get(task) for r in results]
            if any(v is not None for v in values):
                ax.plot(
                    stages, values,
                    color=TASK_COLORS[task],
                    marker="o",
                    label=TASK_LABELS[task],
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel("Training Stage")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(method_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if results:
            ax.set_xticks(stages)
            ax.set_xticklabels([r["stage"] for r in results], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
