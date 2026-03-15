"""Figure 3 right: pass@k curves for tool-use task.

Plots pass@k for different k values, comparing SDFT vs SFT vs DFT vs base.

Usage:
    python analysis/plot_pass_at_k.py --results_dir results/ --output figures/pass_at_k.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {"sdft": "#2196F3", "sft": "#FF9800", "dft": "#4CAF50", "base": "#9E9E9E"}
METHOD_LABELS = {"sdft": "SDFT", "sft": "SFT", "dft": "DFT", "base": "Base"}


def load_pass_at_k(results_file):
    """Load pass@k scores from tooluse results JSON."""
    with open(results_file) as f:
        data = json.load(f)
    pak = data.get("pass_at_k", {})
    return {int(k): v for k, v in pak.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/pass_at_k.pdf")
    args = parser.parse_args()

    results_base = Path(args.results_dir)
    fig, ax = plt.subplots(figsize=(7, 5))

    for run_dir in sorted(results_base.iterdir()):
        if not run_dir.is_dir():
            continue

        tooluse_file = run_dir / "tooluse_results.json"
        if not tooluse_file.exists():
            continue

        name = run_dir.name
        # Determine method from name
        method = "base"
        for m in ["sdft", "sft", "dft"]:
            if m in name.lower():
                method = m
                break

        pak = load_pass_at_k(tooluse_file)
        if not pak:
            continue

        k_values = sorted(pak.keys())
        scores = [pak[k] for k in k_values]

        ax.plot(
            k_values, scores,
            color=METHOD_COLORS.get(method, "gray"),
            marker="o",
            label=f"{METHOD_LABELS.get(method, name)} ({name})",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("k")
    ax.set_ylabel("pass@k (%)")
    ax.set_title("Tool-Use pass@k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
