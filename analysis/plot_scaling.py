"""Figure 3 left: scaling experiment — Science accuracy vs model size.

Compares SDFT vs SFT across 3B/7B/14B Qwen models.

Usage:
    python analysis/plot_scaling.py --csv results.csv --output figures/scaling.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_COLORS = {"sdft": "#2196F3", "sft": "#FF9800", "dft": "#4CAF50"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--output", type=str, default="figures/scaling.pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Filter science task results
    science_df = df[df["task"] == "science"].copy()
    if science_df.empty:
        print("No science results found.")
        return

    # Try to extract model size from run_name
    def extract_size(name):
        name = str(name).lower()
        if "3b" in name:
            return 3
        elif "7b" in name:
            return 7
        elif "14b" in name:
            return 14
        return None

    science_df["model_size"] = science_df["run_name"].apply(extract_size)
    science_df = science_df.dropna(subset=["model_size", "science_accuracy"])

    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ["sdft", "sft", "dft"]:
        method_df = science_df[science_df["method"] == method].sort_values("model_size")
        if method_df.empty:
            continue
        ax.plot(
            method_df["model_size"],
            method_df["science_accuracy"],
            color=METHOD_COLORS.get(method, "gray"),
            marker="o",
            label=method.upper(),
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Model Size (B)")
    ax.set_ylabel("Science Accuracy (%)")
    ax.set_title("Scaling: Science Q&A")
    ax.set_xticks([3, 7, 14])
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
