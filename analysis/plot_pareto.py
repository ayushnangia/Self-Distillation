"""Figure 2: Pareto frontier — new task accuracy vs prior capabilities average.

3-panel plot (one per task: tool-use, science, medical).
X-axis: average of 6 benchmark scores (prior capabilities).
Y-axis: new task accuracy.
Points colored by method (SDFT, SFT, DFT).

Usage:
    python analysis/plot_pareto.py --csv results.csv --output figures/pareto.pdf
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


BENCHMARK_COLS = [
    "hellaswag/acc_norm",
    "truthfulqa_mc2/acc",
    "mmlu/acc",
    "winogrande/acc",
    "humaneval/pass@1",
    "ifeval/prompt_level_strict_acc",
]

TASK_ACC_COL = {
    "tooluse": "tooluse_accuracy",
    "science": "science_accuracy",
    "medical": "medical_accuracy",
}

METHOD_COLORS = {"sdft": "#2196F3", "sft": "#FF9800", "dft": "#4CAF50"}
METHOD_MARKERS = {"sdft": "o", "sft": "s", "dft": "^"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--output", type=str, default="figures/pareto.pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Compute prior capabilities average
    available_benchmarks = [c for c in BENCHMARK_COLS if c in df.columns]
    if available_benchmarks:
        df["prior_avg"] = df[available_benchmarks].mean(axis=1) * 100
    else:
        print("Warning: no benchmark columns found, using placeholder")
        df["prior_avg"] = 50.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tasks = ["tooluse", "science", "medical"]

    for ax, task in zip(axes, tasks):
        acc_col = TASK_ACC_COL[task]
        task_df = df[df["task"] == task].dropna(subset=[acc_col])

        for method in ["sdft", "sft", "dft"]:
            method_df = task_df[task_df["method"] == method]
            if method_df.empty:
                continue
            ax.scatter(
                method_df["prior_avg"],
                method_df[acc_col],
                c=METHOD_COLORS.get(method, "gray"),
                marker=METHOD_MARKERS.get(method, "o"),
                label=method.upper(),
                s=80,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

        ax.set_xlabel("Prior Capabilities Avg (%)")
        ax.set_ylabel(f"{task.title()} Accuracy (%)")
        ax.set_title(task.replace("tooluse", "Tool Use").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


from pathlib import Path

if __name__ == "__main__":
    main()
