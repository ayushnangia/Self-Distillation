"""Table 1: Main results table in LaTeX format.

Generates a LaTeX table comparing SDFT, SFT, DFT across tasks and benchmarks.

Usage:
    python analysis/generate_table.py --csv results.csv --output figures/table1.tex
"""

import argparse
from pathlib import Path

import pandas as pd


BENCHMARK_DISPLAY = {
    "hellaswag/acc_norm": "HellaSwag",
    "truthfulqa_mc2/acc": "TruthfulQA",
    "mmlu/acc": "MMLU",
    "winogrande/acc": "Winogrande",
    "humaneval/pass@1": "HumanEval",
    "ifeval/prompt_level_strict_acc": "IFEval",
}

TASK_DISPLAY = {
    "tooluse_accuracy": "Tool Use",
    "science_accuracy": "Science Q&A",
    "medical_accuracy": "Medical",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--output", type=str, default="figures/table1.tex")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Group by method and task, take best result per group
    all_cols = list(TASK_DISPLAY.keys()) + list(BENCHMARK_DISPLAY.keys())
    available_cols = [c for c in all_cols if c in df.columns]

    if not available_cols:
        print("No metric columns found in CSV.")
        return

    # Build table: rows = method × task, columns = metrics
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results across tasks and prior capabilities.}")
    lines.append(r"\label{tab:main}")

    # Header
    metric_headers = []
    for col in available_cols:
        display = TASK_DISPLAY.get(col, BENCHMARK_DISPLAY.get(col, col))
        metric_headers.append(display)

    ncols = 3 + len(metric_headers)  # method, task, epochs, metrics
    lines.append(r"\begin{tabular}{ll" + "c" * (1 + len(metric_headers)) + "}")
    lines.append(r"\toprule")
    header = "Method & Task & Epochs & " + " & ".join(metric_headers) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for method in ["sdft", "sft", "dft"]:
        method_df = df[df["method"] == method]
        for task in ["tooluse", "science", "medical"]:
            task_df = method_df[method_df["task"] == task]
            if task_df.empty:
                continue

            # Take best by task accuracy
            task_acc_col = f"{task}_accuracy"
            if task_acc_col in task_df.columns:
                best = task_df.loc[task_df[task_acc_col].idxmax()]
            else:
                best = task_df.iloc[0]

            epochs = best.get("epochs", "?")
            values = []
            for col in available_cols:
                v = best.get(col)
                if pd.notna(v) and isinstance(v, (int, float)):
                    values.append(f"{v:.1f}")
                else:
                    values.append("--")

            row = f"{method.upper()} & {task.title()} & {epochs} & " + " & ".join(values) + r" \\"
            lines.append(row)
        lines.append(r"\midrule")

    # Remove last midrule, add bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(latex)
    print(latex)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
