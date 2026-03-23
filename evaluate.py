"""Unified evaluation entry point.

Usage:
    python evaluate.py --model_path /path/to/ckpt --task tooluse --output_dir results/
    python evaluate.py --model_path /path/to/ckpt --task science --output_dir results/
    python evaluate.py --model_path /path/to/ckpt --task medical --output_dir results/
    python evaluate.py --model_path /path/to/ckpt --task medical --judge_only --generations_file results/medical_generations.json
    python evaluate.py --model_path /path/to/ckpt --task benchmarks --output_dir results/
    python evaluate.py --model_path /path/to/ckpt --task all --output_dir results/
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=["tooluse", "science", "medical", "benchmarks", "all"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Tool-use specific
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--k_values", type=str, default="1,5,10")
    parser.add_argument("--skip_pass_at_k", action="store_true")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Override tooluse eval data path (JSON or Arrow dir)")

    # Medical specific
    parser.add_argument("--generate_only", action="store_true",
                        help="Medical: only generate responses, skip judge")
    parser.add_argument("--judge_only", action="store_true",
                        help="Medical: only run judge on existing generations")
    parser.add_argument("--generations_file", type=str, default=None,
                        help="Medical: path to existing generations JSON for --judge_only")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")

    # Benchmark specific
    parser.add_argument("--benchmark_tasks", type=str, default=None,
                        help="Override lm-eval tasks (comma-separated)")

    return parser.parse_args()


def run_tooluse(args):
    from src.eval_tooluse import run_eval
    k_values = tuple(int(k) for k in args.k_values.split(","))
    output_file = str(Path(args.output_dir) / "tooluse_results.json")
    kwargs = dict(
        model_path=args.model_path,
        output_file=output_file,
        gpu_memory_utilization=args.gpu_memory_utilization,
        n_samples=args.n_samples,
        k_values=k_values,
        skip_pass_at_k=args.skip_pass_at_k,
    )
    if args.eval_data:
        kwargs["eval_data_path"] = args.eval_data
    return run_eval(**kwargs)


def run_science(args):
    from src.eval_science import eval_science
    output_file = str(Path(args.output_dir) / "science_results.json")
    return eval_science(
        model_path=args.model_path,
        seed=args.seed,
        output_file=output_file,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def run_medical(args):
    from src.eval_medical import generate_responses, judge_responses
    output_dir = Path(args.output_dir)

    if args.judge_only:
        gen_file = args.generations_file or str(output_dir / "medical_generations.json")
        judge_file = str(output_dir / "medical_judge_results.json")
        return judge_responses(gen_file, output_file=judge_file, judge_model=args.judge_model)

    gen_file = str(output_dir / "medical_generations.json")
    generate_responses(
        model_path=args.model_path,
        seed=args.seed,
        output_file=gen_file,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.generate_only:
        return None

    judge_file = str(output_dir / "medical_judge_results.json")
    return judge_responses(gen_file, output_file=judge_file, judge_model=args.judge_model)


def run_benchmarks(args):
    from src.eval_benchmarks import run_lm_eval
    output_dir = str(Path(args.output_dir) / "lm_eval")
    return run_lm_eval(
        model_path=args.model_path,
        output_dir=output_dir,
        tasks=args.benchmark_tasks,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Evaluating: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Output: {args.output_dir}")

    if args.task == "all":
        for task in ["tooluse", "science", "medical", "benchmarks"]:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}")
            args_copy = argparse.Namespace(**vars(args))
            args_copy.task = task
            if task == "tooluse":
                run_tooluse(args_copy)
            elif task == "science":
                run_science(args_copy)
            elif task == "medical":
                run_medical(args_copy)
            elif task == "benchmarks":
                run_benchmarks(args_copy)
    elif args.task == "tooluse":
        run_tooluse(args)
    elif args.task == "science":
        run_science(args)
    elif args.task == "medical":
        run_medical(args)
    elif args.task == "benchmarks":
        run_benchmarks(args)
