"""Pre-cache all HF datasets as local Arrow files for offline compute node use.

Run on a login node (has internet) before submitting training jobs:
    module load gcc/12.3 arrow python/3.11
    source ~/sdft_env/bin/activate
    python scripts/precache_datasets.py

This saves processed datasets (with prompt/teacher_prompt/messages columns)
to data/{task}_data/{train,eval}_processed/ so that data loaders work
offline on compute nodes without downloading from HuggingFace.
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_science, load_medical, load_tooluse


def save_processed(task_name, loader, seed=42):
    """Load dataset via HF, process it, and save as Arrow."""
    data_dir = os.environ.get("SDFT_DATA_DIR", "data")
    out_dir = os.path.join(data_dir, f"{task_name}_data")
    train_path = os.path.join(out_dir, "train_processed")
    eval_path = os.path.join(out_dir, "eval_processed")

    if os.path.isdir(train_path) and os.path.isdir(eval_path):
        print(f"  {task_name}: already cached at {out_dir}, skipping")
        return

    print(f"  {task_name}: downloading and processing...")
    train, val, test = loader(seed)

    os.makedirs(out_dir, exist_ok=True)
    train.save_to_disk(train_path)
    print(f"    train: {len(train)} examples -> {train_path}")

    if test is not None:
        test.save_to_disk(eval_path)
        print(f"    eval:  {len(test)} examples -> {eval_path}")

    if val is not None:
        val_path = os.path.join(out_dir, "val_processed")
        val.save_to_disk(val_path)
        print(f"    val:   {len(val)} examples -> {val_path}")


if __name__ == "__main__":
    print("Pre-caching datasets for offline use on compute nodes...\n")

    print("1/3 Tooluse (local Arrow already exists, verifying...)")
    try:
        train, _, test = load_tooluse(42)
        print(f"  tooluse: OK ({len(train)} train, {len(test)} test)\n")
    except Exception as e:
        print(f"  tooluse: ERROR - {e}\n")

    print("2/3 Science (downloading from HuggingFace)...")
    try:
        save_processed("science", load_science)
        print()
    except Exception as e:
        print(f"  science: ERROR - {e}\n")

    print("3/3 Medical (downloading from HuggingFace)...")
    try:
        save_processed("medical", load_medical)
        print()
    except Exception as e:
        print(f"  medical: ERROR - {e}\n")

    print("Done. Datasets are ready for offline compute node use.")
