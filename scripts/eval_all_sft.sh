#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --time=0-02:00
#SBATCH --output=logs/eval-all-sft-%j.out
#SBATCH --error=logs/eval-all-sft-%j.err
#SBATCH --job-name=eval-all-sft

cd ~/Self-Distillation
source setup_env.sh --job

RESULTS_BASE="$SCRATCH/sft_eval_results"
CACHE="$SCRATCH/hf_cache"

# Resolve HF cache dirs to snapshot paths
for MODEL_DIR in "$CACHE"/models--ayushnangia-sdft--qwen2.5-7b-instruct-sft-tooluse-*; do
    [ -d "$MODEL_DIR/snapshots" ] || continue
    SNAPSHOT=$(ls -d "$MODEL_DIR"/snapshots/*/ 2>/dev/null | head -1)
    [ -d "$SNAPSHOT" ] || continue

    # Skip incomplete checkpoints (missing tokenizer or shards)
    if [ ! -f "$SNAPSHOT/tokenizer.json" ]; then
        echo "SKIP: $(basename "$MODEL_DIR") (incomplete — missing tokenizer)"
        continue
    fi

    # Extract short name from cache dir
    NAME=$(basename "$MODEL_DIR" | sed 's/models--ayushnangia-sdft--//')
    RESULTS_DIR="$RESULTS_BASE/$NAME"

    if [ -f "$RESULTS_DIR/tooluse_results.json" ]; then
        echo "SKIP: $NAME (already evaluated)"
        continue
    fi

    echo "========================================"
    echo "Evaluating: $NAME"
    echo "Model path: $SNAPSHOT"
    echo "Output: $RESULTS_DIR"
    echo "========================================"

    mkdir -p "$RESULTS_DIR"

    python evaluate.py \
        --model_path "$SNAPSHOT" \
        --task tooluse \
        --output_dir "$RESULTS_DIR" \
        --n_samples 50 \
        --k_values 1,5,10,20,50

    echo "Done: $NAME"
    echo ""
done

echo "========================================"
echo "ALL SFT EVALUATIONS COMPLETE"
echo "========================================"

# Print summary
python3 -c "
import json, os, glob
results_base = os.environ.get('SCRATCH', '/scratch') + '/sft_eval_results'
print(f'{\"Config\":<55} {\"Greedy Acc\":>10} {\"Pass@1\":>8} {\"Pass@5\":>8} {\"Pass@10\":>8}')
print('-' * 95)
for d in sorted(glob.glob(os.path.join(results_base, '*'))):
    name = os.path.basename(d)
    f = os.path.join(d, 'tooluse_results.json')
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    ga = r.get('greedy_accuracy', 0)
    pk = r.get('pass_at_k', {})
    print(f'{name:<55} {ga:>9.1f}% {pk.get(\"1\",0):>7.1f}% {pk.get(\"5\",0):>7.1f}% {pk.get(\"10\",0):>7.1f}%')
"
