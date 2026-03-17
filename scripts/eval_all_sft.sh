#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=0-08:00
#SBATCH --output=logs/eval-all-sft-%j.out
#SBATCH --error=logs/eval-all-sft-%j.err
#SBATCH --job-name=eval-all-sft

cd ~/Self-Distillation
source setup_env.sh --job

SFT_DIR="$SCRATCH/sft_output"
RESULTS_BASE="$SCRATCH/sft_eval_results"

for CKPT_DIR in "$SFT_DIR"/lr*; do
    NAME=$(basename "$CKPT_DIR")
    RESULTS_DIR="$RESULTS_BASE/$NAME"

    if [ -f "$RESULTS_DIR/tooluse_results.json" ]; then
        echo "SKIP: $NAME (already evaluated)"
        continue
    fi

    echo "========================================"
    echo "Evaluating: $NAME"
    echo "Model: $CKPT_DIR"
    echo "Output: $RESULTS_DIR"
    echo "========================================"

    mkdir -p "$RESULTS_DIR"

    python evaluate.py \
        --model_path "$CKPT_DIR" \
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
print(f'{'Config':<35} {'Greedy Acc':>10} {'Pass@1':>8} {'Pass@5':>8} {'Pass@10':>8}')
print('-' * 75)
for d in sorted(glob.glob(os.path.join(results_base, 'lr*'))):
    name = os.path.basename(d)
    f = os.path.join(d, 'tooluse_results.json')
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    ga = r.get('greedy_accuracy', 0)
    pk = r.get('pass_at_k', {})
    print(f'{name:<35} {ga:>9.1f}% {pk.get(\"1\",0):>7.1f}% {pk.get(\"5\",0):>7.1f}% {pk.get(\"10\",0):>7.1f}%')
"
