#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=0-08:00
#SBATCH --output=logs/eval-all-sft-%j.out
#SBATCH --error=logs/eval-all-sft-%j.err
#SBATCH --job-name=eval-all-sft

module load StdEnv/2023 gcc/12.3 arrow/18.1.0 opencv/4.11.0 python/3.11 cuda/12.6
source ~/sdft_env/bin/activate

export HF_HOME=/scratch/anangia/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$SLURM_TMPDIR/.triton

cd /home/anangia/Self-Distillation

SFT_DIR="/scratch/anangia/sft_output"
RESULTS_BASE="/scratch/anangia/sft_eval_results"

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
results_base = '$RESULTS_BASE'
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
