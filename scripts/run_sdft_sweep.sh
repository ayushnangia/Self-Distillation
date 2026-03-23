#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --time=0-03:00
#SBATCH --output=logs/sdft-sweep-%A_%a.out
#SBATCH --error=logs/sdft-sweep-%A_%a.err
#SBATCH --job-name=sdft-sweep
#SBATCH --array=0-53
#SBATCH --exclude=rg31701

# SDFT sweep: 3 LR × 2 epochs × 3 BS × 3 alpha = 54 configs
#
# Usage:
#   sbatch --array=0-53%20 scripts/run_sdft_sweep.sh [seed]

SEED=${1:-42}

# Grid values
LRS=(5e-6 1e-5 5e-5)
EPOCHS_LIST=(1 2)
BSS=(16 32 64)
ALPHAS=(0.01 0.02 0.05)

# Decode SLURM_ARRAY_TASK_ID into grid indices
# Layout: alpha varies fastest, then bs, then epochs, then lr
# index = lr_i * (2*3*3) + ep_i * (3*3) + bs_i * 3 + alpha_i
IDX=$SLURM_ARRAY_TASK_ID
ALPHA_I=$((IDX % 3))
IDX=$((IDX / 3))
BS_I=$((IDX % 3))
IDX=$((IDX / 3))
EP_I=$((IDX % 2))
LR_I=$((IDX / 2))

LR=${LRS[$LR_I]}
EPOCHS=${EPOCHS_LIST[$EP_I]}
BS=${BSS[$BS_I]}
ALPHA=${ALPHAS[$ALPHA_I]}

cd ~/Self-Distillation
source setup_env.sh --job

OUTPUT_DIR="$SCRATCH/sdft_sweep/tooluse/lr${LR}_bs${BS}_ep${EPOCHS}_alpha${ALPHA}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "=== SDFT Sweep ==="
echo "Array task: $SLURM_ARRAY_TASK_ID | LR: $LR | BS: $BS | Epochs: $EPOCHS | Alpha: $ALPHA | Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"

# Auto-resume from latest checkpoint if exists
RESUME_FLAG=""
LATEST_CKPT=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_FLAG="--resume"
fi

accelerate launch --config_file configs/accelerate/sdft.yaml \
    train.py \
    --method sdft \
    --task tooluse \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LR" \
    --batch_size "$BS" \
    --num_train_epochs "$EPOCHS" \
    --ref_model_mixup_alpha "$ALPHA" \
    --seed "$SEED" \
    --save_steps 9999 \
    $RESUME_FLAG

echo ""
echo "=== Job complete. Run 'seff $SLURM_JOB_ID' for resource usage ==="
