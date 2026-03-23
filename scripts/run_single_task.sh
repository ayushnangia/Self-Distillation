#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --time=0-08:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Usage:
#   sbatch --job-name=sdft-tooluse scripts/run_single_task.sh sdft tooluse
#   sbatch --job-name=sft-science  scripts/run_single_task.sh sft science
#   sbatch --job-name=dft-medical  scripts/run_single_task.sh dft medical
#
# Arguments:
#   $1 = method (sdft, sft, dft)
#   $2 = task (tooluse, science, medical)
#   $3 = learning rate (optional, default: 1e-5)
#   $4 = batch size (optional, default: 32)
#   $5 = epochs (optional, default: 1)
#   $6 = alpha (optional, default: 0.01, SDFT/DFT only)
#   $7 = seed (optional, default: 42)

METHOD=${1:?Usage: $0 METHOD TASK [LR] [BS] [EPOCHS] [ALPHA] [SEED]}
TASK=${2:?Usage: $0 METHOD TASK [LR] [BS] [EPOCHS] [ALPHA] [SEED]}
LR=${3:-1e-5}
BS=${4:-32}
EPOCHS=${5:-1}
ALPHA=${6:-0.01}
SEED=${7:-42}

cd ~/Self-Distillation
source setup_env.sh --job

OUTPUT_DIR="$SCRATCH/${METHOD}_${TASK}_output/lr${LR}_bs${BS}_ep${EPOCHS}_alpha${ALPHA}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "Method: $METHOD | Task: $TASK | LR: $LR | BS: $BS | Epochs: $EPOCHS | Alpha: $ALPHA | Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Data dir: $SDFT_DATA_DIR"

# Auto-resume from latest checkpoint if exists
RESUME_FLAG=""
LATEST_CKPT=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_FLAG="--resume"
fi

if [[ "$METHOD" == "sft" ]]; then
    # SFT uses accelerate + ZeRO-2
    accelerate launch --config_file configs/accelerate/sft.yaml \
        train.py \
        --method sft \
        --task "$TASK" \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate "$LR" \
        --batch_size "$BS" \
        --num_train_epochs "$EPOCHS" \
        --seed "$SEED" \
        $RESUME_FLAG
else
    # SDFT and DFT use accelerate + ZeRO-3
    accelerate launch --config_file configs/accelerate/sdft.yaml \
        train.py \
        --method "$METHOD" \
        --task "$TASK" \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate "$LR" \
        --batch_size "$BS" \
        --num_train_epochs "$EPOCHS" \
        --ref_model_mixup_alpha "$ALPHA" \
        --seed "$SEED" \
        $RESUME_FLAG
fi

# Print resource usage
echo ""
echo "=== Job complete. Run 'seff $SLURM_JOB_ID' for resource usage ==="
