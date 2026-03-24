#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=256000M
#SBATCH --time=1-00:00
#SBATCH --output=logs/seq-%x-%j.out
#SBATCH --error=logs/seq-%x-%j.err

# Usage:
#   sbatch --job-name=seq-sdft scripts/train_sequential.sh sdft [LR] [BS] [EPOCHS] [ALPHA] [SEED]
#   sbatch --job-name=seq-sft  scripts/train_sequential.sh sft
#   sbatch --job-name=seq-dft  scripts/train_sequential.sh dft
#
# Arguments:
#   $1 = method (sdft, sft, dft)
#   $2 = learning rate (optional, default: 1e-5)
#   $3 = batch size (optional, default: 32)
#   $4 = epochs per task (optional, default: 2)
#   $5 = alpha (optional, default: 0.01, SDFT only)
#   $6 = seed (optional, default: 42)
#   $7 = task_order (optional, default: science,tooluse,medical)

METHOD=${1:?Usage: $0 METHOD [LR] [BS] [EPOCHS] [ALPHA] [SEED] [TASK_ORDER]}
LR=${2:-1e-5}
BS=${3:-32}
EPOCHS=${4:-2}
ALPHA=${5:-0.01}
SEED=${6:-42}
TASK_ORDER=${7:-science,tooluse,medical}

cd ~/Self-Distillation
source setup_env.sh --job

OUTPUT_DIR="$SCRATCH/seq_${METHOD}_output/lr${LR}_bs${BS}_ep${EPOCHS}_alpha${ALPHA}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "Sequential CL: method=$METHOD | LR=$LR | BS=$BS | Epochs=$EPOCHS | Alpha=$ALPHA | Seed=$SEED"
echo "Task order: Science -> Tool Use -> Medical"
echo "Output: $OUTPUT_DIR"

# SFT uses ZeRO-2 (needs DeepSpeed for 7B model), SDFT/DFT use ZeRO-3
if [[ "$METHOD" == "sft" ]]; then
    ACCEL_CONFIG=configs/accelerate/sft.yaml
else
    ACCEL_CONFIG=configs/accelerate/sdft.yaml
fi

accelerate launch --config_file "$ACCEL_CONFIG" \
    train_sequential.py \
    --method "$METHOD" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LR" \
    --batch_size "$BS" \
    --num_train_epochs "$EPOCHS" \
    --ref_model_mixup_alpha "$ALPHA" \
    --seed "$SEED" \
    --task_order "$TASK_ORDER"
