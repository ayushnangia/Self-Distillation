#!/bin/bash
# Minimal paper replication: SDFT vs SFT on all 3 tasks + Sequential CL
#
# Hyperparams (paper's typical best, no sweep):
#   SDFT: lr=1e-5, bs=32, epochs=2, alpha=0.01
#   SFT:  lr=1e-5, bs=32, epochs=2 (same as SDFT for fair comparison)
#
# Phases:
#   1. train     — Single-task: SDFT & SFT on tooluse, science, medical (6 jobs)
#   2. eval      — Eval all Phase 1 models + base on benchmarks + task accuracy (7 jobs)
#   3. seq       — Sequential CL: science -> tooluse -> medical (2 jobs)
#   4. seq-eval  — Eval sequential checkpoints (6 jobs)
#   5. med-judge — Medical GPT judge (run on LOGIN NODE with OPENAI_API_KEY)
#
# Usage:
#   bash scripts/run_replication.sh          # Phase 1
#   bash scripts/run_replication.sh eval     # Phase 2
#   bash scripts/run_replication.sh seq      # Phase 3
#   bash scripts/run_replication.sh seq-eval # Phase 4
#   bash scripts/run_replication.sh med-judge # Phase 5 (login node only)

set -e
cd ~/Self-Distillation

PHASE=${1:-train}
SEED=42

# --- Hyperparams ---
SDFT_LR=1e-5
SDFT_BS=32
SDFT_EP=2
SDFT_ALPHA=0.01

SFT_LR=1e-5
SFT_BS=32
SFT_EP=2

# --- Path helpers ---
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

sdft_dir() { echo "$SCRATCH/sdft_${1}_output/lr${SDFT_LR}_bs${SDFT_BS}_ep${SDFT_EP}_alpha${SDFT_ALPHA}_seed${SEED}"; }
sft_dir()  { echo "$SCRATCH/sft_${1}_output/lr${SFT_LR}_bs${SFT_BS}_ep${SFT_EP}_alpha0.01_seed${SEED}"; }

SEQ_SDFT_DIR="$SCRATCH/seq_sdft_output/lr${SDFT_LR}_bs${SDFT_BS}_ep${SDFT_EP}_alpha${SDFT_ALPHA}_seed${SEED}"
SEQ_SFT_DIR="$SCRATCH/seq_sft_output/lr${SFT_LR}_bs${SFT_BS}_ep${SFT_EP}_alpha0.01_seed${SEED}"

check_dir() {
    if [ ! -d "$1" ]; then
        echo "  MISSING: $1"
        echo "  Previous phase may not be done. Run 'sq' to check."
        exit 1
    fi
    echo "  OK: $(basename $(dirname $1))/$(basename $1)"
}

# ============================================================
if [[ "$PHASE" == "train" ]]; then
    echo "=== Phase 1: Single-task training (6 jobs) ==="
    mkdir -p logs

    for TASK in tooluse science medical; do
        J=$(sbatch --parsable --job-name=sdft-$TASK \
            scripts/run_single_task.sh sdft $TASK $SDFT_LR $SDFT_BS $SDFT_EP $SDFT_ALPHA $SEED)
        echo "  SDFT $TASK: $J -> $(sdft_dir $TASK)"

        J=$(sbatch --parsable --job-name=sft-$TASK \
            scripts/run_single_task.sh sft $TASK $SFT_LR $SFT_BS $SFT_EP 0.01 $SEED)
        echo "  SFT  $TASK: $J -> $(sft_dir $TASK)"
    done

    echo ""
    echo "Monitor: sq"
    echo "Next:    bash scripts/run_replication.sh eval"

# ============================================================
elif [[ "$PHASE" == "eval" ]]; then
    echo "=== Phase 2: Eval single-task models + base (7 jobs) ==="
    echo "Checking model dirs..."

    for TASK in tooluse science medical; do
        check_dir "$(sdft_dir $TASK)"
        check_dir "$(sft_dir $TASK)"
    done

    echo ""
    echo "Submitting eval jobs..."

    J=$(sbatch --parsable scripts/eval_corrected.sh "$BASE_MODEL" "base")
    echo "  Base:          $J"

    for TASK in tooluse science medical; do
        J=$(sbatch --parsable scripts/eval_corrected.sh "$(sdft_dir $TASK)" "sdft_${TASK}")
        echo "  SDFT $TASK: $J"

        J=$(sbatch --parsable scripts/eval_corrected.sh "$(sft_dir $TASK)" "sft_${TASK}")
        echo "  SFT  $TASK: $J"
    done

    echo ""
    echo "Results: \$SCRATCH/corrected_results/"
    echo "Next:    bash scripts/run_replication.sh seq"

# ============================================================
elif [[ "$PHASE" == "seq" ]]; then
    echo "=== Phase 3: Sequential CL — Science -> Tooluse -> Medical (2 jobs) ==="

    J=$(sbatch --parsable --job-name=seq-sdft \
        scripts/train_sequential.sh sdft $SDFT_LR $SDFT_BS $SDFT_EP $SDFT_ALPHA $SEED)
    echo "  SDFT sequential: $J -> $SEQ_SDFT_DIR"

    J=$(sbatch --parsable --job-name=seq-sft \
        scripts/train_sequential.sh sft $SFT_LR $SFT_BS $SFT_EP 0.01 $SEED)
    echo "  SFT sequential:  $J -> $SEQ_SFT_DIR"

    echo ""
    echo "Next: bash scripts/run_replication.sh seq-eval"

# ============================================================
elif [[ "$PHASE" == "seq-eval" ]]; then
    echo "=== Phase 4: Eval sequential CL checkpoints (up to 6 jobs) ==="

    for DIR in "$SEQ_SDFT_DIR" "$SEQ_SFT_DIR"; do
        METHOD=$(echo "$DIR" | grep -o 'seq_[a-z]*' | sed 's/seq_//')
        echo "--- $METHOD ---"

        for TASK_DIR in "$DIR"/task*; do
            [ -d "$TASK_DIR" ] || continue
            TASK_NAME=$(basename "$TASK_DIR")
            J=$(sbatch --parsable scripts/eval_corrected.sh "$TASK_DIR" "seq_${METHOD}_${TASK_NAME}")
            echo "  $TASK_NAME: $J"
        done
    done

    echo ""
    echo "Results: \$SCRATCH/corrected_results/seq_*"
    echo ""
    echo "For medical GPT judge (login node only):"
    echo "  bash scripts/run_replication.sh med-judge"

# ============================================================
elif [[ "$PHASE" == "med-judge" ]]; then
    echo "=== Phase 5: Medical GPT judge (login node, needs OPENAI_API_KEY) ==="

    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: Set OPENAI_API_KEY first:"
        echo "  export OPENAI_API_KEY=sk-..."
        exit 1
    fi

    module load gcc/12.3 arrow python/3.11 2>/dev/null
    source ~/sdft_env/bin/activate

    RESULTS_BASE="$SCRATCH/corrected_results"

    for NAME in sdft_medical sft_medical seq_sdft_task3_medical seq_sft_task3_medical base; do
        GEN_FILE="$RESULTS_BASE/$NAME/medical_generations.json"
        if [ -f "$GEN_FILE" ]; then
            echo "Judging: $NAME"
            python evaluate.py \
                --model_path dummy \
                --task medical \
                --judge_only \
                --generations_file "$GEN_FILE" \
                --output_dir "$RESULTS_BASE/$NAME"
        else
            echo "  SKIP $NAME — no generations file at $GEN_FILE"
        fi
    done

    echo "Done. Check results in \$SCRATCH/corrected_results/*/medical_judge_results.json"

# ============================================================
else
    echo "Usage: bash scripts/run_replication.sh [train|eval|seq|seq-eval|med-judge]"
    echo ""
    echo "  train     - Phase 1: Single-task SDFT & SFT on tooluse + science + medical"
    echo "  eval      - Phase 2: Eval Phase 1 models + base (7 benchmarks + task acc)"
    echo "  seq       - Phase 3: Sequential CL: science -> tooluse -> medical"
    echo "  seq-eval  - Phase 4: Eval sequential checkpoints"
    echo "  med-judge - Phase 5: Run medical GPT judge (LOGIN NODE, needs OPENAI_API_KEY)"
fi
