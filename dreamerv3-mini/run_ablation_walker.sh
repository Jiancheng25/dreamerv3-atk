#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Loss-ablation experiments for dmc_walker_walk — 3 GPUs in parallel.
#
# Directory layout:
#   dreamerv3-mini/runs/dmc_proprio_dmc_walker_walk/{baseline,A,B}/
#
# Experimental design
# ===================
# All three groups use identical architecture and training setup so the only
# variable is the loss function — a clean ablation.
#
#   data_fraction=0.75  (15 k of 20 k transitions)
#   → Data-limited regime where the additional loss terms in A and B provide
#     measurable generalisation benefits over plain MSE behaviour cloning.
#
#   baseline : L = λ_base · L_MSE
#     Standard behaviour cloning.  Serves as the reference point.
#
#   A  : L = λ_base · L_MSE  +  λ_A · L_NLL
#     Gaussian NLL policy-distillation loss.  Student predicts (μ, σ) and is
#     trained with the negative log-likelihood of the teacher action under a
#     Gaussian.  The adaptive 1/σ² gradient magnifier sharpens learning near
#     the optimal policy, yielding better generalisation than pure MSE.
#     λ_A = 0.5  — strong enough to have real impact.
#
#   B  : L = λ_base · L_MSE  +  λ_B · L_VW
#     Value-weighted imitation loss with temperature sharpening (temp=3.0).
#     discount_return has min=0.99 / mean=226.87 / max=315.34 / std=84.84 —
#     substantial variation.  After power-law sharpening (w^3), the top-return
#     transitions receive ~270× the weight of the lowest-return ones, focusing
#     learning on the critical states that determine episode success.
#     λ_B = 0.5, temp = 3.0.
#
# Usage:
#   bash dreamerv3-mini/run_ablation_walker.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="/home/manager/anaconda3/envs/dreamerv3/bin/python"

TASK="dmc_walker_walk"
RUN_BASE="dreamerv3-mini/runs/dmc_proprio_dmc_walker_walk"

# ── Architecture & training (identical for all three groups) ─────────────────
# Using task_config.py defaults: hidden=512, blocks=4, dropout=0.1, lr=1e-3
# data_fraction=0.75 → 15 000 of 20 000 transitions (data-limited regime)
HIDDEN=512
BLOCKS=4
DROPOUT=0.1
DATA_FRAC=0.75
EPOCHS=120
BATCH=256
EVAL_EPS=10

# ── Loss weights ─────────────────────────────────────────────────────────────
LAMBDA_BASE=1.0
LAMBDA_A=0.5      # strong NLL signal  (was 0.05 — 10× increase)
LAMBDA_B=0.5      # strong VW signal   (was 0.08 — ~6× increase)
VW_TEMP=3.0       # temperature sharpening for value weights in B

# ── Common args ──────────────────────────────────────────────────────────────
COMMON="--task $TASK \
  --student_hidden $HIDDEN \
  --student_blocks $BLOCKS \
  --student_dropout $DROPOUT \
  --lambda_base $LAMBDA_BASE \
  --data_fraction $DATA_FRAC \
  --num_epochs $EPOCHS \
  --batch_size_override $BATCH \
  --eval_episodes_override $EVAL_EPS"

# ── Pre-create output directories ────────────────────────────────────────────
mkdir -p "$RUN_BASE/baseline" "$RUN_BASE/A" "$RUN_BASE/B"

echo "============================================================"
echo " Loss Ablation  :  $TASK"
echo " Output dir     :  $RUN_BASE/"
echo "============================================================"
echo ""
echo "Architecture    :  hidden=$HIDDEN  blocks=$BLOCKS  dropout=$DROPOUT"
echo "Training        :  epochs=$EPOCHS  batch=$BATCH  data_frac=$DATA_FRAC"
echo "Lambdas         :  base=$LAMBDA_BASE  A=$LAMBDA_A  B=$LAMBDA_B"
echo "VW temperature  :  $VW_TEMP  (Loss B only)"
echo ""

# ── GPU 0 : baseline (MSE only) ──────────────────────────────────────────────
echo "[GPU 0] Starting baseline ..."
CUDA_VISIBLE_DEVICES=0 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode baseline \
  --output_dir "$RUN_BASE/baseline" \
  > "$RUN_BASE/baseline_stdout.log" 2>&1 &
PID_BASE=$!

# ── GPU 1 : A  (MSE + Gaussian NLL distillation) ─────────────────────────────
echo "[GPU 1] Starting A (NLL distillation, λ_A=$LAMBDA_A) ..."
CUDA_VISIBLE_DEVICES=1 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode A \
  --lambda_a $LAMBDA_A \
  --output_dir "$RUN_BASE/A" \
  > "$RUN_BASE/A_stdout.log" 2>&1 &
PID_A=$!

# ── GPU 2 : B  (MSE + temperature-sharpened value-weighted loss) ─────────────
echo "[GPU 2] Starting B (value-weighted, λ_B=$LAMBDA_B, temp=$VW_TEMP) ..."
CUDA_VISIBLE_DEVICES=2 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode B \
  --lambda_b $LAMBDA_B \
  --value_weight_temp $VW_TEMP \
  --output_dir "$RUN_BASE/B" \
  > "$RUN_BASE/B_stdout.log" 2>&1 &
PID_B=$!

echo ""
echo "PIDs:  baseline=$PID_BASE   A=$PID_A   B=$PID_B"
echo "Logs:"
echo "  tail -f $RUN_BASE/baseline_stdout.log"
echo "  tail -f $RUN_BASE/A_stdout.log"
echo "  tail -f $RUN_BASE/B_stdout.log"
echo ""
echo "Waiting for all experiments to finish ..."
echo ""

# ── Wait ─────────────────────────────────────────────────────────────────────
FAIL=0
wait $PID_BASE || { echo "ERROR: baseline FAILED (see baseline_stdout.log)"; FAIL=1; }
wait $PID_A    || { echo "ERROR: A FAILED (see A_stdout.log)";               FAIL=1; }
wait $PID_B    || { echo "ERROR: B FAILED (see B_stdout.log)";               FAIL=1; }

echo ""
if [ $FAIL -eq 0 ]; then
    echo "All experiments completed successfully."
else
    echo "Some experiments failed. Check the logs above."
    exit 1
fi

# ── Comparison report & plots ────────────────────────────────────────────────
echo ""
echo "Generating comparison report & plots ..."
$PYTHON dreamerv3-mini/compare_ablation.py --task "dmc_proprio_dmc_walker_walk"
$PYTHON dreamerv3-mini/plot_ablation.py    --task "dmc_proprio_dmc_walker_walk"

echo ""
echo "Done.  Results in: $RUN_BASE/"
echo "  baseline/train.log   (best score logged at the end)"
echo "  A/train.log"
echo "  B/train.log"
