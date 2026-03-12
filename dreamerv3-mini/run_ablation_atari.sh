#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Loss-ablation experiments for atari_pong — 3 GPUs in parallel.
#
# Directory layout:
#   dreamerv3-mini/runs/atari_pong/{baseline,A,B}/
#
# Task specifics
# ==============
# atari_pong: image (64x64x4 grayscale, 4-frame stack) + discrete (18 actions)
#   → L_base = CrossEntropyLoss (hard label)
#   → L_A    = Soft-label KL distillation  (teacher logits stored as act_logits)
#              Temperature T=5 softens near-one-hot teacher distribution,
#              exposing action uncertainty across 18 Atari actions.
#   → L_B    = Value-weighted CrossEntropy (temp=2.0 moderate sharpening)
#              discount_return: min=0.73 / mean=3.47 / max=4.75 / std=0.96
#              All positive returns → standard normalisation works correctly.
#              temp=2.0 amplifies high-return (max-score) transitions ~2x over
#              low-return (surviving-only) transitions.
#
# All three groups: identical architecture (hidden=512, blocks=4) and
# training setup — only the loss function differs.
# data_fraction=0.75 creates a data-limited regime where innovations matter.
#
# Usage:
#   bash dreamerv3-mini/run_ablation_atari.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="python"

TASK="atari_pong"
RUN_BASE="dreamerv3-mini/runs/atari_pong"

# ── Architecture & training (identical for all groups) ───────────────────────
# task_config defaults: hidden=512, blocks=4, dropout=0.1, lr=3e-4, batch=128
HIDDEN=512
BLOCKS=4
DROPOUT=0.1
DATA_FRAC=0.75
EPOCHS=100
BATCH=128
EVAL_EPS=15

# ── Loss weights ─────────────────────────────────────────────────────────────
LAMBDA_BASE=1.0
LAMBDA_A=0.8
KD_TEMP=5.0       # soften near-one-hot teacher logits for meaningful soft labels
LAMBDA_B=0.3
VW_TEMP=2.0       # moderate sharpening for well-distributed positive returns

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
echo "Loss A          :  λ_A=$LAMBDA_A  kd_temp=$KD_TEMP"
echo "Loss B          :  λ_B=$LAMBDA_B  vw_temp=$VW_TEMP"
echo ""

# ── GPU 0 : baseline (CrossEntropy only) ─────────────────────────────────────
echo "[GPU 0] Starting baseline ..."
CUDA_VISIBLE_DEVICES=0 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode baseline \
  --output_dir "$RUN_BASE/baseline" \
  > "$RUN_BASE/baseline_stdout.log" 2>&1 &
PID_BASE=$!

# ── GPU 1 : A  (CE + soft-label KL distillation, T=5) ────────────────────────
echo "[GPU 1] Starting A (soft-label KL, λ_A=$LAMBDA_A, T=$KD_TEMP) ..."
CUDA_VISIBLE_DEVICES=1 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode A \
  --lambda_a $LAMBDA_A \
  --kd_temperature $KD_TEMP \
  --output_dir "$RUN_BASE/A" \
  > "$RUN_BASE/A_stdout.log" 2>&1 &
PID_A=$!

# ── GPU 2 : B  (CE + value-weighted CE, temp=2.0) ────────────────────────────
echo "[GPU 2] Starting B (value-weighted CE, λ_B=$LAMBDA_B, vw_temp=$VW_TEMP) ..."
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
wait $PID_BASE || { echo "ERROR: baseline FAILED"; FAIL=1; }
wait $PID_A    || { echo "ERROR: A FAILED";        FAIL=1; }
wait $PID_B    || { echo "ERROR: B FAILED";        FAIL=1; }

echo ""
if [ $FAIL -eq 0 ]; then
    echo "All experiments completed successfully."
else
    echo "Some experiments failed. Check logs."
    exit 1
fi

echo ""
echo "Done. Results in: $RUN_BASE/"
