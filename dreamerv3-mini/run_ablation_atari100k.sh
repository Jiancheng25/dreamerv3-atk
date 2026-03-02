#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Loss-ablation experiments for atari100k_pong — 3 GPUs in parallel.
#
# Directory layout:
#   dreamerv3-mini/runs/atari100k_pong/{baseline,A,B}/
#
# Task specifics
# ==============
# atari100k_pong: image (64x64x12 RGB, 4-frame stack) + discrete (6 actions)
#   → L_base = CrossEntropyLoss (hard label)
#   → L_A    = Soft-label KL distillation (teacher logits stored as act_logits)
#              Temperature T=5; 6 minimal actions, more concentrated teacher.
#   → L_B    = Value-weighted CrossEntropy (temp=1.5 moderate sharpening)
#              discount_return: min=-5.50 / mean=-0.50 / max=2.39 / std=0.96
#              Negative mean: return_mean=abs(mean)=0.50 (see train_student.py fix).
#              Positive-return states (~30%) get upweighted; negative-return
#              states (~70%) get weight→0.01 (effectively downweighted).
#
# data_fraction=0.25 creates a data-limited regime (≈97k transitions).
# Architecture kept small (hidden=256, blocks=2) matching atari100k budget.
#
# Usage:
#   bash dreamerv3-mini/run_ablation_atari100k.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="/home/manager/anaconda3/envs/dreamerv3/bin/python"

TASK="atari100k_pong"
RUN_BASE="dreamerv3-mini/runs/atari100k_pong"

# ── Architecture & training (identical for all groups) ───────────────────────
# task_config defaults: hidden=256, blocks=2, dropout=0.1, lr=3e-4, batch=128
HIDDEN=256
BLOCKS=2
DROPOUT=0.1
DATA_FRAC=0.25
EPOCHS=150
BATCH=128
EVAL_EPS=10

# ── Loss weights ─────────────────────────────────────────────────────────────
LAMBDA_BASE=1.0
LAMBDA_A=0.8
KD_TEMP=5.0       # soften 6-action teacher distribution
LAMBDA_B=0.3
VW_TEMP=1.5       # moderate sharpening; return_mean fixed to abs(-0.50)=0.50

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

# ── GPU 3 : baseline (CrossEntropy only) ─────────────────────────────────────
echo "[GPU 3] Starting baseline ..."
CUDA_VISIBLE_DEVICES=3 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode baseline \
  --output_dir "$RUN_BASE/baseline" \
  > "$RUN_BASE/baseline_stdout.log" 2>&1 &
PID_BASE=$!

# ── GPU 4 : A  (CE + soft-label KL distillation, T=5) ────────────────────────
echo "[GPU 4] Starting A (soft-label KL, λ_A=$LAMBDA_A, T=$KD_TEMP) ..."
CUDA_VISIBLE_DEVICES=4 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode A \
  --lambda_a $LAMBDA_A \
  --kd_temperature $KD_TEMP \
  --output_dir "$RUN_BASE/A" \
  > "$RUN_BASE/A_stdout.log" 2>&1 &
PID_A=$!

# ── GPU 6 : B  (CE + value-weighted CE, temp=1.5) ────────────────────────────
echo "[GPU 6] Starting B (value-weighted CE, λ_B=$LAMBDA_B, vw_temp=$VW_TEMP) ..."
CUDA_VISIBLE_DEVICES=6 $PYTHON dreamerv3-mini/train_student.py \
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
