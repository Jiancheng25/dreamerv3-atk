#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Loss-ablation experiments for procgen_coinrun — 3 GPUs in parallel.
#
# Directory layout:
#   dreamerv3-mini/runs/procgen_coinrun/{baseline,A,B}/
#
# Task specifics
# ==============
# procgen_coinrun: image (64x64x3 RGB, single frame) + discrete (15 actions)
#   obs: 29,152 transitions | return: min=0 mean=8.70 max=10 std=1.67
#   → L_base = CrossEntropyLoss (hard label)
#
#   → L_A = Soft-label KL distillation
#       Problem v1: T=5 → teacher peak≈0.9997 (near one-hot), KD dominated 85%
#       Fix v2:     T=10, λ_A=0.2 → meaningful soft uncertainty at decision
#                   points (gap≈8 gives peak≈0.88), KD ≈15% of total loss
#
#   → L_B = Value-weighted CrossEntropy
#       Problem v1: vw_temp=1.5 → weights in [0.72,1.23], 5:4 ratio (useless)
#                   because 83.5% of transitions have return>8.0 (narrow spread)
#       Fix v2:     vw_temp=5.0 → weights in [0.35,2.0] at [p5-p95], 5.7:1 ratio
#                   Strongly emphasizes critical states near coin collection.
#                   λ_B=0.5 for sufficient signal.
#
# Additional fixes:
#   - data_fraction=1.0 (29k transitions already small, don't subsample)
#   - eval_episodes=30  (CoinRun is Bernoulli 0/10; 10 eps gave ±4.6 noise,
#                        30 eps gives ±2.7, enough to detect ~3pt differences)
#   - epochs=150        (more training time for convergence)
#
# Usage:
#   bash dreamerv3-mini/run_ablation_procgen.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="/home/manager/anaconda3/envs/dreamerv3/bin/python"

TASK="procgen_coinrun"
RUN_BASE="dreamerv3-mini/runs/procgen_coinrun"

# ── Architecture & training (identical for all groups) ───────────────────────
HIDDEN=256
BLOCKS=3
DROPOUT=0.1
DATA_FRAC=1.0      # use all 29k transitions (dataset already small)
EPOCHS=150
BATCH=128
EVAL_EPS=30        # Bernoulli 0/10 per ep; 30 eps → ±2.7 noise vs ±4.6 at 10

# ── Loss weights ─────────────────────────────────────────────────────────────
LAMBDA_BASE=1.0

# A: T=10 gives peak≈0.88 for gap≈8 logits (vs 0.9997 at T=5) — meaningful
#    soft distribution.  λ_A=0.2 → KD ≈ 15% of total loss, won't dominate CE.
LAMBDA_A=0.2
KD_TEMP=10.0

# B: vw_temp=5.0 → [p5=7.1→w=0.35, p95=9.9→w=1.95], 5.6:1 ratio —
#    strongly emphasises critical states near coin.  λ_B=0.5 for clear signal.
LAMBDA_B=0.5
VW_TEMP=5.0

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
echo " Loss Ablation  :  $TASK  (v2 — optimised settings)"
echo " Output dir     :  $RUN_BASE/"
echo "============================================================"
echo ""
echo "Architecture    :  hidden=$HIDDEN  blocks=$BLOCKS  dropout=$DROPOUT"
echo "Training        :  epochs=$EPOCHS  batch=$BATCH  data_frac=$DATA_FRAC"
echo "Eval episodes   :  $EVAL_EPS  (reduced noise: ±2.7 vs ±4.6 at 10 eps)"
echo "Loss A          :  λ_A=$LAMBDA_A  kd_temp=$KD_TEMP  (peak≈0.88 at gap=8)"
echo "Loss B          :  λ_B=$LAMBDA_B  vw_temp=$VW_TEMP  (5.6:1 ratio [p5→p95])"
echo ""

# ── GPU 3 : baseline (CrossEntropy only) ─────────────────────────────────────
echo "[GPU 3] Starting baseline ..."
CUDA_VISIBLE_DEVICES=3 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode baseline \
  --output_dir "$RUN_BASE/baseline" \
  > "$RUN_BASE/baseline_stdout.log" 2>&1 &
PID_BASE=$!

# ── GPU 4 : A  (CE + soft-label KL, T=10, λ_A=0.2) ──────────────────────────
echo "[GPU 4] Starting A (soft-label KL, λ_A=$LAMBDA_A, T=$KD_TEMP) ..."
CUDA_VISIBLE_DEVICES=4 $PYTHON dreamerv3-mini/train_student.py \
  $COMMON \
  --loss_mode A \
  --lambda_a $LAMBDA_A \
  --kd_temperature $KD_TEMP \
  --output_dir "$RUN_BASE/A" \
  > "$RUN_BASE/A_stdout.log" 2>&1 &
PID_A=$!

# ── GPU 6 : B  (CE + value-weighted CE, vw_temp=5.0, λ_B=0.5) ───────────────
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
