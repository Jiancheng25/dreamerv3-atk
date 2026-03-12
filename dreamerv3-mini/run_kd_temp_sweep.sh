#!/bin/bash
# KD Temperature sweep for Loss A on Crafter
# 8 values: 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0

set -e
cd "$(dirname "$0")/.."

PYTHON=python
SCRIPT=dreamerv3-mini/train_student.py
TASK=crafter_reward

TEMPS=(0.1 0.5 1.0 2.0 3.0 5.0 8.0 15.0)

for T in "${TEMPS[@]}"; do
    TAG="A_kdT${T}"
    echo "============================================================"
    echo "  kd_temperature = ${T}  (run_name = ${TAG})"
    echo "============================================================"
    $PYTHON $SCRIPT \
        --task $TASK \
        --loss_mode A \
        --kd_temperature $T \
        --run_name "$TAG"
    echo ""
done

echo "===== All kd_temperature sweep runs complete ====="

# Collect best scores
echo ""
echo "===== Score Summary ====="
printf "%-15s %s\n" "kd_temperature" "best_score"
printf "%-15s %s\n" "---------------" "----------"
for T in "${TEMPS[@]}"; do
    TAG="A_kdT${T}"
    CKPT="dreamerv3-mini/runs/crafter_reward/${TAG}/best_student.pth"
    if [ -f "$CKPT" ]; then
        SCORE=$($PYTHON -c "
import torch
d = torch.load('$CKPT', map_location='cpu', weights_only=False)
print(f\"{d['score']:.1f}\")
")
        printf "%-15s %s\n" "$T" "$SCORE"
    else
        printf "%-15s %s\n" "$T" "N/A"
    fi
done
