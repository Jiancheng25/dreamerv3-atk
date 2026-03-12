#!/bin/bash
# Value Weight Temperature sweep for Loss B on Crafter
# 8 values: 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0

set -e
cd "$(dirname "$0")/.."

PYTHON=python
SCRIPT=dreamerv3-mini/train_student.py
TASK=crafter_reward

TEMPS=(0.1 0.3 0.5 1.0 2.0 3.0 5.0 10.0)

for T in "${TEMPS[@]}"; do
    TAG="B_vwT${T}"
    echo "============================================================"
    echo "  value_weight_temp = ${T}  (run_name = ${TAG})"
    echo "============================================================"
    $PYTHON $SCRIPT \
        --task $TASK \
        --loss_mode B \
        --value_weight_temp $T \
        --run_name "$TAG"
    echo ""
done

echo "===== All value_weight_temp sweep runs complete ====="
