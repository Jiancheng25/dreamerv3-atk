#!/bin/bash
# Architecture ablation: Crafter-DreamerV3 166M
# Compare encoder architectures: ResNet / VGG / Transformer
# ("our" CNN+ResMLP already run as crafter_reward/AB_vwT1.6)
#
# All use AB loss with: kd_temp=5, vw_temp=1.6, lambda_a=1, lambda_b=0.15
# Backbone: hidden=512, blocks=3 (same as "our" baseline)
# Data: teacher_data_crafter_reward.npz (shared)
#
# Runs in parallel on 3 GPUs.

set -e
cd "$(dirname "$0")"

PYTHON="python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Shared hyperparams (matching crafter_reward/AB_vwT1.6)
KD_TEMP=5
VW_TEMP=1.6
LAMBDA_A=1.0
LAMBDA_B=0.15
DATA_PATH="data/teacher_data_crafter_reward.npz"

COMMON_ARGS="--loss_mode AB \
  --lambda_a $LAMBDA_A --lambda_b $LAMBDA_B \
  --kd_temperature $KD_TEMP --value_weight_temp $VW_TEMP \
  --data_path $DATA_PATH"

echo "================================================================"
echo "  Architecture Ablation — launching 3 encoders in parallel"
echo "  Loss: AB  kd_temp=$KD_TEMP  vw_temp=$VW_TEMP"
echo "  lambda_a=$LAMBDA_A  lambda_b=$LAMBDA_B"
echo "================================================================"

# ResNet — GPU 0
echo "[GPU 0] Starting ResNet encoder..."
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train_student.py \
  --task crafter_reward_resnet \
  --run_name AB_arch \
  $COMMON_ARGS \
  > runs/crafter_reward_resnet_launch.log 2>&1 &
PID_RESNET=$!

# VGG — GPU 1
echo "[GPU 1] Starting VGG encoder..."
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train_student.py \
  --task crafter_reward_vgg \
  --run_name AB_arch \
  $COMMON_ARGS \
  > runs/crafter_reward_vgg_launch.log 2>&1 &
PID_VGG=$!

# Transformer — GPU 2
echo "[GPU 2] Starting Transformer encoder..."
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON train_student.py \
  --task crafter_reward_transformer \
  --run_name AB_arch \
  $COMMON_ARGS \
  > runs/crafter_reward_transformer_launch.log 2>&1 &
PID_TRANSFORMER=$!

echo ""
echo "PIDs: ResNet=$PID_RESNET  VGG=$PID_VGG  Transformer=$PID_TRANSFORMER"
echo "Waiting for all to finish..."
wait $PID_RESNET $PID_VGG $PID_TRANSFORMER

echo ""
echo "All architecture ablation runs complete!"
echo "Results in:"
echo "  runs/crafter_reward_resnet/AB_arch/"
echo "  runs/crafter_reward_vgg/AB_arch/"
echo "  runs/crafter_reward_transformer/AB_arch/"
