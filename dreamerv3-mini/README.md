# DreamerV3-Mini: Lightweight Student Policy Distillation

A lightweight behavior cloning framework that distills DreamerV3 teacher policies into efficient student networks for real-time deployment.

## Overview

This project implements a two-stage knowledge distillation pipeline:
1. **Data Collection**: Collect expert demonstrations from a trained DreamerV3 agent
2. **Student Training**: Train lightweight ResMLP/CNN student policies via behavior cloning

### Key Features

- **Multi-task Support**: Crafter, Procgen, Atari, DMC Walker
- **Architecture Variants**: ResMLP (default), ResNet, VGG, Transformer
- **Loss Functions**: 
  - Layer A: Distribution fitting (NLL)
  - Layer B: Value-weighted action ranking
  - Combined A+B for optimal performance
- **Comprehensive Metrics**: 1-step error, k-step rollout error, NDCG, Top-1 Hit, Kendall τ

## Installation

```bash
# Clone the repository
git clone https://github.com/Jiancheng25/dreamerv3-atk.git
cd dreamerv3-atk

# Create conda environment
conda create -n dreamerv3 python=3.10
conda activate dreamerv3

# Install dependencies
pip install torch numpy gym crafter procgen
pip install tensorboard imageio imageio-ffmpeg
```

## Quick Start

### 1. Collect Expert Data

```bash
python dreamerv3-mini/collect_data.py --task crafter_reward --episodes 400
```

### 2. Train Student Model

```bash
python dreamerv3-mini/train_student.py --task crafter_reward --loss_mode AB
```

### 3. Evaluate Metrics

```bash
python dreamerv3-mini/compute_metrics.py --task crafter_reward
```

## Project Structure

```
dreamerv3-mini/
├── collect_data.py          # Expert data collection from teacher
├── train_student.py         # Student training with BC
├── student_model.py         # Model architectures (ResMLP, ResNet, VGG, ViT)
├── task_config.py           # Task-specific configurations
├── compute_metrics.py       # 3-layer evaluation metrics
├── run_ablation_*.sh        # Ablation experiment scripts
└── data/                    # Expert demonstration data (not tracked)
```

## Supported Tasks

| Task | Observation | Action | Teacher Checkpoint |
|------|-------------|--------|-------------------|
| Crafter | 64×64 RGB | Discrete (17) | DreamerV3 166M |
| Procgen CoinRun | 64×64 RGB | Discrete (15) | DreamerV3 166M |
| Atari Pong | 64×64 RGB | Discrete (6) | DreamerV3 166M |
| DMC Walker Walk | Proprio (24) | Continuous (6) | DreamerV3 18M |

## Results

### Crafter (DreamerV3 166M Teacher)

| Method | Score | 1-step Err ↓ | NDCG ↑ | Top-1 Hit ↑ |
|--------|-------|--------------|--------|-------------|
| Baseline (CE) | 5.9 | 0.1147 | 0.9512 | 0.8039 |
| A-only (NLL) | 6.8 | 0.1052 | 0.9543 | 0.8127 |
| B-only (VW) | 7.2 | 0.1089 | 0.9521 | 0.8065 |
| **Ours (A+B)** | **9.4** | **0.0998** | **0.9569** | **0.8190** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{dreamerv3mini2026,
  title={Efficient Policy Distillation from World Model Agents},
  author={},
  year={2026}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

- [DreamerV3](https://github.com/danijar/dreamerv3) by Danijar Hafner
- [Crafter](https://github.com/danijar/crafter) environment
