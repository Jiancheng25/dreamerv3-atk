# 快速开始指南 (Quick Start Guide)

## 5 分钟快速体验

### 前置条件

```bash
# 检查 Python 版本 (需要 3.8+)
python --version

# 检查 GPU (可选，但强烈推荐)
nvidia-smi
```

### 安装依赖

```bash
# 克隆仓库 (如果还没有)
git clone https://github.com/Jiancheng25/dreamerv3-atk.git
cd dreamerv3-atk

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 选项 1: 快速测试 DreamerV3

```bash
# 在 DMC Walker Walk 任务上训练 1000 步 (约 5 分钟)
python -m dreamerv3.main \
  --configs defaults dmc_proprio debug \
  --task dmc_walker_walk \
  --logdir /tmp/quick_test \
  --run.steps 1000

# 监控训练
tensorboard --logdir /tmp/quick_test
```

### 选项 2: 快速测试知识蒸馏

```bash
cd dreamerv3-mini

# 步骤 1: 收集少量教师数据 (约 2 分钟)
python collect_data.py \
  --task dmc_walker_walk \
  --num_episodes 5 \
  --output_path data/quick_test.npz

# 步骤 2: 训练学生策略 (约 3 分钟)
python train_student.py \
  --task dmc_walker_walk \
  --data_path data/quick_test.npz \
  --loss_mode baseline \
  --epochs 10 \
  --batch_size 128

# 步骤 3: 评估学生
python compute_metrics.py \
  --task dmc_walker_walk \
  --variants baseline
```

## 完整实验示例

### DMC Walker Walk (本体感觉, 连续控制)

```bash
# 1. 训练教师 (约 2 小时, 需要 GPU)
python -m dreamerv3.main \
  --configs defaults dmc_proprio \
  --task dmc_walker_walk \
  --logdir ./logdir/walker_teacher \
  --batch_size 16

# 2. 运行蒸馏管道
cd dreamerv3-mini

# 收集数据
python collect_data.py \
  --task dmc_walker_walk \
  --num_episodes 20

# 训练基线学生
CUDA_VISIBLE_DEVICES=0 python train_student.py \
  --task dmc_walker_walk \
  --loss_mode baseline \
  --epochs 50

# 训练变体 A (NLL)
CUDA_VISIBLE_DEVICES=1 python train_student.py \
  --task dmc_walker_walk \
  --loss_mode A \
  --lambda_A 0.1 \
  --epochs 50

# 训练变体 B (价值加权)
CUDA_VISIBLE_DEVICES=2 python train_student.py \
  --task dmc_walker_walk \
  --loss_mode B \
  --lambda_B 0.5 \
  --temperature 3.0 \
  --epochs 50

# 评估和对比
python compute_metrics.py --task dmc_walker_walk
python compare_ablation.py --task dmc_walker_walk
python plot_ablation.py --task dmc_walker_walk
```

### Crafter (图像输入, 离散动作)

```bash
# 1. 训练教师 (约 6-12 小时)
python -m dreamerv3.main \
  --configs defaults crafter \
  --task crafter_reward \
  --logdir ./logdir/crafter_teacher

# 2. 运行蒸馏
cd dreamerv3-mini
bash run_ablation_crafter.sh  # 需要 3 个 GPU
```

## 配置说明

### 任务配置

| 任务 | 观察类型 | 动作类型 | 配置 |
|-----|---------|---------|------|
| `dmc_walker_walk` | 本体感觉 (24维) | 连续 (6维) | `dmc_proprio` |
| `dmc_walker_walk` | 图像 (64×64×3) | 连续 (6维) | `dmc_vision` |
| `crafter_reward` | 图像 (64×64×3) | 离散 (17) | `crafter` |
| `atari_pong` | 图像 (84×84×4) | 离散 (18) | `atari` |
| `procgen_coinrun` | 图像 (64×64×3) | 离散 (15) | `procgen` |

### 常用参数

```bash
# DreamerV3 训练
--batch_size 16              # 批量大小
--batch_length 64            # 序列长度
--run.train_ratio 512        # 训练频率
--learning_rate 1e-4         # 学习率
--run.steps 1000000          # 总训练步数

# 学生训练
--loss_mode baseline|A|B|AB  # 损失模式
--epochs 50                  # 训练轮数
--learning_rate 3e-4         # 学习率
--lambda_A 0.1               # NLL 损失权重
--lambda_B 0.5               # 价值加权损失权重
--temperature 3.0            # 温度参数
```

## 目录结构

```
dreamerv3-atk/
├── logdir/                  # DreamerV3 训练日志和检查点
│   └── walker_teacher/
│       ├── checkpoint.pkl
│       └── events.out.tfevents.*
│
└── dreamerv3-mini/
    ├── data/                # 教师演示数据
    │   └── teacher_data.npz
    ├── runs/                # 学生训练输出
    │   └── dmc_proprio_dmc_walker_walk/
    │       ├── baseline/
    │       │   ├── best_student.pth
    │       │   ├── config.json
    │       │   └── metrics.json
    │       ├── A/
    │       └── B/
    ├── logs/                # TensorBoard 日志
    └── plots/               # 可视化图表
```

## 监控和可视化

### TensorBoard

```bash
# DreamerV3 训练
tensorboard --logdir ./logdir --port 6006

# 学生训练
tensorboard --logdir ./dreamerv3-mini/logs --port 6007
```

### 关键指标

**DreamerV3 (教师):**
- `train/return`: 训练回报
- `eval/return`: 评估回报
- `loss/total`: 总损失
- `loss/recon`: 重构损失
- `loss/dyn`: 动态损失

**学生蒸馏:**
- `train/loss`: 训练损失
- `val/loss`: 验证损失
- `eval/return`: 评估回报
- `metrics/action_error`: 动作误差
- `metrics/ndcg`: 排序相关性

## 常见问题排查

### 问题 1: CUDA Out of Memory

```bash
# 解决方法 1: 减小批量大小
--batch_size 8

# 解决方法 2: 减小序列长度
--batch_length 32

# 解决方法 3: 使用小型模型
--configs defaults dmc_proprio size1m
```

### 问题 2: 训练不稳定 (Loss = NaN)

```bash
# 解决方法: 降低学习率和梯度裁剪
--learning_rate 1e-5
--grad_clip 100.0
```

### 问题 3: 环境安装失败

```bash
# 分步安装
pip install jax jaxlib  # 先安装 JAX
pip install dm-control   # 再安装环境
pip install -e .         # 最后安装项目
```

### 问题 4: 找不到教师检查点

```bash
# 确保教师已训练完成
ls ./logdir/walker_teacher/checkpoint.pkl

# 如果没有，先训练教师
python -m dreamerv3.main --configs defaults dmc_proprio --task dmc_walker_walk
```

## 下一步

完成快速开始后，建议：

1. 📖 阅读 [LEARNING_GUIDE.md](LEARNING_GUIDE.md) 了解详细概念
2. 🔍 探索 `dreamerv3/agent.py` 理解 DreamerV3 实现
3. 🔬 运行完整消融实验 `bash run_ablation_walker.sh`
4. 🎓 尝试自定义损失函数和学生架构
5. 🌟 在新任务上测试蒸馏管道

## 有用的命令

```bash
# 列出可用任务
python -m dreamerv3.main --help | grep task

# 查看配置选项
python -m dreamerv3.main --configs defaults --help

# 从检查点恢复训练
python -m dreamerv3.main --configs defaults dmc_proprio \
  --task dmc_walker_walk \
  --logdir ./logdir/walker_teacher  # 自动从最后检查点恢复

# 仅评估 (不训练)
python -m embodied.run eval_only \
  --task dmc_walker_walk \
  --logdir ./logdir/walker_teacher

# 并行训练 (多 GPU)
python -m dreamerv3.main --configs defaults dmc_proprio \
  --script parallel \
  --run.envs 16
```

## 资源需求参考

| 任务 | 训练时间 | GPU 显存 | 磁盘空间 |
|-----|---------|---------|---------|
| DMC (debug) | 5 分钟 | 2 GB | 500 MB |
| DMC (完整) | 2 小时 | 6 GB | 5 GB |
| Crafter | 6-12 小时 | 12 GB | 10 GB |
| Atari | 24-48 小时 | 20 GB | 50 GB |
| 学生蒸馏 | 30-60 分钟 | 4 GB | 1 GB |

祝实验顺利! 🚀
