# DreamerV3-ATK

基于 DreamerV3 的知识蒸馏和对抗训练框架

## 📚 文档导航

- **[学习指南 (LEARNING_GUIDE.md)](LEARNING_GUIDE.md)** - 详细的项目学习指南，包含核心概念、代码结构和技术细节
- **[快速开始 (QUICKSTART.md)](QUICKSTART.md)** - 5分钟快速体验和完整实验示例

## 🎯 项目概述

DreamerV3-ATK 是一个研究项目，专注于将强大但计算昂贵的 DreamerV3 世界模型智能体的知识蒸馏到轻量级学生策略中。

### 核心功能

- 🤖 **DreamerV3 智能体**: 世界模型强化学习算法的完整实现
- 📦 **知识蒸馏**: 训练轻量级学生策略模仿专家教师
- 🔬 **损失函数消融**: 系统研究不同损失设计的影响
- 🌍 **多领域支持**: 支持 8 个任务族 (DMC, Atari, Crafter, Procgen 等)
- 📊 **三层评估**: 预测精度、可替代性、任务性能

### 主要组件

```
dreamerv3-atk/
├── dreamerv3/          # DreamerV3 智能体实现 (JAX)
├── embodied/           # 强化学习训练框架
├── dreamerv3-mini/     # 知识蒸馏管道 (PyTorch)
└── 文档/
    ├── LEARNING_GUIDE.md  # 详细学习指南
    └── QUICKSTART.md      # 快速开始
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/Jiancheng25/dreamerv3-atk.git
cd dreamerv3-atk

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

### 5 分钟快速测试

```bash
# 测试 DreamerV3
python -m dreamerv3.main \
  --configs defaults dmc_proprio debug \
  --task dmc_walker_walk \
  --logdir /tmp/quick_test \
  --run.steps 1000
```

更多示例请查看 [QUICKSTART.md](QUICKSTART.md)

## 📖 学习路径

### 初学者 (1-2 天)
1. 阅读 [LEARNING_GUIDE.md](LEARNING_GUIDE.md) 的"项目概述"和"核心概念"
2. 运行 [QUICKSTART.md](QUICKSTART.md) 中的快速测试
3. 浏览 `dreamerv3/configs.yaml` 了解配置系统

### 中级 (3-5 天)
1. 阅读 `dreamerv3/agent.py` 理解 DreamerV3 实现
2. 阅读 `dreamerv3-mini/train_student.py` 理解蒸馏流程
3. 运行完整的蒸馏实验

### 高级 (1-2 周)
1. 运行消融实验 `bash dreamerv3-mini/run_ablation_walker.sh`
2. 实现自定义损失函数
3. 在新任务上测试蒸馏管道

## 🎓 支持的任务

| 任务族 | 示例 | 观察类型 | 动作类型 |
|-------|------|---------|---------|
| **DMC** | `dmc_walker_walk` | 本体感觉/视觉 | 连续 |
| **Atari** | `atari_pong` | 图像 (84×84) | 离散 |
| **Crafter** | `crafter_reward` | 图像 (64×64) | 离散 |
| **Procgen** | `procgen_coinrun` | 图像 (64×64) | 离散 |
| **DeepMind Lab** | `dmlab_rooms_watermaze` | 图像 (72×96) | 离散 |
| **Minecraft** | `minecraft_diamond` | 图像 (64×64) | 离散 |

## 🔬 研究重点

本项目研究三种损失函数变体：

1. **Baseline**: 标准 MSE 行为克隆
2. **Variant A**: MSE + NLL 分布拟合 (捕捉教师不确定性)
3. **Variant B**: MSE + 价值加权模仿 (强调关键决策)
4. **Variant AB**: 组合变体

详细说明请参考 [LEARNING_GUIDE.md](LEARNING_GUIDE.md#损失函数消融)

## 📊 评估框架

三层评估体系：

- **层次 A - 预测精度**: 1-步动作误差，rollout 误差
- **层次 B - 可替代性**: NDCG, Top-1 命中率, Kendall τ
- **层次 C - 任务性能**: 真实环境评估回报

## 🛠️ 使用示例

### 训练 DreamerV3 教师

```bash
python -m dreamerv3.main \
  --configs defaults dmc_proprio \
  --task dmc_walker_walk \
  --logdir ./logdir/walker_teacher
```

### 运行知识蒸馏

```bash
cd dreamerv3-mini

# 1. 收集教师数据
python collect_data.py --task dmc_walker_walk --num_episodes 20

# 2. 训练学生
python train_student.py --task dmc_walker_walk --loss_mode baseline --epochs 50

# 3. 评估
python compute_metrics.py --task dmc_walker_walk
```

### 运行消融实验

```bash
# 并行运行所有损失变体 (需要 3 个 GPU)
cd dreamerv3-mini
bash run_ablation_walker.sh

# 生成对比报告
python compare_ablation.py --task dmc_walker_walk
python plot_ablation.py --task dmc_walker_walk
```

## 📁 项目结构

```
dreamerv3-atk/
│
├── dreamerv3/                    # DreamerV3 智能体
│   ├── agent.py                 # 智能体实现
│   ├── rssm.py                  # 世界模型 (RSSM)
│   ├── main.py                  # 训练入口
│   └── configs.yaml             # 配置预设
│
├── embodied/                     # RL 框架
│   ├── core/                    # 核心抽象 (Agent, Env, Replay)
│   ├── jax/                     # JAX 神经网络
│   ├── envs/                    # 环境实现
│   └── run/                     # 训练脚本
│
├── dreamerv3-mini/              # 知识蒸馏管道
│   ├── collect_data.py         # 数据收集
│   ├── train_student.py        # 学生训练
│   ├── student_model.py        # 学生架构
│   ├── compute_metrics.py      # 评估
│   ├── compare_ablation.py     # 消融对比
│   └── run_ablation_*.sh       # 实验脚本
│
├── LEARNING_GUIDE.md            # 详细学习指南 ⭐
├── QUICKSTART.md                # 快速开始指南 ⭐
├── requirements.txt             # 依赖
└── setup.py                     # 安装脚本
```

## 💡 关键技术点

- **RSSM (递归状态空间模型)**: 结合确定性 (8192 维) 和随机 (32×64 类别) 状态
- **想象轨迹**: 在潜在空间优化策略，无需真实环境交互
- **NLL 损失**: 自适应学习率，通过 1/σ² 缩放梯度
- **价值加权**: 用 (return)^temperature 加权高回报转换

详细技术解析请参考 [LEARNING_GUIDE.md](LEARNING_GUIDE.md#关键技术点)

## 📦 依赖项

主要依赖:
- **JAX** (DreamerV3 实现)
- **PyTorch** (学生训练)
- **DM-Control**, **Atari**, **Crafter** (环境)
- **Optax** (优化器)
- **TensorBoard** (可视化)

完整列表见 `requirements.txt`

## 🐛 常见问题

### CUDA Out of Memory
```bash
--batch_size 8 --batch_length 32
```

### Loss = NaN
```bash
--learning_rate 1e-5 --grad_clip 100.0
```

### 找不到教师检查点
```bash
# 确保先训练教师
python -m dreamerv3.main --configs defaults dmc_proprio --task dmc_walker_walk
```

更多问题请参考 [LEARNING_GUIDE.md](LEARNING_GUIDE.md#常见问题)

## 📚 参考文献

**DreamerV3**:
```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```

**知识蒸馏**:
```bibtex
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

本项目基于 MIT 许可证 (继承自原始 DreamerV3 实现)

## 🔗 相关链接

- **DreamerV3 官方**: https://github.com/danijar/dreamerv3
- **论文**: https://arxiv.org/abs/2301.04104
- **项目主页**: https://github.com/Jiancheng25/dreamerv3-atk

---

⭐ **开始学习**: 访问 [LEARNING_GUIDE.md](LEARNING_GUIDE.md) 获取完整学习路径

🚀 **快速体验**: 访问 [QUICKSTART.md](QUICKSTART.md) 立即开始实验
