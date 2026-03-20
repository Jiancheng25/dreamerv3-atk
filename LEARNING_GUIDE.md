# DreamerV3-ATK 学习指南 (Learning Guide)

欢迎! 这是一个详细的学习指南，帮助您理解和学习 DreamerV3-ATK 项目。

## 目录 (Table of Contents)

1. [项目概述](#项目概述)
2. [核心概念](#核心概念)
3. [代码结构](#代码结构)
4. [学习路径](#学习路径)
5. [实战示例](#实战示例)
6. [关键技术点](#关键技术点)
7. [常见问题](#常见问题)

---

## 项目概述

### 这是什么项目？

**DreamerV3-ATK** 是一个基于 **DreamerV3** (世界模型强化学习算法) 的知识蒸馏和对抗训练框架。

**核心功能：**
- 🎯 **行为克隆与蒸馏**: 训练轻量级学生策略来模仿专家教师智能体 (DreamerV3)
- 🔬 **损失函数消融研究**: 系统性分析新颖损失函数的效果
- 🌍 **多领域评估**: 在不同环境中测试 (连续控制、离散游戏、图像任务)
- 📦 **知识迁移**: 用高效的学生模型替代昂贵的训练世界模型进行部署

**论文引用**: "Mastering Diverse Domains through World Models" (DreamerV3 by Danijar Hafner, v3.3.1)

### 为什么要知识蒸馏？

DreamerV3 虽然强大，但：
- ⚠️ **计算开销大**: 需要大型世界模型 (数百万参数)
- ⚠️ **部署困难**: 难以在资源受限设备上运行
- ✅ **解决方案**: 训练小型学生策略，只需观察→动作映射，无需世界模型

---

## 核心概念

### 1. DreamerV3 世界模型

DreamerV3 是一种基于世界模型的强化学习算法：

```
观察 → 编码器 → 潜在状态 (RSSM) → 解码器 → 预测
              ↓
         策略网络 → 动作
              ↓
         价值网络 → 状态价值
```

**关键组件：**

| 组件 | 作用 | 维度 |
|-----|------|------|
| **RSSM** | 递归状态空间模型 | 确定性状态: 8192 维<br>随机状态: 32×64 类别 |
| **编码器** | 观察 → 潜在表示 | 视觉: CNN<br>本体感觉: MLP |
| **解码器** | 潜在表示 → 重构观察 | 与编码器对称 |
| **策略网络** | 状态 → 动作分布 | 连续: Gaussian<br>离散: Categorical |
| **价值网络** | 状态 → 预期回报 | 标量输出 |

### 2. 知识蒸馏流程

```
步骤 1: 训练 DreamerV3 教师
  → 在环境中学习世界模型和策略

步骤 2: 收集专家数据
  → 让教师执行并记录 (观察, 动作, 奖励)

步骤 3: 训练学生策略
  → 输入: 观察
  → 输出: 动作
  → 损失: 模仿教师行为

步骤 4: 评估学生
  → 层次 A: 预测精度 (行为匹配)
  → 层次 B: 可替代性 (排序相关性)
  → 层次 C: 任务性能 (真实环境成功率)
```

### 3. 损失函数消融

本项目研究三种损失变体：

#### **基线 (Baseline)**
```python
L = MSE(student_action, teacher_action)
```
标准行为克隆，均方误差。

#### **变体 A: NLL 分布拟合**
```python
L = L_MSE + λ_A * NLL(teacher_action | student_μ, student_σ)
```
- 学生预测动作分布 (μ, σ)
- 教师动作作为采样点
- **优势**: 捕捉教师不确定性，接近最优策略时自适应梯度缩放 (1/σ²)

#### **变体 B: 价值加权模仿**
```python
L = L_MSE + λ_B * Σ w_i * MSE(action_i)
其中 w_i = (return_i)^temperature
```
- 根据回报加权每个转换
- temperature=3.0 时，高回报转换权重可达 270×
- **优势**: 聚焦关键决策状态

#### **变体 AB: 组合**
```python
L = L_MSE + λ_A * L_NLL + λ_B * L_VW
```
结合两种优势。

---

## 代码结构

### 目录组织

```
dreamerv3-atk/
│
├── dreamerv3/                    # DreamerV3 智能体实现
│   ├── agent.py                 # 智能体类，损失计算，训练步骤
│   ├── rssm.py                  # RSSM 模块 (编码器，动态模型，解码器)
│   ├── main.py                  # 入口点，配置解析
│   └── configs.yaml             # 配置预设 (默认，任务专用)
│
├── embodied/                     # 强化学习训练框架
│   ├── core/                    # 核心抽象
│   │   ├── base.py             # Agent, Env 接口
│   │   ├── driver.py           # 交互循环 (环境 ↔ 智能体)
│   │   ├── replay.py           # 经验回放缓冲区
│   │   └── streams.py          # 数据管道
│   ├── jax/                    # 基于 JAX 的神经网络
│   │   ├── nets.py             # 网络构建块 (MLP, CNN, 注意力)
│   │   ├── heads.py            # 输出头 (策略, 价值)
│   │   ├── outs.py             # 输出分布
│   │   └── opt.py              # 优化器工具
│   ├── envs/                   # 环境实现
│   │   ├── dmc.py              # DeepMind Control Suite
│   │   ├── atari.py            # Atari 游戏
│   │   ├── crafter.py          # Crafter 环境
│   │   └── ...
│   └── run/                    # 训练编排
│       ├── train.py            # 标准训练
│       ├── eval_only.py        # 仅评估
│       └── parallel.py         # 分布式训练
│
├── dreamerv3-mini/              # 知识蒸馏管道 (核心研究部分)
│   ├── collect_data.py         # 收集教师演示数据
│   ├── train_student.py        # 训练学生策略 (BC + 损失消融)
│   ├── student_model.py        # 学生架构 (MLP/CNN + ResBlocks)
│   ├── task_config.py          # 任务配置注册表
│   ├── compute_metrics.py      # 多层评估
│   ├── compare_ablation.py     # 解析和比较消融结果
│   ├── plot_ablation.py        # 生成消融图
│   ├── run_ablation_*.sh       # 消融实验脚本
│   ├── runs/                   # 输出目录 (模型检查点)
│   ├── data/                   # 专家演示数据
│   └── logs/                   # TensorBoard 日志
│
├── setup.py                    # 包安装
├── requirements.txt            # 依赖项 (JAX, Optax, DM-Control 等)
├── baselines.yaml              # 基线性能分数
└── plot.py                     # 结果绘图工具
```

### 关键文件说明

#### 1. `dreamerv3/agent.py` (约 700 行)
DreamerV3 智能体的核心实现。

**关键类和方法：**
```python
class Agent:
    def __init__(self, config):
        # 初始化 RSSM, 编码器, 解码器, 策略, 价值网络

    def policy(self, obs, state, mode='train'):
        # 给定观察和状态，输出动作

    def train(self, data, state):
        # 训练步骤: 计算所有损失，更新参数
        # 损失包括: 重构, 奖励, 继续, 动态, 策略, 价值
```

**学习建议**: 从 `policy()` 开始，了解推理流程，然后研究 `train()` 理解训练过程。

#### 2. `dreamerv3/rssm.py`
递归状态空间模型 (RSSM) 实现。

**核心概念：**
```python
# RSSM 状态包含两部分:
state = {
    'deter': deterministic_state,  # 8192 维，GRU 输出
    'stoch': stochastic_state,     # 32×64 类别，离散潜在变量
}

# 前向传播:
def observe(self, embed, action, state):
    # 1. 确定性转换: deter' = GRU(deter, [stoch, action, embed])
    # 2. 随机采样: stoch' ~ p(stoch | deter')
    return next_state

def imagine(self, action, state):
    # 想象未来状态 (无观察)
    # 用于策略优化的想象轨迹
    return next_state
```

#### 3. `dreamerv3-mini/train_student.py` (约 400 行)
学生训练的核心脚本。

**训练循环：**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        obs, actions, returns = batch

        # 前向传播
        pred_actions = student(obs)

        # 计算损失
        if loss_mode == 'baseline':
            loss = mse_loss(pred_actions, actions)
        elif loss_mode == 'A':
            loss = mse_loss + nll_loss
        elif loss_mode == 'B':
            weights = (returns ** temperature)
            loss = weighted_mse_loss

        # 反向传播
        loss.backward()
        optimizer.step()
```

**支持的损失模式：**
- `baseline`: 标准 MSE
- `A`: MSE + NLL 分布拟合
- `B`: MSE + 价值加权
- `AB`: 组合

#### 4. `dreamerv3-mini/student_model.py`
学生策略网络架构。

**架构选择：**
```python
class StudentPolicy:
    def __init__(self, config):
        if encoder_type == 'mlp':
            # 适用于本体感觉 (低维观察)
            self.encoder = MLP(obs_dim, hidden_dim)
        elif encoder_type == 'cnn':
            # 适用于图像观察
            self.encoder = CNN(in_channels=3, out_channels=[32,64,64])

        # 残差块 (提升表达能力)
        self.res_blocks = [ResBlock(hidden_dim) for _ in range(num_blocks)]

        # 输出头
        self.action_head = Linear(hidden_dim, action_dim)
```

**配置示例：**
```python
StudentConfig:
    encoder_type: 'mlp' | 'cnn' | 'resnet'
    hidden: 512           # 隐藏层维度
    blocks: 4             # ResBlock 数量
    dropout: 0.1
    learning_rate: 3e-4
```

---

## 学习路径

### 🟢 初级: 理解基础概念 (1-2 天)

#### 任务清单:
- [ ] 阅读 DreamerV3 论文摘要和引言
- [ ] 了解世界模型 vs 无模型 RL 的区别
- [ ] 理解知识蒸馏的动机
- [ ] 浏览 `dreamerv3/configs.yaml`，了解超参数

#### 推荐阅读顺序:
1. **本文档** (LEARNING_GUIDE.md)
2. `dreamerv3/configs.yaml` - 查看默认配置
3. `embodied/core/base.py` - 理解 Agent/Env 接口
4. `dreamerv3-mini/task_config.py` - 查看支持的任务

#### 实验练习:
```bash
# 运行一个简单的评估任务
python -m embodied.run eval_only \
  --env dmc_walker_walk \
  --logdir /tmp/test_run

# 查看配置解析
python -m dreamerv3.main --help
```

### 🟡 中级: 深入代码实现 (3-5 天)

#### 任务清单:
- [ ] 阅读 `dreamerv3/agent.py` - 理解训练循环
- [ ] 阅读 `dreamerv3/rssm.py` - 理解 RSSM 前向/后向传播
- [ ] 阅读 `embodied/jax/nets.py` - 理解网络构建块
- [ ] 阅读 `dreamerv3-mini/train_student.py` - 理解蒸馏训练
- [ ] 阅读 `dreamerv3-mini/student_model.py` - 理解学生架构

#### 关键代码片段分析:

**1. RSSM 状态转换** (`dreamerv3/rssm.py`):
```python
# 观察模式 (有环境反馈)
def observe(self, embed, action, state):
    # embed: 编码后的观察 (来自编码器)
    # action: 上一步动作
    # state: 上一步 RSSM 状态 {'deter', 'stoch'}

    # 1. 确定性转换
    x = jnp.concatenate([state['stoch'], action, embed], -1)
    deter = self.gru(x, state['deter'])

    # 2. 随机采样
    logits = self.dense(deter)  # → [batch, 32, 64]
    stoch = jax.random.categorical(key, logits)

    return {'deter': deter, 'stoch': stoch}

# 想象模式 (无环境反馈)
def imagine(self, action, state):
    # 仅用动作和当前状态预测下一状态
    # 用于策略优化时的想象轨迹
    x = jnp.concatenate([state['stoch'], action], -1)
    deter = self.gru(x, state['deter'])
    logits = self.dense(deter)
    stoch = jax.random.categorical(key, logits)
    return {'deter': deter, 'stoch': stoch}
```

**2. 损失计算** (`dreamerv3/agent.py`):
```python
def train(self, data, state):
    # 数据: {'obs', 'action', 'reward', 'cont'}

    # 1. 编码观察
    embed = self.encoder(data['obs'])

    # 2. RSSM 推断 (后验)
    post = self.rssm.observe(embed, data['action'], state)

    # 3. RSSM 预测 (先验)
    prior = self.rssm.imagine(data['action'], state)

    # 4. 解码重构
    recon = self.decoder(post)

    # 5. 计算损失
    losses = {}
    losses['recon'] = -recon.log_prob(data['obs']).mean()  # 重构
    losses['dyn'] = kl_divergence(post, prior).mean()       # 动态
    losses['reward'] = -self.reward_head(post).log_prob(data['reward']).mean()
    losses['continue'] = -self.cont_head(post).log_prob(data['cont']).mean()

    # 6. 策略和价值优化 (基于想象轨迹)
    imag_states = self.imagine_trajectory(post, horizon=15)
    losses['policy'] = -self.policy_loss(imag_states).mean()
    losses['value'] = self.value_loss(imag_states).mean()

    total_loss = sum(losses.values())
    return total_loss, losses
```

**3. 学生训练损失** (`dreamerv3-mini/train_student.py`):
```python
def compute_loss(student, obs, teacher_actions, returns, config):
    pred_actions = student(obs)

    # 基线损失
    base_loss = F.mse_loss(pred_actions, teacher_actions)

    # 变体 A: NLL 分布拟合
    if config.use_nll:
        mu, log_std = pred_actions.chunk(2, dim=-1)
        std = torch.exp(log_std)
        nll = 0.5 * (((teacher_actions - mu) / std) ** 2 + 2 * log_std)
        nll_loss = nll.mean()
    else:
        nll_loss = 0.0

    # 变体 B: 价值加权
    if config.use_value_weight:
        weights = (returns ** config.temperature)
        weights = weights / weights.mean()  # 归一化
        vw_loss = (weights * F.mse_loss(pred_actions, teacher_actions, reduction='none')).mean()
    else:
        vw_loss = 0.0

    total_loss = (config.lambda_base * base_loss +
                  config.lambda_A * nll_loss +
                  config.lambda_B * vw_loss)

    return total_loss, {'base': base_loss, 'nll': nll_loss, 'vw': vw_loss}
```

#### 实验练习:
```bash
# 1. 收集少量教师数据
cd dreamerv3-mini
python collect_data.py --task dmc_walker_walk --num_episodes 5

# 2. 训练基线学生
python train_student.py --task dmc_walker_walk --loss_mode baseline --epochs 20

# 3. 训练变体 A 学生
python train_student.py --task dmc_walker_walk --loss_mode A --epochs 20

# 4. 比较结果
python compute_metrics.py --task dmc_walker_walk
```

### 🔴 高级: 实验和研究 (1-2 周)

#### 任务清单:
- [ ] 运行完整消融实验 (3 个 GPU, 24 小时)
- [ ] 分析不同损失函数的效果
- [ ] 尝试修改学生架构 (更深/更宽/不同编码器)
- [ ] 在新任务上测试蒸馏管道
- [ ] 实现新的损失函数变体
- [ ] 优化超参数 (学习率, 温度, λ 权重)

#### 研究问题:
1. **数据效率**: 需要多少教师演示才能训练有效的学生？
2. **泛化能力**: 学生在分布外状态的表现如何？
3. **架构敏感性**: 学生网络深度/宽度如何影响性能？
4. **损失协同**: 变体 A 和 B 结合时是否有协同效应？
5. **跨任务迁移**: 在一个任务上训练的学生能否迁移到相似任务？

#### 实验脚本:
```bash
# 运行完整消融研究 (Walker Walk 任务)
cd dreamerv3-mini
bash run_ablation_walker.sh  # 需要 3 个 GPU

# 运行 Crafter 任务消融
bash run_ablation_crafter.sh

# 生成对比图表
python compare_ablation.py --task dmc_walker_walk
python plot_ablation.py --task dmc_walker_walk
```

#### 自定义实验示例:

**修改学生架构**:
```python
# dreamerv3-mini/student_model.py
class StudentPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 实验: 使用更深的网络
        self.encoder = nn.Sequential(
            nn.Linear(config.obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        # 实验: 添加注意力机制
        self.attention = nn.MultiheadAttention(512, num_heads=8)

        # 实验: 更多残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(512) for _ in range(8)  # 从 4 增加到 8
        ])
```

**实现新损失函数**:
```python
# dreamerv3-mini/train_student.py
def compute_obd_loss(student, obs, teacher_actions, teacher_states, config):
    """
    在线蒸馏损失 (Online Behaviour Distillation)
    利用教师状态特征作为额外监督信号
    """
    pred_actions, student_features = student(obs, return_features=True)

    # 动作损失
    action_loss = F.mse_loss(pred_actions, teacher_actions)

    # 特征对齐损失
    feature_loss = F.mse_loss(student_features, teacher_states)

    total_loss = action_loss + config.lambda_feature * feature_loss
    return total_loss
```

---

## 实战示例

### 示例 1: 训练 DreamerV3 教师

```bash
# 任务: DMC Walker Walk (连续控制, 本体感觉)
python -m dreamerv3.main \
  --configs defaults dmc_proprio \
  --task dmc_walker_walk \
  --logdir ./logdir/walker_teacher \
  --batch_size 16 \
  --run.train_ratio 512

# 任务: Crafter (图像输入, 离散动作)
python -m dreamerv3.main \
  --configs defaults crafter \
  --task crafter_reward \
  --logdir ./logdir/crafter_teacher \
  --batch_size 16 \
  --run.train_ratio 1024

# 监控训练进度
tensorboard --logdir ./logdir
```

### 示例 2: 完整蒸馏管道

```bash
cd dreamerv3-mini

# 步骤 1: 收集教师数据
python collect_data.py \
  --task dmc_walker_walk \
  --num_episodes 20 \
  --output_path data/walker_teacher.npz

# 步骤 2: 训练学生 (基线)
python train_student.py \
  --task dmc_walker_walk \
  --data_path data/walker_teacher.npz \
  --loss_mode baseline \
  --epochs 50 \
  --batch_size 256 \
  --learning_rate 3e-4

# 步骤 3: 训练学生 (变体 A)
python train_student.py \
  --task dmc_walker_walk \
  --data_path data/walker_teacher.npz \
  --loss_mode A \
  --lambda_A 0.1 \
  --epochs 50

# 步骤 4: 训练学生 (变体 B)
python train_student.py \
  --task dmc_walker_walk \
  --data_path data/walker_teacher.npz \
  --loss_mode B \
  --lambda_B 0.5 \
  --temperature 3.0 \
  --epochs 50

# 步骤 5: 评估所有变体
python compute_metrics.py \
  --task dmc_walker_walk \
  --variants baseline A B AB

# 步骤 6: 生成对比报告
python compare_ablation.py --task dmc_walker_walk
python plot_ablation.py --task dmc_walker_walk
```

### 示例 3: 并行消融实验 (多 GPU)

```bash
# 假设您有 3 个 GPU (CUDA:0, CUDA:1, CUDA:2)

# run_ablation_custom.sh
#!/bin/bash

# GPU 0: 基线
CUDA_VISIBLE_DEVICES=0 python train_student.py \
  --task dmc_walker_walk \
  --loss_mode baseline \
  --epochs 100 &

# GPU 1: 变体 A
CUDA_VISIBLE_DEVICES=1 python train_student.py \
  --task dmc_walker_walk \
  --loss_mode A \
  --lambda_A 0.1 \
  --epochs 100 &

# GPU 2: 变体 B
CUDA_VISIBLE_DEVICES=2 python train_student.py \
  --task dmc_walker_walk \
  --loss_mode B \
  --lambda_B 0.5 \
  --temperature 3.0 \
  --epochs 100 &

wait  # 等待所有任务完成
echo "所有实验完成！"
```

### 示例 4: 评估指标详解

```bash
# 运行三层评估
python compute_metrics.py --task dmc_walker_walk

# 输出示例:
"""
=== 层次 A: 预测精度 ===
1-Step Action Error (MSE): 0.0234
10-Step Rollout Error: 0.1456

=== 层次 B: 可替代性 ===
Ranker NDCG@10: 0.8234
Top-1 Hit Rate: 0.7890
Kendall τ: 0.6543

=== 层次 C: 任务性能 ===
Average Return: 856.3 (Teacher: 923.4)
Success Rate: 78.2%
Episode Length: 987 steps
"""
```

---

## 关键技术点

### 1. RSSM 的双重状态表示

**为什么需要确定性 + 随机状态？**

```python
state = {
    'deter': deterministic,  # 8192 维
    'stoch': stochastic,     # 32×64 = 2048 类别
}
```

**优势：**
- **确定性状态**: 捕捉长期依赖，GRU 保证梯度流动
- **随机状态**: 建模不确定性，离散表示提升鲁棒性
- **组合**: 充分表达能力 + 稳定训练

**类比**: 像是 VAE 的潜在空间 + RNN 的记忆单元的结合。

### 2. 损失函数设计哲学

**基线 MSE 的局限：**
- 忽略教师不确定性 (在模糊状态，教师可能探索多个动作)
- 忽略状态重要性 (关键决策点 vs 常规状态)

**变体 A (NLL) 的直觉：**
```python
# 学生预测分布
student_action ~ N(μ, σ²)

# NLL 损失
L_NLL = 0.5 * ((teacher_action - μ) / σ)² + log(σ)

# 效果:
# - 接近最优时, σ → 小 → 梯度大 (1/σ² 缩放)
# - 不确定状态, σ → 大 → 容忍偏差
```

**变体 B (价值加权) 的直觉：**
```python
# 回报加权
w_i = (return_i / max_return)^temperature

# temperature = 3.0 示例:
# return=0.9 → w=0.729
# return=1.0 → w=1.000
# return=0.5 → w=0.125

# 效果: 高回报轨迹占主导，学生优先学习成功案例
```

### 3. JAX vs PyTorch 的权衡

**DreamerV3 使用 JAX**:
- ✅ JIT 编译加速
- ✅ 自动向量化 (vmap)
- ✅ 灵活的自动微分
- ❌ 学习曲线陡峭
- ❌ 调试困难

**蒸馏管道使用 PyTorch**:
- ✅ 简单易用
- ✅ 丰富的生态系统
- ✅ 方便调试
- ❌ 性能略逊 JAX

**为什么不统一？**
- DreamerV3 原始实现是 JAX (继承自原始论文实现)
- 蒸馏是研究扩展，使用 PyTorch 降低入门门槛

### 4. 数据采样策略

**Replay Buffer 采样器** (`embodied/core/selectors.py`):

```python
# 1. 均匀采样 (Uniform)
class Uniform:
    def sample(self, size):
        return np.random.choice(len(buffer), size)

# 2. 优先级采样 (Prioritized)
class Prioritized:
    def sample(self, size):
        # 高 TD-error 转换优先
        probs = priorities / sum(priorities)
        return np.random.choice(len(buffer), size, p=probs)

# 3. 时间加权 (Recency)
class Recency:
    def sample(self, size):
        # 最近转换权重更高
        weights = np.exp(-age / tau)
        probs = weights / sum(weights)
        return np.random.choice(len(buffer), size, p=probs)

# 4. 混合采样 (Mixture)
class Mixture:
    def sample(self, size):
        # 组合多种策略
        uniform_frac = 0.5
        priority_frac = 0.3
        recency_frac = 0.2
        ...
```

**选择建议:**
- **均匀**: 稳定, 适合离线数据集
- **优先级**: 难样本, 提升样本效率
- **时间加权**: 非平稳环境, 适应分布变化
- **混合**: 平衡探索与利用

### 5. 配置系统详解

**分层配置** (`dreamerv3/configs.yaml`):

```yaml
# 第 1 层: 默认值 (所有任务共享)
defaults:
  batch_size: 16
  batch_length: 64
  learning_rate: 1e-4
  ...

# 第 2 层: 任务族配置
dmc_proprio:
  encoder: mlp
  decoder: mlp
  units: 1024
  ...

dmc_vision:
  encoder: cnn
  decoder: cnn
  units: 1024
  cnn_depth: 96
  ...

# 第 3 层: 模型大小配置
size1m:
  rssm.deter: 512
  rssm.units: 512
  .**.*_head.layers: 2
  ...

size200m:
  rssm.deter: 8192
  rssm.units: 1024
  .**.*_head.layers: 5
  ...

# 使用:
python -m dreamerv3.main \
  --configs defaults dmc_vision size200m \
  --task dmc_walker_walk
# 配置合并顺序: defaults → dmc_vision → size200m → 命令行参数
```

**自定义配置:**
```bash
# 方法 1: 命令行覆盖
python -m dreamerv3.main \
  --configs defaults dmc_proprio \
  --batch_size 32 \
  --learning_rate 3e-4

# 方法 2: 创建自定义 YAML
# my_config.yaml
custom_walker:
  batch_size: 32
  imag_horizon: 20
  rssm.deter: 4096

# 使用自定义配置
python -m dreamerv3.main \
  --configs defaults my_config.custom_walker \
  --task dmc_walker_walk
```

---

## 常见问题

### Q1: 如何选择合适的任务开始学习？

**推荐顺序:**

1. **入门**: `dmc_walker_walk` (DMC 本体感觉)
   - ✅ 观察空间小 (24 维)
   - ✅ 动作空间连续但低维 (6 维)
   - ✅ 训练快 (1-2 小时)
   - ✅ 结果稳定

2. **进阶**: `crafter_reward` (图像 + 离散)
   - ✅ 图像观察 (64×64×3)
   - ✅ 离散动作 (17 类)
   - ⚠️ 训练慢 (6-12 小时)
   - ✅ 更具挑战性

3. **高级**: `atari_pong` 或 `procgen_coinrun`
   - ⚠️ 大型网络 (200M 参数)
   - ⚠️ 训练很慢 (24-48 小时)
   - ⚠️ 需要多 GPU

### Q2: 需要什么硬件？

**最低配置:**
- CPU: 4 核
- RAM: 16 GB
- GPU: NVIDIA GTX 1660 (6 GB VRAM)
- 磁盘: 50 GB

**推荐配置:**
- CPU: 8 核+
- RAM: 32 GB+
- GPU: NVIDIA RTX 3090 (24 GB VRAM) × 3
- 磁盘: 200 GB SSD

**各任务 VRAM 需求:**
| 任务 | 模型大小 | VRAM |
|-----|---------|------|
| DMC Proprio | 1M | 4 GB |
| DMC Vision | 12M | 8 GB |
| Crafter | 25M | 12 GB |
| Atari | 200M | 20 GB |

### Q3: 训练多久才能看到效果？

**DreamerV3 教师:**
- DMC: 0.5-1M 步 (~2-4 小时, 单 GPU)
- Crafter: 1-2M 步 (~6-12 小时)
- Atari: 10-50M 步 (~24-72 小时)

**学生蒸馏:**
- 收集数据: 20 episodes (~10 分钟)
- 训练学生: 50-100 epochs (~30-60 分钟)
- 评估: ~10 分钟

**提示**: 使用 TensorBoard 实时监控:
```bash
tensorboard --logdir ./logdir --port 6006
# 浏览器打开 http://localhost:6006
```

### Q4: 如何调试训练失败？

**常见问题和解决方案:**

**问题 1: Loss = NaN**
```python
# 原因: 梯度爆炸, 学习率过高
# 解决:
--learning_rate 1e-5  # 降低学习率
--grad_clip 1000.0    # 梯度裁剪
```

**问题 2: Reward 不增长**
```python
# 原因: 探索不足, 训练不足
# 解决:
--run.train_ratio 512  # 增加训练频率 (默认 32)
--imag_horizon 20      # 增加想象长度 (默认 15)
```

**问题 3: OOM (显存不足)**
```python
# 解决:
--batch_size 8         # 减小批量大小
--batch_length 32      # 减小序列长度
--encoder.cnn_depth 48 # 减小网络宽度
```

**问题 4: 学生性能差**
```python
# 原因: 数据不足, 架构不匹配
# 解决:
--num_episodes 50      # 收集更多数据
--student.blocks 6     # 增加学生容量
--learning_rate 1e-4   # 调整学习率
```

### Q5: 如何可视化结果？

**方法 1: TensorBoard**
```bash
tensorboard --logdir ./logdir
```

**方法 2: 视频录制**
```python
# dreamerv3-mini/record_teacher_video.py
python record_teacher_video.py \
  --task dmc_walker_walk \
  --checkpoint ./logdir/checkpoint.pkl \
  --output ./videos/walker.mp4
```

**方法 3: 消融对比图**
```python
# 生成对比图表
python plot_ablation.py --task dmc_walker_walk
# 输出: ./plots/ablation_comparison.png
```

**方法 4: 自定义可视化**
```python
import matplotlib.pyplot as plt
import numpy as np

# 加载学生训练日志
data = np.load('runs/dmc_walker_walk/baseline/metrics.npz')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(data['train_loss'], label='Train Loss')
plt.plot(data['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(data['eval_return'], label='Evaluation Return')
plt.axhline(y=data['teacher_return'], color='r', linestyle='--', label='Teacher')
plt.xlabel('Epoch')
plt.ylabel('Return')
plt.legend()

plt.tight_layout()
plt.savefig('training_progress.png')
```

### Q6: 如何扩展到新任务？

**步骤:**

1. **添加任务配置** (`dreamerv3-mini/task_config.py`):
```python
TASK_CONFIGS['my_custom_task'] = TaskConfig(
    env_name='my_env',
    obs_shape=(64, 64, 3),      # 图像
    action_dim=5,                # 离散动作
    action_type='discrete',
    encoder_type='cnn',
    teacher_logdir='./logdir/my_teacher',
)
```

2. **训练教师** (如果还没有):
```bash
python -m dreamerv3.main \
  --configs defaults custom_config \
  --task my_custom_task \
  --logdir ./logdir/my_teacher
```

3. **运行蒸馏管道**:
```bash
cd dreamerv3-mini
python collect_data.py --task my_custom_task
python train_student.py --task my_custom_task --loss_mode baseline
python compute_metrics.py --task my_custom_task
```

### Q7: 论文复现清单

如果您想完整复现论文实验:

- [ ] **硬件**: 3× NVIDIA RTX 3090 (24 GB)
- [ ] **任务**:
  - [ ] `dmc_walker_walk` (本体感觉)
  - [ ] `crafter_reward` (图像 + 离散)
  - [ ] `atari_pong` (可选)
- [ ] **变体**: baseline, A, B, AB (4 种)
- [ ] **运行**: 每个任务×变体 3 次 (取平均)
- [ ] **总训练时间**: ~72 小时 (并行)
- [ ] **总存储**: ~100 GB (检查点 + 日志)

**脚本:**
```bash
# 完整复现脚本
for task in dmc_walker_walk crafter_reward; do
    for variant in baseline A B AB; do
        for seed in 1 2 3; do
            python train_student.py \
              --task $task \
              --loss_mode $variant \
              --seed $seed \
              --epochs 100
        done
    done
done

# 聚合结果
python aggregate_results.py --tasks dmc_walker_walk crafter_reward
```

---

## 进一步学习资源

### 论文阅读列表

1. **DreamerV3 原论文**:
   - "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
   - arXiv: https://arxiv.org/abs/2301.04104

2. **知识蒸馏基础**:
   - "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

3. **世界模型**:
   - "World Models" (Ha & Schmidhuber, 2018)
   - "Dream to Control" (Hafner et al., 2020) - DreamerV1
   - "Mastering Atari with Discrete World Models" (Hafner et al., 2021) - DreamerV2

4. **强化学习基础**:
   - Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)

### 代码阅读顺序

**第 1 周**: 接口和抽象
1. `embodied/core/base.py` - Agent/Env 接口
2. `embodied/core/driver.py` - 交互循环
3. `embodied/core/replay.py` - 经验回放

**第 2 周**: DreamerV3 核心
1. `dreamerv3/configs.yaml` - 配置系统
2. `dreamerv3/rssm.py` - RSSM 模块
3. `dreamerv3/agent.py` - 智能体实现

**第 3 周**: 网络和优化
1. `embodied/jax/nets.py` - 网络构建块
2. `embodied/jax/heads.py` - 输出头
3. `embodied/jax/opt.py` - 优化器

**第 4 周**: 蒸馏管道
1. `dreamerv3-mini/task_config.py` - 任务配置
2. `dreamerv3-mini/student_model.py` - 学生架构
3. `dreamerv3-mini/train_student.py` - 训练循环
4. `dreamerv3-mini/compute_metrics.py` - 评估

### 社区资源

- **GitHub Issues**: https://github.com/Jiancheng25/dreamerv3-atk/issues
- **DreamerV3 官方实现**: https://github.com/danijar/dreamerv3
- **Discord**: (如果有社区频道)

---

## 总结

这个项目是一个完整的强化学习研究框架，包含:

1. **DreamerV3**: 最先进的世界模型 RL 算法
2. **知识蒸馏**: 将大型模型压缩为轻量部署版本
3. **消融研究**: 系统性评估不同损失函数设计
4. **多领域评估**: 8 个任务族，涵盖视觉、控制、游戏

**核心价值:**
- 🔬 **研究**: 探索知识蒸馏在 RL 中的应用
- 🚀 **部署**: 用小型学生模型替代大型世界模型
- 📚 **学习**: 理解世界模型 RL 和蒸馏技术

**学习建议:**
1. 从简单任务开始 (DMC Proprio)
2. 先理解接口和流程，再深入实现细节
3. 运行小规模实验验证理解
4. 逐步增加复杂度和实验规模

祝学习顺利! 如有问题，请查阅代码注释或提交 Issue。
