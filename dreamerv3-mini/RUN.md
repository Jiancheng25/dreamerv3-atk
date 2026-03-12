# DreamerV3 -> Student ResMLP: Behaviour Cloning Pipeline

## Prerequisites

- Conda environment: `dreamerv3`
- Trained Teacher checkpoint at:
  `logdir/dreamer/20260128T193932-dmc_proprio_dmc_walker_walk/ckpt/`

## Pipeline (run in order)

```bash
# Activate the conda environment
conda activate dreamerv3

# All commands below assume you are in the dreamerv3-main root
cd dreamerv3-main  # or your project root
```

### Step 1 -- Collect expert data

```bash
CUDA_VISIBLE_DEVICES=1 \
MUJOCO_GL=osmesa \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python dreamerv3-mini/collect_data.py
```

**Output:** `dreamerv3-mini/data/teacher_data.npz`
(20 episodes, ~20 000 transitions, includes obs/act/mean/std)

### Step 2 -- Train the Student

```bash
CUDA_VISIBLE_DEVICES=1 \
MUJOCO_GL=osmesa \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python dreamerv3-mini/train_student.py
```

**Output:**
- Best model: `dreamerv3-mini/models/best_student.pth`
- TensorBoard logs: `dreamerv3-mini/logs/student_train/`

### Step 3 -- Monitor training (optional, separate terminal)

```bash
tensorboard --logdir dreamerv3-mini/logs/student_train --port 6006
```

## File layout

```
dreamerv3-mini/
  collect_data.py      # Step 1: expert data collection
  student_model.py     # ResMLP architecture definition
  train_student.py     # Step 2: BC training + real-env eval
  data/
    teacher_data.npz   # expert demonstrations
  models/
    best_student.pth   # best student checkpoint (by eval score)
  logs/
    teacher_eval/      # teacher collection logs
    student_train/     # TensorBoard events
```

## Key design decisions

| Aspect | Detail |
|--------|--------|
| Obs keys | `height`(1) + `orientations`(14) + `velocity`(9) = **24-dim** |
| Action | 6-dim continuous, Tanh-bounded to [-1, 1] |
| Student arch | ResMLP: 512-hidden, 4 ResBlocks, ~2.1M params |
| Normalisation | Fixed input normalisation using expert data mean/std |
| Eval criterion | Mean episode score on real env (not training loss) |
| Save policy | Only save when eval score improves (best-score gating) |
