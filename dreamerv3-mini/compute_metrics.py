#!/usr/bin/env python3
"""
Compute the 3-layer evaluation metrics for student models.

Layer A — Predictive Fidelity (预测一致性):
  - 1-step Error:  Action prediction error on held-out test data     (↓)
  - Multi-step Rollout Error (k=2, k=5): Cumulative reward gap      (↓)

Layer B — Substitutability (功能替代性) [discrete tasks]:
  - Ranker NDCG:   Action-ranking quality vs teacher                 (↑)
  - Top-1 Hit:     Fraction where student picks same action          (↑)
  - Kendall τ:     Rank correlation of action preferences            (↑)

Layer C — End-task Performance (最终任务结果):
  - Score:  Best eval score from training (from checkpoint)          (↑)

Usage:
    python compute_metrics.py --task crafter_reward
    python compute_metrics.py --task crafter_reward --run_online --online_episodes 20
    python compute_metrics.py --task dmc_walker_walk
"""

# ── Environment variables (MUST precede other imports) ───────────────────────
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import sys
import pathlib
import argparse
import time
import collections
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import torch

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from student_model import StudentConfig, StudentPolicy
from task_config import TASK_CONFIGS

# ─── Configuration ───────────────────────────────────────────────────────────
VARIANTS = ['baseline', 'A', 'B', 'AB']
VARIANT_LABELS = {
    'baseline': 'Baseline (MSE/CE)',
    'A':        'A-only (分布拟合NLL)',
    'B':        'B-only (价值加权)',
    'AB':       'Ours (A+B)',
}

RUNS_DIR = SCRIPT_DIR / 'runs'
DATA_DIR = SCRIPT_DIR / 'data'

GAMMA = 0.997
TEST_FRACTION = 0.2

# Mapping: task_config name → actual run sub-directory name
TASK_RUN_DIRS = {
    'dmc_walker_walk':  'dmc_proprio_dmc_walker_walk',
    'crafter_reward':   'crafter_reward',
    'atari_pong':       'atari_pong',
    'atari100k_pong':   'atari100k_pong',
    'procgen_coinrun':  'procgen_coinrun',
}


def parse_args():
    ap = argparse.ArgumentParser(description='Compute 3-layer evaluation metrics')
    ap.add_argument('--task', required=True, choices=list(TASK_CONFIGS.keys()))
    ap.add_argument('--batch_size', type=int, default=512,
                    help='Batch size for model inference')
    return ap.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
#  Episode boundary detection
# ═════════════════════════════════════════════════════════════════════════════
def detect_episodes(reward, discount_return, gamma=GAMMA):
    """Detect episode boundaries using Bellman residual.

    Within an episode: DR[i] = reward[i] + γ·DR[i+1]  (exact by construction).
    At episode boundary the relation breaks → residual > 0.

    Returns list of (start_idx, end_idx) tuples.
    """
    n = len(reward)
    episode_ends = []
    for i in range(n - 1):
        expected = reward[i] + gamma * discount_return[i + 1]
        residual = abs(float(discount_return[i]) - float(expected))
        if residual > 1e-3:
            episode_ends.append(i)

    starts = [0] + [e + 1 for e in episode_ends]
    episodes = []
    for i in range(len(starts)):
        s = starts[i]
        e = starts[i + 1] if i + 1 < len(starts) else n
        if e > s:
            episodes.append((s, e))
    return episodes


# ═════════════════════════════════════════════════════════════════════════════
#  A-layer: Predictive Fidelity metrics
# ═════════════════════════════════════════════════════════════════════════════
def compute_1step_error_discrete(student_logits, teacher_actions):
    """1-step Error for discrete tasks: action mismatch rate (↓ better)."""
    student_actions = student_logits.argmax(axis=-1)
    return 1.0 - float((student_actions == teacher_actions).mean())


def compute_1step_error_continuous(student_actions, teacher_actions):
    """1-step Error for continuous tasks: MSE (↓ better)."""
    return float(np.mean((student_actions - teacher_actions) ** 2))


def compute_kstep_rollout_error(model, obs, act, episodes, device,
                                is_discrete, k, batch_size=512):
    """Multi-step Rollout Error (k=1,2,5).

    For each starting position t within an episode, take k consecutive
    teacher states [s_t, s_{t+1}, ..., s_{t+k-1}] and teacher actions
    [a_t, ..., a_{t+k-1}].  Run the student on each of those states
    and compute the CUMULATIVE per-step action error up to step k.

    This measures: if teacher and student start from the same state
    and both follow k steps, how much do their ACTION sequences diverge?

    For discrete:   error = (1/k) Σ_{i=0}^{k-1} 1[student(s_{t+i}) ≠ a_{t+i}]
    For continuous:  error = (1/k) Σ_{i=0}^{k-1} MSE(student(s_{t+i}), a_{t+i})

    Returns: (mean_error, std_error) across all windows.
    """
    # Collect all valid k-step windows within episodes
    window_starts = []
    for ep_s, ep_e in episodes:
        ep_len = ep_e - ep_s
        if ep_len >= k:
            for t in range(ep_s, ep_e - k + 1):
                window_starts.append(t)

    if not window_starts:
        return None, None

    window_starts = np.array(window_starts)
    n_windows = len(window_starts)

    # Build index array: (n_windows, k)
    offsets = np.arange(k)[None, :]  # (1, k)
    indices = window_starts[:, None] + offsets  # (n_windows, k)
    flat_idx = indices.ravel()

    # Batch inference on all needed observations
    flat_obs = obs[flat_idx]
    flat_act = act[flat_idx]

    outputs = []
    for i in range(0, len(flat_obs), batch_size):
        batch = torch.as_tensor(flat_obs[i:i + batch_size]).to(device)
        with torch.no_grad():
            out = model(batch)
        outputs.append(out.cpu().numpy())
    student_out = np.concatenate(outputs, axis=0)  # (n_windows * k, ...)

    # Reshape to (n_windows, k, ...)
    if is_discrete:
        student_acts = student_out.argmax(axis=-1).reshape(n_windows, k)
        teacher_acts = flat_act.reshape(n_windows, k)
        # Per-step mismatch: 1 if different, 0 if same
        per_step_err = (student_acts != teacher_acts).astype(np.float32)
    else:
        student_acts = student_out.reshape(n_windows, k, -1)
        teacher_acts = flat_act.reshape(n_windows, k, -1)
        # Per-step MSE
        per_step_err = ((student_acts - teacher_acts) ** 2).mean(axis=-1)

    # Cumulative error up to step k:  mean over the k steps per window
    # Shape: per_step_err is (n_windows, k)
    window_errors = per_step_err.mean(axis=1)  # (n_windows,)

    return float(window_errors.mean()), float(window_errors.std())


# ═════════════════════════════════════════════════════════════════════════════
#  B-layer: Substitutability metrics (discrete tasks only)
# ═════════════════════════════════════════════════════════════════════════════
def compute_top1_hit(student_logits, teacher_logits):
    """Top-1 Hit Rate: P(argmax_student == argmax_teacher) (↑ better)."""
    t_best = teacher_logits.argmax(axis=-1)
    s_best = student_logits.argmax(axis=-1)
    return float((t_best == s_best).mean())


def compute_ndcg(teacher_logits, student_logits):
    """Ranker NDCG: action-ranking quality vs teacher (↑ better).

    Uses teacher softmax probabilities as relevance scores.
    Fully vectorised over the observation dimension.
    """
    # Teacher probs as relevance (numerically stable softmax)
    t_max = teacher_logits.max(axis=-1, keepdims=True)
    t_exp = np.exp(teacher_logits - t_max)
    t_probs = t_exp / t_exp.sum(axis=-1, keepdims=True)

    _n, k = teacher_logits.shape
    discount = 1.0 / np.log2(np.arange(k) + 2)  # (k,)

    # Rankings
    s_rank = np.argsort(-student_logits, axis=-1)  # student's ranking
    t_rank = np.argsort(-t_probs, axis=-1)          # ideal (teacher) ranking

    # Gather relevance along rankings
    s_rel = np.take_along_axis(t_probs, s_rank, axis=-1)
    t_rel = np.take_along_axis(t_probs, t_rank, axis=-1)

    dcg  = (s_rel * discount).sum(axis=-1)
    idcg = (t_rel * discount).sum(axis=-1)

    ndcg = dcg / (idcg + 1e-10)
    return float(ndcg.mean())


def compute_kendall_tau(teacher_logits, student_logits):
    """Kendall τ: rank correlation between teacher and student (↑ better).

    Vectorised pairwise comparison over actions.
    """
    _n, k = teacher_logits.shape
    idx_i, idx_j = np.triu_indices(k, k=1)

    t_diff = teacher_logits[:, idx_i] - teacher_logits[:, idx_j]
    s_diff = student_logits[:, idx_i] - student_logits[:, idx_j]

    product = t_diff * s_diff
    concordant = (product > 0).sum(axis=1).astype(float)
    discordant = (product < 0).sum(axis=1).astype(float)
    total = concordant + discordant

    taus = np.where(total > 0, (concordant - discordant) / total, 0.0)
    return float(taus.mean())


# ═════════════════════════════════════════════════════════════════════════════
#  Model loading & batched inference
# ═════════════════════════════════════════════════════════════════════════════
def load_student_model(ckpt_path, device):
    """Load student model from checkpoint, return (model, config, score)."""
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    config = ckpt['config']

    kwargs = {}
    if config.obs_type == 'proprio':
        kwargs['obs_mean'] = torch.tensor(ckpt['obs_mean'])
        kwargs['obs_std']  = torch.tensor(ckpt['obs_std'])

    model = StudentPolicy(config, **kwargs).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, config, ckpt.get('score', None), ckpt.get('proprio_keys', None)


@torch.no_grad()
def run_model_batched(model, obs, device, batch_size=512):
    """Run model on observations in batches; return numpy output."""
    outputs = []
    for i in range(0, len(obs), batch_size):
        batch = torch.as_tensor(obs[i:i + batch_size]).to(device)
        out = model(batch)
        outputs.append(out.cpu().numpy())
    return np.concatenate(outputs, axis=0)


# ═════════════════════════════════════════════════════════════════════════════
#  Online evaluation (multi-step rollout in real environment)
# ═════════════════════════════════════════════════════════════════════════════
def make_eval_env(task_cfg):
    """Create eval environment (same wrapper stack as Teacher training)."""
    import embodied

    suite = task_cfg['env_suite']
    if suite == 'dmc':
        from embodied.envs.dmc import DMC
        env = DMC(task_cfg['env_task'], **task_cfg['env_kwargs'])
        for name, space in env.act_space.items():
            if not space.discrete:
                env = embodied.wrappers.NormalizeAction(env, name)
        env = embodied.wrappers.UnifyDtypes(env)
        env = embodied.wrappers.CheckSpaces(env)
        for name, space in env.act_space.items():
            if not space.discrete:
                env = embodied.wrappers.ClipAction(env, name)
    elif suite == 'crafter':
        from embodied.envs.crafter import Crafter
        env = Crafter(task=task_cfg['env_task'], **task_cfg['env_kwargs'])
        env = embodied.wrappers.UnifyDtypes(env)
        env = embodied.wrappers.CheckSpaces(env)
    elif suite == 'procgen':
        from embodied.envs.procgen import ProcGen
        env = ProcGen(task_cfg['env_task'], **task_cfg['env_kwargs'])
        env = embodied.wrappers.UnifyDtypes(env)
        env = embodied.wrappers.CheckSpaces(env)
    elif suite == 'atari':
        from embodied.envs.atari import Atari
        env = Atari(task_cfg['env_task'], **task_cfg['env_kwargs'])
        env = embodied.wrappers.UnifyDtypes(env)
        env = embodied.wrappers.CheckSpaces(env)
    else:
        raise ValueError(f"Unknown env_suite: {suite}")
    return env


def _extract_proprio(obs, proprio_keys):
    parts = []
    for k in proprio_keys:
        v = np.asarray(obs[k], dtype=np.float32).flatten()
        parts.append(v)
    return np.concatenate(parts)


@torch.no_grad()
def run_online_episodes(model, task_cfg, device, num_episodes,
                        proprio_keys=None):
    """Run student in real environment.

    Returns list of per-episode reward lists: [[r0, r1, ...], ...]
    """
    model.eval()
    env = make_eval_env(task_cfg)

    is_proprio    = task_cfg['obs_type'] == 'proprio'
    is_continuous = task_cfg['act_type'] == 'continuous'
    act_dim       = task_cfg['act_dim']
    frame_stack   = task_cfg.get('frame_stack', 1)
    ep_max        = task_cfg['ep_max_steps']

    reset_action = (np.zeros(act_dim, dtype=np.float32) if is_continuous
                    else np.int32(0))

    all_ep_rewards = []

    for ep in range(num_episodes):
        obs = env.step({'action': reset_action, 'reset': np.array(True)})
        obs_f = {k: v for k, v in obs.items() if not k.startswith('log/')}

        frame_buf = None
        if frame_stack > 1 and not is_proprio:
            init_img = np.array(obs_f['image'], dtype=np.uint8)
            frame_buf = collections.deque(
                [init_img] * frame_stack, maxlen=frame_stack)

        ep_rewards = []
        steps = 0

        while not obs_f['is_last'] and steps < ep_max:
            if is_proprio:
                obs_vec = _extract_proprio(obs_f, proprio_keys)
                obs_t = torch.tensor(
                    obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_img = np.array(obs_f['image'], dtype=np.uint8)
                if frame_buf is not None:
                    frame_buf.append(obs_img)
                    obs_img = np.concatenate(list(frame_buf), axis=-1)
                obs_t = torch.as_tensor(obs_img, device=device).unsqueeze(0)

            output = model(obs_t).squeeze(0)

            if is_continuous:
                action = np.clip(output.cpu().numpy(), -1.0, 1.0)
                step_action = action
            else:
                step_action = np.int32(int(output.argmax(dim=-1).item()))

            obs = env.step({'action': step_action, 'reset': np.array(False)})
            obs_f = {k: v for k, v in obs.items() if not k.startswith('log/')}
            ep_rewards.append(float(obs_f['reward']))
            steps += 1

        all_ep_rewards.append(ep_rewards)

    env.close()
    return all_ep_rewards


# ═════════════════════════════════════════════════════════════════════════════
#  Table formatting & output
# ═════════════════════════════════════════════════════════════════════════════
def _fv(v, fmt='.4f'):
    """Format a value; None → '--'."""
    if v is None:
        return '--'
    return f'{v:{fmt}}'


def print_table(task_name, results, is_discrete):
    """Print the evaluation matrix table."""
    if is_discrete:
        col_names = ['score', 'k=1 Err(↓)', 'k=2 Err(↓)', 'k=5 Err(↓)',
                     'NDCG(↑)', 'Top1 Hit(↑)', 'Kendall τ(↑)']
        col_keys  = ['score', 'k1_error', 'k2_error', 'k5_error',
                     'ranker_ndcg', 'top1_hit', 'kendall_tau']
        col_fmts  = ['.1f', '.4f', '.4f', '.4f', '.4f', '.4f', '.4f']
    else:
        col_names = ['score', 'k=1 Err(↓)', 'k=2 Err(↓)', 'k=5 Err(↓)']
        col_keys  = ['score', 'k1_error', 'k2_error', 'k5_error']
        col_fmts  = ['.1f', '.6f', '.6f', '.6f']

    lw = 24                         # label column width
    cw = 14                         # data column width
    ncols = len(col_names)

    hline = '─' * lw + '┼' + ('─' * cw + '┼') * (ncols - 1) + '─' * cw

    print(f"\n{'═' * (lw + ncols * (cw + 1))}")
    print(f"  学生模型评估矩阵 — {task_name}")
    print(f"{'═' * (lw + ncols * (cw + 1))}")

    # Header
    print(f"{'':>{lw}s} │", end='')
    for c in col_names:
        print(f'{c:^{cw}s}│', end='')
    print()
    print(hline)

    # Rows
    for v in VARIANTS:
        r = results.get(v)
        label = VARIANT_LABELS[v]
        print(f' {label:<{lw - 1}s} │', end='')
        if r is None:
            for _ in col_names:
                print(f"{'--':^{cw}s}│", end='')
        else:
            for key, fmt in zip(col_keys, col_fmts):
                val = r.get(key)
                print(f'{_fv(val, fmt):^{cw}s}│', end='')
        print()

    print(hline)

    # ── Relative improvement over baseline ────────────────────────────
    base = results.get('baseline')
    if base is None:
        return

    print(f"\n  相对 Baseline 提升 (↑ for score/NDCG/Hit/τ, ↓ for errors):")
    for v in ['A', 'B', 'AB']:
        r = results.get(v)
        if r is None:
            continue
        parts = [f"  {VARIANT_LABELS[v]:24s}"]
        base_score = base.get('score')
        v_score = r.get('score')
        if base_score and v_score:
            pct = (v_score - base_score) / abs(base_score) * 100
            parts.append(f"score {pct:+.1f}%")
        base_err = base.get('k1_error')
        v_err = r.get('k1_error')
        if base_err is not None and v_err is not None and base_err > 0:
            pct = (base_err - v_err) / base_err * 100
            parts.append(f"1step_err {pct:+.1f}%")
        if is_discrete:
            for key, name in [('top1_hit', 'Top1'), ('ranker_ndcg', 'NDCG'),
                              ('kendall_tau', 'τ')]:
                bv = base.get(key)
                vv = r.get(key)
                if bv is not None and vv is not None and bv > 0:
                    pct = (vv - bv) / bv * 100
                    parts.append(f"{name} {pct:+.1f}%")
        print('  |  '.join(parts))


def print_csv(task_name, results, is_discrete):
    """Print CSV for pasting into a spreadsheet."""
    if is_discrete:
        cols = ['variant', 'score', 'k1_error', 'k2_error', 'k5_error',
                'ranker_ndcg', 'top1_hit', 'kendall_tau']
    else:
        cols = ['variant', 'score', 'k1_error', 'k2_error', 'k5_error']

    print(f"\n--- CSV ({task_name}) ---")
    print(','.join(cols))
    for v in VARIANTS:
        r = results.get(v)
        if r is None:
            print(f"{v}," + ','*(len(cols)-2))
            continue
        vals = [v]
        for c in cols[1:]:
            val = r.get(c)
            vals.append(f'{val}' if val is not None else '')
        print(','.join(vals))


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    task_name = args.task
    task_cfg  = TASK_CONFIGS[task_name]

    is_discrete = task_cfg['act_type'] == 'discrete'
    is_proprio  = task_cfg['obs_type'] == 'proprio'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir_name = TASK_RUN_DIRS.get(task_name, task_name)

    print(f"{'═' * 70}")
    print(f"  Task   : {task_name}")
    print(f"  Type   : {'discrete' if is_discrete else 'continuous'}")
    print(f"  Device : {device}")
    print(f"{'═' * 70}")

    # ── 1. Load expert data ──────────────────────────────────────────────
    data_path = DATA_DIR / f'teacher_data_{task_name}.npz'
    print(f"\nLoading expert data: {data_path}")
    data = np.load(str(data_path), allow_pickle=True)

    obs = data['obs']
    act = data['act']
    reward = data['reward']
    discount_return = data['discount_return']

    teacher_logits = data['act_logits'] if 'act_logits' in data else None
    proprio_keys = (list(data['proprio_keys'])
                    if 'proprio_keys' in data else None)

    n_total = len(obs)
    print(f"  Transitions : {n_total:,}")

    # ── 2. Detect episode boundaries ─────────────────────────────────────
    episodes = detect_episodes(reward, discount_return)
    ep_lens = [e - s for s, e in episodes]
    print(f"  Episodes    : {len(episodes)}")
    print(f"  Ep length   : mean={np.mean(ep_lens):.0f}  "
          f"min={np.min(ep_lens)}  max={np.max(ep_lens)}")

    # Episode stats for k-step windowing
    for k_val in [1, 2, 5]:
        n_windows = sum(max(0, (e - s) - k_val + 1) for s, e in episodes)
        print(f"  k={k_val} windows : {n_windows:,}")

    # ── 3. Held-out test set ─────────────────────────────────────────────
    n_test = max(1, int(n_total * TEST_FRACTION))
    test_idx = np.arange(n_total - n_test, n_total)

    test_obs = obs[test_idx]
    test_act = act[test_idx]
    test_teacher_logits = (teacher_logits[test_idx]
                           if teacher_logits is not None else None)

    print(f"  Test set    : {len(test_idx):,} samples "
          f"(last {TEST_FRACTION * 100:.0f}%)")

    # ── 4. Evaluate each variant ─────────────────────────────────────────
    results = {}

    for variant in VARIANTS:
        ckpt_path = RUNS_DIR / run_dir_name / variant / 'best_student.pth'
        if not ckpt_path.exists():
            print(f"\n  [{variant}] checkpoint not found — skipping")
            continue

        print(f"\n{'─' * 70}")
        print(f"  [{variant}] {ckpt_path}")

        model, config, score, ckpt_proprio_keys = \
            load_student_model(ckpt_path, device)
        pk = (list(ckpt_proprio_keys) if ckpt_proprio_keys is not None
              else proprio_keys)

        r = {'score': score}
        if score is not None:
            print(f"  C层 Score (from training) : {score:.1f}")

        # ── A层: k-step Rollout Error (k=1, 2, 5) ─────────────────
        t0 = time.time()
        for k_val in [1, 2, 5]:
            k_mean, k_std = compute_kstep_rollout_error(
                model, obs, act, episodes, device,
                is_discrete, k=k_val, batch_size=args.batch_size)
            key = f'k{k_val}_error'
            r[key] = k_mean
            if k_mean is not None:
                err_type = 'mismatch' if is_discrete else 'MSE'
                print(f"  A层 k={k_val} Rollout Error ({err_type}) : "
                      f"{k_mean:.4f} ± {k_std:.4f}")
        print(f"  (A-layer k-step: {time.time() - t0:.1f}s)")

        # ── B层: Substitutability (discrete only) ────────────────────
        if is_discrete and test_teacher_logits is not None:
            t0 = time.time()

            # Run model on test set for B-layer metrics
            student_test_out = run_model_batched(
                model, test_obs, device, args.batch_size)

            hit = compute_top1_hit(student_test_out, test_teacher_logits)
            print(f"  B层 Top-1 Hit Rate   : {hit:.4f}")
            r['top1_hit'] = hit

            ndcg = compute_ndcg(test_teacher_logits, student_test_out)
            print(f"  B层 Ranker NDCG      : {ndcg:.4f}")
            r['ranker_ndcg'] = ndcg

            tau = compute_kendall_tau(test_teacher_logits, student_test_out)
            print(f"  B层 Kendall τ        : {tau:.4f}")
            r['kendall_tau'] = tau

            print(f"  (B-layer: {time.time() - t0:.1f}s)")

        results[variant] = r

    # ── 5. Output ────────────────────────────────────────────────────────
    print_table(task_name, results, is_discrete)
    print_csv(task_name, results, is_discrete)

    print(f"\n{'═' * 70}")
    print(f"  Done. Variants evaluated: "
          f"{', '.join(v for v in VARIANTS if v in results)}")
    if 'AB' not in results:
        print(f"  Note: 'Ours (A+B)' not found. Train with --loss_mode AB.")
    print(f"{'═' * 70}")


if __name__ == '__main__':
    main()
