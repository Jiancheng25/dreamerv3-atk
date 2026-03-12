#!/usr/bin/env python3
"""
Collect expert demonstrations from a fully-trained DreamerV3 Teacher agent.

Supports multiple tasks via ``--task``:
  - dmc_walker_walk  : proprioceptive obs (24-dim) + continuous actions (6-dim)
  - crafter_reward   : image obs (64x64x3)        + discrete actions (17 classes)

Usage:
    python collect_data.py --task dmc_walker_walk
    python collect_data.py --task crafter_reward
"""

# ── Environment variables (MUST precede every other import) ──────────────────
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import sys
import pathlib
import time
import collections
import numpy as np
from PIL import Image as PILImage

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent        # dreamerv3-main/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dreamerv3-mini'))

import elements
import ruamel.yaml as yaml
from dreamerv3.main import make_agent, make_env
from task_config import parse_task_arg

# ── Parse task ───────────────────────────────────────────────────────────────
import argparse as _argparse
TASK_NAME, TASK_CFG, _remaining = parse_task_arg()

_ep_parser = _argparse.ArgumentParser(add_help=False)
_ep_parser.add_argument('--num_episodes', type=int, default=None,
                        help='Override num_episodes from task_config')
_ep_args, _ = _ep_parser.parse_known_args(_remaining)

# ── Directories ──────────────────────────────────────────────────────────────
OUT_DIR  = ROOT / 'dreamerv3-mini'
DATA_DIR = OUT_DIR / 'data'
LOG_DIR  = OUT_DIR / 'logs' / 'teacher_eval'

CKPT_DIR     = TASK_CFG['checkpoint_dir']
NUM_EPISODES = _ep_args.num_episodes if _ep_args.num_episodes is not None else TASK_CFG['num_episodes']
MIN_SCORE    = 0.0
EP_MAX_STEPS = TASK_CFG['ep_max_steps']


# ─────────────────────────────────────────────────────────────────────────────
def build_config(task_cfg):
    """Reconstruct the DreamerV3 config using the task's config presets."""
    configs_path = ROOT / 'dreamerv3' / 'configs.yaml'
    raw = elements.Path(str(configs_path)).read()
    configs = yaml.YAML(typ='safe').load(raw)

    preset_names = task_cfg['dreamer_config_presets']
    argv = [
        '--configs', *preset_names,
        '--jax.compute_dtype', 'float32',
        '--jax.prealloc',      'False',
        '--run.debug',         'True',
        '--logdir',            str(LOG_DIR),
    ]

    parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
    return config


def get_proprio_keys(obs_space):
    """Return sorted list of proprioceptive observation keys."""
    exclude = {'reward', 'is_first', 'is_last', 'is_terminal'}
    keys = []
    for k in sorted(obs_space.keys()):
        if k in exclude or k.startswith('log/'):
            continue
        space = obs_space[k]
        if hasattr(space, 'dtype') and space.dtype == np.uint8:
            continue
        keys.append(k)
    return keys


def extract_proprio(obs, keys):
    """Flatten and concatenate selected obs keys into a single float32 vector."""
    parts = []
    for k in keys:
        v = np.asarray(obs[k], dtype=np.float32).flatten()
        parts.append(v)
    return np.concatenate(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics printing helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_proprio_summary(obs_arr, act_arr, proprio_keys, obs_space):
    """Print detailed statistics for proprioceptive + continuous tasks."""
    act_dim = act_arr.shape[1]

    # Per-key dimension offsets
    key_dims = []
    for k in proprio_keys:
        sp = obs_space[k]
        key_dims.append(int(np.prod(sp.shape)) if sp.shape else 1)
    key_offsets = np.cumsum([0] + key_dims)

    obs_mean = obs_arr.mean(axis=0)
    obs_std  = obs_arr.std(axis=0) + 1e-8

    print(f"\n  \u250c\u2500 Observations (Input: {obs_arr.shape[1]}-dim) "
          f"{'─'*37}")
    print(f"  \u2502  Shape          : {obs_arr.shape}")
    print(f"  \u2502  Global mean    : {obs_arr.mean():.6f}")
    print(f"  \u2502  Global std     : {obs_arr.std():.6f}")
    print(f"  \u2502  Global range   : [{obs_arr.min():.4f}, {obs_arr.max():.4f}]")
    print(f"  \u2502")
    print(f"  \u2502  Per-key breakdown:")
    print(f"  \u2502  {'Key':20s}  {'Dims':>4s}  {'Mean':>9s}  {'Std':>9s}  "
          f"{'Min':>9s}  {'Max':>9s}")
    print(f"  \u2502  {'─'*20}  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
    for i, k in enumerate(proprio_keys):
        lo, hi = key_offsets[i], key_offsets[i + 1]
        col = obs_arr[:, lo:hi]
        print(f"  \u2502  {k:20s}  {hi-lo:4d}  {col.mean():9.4f}  {col.std():9.4f}  "
              f"{col.min():9.4f}  {col.max():9.4f}")
    print(f"  \u2514{'─'*68}")

    print(f"\n  \u250c\u2500 Actions (Label: {act_dim}-dim) {'─'*40}")
    print(f"  \u2502  Shape          : {act_arr.shape}")
    print(f"  \u2502  Global mean    : {act_arr.mean():.6f}")
    print(f"  \u2502  Global std     : {act_arr.std():.6f}")
    print(f"  \u2502  Global range   : [{act_arr.min():.4f}, {act_arr.max():.4f}]")
    print(f"  \u2502  Mean |action|  : {np.abs(act_arr).mean():.6f}")
    print(f"  \u2502")
    print(f"  \u2502  Per-dimension breakdown:")
    abs_mu = '\u2502\u03bc\u2502'
    print(f"  \u2502  {'Dim':>4s}  {'Mean':>9s}  {'Std':>9s}  {'Min':>9s}  "
          f"{'Max':>9s}  {abs_mu:>9s}  {'Sat%':>9s}")
    print(f"  \u2502  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  "
          f"{'─'*9}  {'─'*9}")
    for d in range(act_dim):
        col = act_arr[:, d]
        saturated = ((np.abs(col) > 0.99).sum() / len(col)) * 100
        print(f"  \u2502  {d:4d}  {col.mean():9.4f}  {col.std():9.4f}  "
              f"{col.min():9.4f}  {col.max():9.4f}  "
              f"{np.abs(col).mean():9.4f}  {saturated:8.1f}%")
    print(f"  \u2514{'─'*68}")

    print(f"\n  \u250c\u2500 Normalisation Stats (saved for Student) {'─'*25}")
    print(f"  \u2502  obs_mean : "
          f"{np.array2string(obs_mean, precision=4, separator=', ')}")
    print(f"  \u2502  obs_std  : "
          f"{np.array2string(obs_std,  precision=4, separator=', ')}")
    print(f"  \u2514{'─'*68}")


def print_image_summary(obs_arr, act_arr, act_dim):
    """Print detailed statistics for image + discrete tasks."""
    print(f"\n  \u250c\u2500 Observations (Image) {'─'*47}")
    print(f"  \u2502  Shape          : {obs_arr.shape}")
    print(f"  \u2502  Dtype          : {obs_arr.dtype}")
    print(f"  \u2502  Pixel range    : [{obs_arr.min()}, {obs_arr.max()}]")
    for ch, name in enumerate(['R', 'G', 'B'] if obs_arr.shape[-1] == 3
                               else ['Gray'] if obs_arr.shape[-1] == 1
                               else [f'Ch{i}' for i in range(obs_arr.shape[-1])]):
        ch_data = obs_arr[..., ch].astype(np.float32)
        print(f"  \u2502  Channel {name}      : "
              f"mean={ch_data.mean():.1f}  std={ch_data.std():.1f}  "
              f"min={ch_data.min():.0f}  max={ch_data.max():.0f}")
    print(f"  \u2514{'─'*68}")

    print(f"\n  \u250c\u2500 Actions (Discrete, {act_dim} classes) {'─'*37}")
    print(f"  \u2502  Shape          : {act_arr.shape}")
    print(f"  \u2502  Dtype          : {act_arr.dtype}")
    print(f"  \u2502  Unique actions : {len(np.unique(act_arr))}")
    print(f"  \u2502")
    print(f"  \u2502  Action distribution:")
    counts = np.bincount(act_arr, minlength=act_dim)
    total = len(act_arr)
    for a in range(act_dim):
        pct = counts[a] / total * 100
        bar = '#' * int(pct / 2)
        print(f"  \u2502  {a:3d}  {counts[a]:7d}  ({pct:5.1f}%)  {bar}")
    print(f"  \u2514{'─'*68}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for d in [DATA_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Config ───────────────────────────────────────────────────────────
    config = build_config(TASK_CFG)
    print(f"Task             : {config.task}  (--task {TASK_NAME})")
    print(f"Obs type         : {TASK_CFG['obs_type']}")
    print(f"Act type         : {TASK_CFG['act_type']} (dim={TASK_CFG['act_dim']})")
    print(f"jax.compute_dtype: {config.jax.compute_dtype}")

    # ── Agent & environment ──────────────────────────────────────────────
    print("\nCreating agent and environment ...")
    agent = make_agent(config)
    env   = make_env(config, 0)

    # ── Load checkpoint ──────────────────────────────────────────────────
    print(f"Loading checkpoint from: {CKPT_DIR}")
    cp = elements.Checkpoint(str(CKPT_DIR))
    cp.agent = agent
    # Temporarily register numpy._core shim for checkpoints pickled with
    # NumPy 2.x (when running under NumPy 1.x).
    _np_shims = {}
    if not hasattr(np, '_core'):
        for name in ['numpy._core', 'numpy._core.multiarray']:
            _np_shims[name] = sys.modules.get(name)
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
    cp.load(keys=['agent'])
    # Remove shims after loading
    for name, orig in _np_shims.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig

    # ── Detect observation / action info ─────────────────────────────────
    obs_space = {k: v for k, v in env.obs_space.items()
                 if not k.startswith('log/')}

    is_proprio = TASK_CFG['obs_type'] == 'proprio'
    is_continuous = TASK_CFG['act_type'] == 'continuous'
    act_dim = TASK_CFG['act_dim']

    proprio_keys = None
    obs_dim = None
    if is_proprio:
        proprio_keys = get_proprio_keys(obs_space)
        obs_dim = sum(
            int(np.prod(obs_space[k].shape)) if obs_space[k].shape else 1
            for k in proprio_keys
        )
        print(f"\nProprioceptive keys ({len(proprio_keys)}):")
        for k in proprio_keys:
            sp = obs_space[k]
            print(f"  {k:20s}  dtype={sp.dtype}  shape={sp.shape}")
        print(f"Total obs dim : {obs_dim}")
    else:
        print(f"\nImage obs     : {obs_space['image'].shape}  "
              f"dtype={obs_space['image'].dtype}")

    print(f"Action        : {'continuous' if is_continuous else 'discrete'} "
          f"(dim={act_dim})")

    # ── Initialise policy carry ──────────────────────────────────────────
    carry = agent.init_policy(1)

    # ── Reset action ─────────────────────────────────────────────────────
    if is_continuous:
        reset_action = np.zeros(act_dim, dtype=np.float32)
    else:
        reset_action = np.int32(0)

    # ── Frame stacking config ─────────────────────────────────────────────
    frame_stack = TASK_CFG.get('frame_stack', 1)
    if frame_stack > 1 and not is_proprio:
        print(f"Frame stack   : {frame_stack}")

    # ── Collection loop ──────────────────────────────────────────────────
    all_obs  = []
    all_acts = []
    all_rewards = []         # per-step reward
    all_returns = []         # discounted return (computed per episode)
    all_logits  = []         # teacher action logits  (discrete tasks)
    all_means   = []         # teacher action mean    (continuous tasks)
    all_stddevs = []         # teacher action stddev  (continuous tasks)
    episode_scores = []
    episode_lengths = []

    t0 = time.time()

    print(f"\nCollecting {NUM_EPISODES} expert episodes "
          f"(max {EP_MAX_STEPS} steps/ep) ...")
    print(f"{'─'*70}")

    for ep in range(NUM_EPISODES):
        obs = env.step({'action': reset_action, 'reset': np.array(True)})
        obs_f = {k: v for k, v in obs.items() if not k.startswith('log/')}

        ep_obs  = []
        ep_acts = []
        ep_rewards = []
        ep_logits  = []
        ep_means   = []
        ep_stddevs = []
        score   = 0.0
        steps   = 0

        # Initialise frame buffer for stacking (image tasks only)
        frame_buf = None
        if frame_stack > 1 and not is_proprio:
            init_img = np.array(obs_f['image'], dtype=np.uint8)
            student_size = tuple(TASK_CFG['env_kwargs'].get(
                'size', init_img.shape[:2]))
            n_ch = init_img.shape[2] if init_img.ndim == 3 else 1
            zero_frame = np.zeros((*student_size, n_ch), dtype=np.uint8)
            frame_buf = collections.deque(
                [zero_frame] * frame_stack, maxlen=frame_stack)

        while not obs_f['is_last'] and steps < EP_MAX_STEPS:
            # ── Teacher policy ────────────────────────────────────────
            batched = {k: v[None] for k, v in obs_f.items()}
            carry, acts, policy_out = agent.policy(
                carry, batched, mode='eval')

            # ── Extract teacher distribution params ────────────────────
            if is_continuous:
                act_mean   = np.array(
                    policy_out['mean/action'][0], dtype=np.float32)
                act_stddev = np.array(
                    policy_out['stddev/action'][0], dtype=np.float32)
            else:
                act_logits = np.array(
                    policy_out['logits/action'][0], dtype=np.float32)

            # ── Extract action ────────────────────────────────────────
            if is_continuous:
                action = np.array(acts['action'][0], dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)
                step_action = action
                stored_action = action.copy()
            else:
                action_int = int(acts['action'][0])
                step_action = np.int32(action_int)
                stored_action = action_int

            # ── Extract observation ───────────────────────────────────
            if is_proprio:
                stored_obs = extract_proprio(obs_f, proprio_keys)
            else:
                stored_obs = np.array(obs_f['image'], dtype=np.uint8)
                # Resize if teacher image size differs from student expected size
                student_size = tuple(TASK_CFG['env_kwargs'].get(
                    'size', stored_obs.shape[:2]))
                if (stored_obs.shape[0], stored_obs.shape[1]) != student_size:
                    n_ch = stored_obs.shape[2]
                    if n_ch == 1:
                        pil_img = PILImage.fromarray(
                            stored_obs[:, :, 0], mode='L')
                    else:
                        pil_img = PILImage.fromarray(stored_obs)
                    pil_img = pil_img.resize(
                        (student_size[1], student_size[0]),
                        PILImage.BILINEAR)
                    resized = np.array(pil_img)
                    if resized.ndim == 2:
                        resized = resized[:, :, None]
                    stored_obs = resized

                # ── Frame stacking ───────────────────────────────────
                if frame_buf is not None:
                    frame_buf.append(stored_obs)
                    stored_obs = np.concatenate(list(frame_buf), axis=-1)

            ep_obs.append(stored_obs)
            ep_acts.append(stored_action)
            if is_continuous:
                ep_means.append(act_mean)
                ep_stddevs.append(act_stddev)
            else:
                ep_logits.append(act_logits)
            steps += 1

            # ── Step environment ──────────────────────────────────────
            obs = env.step({'action': step_action, 'reset': np.array(False)})
            obs_f = {k: v for k, v in obs.items() if not k.startswith('log/')}
            step_reward = float(obs_f['reward'])
            ep_rewards.append(step_reward)
            score += step_reward

        # ── Compute discounted returns for this episode ──────────────
        gamma = 0.997
        ep_returns = np.zeros(len(ep_rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(ep_rewards))):
            running = ep_rewards[t] + gamma * running
            ep_returns[t] = running

        episode_scores.append(score)
        episode_lengths.append(steps)
        all_obs.extend(ep_obs)
        all_acts.extend(ep_acts)
        all_rewards.extend(ep_rewards)
        all_returns.extend(ep_returns.tolist())
        if is_continuous:
            all_means.extend(ep_means)
            all_stddevs.extend(ep_stddevs)
        else:
            all_logits.extend(ep_logits)

        elapsed = time.time() - t0
        print(f"  Ep {ep+1:3d}/{NUM_EPISODES}  |  score={score:8.2f}  |  "
              f"steps={steps:5d}  |  total_trans={len(all_obs):,}  |  "
              f"elapsed={elapsed:.0f}s")

    env.close()
    elapsed = time.time() - t0
    print(f"{'─'*70}")

    # ── Convert to arrays ─────────────────────────────────────────────────
    reward_arr  = np.array(all_rewards, dtype=np.float32)
    return_arr  = np.array(all_returns, dtype=np.float32)

    if is_proprio:
        obs_arr = np.array(all_obs, dtype=np.float32)
        act_arr = np.array(all_acts, dtype=np.float32)
        obs_mean = obs_arr.mean(axis=0)
        obs_std  = obs_arr.std(axis=0) + 1e-8
        mean_arr   = np.array(all_means,   dtype=np.float32)
        stddev_arr = np.array(all_stddevs, dtype=np.float32)
    else:
        obs_arr = np.array(all_obs, dtype=np.uint8)
        act_arr = np.array(all_acts, dtype=np.int32)
        obs_mean = None
        obs_std  = None
        logits_arr = np.array(all_logits, dtype=np.float32)

    # ── Save ──────────────────────────────────────────────────────────────
    _ep_suffix = f'_{NUM_EPISODES}eps' if _ep_args.num_episodes is not None else ''
    save_path = str(DATA_DIR / f'teacher_data_{TASK_NAME}{_ep_suffix}.npz')
    save_dict = dict(
        obs=obs_arr,
        act=act_arr,
        reward=reward_arr,
        discount_return=return_arr,
        task=TASK_NAME,
        obs_type=TASK_CFG['obs_type'],
        act_type=TASK_CFG['act_type'],
    )
    if is_proprio:
        save_dict['obs_mean'] = obs_mean
        save_dict['obs_std']  = obs_std
        save_dict['proprio_keys'] = np.array(proprio_keys)
        save_dict['act_mean']   = mean_arr
        save_dict['act_stddev'] = stddev_arr
    else:
        save_dict['act_logits'] = logits_arr
    np.savez_compressed(save_path, **save_dict)

    # ── Summary ───────────────────────────────────────────────────────────
    scores = np.array(episode_scores)
    lengths = np.array(episode_lengths)

    print(f"\n{'═'*70}")
    print(f"  COLLECTION SUMMARY  ({TASK_NAME})")
    print(f"{'═'*70}")
    print(f"  Time elapsed      : {elapsed:.1f}s")
    print(f"  Episodes          : {NUM_EPISODES}")
    print(f"  Total transitions : {len(obs_arr):,}")

    print(f"\n  ┌─ Episode Scores {'─'*51}")
    print(f"  │  Mean ± Std     : {scores.mean():.2f} ± {scores.std():.2f}")
    print(f"  │  Median         : {np.median(scores):.2f}")
    print(f"  │  Min / Max      : {scores.min():.2f} / {scores.max():.2f}")
    print(f"  │  Ep length      : {lengths.mean():.0f} "
          f"(min {lengths.min()}, max {lengths.max()})")
    print(f"  └{'─'*68}")

    print(f"\n  ┌─ Teacher Extra Info {'─'*46}")
    print(f"  │  Reward          : shape={reward_arr.shape}  "
          f"mean={reward_arr.mean():.4f}  std={reward_arr.std():.4f}")
    print(f"  │  Disc. return    : shape={return_arr.shape}  "
          f"mean={return_arr.mean():.4f}  std={return_arr.std():.4f}  "
          f"gamma={0.997}")
    if is_continuous:
        print(f"  │  Act mean       : shape={mean_arr.shape}  "
              f"range=[{mean_arr.min():.4f}, {mean_arr.max():.4f}]")
        print(f"  │  Act stddev     : shape={stddev_arr.shape}  "
              f"range=[{stddev_arr.min():.4f}, {stddev_arr.max():.4f}]")
    else:
        print(f"  │  Act logits     : shape={logits_arr.shape}  "
              f"range=[{logits_arr.min():.4f}, {logits_arr.max():.4f}]")
    print(f"  └{'─'*68}")

    if is_proprio:
        print_proprio_summary(obs_arr, act_arr, proprio_keys, obs_space)
    else:
        print_image_summary(obs_arr, act_arr, act_dim)

    file_size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"\n  Saved to: {save_path}  ({file_size_mb:.1f} MB)")
    print(f"{'═'*70}")


if __name__ == '__main__':
    main()
