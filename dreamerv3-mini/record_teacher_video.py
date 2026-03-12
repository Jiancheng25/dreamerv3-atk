#!/usr/bin/env python3
"""
Record high-resolution (256x256) teacher evaluation videos.

Uses the same rendering pipeline as student eval so that
teacher / baseline / ours frames have identical visual quality.

Usage:
    python dreamerv3-mini/record_teacher_video.py --task dmc_walker_walk
    python dreamerv3-mini/record_teacher_video.py --task atari_pong
"""

# ── Environment variables ────────────────────────────────────────────────────
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
import av
from PIL import Image as PILImage

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dreamerv3-mini'))

import elements
import ruamel.yaml as yaml
from dreamerv3.main import make_agent, make_env
from task_config import parse_task_arg

TASK_NAME, TASK_CFG, _remaining = parse_task_arg()

OUT_DIR  = ROOT / 'dreamerv3-mini'
LOG_DIR  = OUT_DIR / 'logs' / 'teacher_eval'
CKPT_DIR = TASK_CFG['checkpoint_dir']


def build_config(task_cfg):
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


# ── Rendering helpers (same as train_student.py) ─────────────────────────────

def get_dmc_physics(env):
    while hasattr(env, 'env'):
        env = env.env
    return env._dmenv.physics


def render_frame_physics(physics, size=(256, 256), camera_id=0):
    return physics.render(*size, camera_id=camera_id)


def upscale_image(img, target_size=(256, 256)):
    h, w = img.shape[:2]
    th, tw = target_size
    # Handle grayscale (H,W,1) -> squeeze to (H,W) for PIL
    if img.ndim == 3 and img.shape[2] == 1:
        img_2d = img[:, :, 0]
        pil_img = PILImage.fromarray(img_2d, mode='L')
    elif img.ndim == 2:
        pil_img = PILImage.fromarray(img, mode='L')
    else:
        pil_img = PILImage.fromarray(img)
    pil_img = pil_img.resize((tw, th), PILImage.LANCZOS)
    # Always convert to RGB for video
    pil_img = pil_img.convert('RGB')
    return np.array(pil_img)


def save_mp4(frames, path, fps=30):
    T, H, W, _ = frames.shape
    container = av.open(path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = 'yuv420p'
    for t in range(T):
        frame = av.VideoFrame.from_ndarray(frames[t], format='rgb24')
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


# ── Eval environment (same wrapper stack as student eval) ────────────────────

def make_eval_env(task_cfg):
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
    elif suite == 'atari':
        from embodied.envs.atari import Atari
        env = Atari(task_cfg['env_task'], **task_cfg['env_kwargs'])
        env = embodied.wrappers.UnifyDtypes(env)
        env = embodied.wrappers.CheckSpaces(env)
    else:
        raise ValueError(f"Unknown suite: {suite}")
    return env


def main():
    os.makedirs(str(LOG_DIR), exist_ok=True)

    task_cfg = TASK_CFG
    is_proprio = task_cfg['obs_type'] == 'proprio'
    is_continuous = task_cfg['act_type'] == 'continuous'
    act_dim = task_cfg['act_dim']
    render_source = task_cfg['render_source']
    render_size = task_cfg['render_size']
    video_fps = task_cfg['video_fps']
    frame_stack = task_cfg.get('frame_stack', 1)

    print(f"Task          : {TASK_NAME}")
    print(f"Obs type      : {task_cfg['obs_type']}")
    print(f"Act type      : {task_cfg['act_type']}")
    print(f"Render source : {render_source}")
    print(f"Render size   : {render_size}")

    # ── Build config & load teacher ──────────────────────────────────────
    config = build_config(task_cfg)
    print("\nCreating agent & environment ...")
    agent = make_agent(config)
    env_dreamer = make_env(config, 0)   # DreamerV3 env (teacher acts here)

    print(f"Loading checkpoint from: {CKPT_DIR}")
    cp = elements.Checkpoint(str(CKPT_DIR))
    cp.agent = agent
    _np_shims = {}
    if not hasattr(np, '_core'):
        for name in ['numpy._core', 'numpy._core.multiarray']:
            _np_shims[name] = sys.modules.get(name)
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
    cp.load(keys=['agent'])
    for name, orig in _np_shims.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig

    # ── Detect obs keys ──────────────────────────────────────────────────
    obs_space = {k: v for k, v in env_dreamer.obs_space.items()
                 if not k.startswith('log/')}

    # ── For physics rendering (DMC): use separate eval env for high-res ──
    # ── For image obs (Atari): use single env (DreamerV3 env) to stay in sync ──
    eval_env = None
    physics = None
    if render_source == 'physics':
        eval_env = make_eval_env(task_cfg)
        physics = get_dmc_physics(eval_env)

    # ── Reset action ─────────────────────────────────────────────────────
    if is_continuous:
        reset_action = np.zeros(act_dim, dtype=np.float32)
    else:
        reset_action = np.int32(0)

    # ── Run 1 episode ────────────────────────────────────────────────────
    carry = agent.init_policy(1)

    # Reset DreamerV3 env
    obs_d = env_dreamer.step({'action': reset_action, 'reset': np.array(True)})
    obs_d_f = {k: v for k, v in obs_d.items() if not k.startswith('log/')}

    # If physics rendering, also reset eval env
    if eval_env is not None:
        eval_env.step({'action': reset_action, 'reset': np.array(True)})

    video_frames = []
    score = 0.0
    steps = 0
    max_steps = task_cfg['ep_max_steps']

    # Record initial frame
    if render_source == 'physics':
        video_frames.append(render_frame_physics(physics, render_size))
    else:
        video_frames.append(
            upscale_image(np.array(obs_d_f['image'], dtype=np.uint8),
                          render_size))

    print(f"\nRunning teacher episode (max {max_steps} steps) ...")
    t0 = time.time()

    while not obs_d_f['is_last'] and steps < max_steps:
        # Teacher policy on DreamerV3 env observations
        batched = {k: v[None] for k, v in obs_d_f.items()}
        carry, acts, _ = agent.policy(carry, batched, mode='eval')

        # Extract action
        if is_continuous:
            action = np.array(acts['action'][0], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            step_action = action
        else:
            action_int = int(acts['action'][0])
            step_action = np.int32(action_int)

        # Step DreamerV3 env
        obs_d = env_dreamer.step({'action': step_action,
                                   'reset': np.array(False)})
        obs_d_f = {k: v for k, v in obs_d.items() if not k.startswith('log/')}

        # For physics rendering, also step eval env to keep physics in sync
        if eval_env is not None:
            eval_env.step({'action': step_action, 'reset': np.array(False)})

        score += float(obs_d_f['reward'])
        steps += 1

        # Record frame
        if render_source == 'physics':
            video_frames.append(render_frame_physics(physics, render_size))
        else:
            video_frames.append(
                upscale_image(np.array(obs_d_f['image'], dtype=np.uint8),
                              render_size))

    elapsed = time.time() - t0
    print(f"Episode done: {steps} steps, score={score:.1f} ({elapsed:.1f}s)")

    # ── Save video ───────────────────────────────────────────────────────
    video = np.stack(video_frames, axis=0)
    out_dir = OUT_DIR / 'runs' / TASK_NAME.replace('dmc_walker_walk',
              'dmc_proprio_dmc_walker_walk')
    os.makedirs(str(out_dir / 'teacher_hires'), exist_ok=True)
    out_path = str(out_dir / 'teacher_hires' / 'teacher_eval.mp4')
    save_mp4(video, out_path, fps=video_fps)
    print(f"Saved: {out_path}  ({video.shape})")

    env_dreamer.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == '__main__':
    main()
