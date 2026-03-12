#!/usr/bin/env python3
"""
Train the Student policy via behaviour cloning on expert demonstrations,
with periodic evaluation on the real environment.

Supports multiple tasks via ``--task``:
  - dmc_walker_walk  : proprio obs + continuous actions (MSE loss)
  - crafter_reward   : image obs   + discrete  actions (CE  loss)

Only the model that achieves the highest *environment eval score* is saved
(not the model with the lowest training loss).

Each training run is stored under a timestamped directory:
  dreamerv3-mini/runs/<task_name>/<YYYYMMDD_HHMMSS>/
    ├── train.log          # full training text log
    ├── best_student.pth   # best-score model checkpoint
    ├── videos/            # per-eval MP4 videos
    │   ├── eval_epoch_005.mp4
    │   └── ...
    └── tb/                # TensorBoard events

Usage:
    python train_student.py --task dmc_walker_walk
    python train_student.py --task crafter_reward
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
import logging
import datetime
import collections
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ── Path setup (must come before task_config import) ────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent         # dreamerv3-mini/
ROOT       = SCRIPT_DIR.parent                               # dreamerv3-main/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from task_config import parse_task_arg

# ── Parse task early (before heavy imports, so we can skip DMC for Crafter) ──
TASK_NAME, TASK_CFG, REMAINING_ARGV = parse_task_arg()

# dm_control MUST be imported before torch, otherwise torch's CUDA init
# loads GL symbols that break OSMesa.  Only needed for DMC tasks.
if TASK_CFG['env_suite'] == 'dmc':
    import embodied.envs.dmc  # noqa: E402,F401  -- force early OSMesa init

import numpy as np
import av
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from student_model import StudentConfig, StudentPolicy

# ── Extra CLI arguments (loss ablation) ──────────────────────────────────────
import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument('--loss_mode', default='baseline',
                choices=['baseline', 'A', 'B', 'AB', 'OBD'])
_p.add_argument('--run_name', default=None, type=str)
_p.add_argument('--output_dir', default=None, type=str,
                help='Override output directory (absolute or relative to cwd)')
_p.add_argument('--lambda_base', default=1.0, type=float,
                help='Weight for base loss (L_base)')
_p.add_argument('--lambda_a', default=0.1, type=float,
                help='Weight for NLL loss (variant A)')
_p.add_argument('--lambda_b', default=0.3, type=float,
                help='Weight for value-weighted loss (variant B)')
# ── Model / training overrides (optional) ─────────────────────────────────────
_p.add_argument('--student_hidden', default=None, type=int,
                help='Override student hidden dim')
_p.add_argument('--student_blocks', default=None, type=int,
                help='Override student residual blocks')
_p.add_argument('--student_dropout', default=None, type=float,
                help='Override student dropout')
_p.add_argument('--num_epochs', default=None, type=int,
                help='Override number of training epochs')
_p.add_argument('--batch_size_override', default=None, type=int,
                help='Override batch size')
_p.add_argument('--eval_episodes_override', default=None, type=int,
                help='Override eval episodes')
_p.add_argument('--data_fraction', default=1.0, type=float,
                help='Fraction of expert data to use (0.0-1.0). '
                     'Values < 1.0 create a data-limited regime.')
_p.add_argument('--data_path', default=None, type=str,
                help='Override path to teacher data .npz file. '
                     'If not set, default is data/teacher_data_{task}.npz')
_p.add_argument('--value_weight_temp', default=1.0, type=float,
                help='Temperature for value-weight sharpening (higher = sharper)')
_p.add_argument('--kd_temperature', default=1.0, type=float,
                help='Softmax temperature for soft-label KL distillation in Loss A '
                     '(discrete tasks). >1 softens teacher distribution.')
_p.add_argument('--seed', default=0, type=int,
                help='Random seed for reproducibility')
_EXTRA = _p.parse_args(REMAINING_ARGV)
LOSS_MODE    = _EXTRA.loss_mode
RUN_NAME     = _EXTRA.run_name
LAMBDA_BASE  = _EXTRA.lambda_base
LAMBDA_A     = _EXTRA.lambda_a
LAMBDA_B     = _EXTRA.lambda_b

# ── Directories ─────────────────────────────────────────────────────────────
OUT_DIR  = ROOT / 'dreamerv3-mini'
DATA_DIR = OUT_DIR / 'data'
RUNS_DIR = OUT_DIR / 'runs'

SEED = _EXTRA.seed


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(log_file: pathlib.Path):
    """Create a logger that writes to both stdout and a log file."""
    logger = logging.getLogger('student_train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(str(log_file), mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ExpertDataset(Dataset):
    """Wrap expert (obs, act) pairs for DataLoader.

    - proprio + continuous : obs float32, act float32
    - image   + discrete   : obs uint8 (model handles /255), act int64

    Optional extra fields for loss ablation:
      teacher_mean, teacher_stddev (continuous), discount_return,
      teacher_logits (discrete: raw logits of teacher policy, shape (n, act_dim))
    """

    def __init__(self, obs: np.ndarray, act: np.ndarray,
                 obs_type: str = 'proprio', act_type: str = 'continuous',
                 teacher_mean: np.ndarray = None,
                 teacher_stddev: np.ndarray = None,
                 discount_return: np.ndarray = None,
                 teacher_logits: np.ndarray = None):
        if obs_type == 'proprio':
            self.obs = torch.as_tensor(obs, dtype=torch.float32)
        else:
            # Keep uint8; ImageEncoder.forward handles conversion
            self.obs = torch.as_tensor(obs)

        if act_type == 'continuous':
            self.act = torch.as_tensor(act, dtype=torch.float32)
        else:
            self.act = torch.as_tensor(act, dtype=torch.long)

        n = len(obs)
        act_d = act.shape[1] if act.ndim > 1 else 1
        self.teacher_mean = torch.as_tensor(
            teacher_mean if teacher_mean is not None
            else np.zeros((n, act_d), dtype=np.float32),
            dtype=torch.float32)
        self.teacher_stddev = torch.as_tensor(
            teacher_stddev if teacher_stddev is not None
            else np.ones((n, act_d), dtype=np.float32),
            dtype=torch.float32)
        self.discount_return = torch.as_tensor(
            discount_return if discount_return is not None
            else np.ones(n, dtype=np.float32),
            dtype=torch.float32)
        # teacher_logits: for discrete tasks (shape n x act_dim).
        # Used by Loss A (soft-label KL distillation).
        # For continuous tasks, stored as zeros placeholder.
        if teacher_logits is not None:
            tlog = teacher_logits
        else:
            log_d = act.shape[1] if act.ndim > 1 else act_d
            tlog  = np.zeros((n, log_d), dtype=np.float32)
        self.teacher_logits = torch.as_tensor(tlog, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (self.obs[idx], self.act[idx],
                self.teacher_mean[idx], self.teacher_stddev[idx],
                self.discount_return[idx], self.teacher_logits[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers (no JAX dependency)
# ─────────────────────────────────────────────────────────────────────────────
def make_eval_env(task_cfg):
    """Create an evaluation environment with the same wrapper stack
    used during Teacher training."""
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


def get_dmc_physics(env):
    """Traverse the wrapper chain to reach the base DMC env's physics."""
    while hasattr(env, 'env'):
        env = env.env
    return env._dmenv.physics


def render_frame_physics(physics, size=(256, 256), camera_id=0):
    """Render a single RGB frame from MuJoCo physics engine."""
    return physics.render(*size, camera_id=camera_id)


def upscale_image(img, target_size=(256, 256)):
    """Upscale (H, W, C) uint8 image to target_size via nearest-neighbor.
    Converts single-channel grayscale to 3-channel RGB for video encoding."""
    h, w = img.shape[:2]
    th, tw = target_size
    fy, fx = th // h, tw // w
    result = img.repeat(fy, axis=0).repeat(fx, axis=1)
    if result.ndim == 3 and result.shape[2] == 1:
        result = np.repeat(result, 3, axis=2)
    return result


def save_mp4(frames: np.ndarray, path: str, fps: int = 30):
    """Encode an (T, H, W, 3) uint8 array to an MP4 file using PyAV."""
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


def extract_proprio(obs: dict, proprio_keys: list) -> np.ndarray:
    parts = []
    for k in proprio_keys:
        v = np.asarray(obs[k], dtype=np.float32).flatten()
        parts.append(v)
    return np.concatenate(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, task_cfg, device, proprio_keys=None,
             num_episodes=5, record_video=True):
    """Run the Student policy on the real environment and return scores.

    Handles both proprio/continuous and image/discrete modalities.
    If record_video, renders frames for the *first* episode.
    """
    model.eval()
    env = make_eval_env(task_cfg)

    is_proprio   = task_cfg['obs_type'] == 'proprio'
    is_continuous = task_cfg['act_type'] == 'continuous'
    act_dim       = task_cfg['act_dim']
    render_source = task_cfg['render_source']
    render_size   = task_cfg['render_size']
    frame_stack   = task_cfg.get('frame_stack', 1)

    # DMC physics handle (only for physics rendering)
    physics = None
    if record_video and render_source == 'physics':
        physics = get_dmc_physics(env)

    # Reset action
    if is_continuous:
        reset_action = np.zeros(act_dim, dtype=np.float32)
    else:
        reset_action = np.int32(0)

    scores = []
    video_frames = []

    for ep in range(num_episodes):
        obs = env.step({'action': reset_action, 'reset': np.array(True)})
        obs_f = {k: v for k, v in obs.items() if not k.startswith('log/')}
        score = 0.0

        # Initialise frame buffer for stacking
        frame_buf = None
        if frame_stack > 1 and not is_proprio:
            init_img = np.array(obs_f['image'], dtype=np.uint8)
            frame_buf = collections.deque(
                [init_img] * frame_stack, maxlen=frame_stack)

        # Render initial frame
        if record_video and ep == 0:
            if render_source == 'physics':
                video_frames.append(render_frame_physics(physics, render_size))
            else:
                video_frames.append(
                    upscale_image(np.array(obs_f['image'], dtype=np.uint8),
                                 render_size))

        while not obs_f['is_last']:
            # ── Build observation tensor ────────────────────────────────
            if is_proprio:
                obs_vec = extract_proprio(obs_f, proprio_keys)
                obs_t = torch.tensor(
                    obs_vec, dtype=torch.float32, device=device,
                ).unsqueeze(0)
            else:
                obs_img = np.array(obs_f['image'], dtype=np.uint8)
                if frame_buf is not None:
                    frame_buf.append(obs_img)
                    obs_img = np.concatenate(list(frame_buf), axis=-1)
                obs_t = torch.as_tensor(obs_img, device=device).unsqueeze(0)

            # ── Model inference ─────────────────────────────────────────
            output = model(obs_t).squeeze(0)

            # ── Extract action ──────────────────────────────────────────
            if is_continuous:
                action = output.cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
                step_action = action
            else:
                action_int = int(output.argmax(dim=-1).item())
                step_action = np.int32(action_int)

            # ── Step environment ────────────────────────────────────────
            obs = env.step({'action': step_action, 'reset': np.array(False)})
            obs_f = {k: v for k, v in obs.items() if not k.startswith('log/')}
            score += float(obs_f['reward'])

            # ── Record video frame ──────────────────────────────────────
            if record_video and ep == 0:
                if render_source == 'physics':
                    video_frames.append(
                        render_frame_physics(physics, render_size))
                else:
                    video_frames.append(
                        upscale_image(
                            np.array(obs_f['image'], dtype=np.uint8),
                            render_size))

        scores.append(score)

    env.close()
    model.train()

    video = np.stack(video_frames, axis=0) if video_frames else None
    return float(np.mean(scores)), float(np.std(scores)), scores, video


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    task_cfg = TASK_CFG

    # ── Read hyper-parameters from task config ─────────────────────────
    BATCH_SIZE    = task_cfg['batch_size']
    NUM_EPOCHS    = task_cfg['num_epochs']
    LR            = task_cfg['lr']
    WEIGHT_DECAY  = task_cfg['weight_decay']
    GRAD_CLIP     = task_cfg['grad_clip']
    EVAL_EVERY    = task_cfg['eval_every']
    EVAL_EPISODES = task_cfg['eval_episodes']
    VIDEO_FPS     = task_cfg['video_fps']

    # ── Apply CLI overrides ──────────────────────────────────────────
    if _EXTRA.student_hidden is not None:
        task_cfg['student_hidden'] = _EXTRA.student_hidden
    if _EXTRA.student_blocks is not None:
        task_cfg['student_blocks'] = _EXTRA.student_blocks
    if _EXTRA.student_dropout is not None:
        task_cfg['student_dropout'] = _EXTRA.student_dropout
    if _EXTRA.num_epochs is not None:
        NUM_EPOCHS = _EXTRA.num_epochs
    if _EXTRA.batch_size_override is not None:
        BATCH_SIZE = _EXTRA.batch_size_override
    if _EXTRA.eval_episodes_override is not None:
        EVAL_EPISODES = _EXTRA.eval_episodes_override

    is_proprio    = task_cfg['obs_type'] == 'proprio'
    is_continuous = task_cfg['act_type'] == 'continuous'

    # ── Create timestamped run directory ───────────────────────────────
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if _EXTRA.output_dir:
        run_dir = pathlib.Path(_EXTRA.output_dir)
    elif RUN_NAME:
        run_dir = RUNS_DIR / TASK_NAME / RUN_NAME
    else:
        run_dir = RUNS_DIR / TASK_NAME / timestamp
    model_dir = run_dir
    video_dir = run_dir / 'videos'
    tb_dir    = run_dir / 'tb'
    log_file  = run_dir / 'train.log'

    for d in [model_dir, video_dir, tb_dir]:
        os.makedirs(d, exist_ok=True)

    log = setup_logger(log_file)

    log.info(f"Task          : {TASK_NAME}")
    log.info(f"Loss mode     : {LOSS_MODE}")
    log.info(f"Run directory : {run_dir}")
    log.info(f"Timestamp     : {timestamp}")

    # ── Seed ──────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device        : {device}")

    # ── Load expert data ──────────────────────────────────────────────
    if _EXTRA.data_path:
        data_path = pathlib.Path(_EXTRA.data_path)
    else:
        data_path = DATA_DIR / f'teacher_data_{TASK_NAME}.npz'
    log.info(f"Data path     : {data_path}")
    data = np.load(str(data_path), allow_pickle=True)
    obs = data['obs']
    act = data['act']

    obs_mean     = data['obs_mean'] if 'obs_mean' in data else None
    obs_std      = data['obs_std']  if 'obs_std'  in data else None
    proprio_keys = list(data['proprio_keys']) if 'proprio_keys' in data else None
    teacher_mean   = data['act_mean']    if 'act_mean'    in data else None
    teacher_stddev = data['act_stddev']  if 'act_stddev'  in data else None
    teacher_logits = data['act_logits']  if 'act_logits'  in data else None
    discount_return = data['discount_return'] if 'discount_return' in data else None
    # Use abs so that tasks with negative mean returns (e.g. atari100k) still
    # produce correct value weights: positive-return states get w>1, negative-
    # return states get w→0 after the clamp in Loss B.
    _raw_mean = float(discount_return.mean()) if discount_return is not None else 1.0
    return_mean = abs(_raw_mean) if abs(_raw_mean) > 0.05 else 1.0

    # ── Data subsampling (data-limited regime) ──────────────────────
    data_frac = _EXTRA.data_fraction
    if data_frac < 1.0:
        n_total = len(obs)
        n_keep = max(1, int(n_total * data_frac))
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n_total, n_keep, replace=False)
        idx.sort()
        obs = obs[idx]
        act = act[idx]
        if teacher_mean is not None:
            teacher_mean = teacher_mean[idx]
        if teacher_stddev is not None:
            teacher_stddev = teacher_stddev[idx]
        if teacher_logits is not None:
            teacher_logits = teacher_logits[idx]
        if discount_return is not None:
            discount_return = discount_return[idx]
            _raw_mean = float(discount_return.mean())
            return_mean = abs(_raw_mean) if abs(_raw_mean) > 0.05 else 1.0

    log.info(f"Obs type      : {task_cfg['obs_type']}")
    log.info(f"Act type      : {task_cfg['act_type']} (dim={task_cfg['act_dim']})")
    log.info(f"Transitions   : {len(obs)}")
    log.info(f"Obs shape     : {obs.shape}  dtype={obs.dtype}")
    log.info(f"Act shape     : {act.shape}  dtype={act.dtype}")
    if proprio_keys:
        log.info(f"Proprio keys  : {proprio_keys}")

    # ── Hyper-parameters ──────────────────────────────────────────────
    log.info(f"Hyper-params  : batch={BATCH_SIZE} epochs={NUM_EPOCHS} "
             f"lr={LR} wd={WEIGHT_DECAY} grad_clip={GRAD_CLIP} "
             f"eval_every={EVAL_EVERY} eval_eps={EVAL_EPISODES} seed={SEED}")
    log.info(f"Lambda_base   : {LAMBDA_BASE}")
    if data_frac < 1.0:
        log.info(f"Data fraction : {data_frac}  ({len(obs)} of {int(len(obs)/data_frac)} transitions)")
    if LOSS_MODE in ('A', 'AB'):
        log.info(f"Lambda_A      : {LAMBDA_A}")
        if not is_continuous:
            log.info(f"KD Temperature: {_EXTRA.kd_temperature}")
    if LOSS_MODE in ('B', 'AB'):
        log.info(f"Lambda_B      : {LAMBDA_B}")
        log.info(f"VW Temp       : {_EXTRA.value_weight_temp}")
    if LOSS_MODE == 'OBD':
        log.info(f"Loss variant  : OBD (Av-PBC — action-value weighted)")
        log.info(f"Lambda_OBD    : {LAMBDA_B}")

    # ── Model ─────────────────────────────────────────────────────────
    config = StudentConfig(
        obs_type     = task_cfg['obs_type'],
        obs_dim      = obs.shape[1] if is_proprio else 0,
        act_type     = task_cfg['act_type'],
        act_dim      = task_cfg['act_dim'],
        encoder_type = task_cfg['encoder_type'],
        hidden       = task_cfg['student_hidden'],
        blocks       = task_cfg['student_blocks'],
        dropout      = task_cfg['student_dropout'],
        cnn_channels = task_cfg.get('cnn_channels'),
        img_channels = task_cfg.get('img_channels', 3),
        vit_patch_size = task_cfg.get('vit_patch_size', 8),
        vit_embed_dim  = task_cfg.get('vit_embed_dim', 256),
        vit_num_heads  = task_cfg.get('vit_num_heads', 4),
        vit_num_layers = task_cfg.get('vit_num_layers', 4),
    )
    log.info(f"StudentConfig : {config}")

    model_kwargs = {}
    if is_proprio:
        model_kwargs['obs_mean'] = torch.tensor(obs_mean)
        model_kwargs['obs_std']  = torch.tensor(obs_std)

    model = StudentPolicy(config, **model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters    : {num_params:,}")

    # ── Data loader ───────────────────────────────────────────────────
    dataset = ExpertDataset(obs, act,
                            obs_type=task_cfg['obs_type'],
                            act_type=task_cfg['act_type'],
                            teacher_mean=teacher_mean,
                            teacher_stddev=teacher_stddev,
                            discount_return=discount_return,
                            teacher_logits=teacher_logits)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5,
    )

    # ── Loss function ─────────────────────────────────────────────────
    if task_cfg['loss_type'] == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # ── TensorBoard ───────────────────────────────────────────────────
    writer = SummaryWriter(str(tb_dir))

    # ── Training loop ─────────────────────────────────────────────────
    best_score  = -float('inf')
    global_step = 0

    log.info(f"")
    log.info(f"{'='*70}")
    log.info(f"Training for {NUM_EPOCHS} epochs  "
             f"(eval every {EVAL_EVERY}, {EVAL_EPISODES} episodes)")
    log.info(f"Loss          : {task_cfg['loss_type']}  (mode={LOSS_MODE})")
    log.info(f"{'='*70}")
    log.info(f"")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_A = 0.0
        epoch_loss_B = 0.0
        n_batches  = 0

        for obs_batch, act_batch, tmean_batch, tstd_batch, ret_batch, tlogits_batch in dataloader:
            obs_batch     = obs_batch.to(device)
            act_batch     = act_batch.to(device)
            tmean_batch   = tmean_batch.to(device)
            tstd_batch    = tstd_batch.to(device)
            ret_batch     = ret_batch.to(device)
            tlogits_batch = tlogits_batch.to(device)

            # ── Forward pass ────────────────────────────────────────
            # For variant A we need (mean, log_std); otherwise just mean
            if LOSS_MODE in ('A', 'AB', 'OBD') and is_continuous:
                pred, log_std = model.forward_with_log_std(obs_batch)
            else:
                pred = model(obs_batch)
                log_std = None

            # ── Base loss (always MSE / CE) ─────────────────────────
            L_base = criterion(pred, act_batch)
            loss = LAMBDA_BASE * L_base

            # ── Variant A ────────────────────────────────────────────
            # Continuous: Gaussian NLL policy-distillation loss.
            #   Student predicts (μ_S, σ_S); teacher action a_T is the
            #   target.  L_A = -log N(a_T; μ_S, σ_S).  The adaptive
            #   1/σ² gradient magnifier sharpens learning near optimum,
            #   avoiding MSE action-averaging.
            # Discrete: Soft-label KL distillation loss.
            #   Teacher logits → p_teacher via softmax.  Student logits
            #   → log p_student via log_softmax.
            #   L_A = -Σ_k p_T(k) log p_S(k)  (soft cross-entropy)
            #   More informative than hard-label CE: captures teacher
            #   uncertainty across actions.
            L_A_val = 0.0
            if LOSS_MODE in ('A', 'AB'):
                if is_continuous:
                    var = torch.exp(2.0 * log_std)           # σ²
                    # Gaussian NLL: 0.5*(log(2π) + 2*log_σ + (a-μ)²/σ²)
                    nll = 0.5 * (1.8378770664093453          # log(2π)
                                 + 2.0 * log_std
                                 + (act_batch - pred) ** 2 / (var + 1e-8))
                    L_A = nll.mean()
                else:
                    # Soft-label cross-entropy from teacher distribution.
                    # Teacher logits are softened with temperature T to expose
                    # "dark knowledge" across non-argmax actions (T=5 →
                    # peak action ~0.84 prob, rest spread to ~0.01-0.03).
                    # Student evaluated at T=1. NO T² rescaling: combining
                    # with hard-label L_base so magnitudes should be comparable.
                    kd_T = _EXTRA.kd_temperature
                    with torch.no_grad():
                        p_teacher = torch.softmax(tlogits_batch / kd_T, dim=-1)
                    log_p_student = torch.log_softmax(pred, dim=-1)
                    L_A = -(p_teacher * log_p_student).sum(dim=-1).mean()
                loss = loss + LAMBDA_A * L_A
                L_A_val = L_A.item()

            # ── Variant B: Value-weighted imitation loss ─────────────
            # w_t = reward-to-go (discount_return from data).
            # Normalised to mean~1, then temperature-sharpened so that
            # high-return critical states receive amplified gradients.
            # temp=1: linear weighting; temp>1: power-law sharpening.
            # Continuous: weighted MSE.
            # Discrete  : weighted cross-entropy (negative returns →
            #   weight clamped to near-zero, downweighting bad states).
            L_B_val = 0.0
            if LOSS_MODE in ('B', 'AB'):
                temp = _EXTRA.value_weight_temp           # default 1.0
                w = ret_batch / (return_mean + 1e-8)      # normalise to mean~1
                w = torch.pow(w.clamp(min=0.01), temp)
                w = torch.clamp(w, 0.01, 20.0)
                w = w / (w.mean() + 1e-8)                 # re-centre so mean≈1
                if is_continuous:
                    L_B = (w.unsqueeze(-1) * (pred - act_batch) ** 2).mean()
                else:
                    # Per-sample CE, then weighted average
                    ce_per = nn.functional.cross_entropy(
                        pred, act_batch, reduction='none')  # (B,)
                    L_B = (w * ce_per).mean()
                loss = loss + LAMBDA_B * L_B
                L_B_val = L_B.item()

            # ── OBD (Av-PBC): Action-value weighted decision diff ────
            # Lei et al. (NeurIPS 2024) — Offline Behavior Distillation.
            # L_OBD = E_s[ sum_a  Q_hat(s,a) * |pi_S(a|s) - pi_T(a|s)| ]
            # We approximate Q(s,a) ~ V(s) * pi_T(a|s) where V(s) ≈
            # discount_return (from data), and pi_T = softmax(teacher_logits).
            # For discrete: action-value weighted L1 over full distribution.
            L_OBD_val = 0.0
            if LOSS_MODE == 'OBD':
                with torch.no_grad():
                    # Teacher policy distribution
                    pi_T = torch.softmax(tlogits_batch, dim=-1)  # (B, A)
                    # V(s) approximation from discount_return
                    V_s = ret_batch / (return_mean + 1e-8)       # normalise
                    V_s = V_s.clamp(min=0.01)                    # non-negative
                    # Q_hat(s,a) ≈ V(s) * pi_T(a|s)
                    Q_hat = V_s.unsqueeze(-1) * pi_T             # (B, A)
                if is_continuous:
                    # Gaussian NLL weighted by V(s)
                    var = torch.exp(2.0 * log_std)
                    nll = 0.5 * (1.8378770664093453
                                 + 2.0 * log_std
                                 + (act_batch - pred) ** 2 / (var + 1e-8))
                    w_obd = V_s.clamp(min=0.01, max=20.0)
                    w_obd = w_obd / (w_obd.mean() + 1e-8)
                    L_OBD = (w_obd.unsqueeze(-1) * nll).mean()
                else:
                    # Student policy distribution
                    pi_S = torch.softmax(pred, dim=-1)           # (B, A)
                    # Action-value weighted L1 decision difference
                    diff = (Q_hat * (pi_S - pi_T).abs()).sum(dim=-1)  # (B,)
                    L_OBD = diff.mean()
                loss = loss + LAMBDA_B * L_OBD
                L_OBD_val = L_OBD.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_loss_A += L_A_val
            epoch_loss_B += L_B_val
            if LOSS_MODE == 'OBD':
                epoch_loss_B += L_OBD_val  # reuse B accumulator for OBD
            n_batches    += 1
            global_step  += 1
            writer.add_scalar('train/loss_step', loss.item(), global_step)

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_LA   = epoch_loss_A / max(n_batches, 1)
        avg_LB   = epoch_loss_B / max(n_batches, 1)
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        if LOSS_MODE in ('A', 'AB'):
            writer.add_scalar('train/L_A_nll', avg_LA, epoch)
        if LOSS_MODE in ('B', 'AB'):
            writer.add_scalar('train/L_B_vw', avg_LB, epoch)
        if LOSS_MODE == 'OBD':
            writer.add_scalar('train/L_OBD_avpbc', avg_LB, epoch)

        line = (f"Epoch {epoch:3d}/{NUM_EPOCHS}  |  "
                f"loss = {avg_loss:.6f}  |  lr = {lr:.2e}")
        if LOSS_MODE in ('A', 'AB'):
            line += f"  |  L_A={avg_LA:.4f}"
        if LOSS_MODE in ('B', 'AB'):
            line += f"  |  L_B={avg_LB:.4f}"
        if LOSS_MODE == 'OBD':
            line += f"  |  L_OBD={avg_LB:.4f}"

        # ── Periodic evaluation ───────────────────────────────────────
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            t0 = time.time()
            mean_score, std_score, scores, video = evaluate(
                model, task_cfg, device,
                proprio_keys=proprio_keys,
                num_episodes=EVAL_EPISODES,
            )
            eval_time = time.time() - t0

            writer.add_scalar('eval/mean_score', mean_score, epoch)
            writer.add_scalar('eval/std_score',  std_score,  epoch)
            for i, s in enumerate(scores):
                writer.add_scalar(f'eval/ep_score_{i}', s, epoch)

            # Log eval video to TensorBoard + save MP4 to disk
            if video is not None:
                vid_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
                vid_tensor = vid_tensor.unsqueeze(0)
                writer.add_video('eval/video', vid_tensor, epoch,
                                 fps=VIDEO_FPS)
                mp4_path = str(video_dir / f'eval_epoch_{epoch:03d}.mp4')
                save_mp4(video, mp4_path, fps=VIDEO_FPS)

            line += (f"  |  eval = {mean_score:.1f} +/- {std_score:.1f}"
                     f"  ({eval_time:.1f}s)")

            if mean_score > best_score:
                best_score = mean_score
                save_path  = str(model_dir / 'best_student.pth')
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'config':           config,
                    'task_name':        TASK_NAME,
                    'loss_mode':        LOSS_MODE,
                    'epoch':            epoch,
                    'score':            mean_score,
                }
                if is_proprio:
                    save_dict['obs_mean']     = obs_mean
                    save_dict['obs_std']      = obs_std
                    save_dict['proprio_keys'] = proprio_keys
                torch.save(save_dict, save_path)
                line += "  ** NEW BEST -- saved **"

        log.info(line)

    writer.close()

    # ── Final summary ─────────────────────────────────────────────────
    log.info(f"")
    log.info(f"{'='*70}")
    log.info(f"Training complete.  ({TASK_NAME})")
    log.info(f"Best eval score : {best_score:.1f}")
    log.info(f"Best model      : {model_dir / 'best_student.pth'}")
    log.info(f"Eval videos     : {video_dir}")
    log.info(f"TensorBoard     : {tb_dir}")
    log.info(f"Full log        : {log_file}")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
