"""
Centralized task configuration registry for dreamerv3-mini.

All task-specific parameters live here.  Downstream scripts import
`parse_task_arg()` and branch on the returned config dict -- they never
hard-code task names, paths, or hyper-parameters themselves.

Usage:
    from task_config import parse_task_arg
    task_name, task_cfg, remaining_argv = parse_task_arg()
"""

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent   # dreamerv3-main/

TASK_CONFIGS = {

    # ── DMC Walker Walk (proprioceptive, continuous) ─────────────────────
    'dmc_walker_walk': {
        # DreamerV3 teacher
        'dreamer_config_presets': ['defaults', 'dmc_proprio'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260128T193932-dmc_proprio_dmc_walker_walk' / 'ckpt',

        # Observation
        'obs_type':   'proprio',        # 'proprio' | 'image'
        'obs_keys':   None,             # auto-detect for proprio

        # Action
        'act_type':   'continuous',     # 'continuous' | 'discrete'
        'act_dim':    6,

        # Environment
        'env_suite':  'dmc',
        'env_task':   'walker_walk',
        'env_kwargs': {'size': (64, 64), 'repeat': 1,
                       'proprio': True, 'image': False},

        # Data collection
        'num_episodes':  20,
        'ep_max_steps':  1000,

        # Student model
        'encoder_type':    'mlp',
        'student_hidden':  512,
        'student_blocks':  4,
        'student_dropout': 0.1,
        'cnn_channels':    None,

        # Training
        'loss_type':    'mse',
        'batch_size':   256,
        'num_epochs':   100,
        'lr':           1e-3,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 5,

        # Video
        'render_source': 'physics',     # 'physics' | 'obs_image'
        'render_size':   (256, 256),
        'video_fps':     30,
    },

    # ── Crafter (image, discrete) ────────────────────────────────────────
    'crafter_reward': {
        'dreamer_config_presets': ['defaults', 'crafter'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260120T215532-crafter' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 3,

        'act_type':   'discrete',
        'act_dim':    17,               # 17 Crafter actions

        'env_suite':  'crafter',
        'env_task':   'reward',
        'env_kwargs': {'size': (64, 64), 'logs': False},

        'num_episodes':  100,
        'ep_max_steps':  10000,

        'encoder_type':    'cnn',
        'student_hidden':  256,
        'student_blocks':  2,
        'student_dropout': 0.1,
        'cnn_channels':    (32, 64, 64),

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 5,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,            # Crafter is slower-paced
    },

    # ── Crafter 25M teacher (image, discrete) ───────────────────────────
    'crafter_reward_25m': {
        'dreamer_config_presets': ['defaults', 'crafter', 'size25m'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260304T042925-crafter-size25m' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 3,

        'act_type':   'discrete',
        'act_dim':    17,

        'env_suite':  'crafter',
        'env_task':   'reward',
        'env_kwargs': {'size': (64, 64), 'logs': False},

        'num_episodes':  100,
        'ep_max_steps':  10000,

        'encoder_type':    'cnn',
        'student_hidden':  256,
        'student_blocks':  2,
        'student_dropout': 0.1,
        'cnn_channels':    (32, 64, 64),

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 5,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,
    },

    # ── Crafter 100M teacher (image, discrete) ──────────────────────────
    'crafter_reward_100m': {
        'dreamer_config_presets': ['defaults', 'crafter', 'size100m'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260304T042953-crafter-size100m' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 3,

        'act_type':   'discrete',
        'act_dim':    17,

        'env_suite':  'crafter',
        'env_task':   'reward',
        'env_kwargs': {'size': (64, 64), 'logs': False},

        'num_episodes':  100,
        'ep_max_steps':  10000,

        'encoder_type':    'cnn',
        'student_hidden':  256,
        'student_blocks':  2,
        'student_dropout': 0.1,
        'cnn_channels':    (32, 64, 64),

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 5,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,
    },

    # ── Atari Pong (image/grayscale, discrete) ────────────────────────────
    'atari_pong': {
        'dreamer_config_presets': ['defaults', 'atari'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260123T042551-atari-pong' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 4,               # 1 grayscale * 4 frame_stack
        'frame_stack': 4,                # stack 4 consecutive frames

        'act_type':   'discrete',
        'act_dim':    18,                 # all Atari actions

        'env_suite':  'atari',
        'env_task':   'pong',
        'env_kwargs': {
            'size': (64, 64),            # student size (teacher used 96x96)
            'repeat': 4,
            'gray': True,
            'sticky': True,
            'actions': 'all',
            'lives': 'unused',
            'noops': 30,
            'autostart': False,
            'pooling': 2,
            'aggregate': 'max',
            'resize': 'pillow',
            'clip_reward': False,
        },

        'num_episodes':  100,
        'ep_max_steps':  10000,

        'encoder_type':    'cnn',
        'student_hidden':  512,
        'student_blocks':  4,
        'student_dropout': 0.1,
        'cnn_channels':    (32, 64, 64),

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 5,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,
    },

    # ── Procgen CoinRun (image/RGB, discrete) ────────────────────────────
    'procgen_coinrun': {
        'dreamer_config_presets': ['defaults', 'procgen'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260220T010858-procgen-coinrun' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 3,               # single RGB frame (no stacking needed)

        'act_type':   'discrete',
        'act_dim':    15,                 # procgen Discrete(15): actions 0-14

        'env_suite':  'procgen',
        'env_task':   'coinrun',
        'env_kwargs': {'size': (64, 64)},

        'num_episodes':  500,
        'ep_max_steps':  1000,

        'encoder_type':    'cnn',
        'student_hidden':  256,
        'student_blocks':  3,
        'student_dropout': 0.1,
        'cnn_channels':    (32, 64, 64),

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 10,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,
    },

    # ── Atari 100k Pong (image/RGB, discrete, low-data budget) ──────────
    'atari100k_pong': {
        'dreamer_config_presets': ['defaults', 'atari100k'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260218T214756-atari100k-pong' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 12,                 # 3 RGB * 4 frame_stack
        'frame_stack': 4,

        'act_type':   'discrete',
        'act_dim':    6,                     # needed actions: NOOP,FIRE,R,L,RF,LF

        'env_suite':  'atari',
        'env_task':   'pong',
        'env_kwargs': {
            'size': (64, 64),
            'repeat': 4,
            'gray': False,                   # RGB (not grayscale)
            'sticky': False,
            'actions': 'needed',
            'lives': 'unused',
            'noops': 30,
            'autostart': False,
            'resize': 'pillow',
            'clip_reward': False,
        },

        'num_episodes':  100,
        'ep_max_steps':  10000,

        'encoder_type':    'cnn',
        'student_hidden':  256,
        'student_blocks':  2,
        'student_dropout': 0.1,
        'cnn_channels':    (32, 64, 64),

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 5,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,
    },

    # ═════════════════════════════════════════════════════════════════════
    #  Architecture ablation: Crafter 166M  (ResNet / VGG / Transformer)
    # ═════════════════════════════════════════════════════════════════════

    # ── Crafter 166M — ResNet encoder ────────────────────────────────────
    'crafter_reward_resnet': {
        'dreamer_config_presets': ['defaults', 'crafter'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260120T215532-crafter' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 3,

        'act_type':   'discrete',
        'act_dim':    17,

        'env_suite':  'crafter',
        'env_task':   'reward',
        'env_kwargs': {'size': (64, 64), 'logs': False},

        'num_episodes':  100,
        'ep_max_steps':  10000,

        'encoder_type':    'resnet',
        'student_hidden':  512,
        'student_blocks':  3,
        'student_dropout': 0.1,
        'cnn_channels':    (32, 64, 64),

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 20,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,
    },

    # ── Crafter 166M — VGG encoder ───────────────────────────────────────
    'crafter_reward_vgg': {
        'dreamer_config_presets': ['defaults', 'crafter'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260120T215532-crafter' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 3,

        'act_type':   'discrete',
        'act_dim':    17,

        'env_suite':  'crafter',
        'env_task':   'reward',
        'env_kwargs': {'size': (64, 64), 'logs': False},

        'num_episodes':  100,
        'ep_max_steps':  10000,

        'encoder_type':    'vgg',
        'student_hidden':  512,
        'student_blocks':  3,
        'student_dropout': 0.1,
        'cnn_channels':    (32, 64, 64),

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 20,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,
    },

    # ── Crafter 166M — Transformer (ViT) encoder ────────────────────────
    'crafter_reward_transformer': {
        'dreamer_config_presets': ['defaults', 'crafter'],
        'checkpoint_dir': ROOT / 'logdir' / 'dreamer'
                          / '20260120T215532-crafter' / 'ckpt',

        'obs_type':   'image',
        'obs_keys':   None,
        'img_channels': 3,

        'act_type':   'discrete',
        'act_dim':    17,

        'env_suite':  'crafter',
        'env_task':   'reward',
        'env_kwargs': {'size': (64, 64), 'logs': False},

        'num_episodes':  100,
        'ep_max_steps':  10000,

        'encoder_type':    'transformer',
        'student_hidden':  512,
        'student_blocks':  3,
        'student_dropout': 0.1,
        'vit_patch_size':  8,
        'vit_embed_dim':   256,
        'vit_num_heads':   4,
        'vit_num_layers':  4,

        'loss_type':    'cross_entropy',
        'batch_size':   128,
        'num_epochs':   100,
        'lr':           3e-4,
        'weight_decay': 1e-4,
        'grad_clip':    1.0,
        'eval_every':   5,
        'eval_episodes': 20,

        'render_source': 'obs_image',
        'render_size':   (256, 256),
        'video_fps':     15,
    },
}


def parse_task_arg(argv=None):
    """Parse ``--task`` from the command line.

    Returns
    -------
    task_name : str
    task_cfg  : dict
    remaining : list[str]   (un-consumed argv, for downstream parsers)
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--task', type=str, default='dmc_walker_walk',
        choices=list(TASK_CONFIGS.keys()),
        help='Task to run (default: dmc_walker_walk)',
    )
    args, remaining = parser.parse_known_args(argv)
    return args.task, TASK_CONFIGS[args.task], remaining
