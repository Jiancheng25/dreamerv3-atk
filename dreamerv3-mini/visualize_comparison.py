#!/usr/bin/env python3
"""Generate publication-quality comparison figure.

Layout per task: 3 rows (Teacher, Baseline, Ours) x 4 columns (time-frames).
Two tasks side by side (or stacked): DMC Walker Walk + Atari Pong.

Usage:
    python visualize_comparison.py
"""

import os
import sys
import numpy as np
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image as PILImage

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, 'runs')

# ── Configuration ────────────────────────────────────────────────────────────

TASKS = [
    {
        'name': 'DMC Walker Walk',
        'dir': 'dmc_proprio_dmc_walker_walk',
        'teacher_video': 'teacher_hires/teacher_eval.mp4',
        'baseline_video': 'baseline/videos/eval_epoch_120.mp4',
        'ours_video': 'AB/videos/eval_epoch_120.mp4',
    },
    {
        'name': 'Atari Pong',
        'dir': 'atari_pong',
        'teacher_video': 'teacher_hires/teacher_eval.mp4',
        'baseline_video': 'baseline/videos/eval_epoch_100.mp4',
        'ours_video': 'AB/videos/eval_epoch_100.mp4',
    },
]

NUM_TIMEFRAMES = 4          # columns per task
ROW_LABELS = ['Teacher', 'Baseline', 'Ours (A+B)']
OUTPUT_DIR = os.path.join(BASE, 'figures')


# ── Helper functions ─────────────────────────────────────────────────────────

def read_video_frames(path, max_frames=None):
    """Read all RGB frames from an MP4 file, return as list of np arrays."""
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
        if max_frames and len(frames) >= max_frames:
            break
    container.close()
    return frames


def sample_frames(frames, n=4):
    """Pick n evenly spaced frames (including first and last)."""
    total = len(frames)
    if total <= n:
        return frames[:n]
    indices = np.linspace(0, total - 1, n, dtype=int)
    return [frames[i] for i in indices]


def ensure_rgb_256(img):
    """Resize to 256x256 and ensure 3-channel RGB."""
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    if h != 256 or w != 256:
        pil = PILImage.fromarray(img)
        pil = pil.resize((256, 256), PILImage.LANCZOS)
        img = np.array(pil)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def make_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_tasks = len(TASKS)
    n_rows = 3   # Teacher / Baseline / Ours
    n_cols = NUM_TIMEFRAMES

    # Gather all frame grids: task_frames[t][r] = list of n_cols images
    task_frames = []
    for task_cfg in TASKS:
        task_dir = os.path.join(RUNS, task_cfg['dir'])
        videos = {
            'Teacher':  os.path.join(task_dir, task_cfg['teacher_video']),
            'Baseline': os.path.join(task_dir, task_cfg['baseline_video']),
            'Ours (A+B)': os.path.join(task_dir, task_cfg['ours_video']),
        }
        rows = []
        for label in ROW_LABELS:
            vpath = videos[label]
            if not os.path.isfile(vpath):
                print(f"WARNING: missing {vpath}")
                rows.append([np.zeros((256, 256, 3), dtype=np.uint8)] * n_cols)
                continue
            all_frames = read_video_frames(vpath)
            selected = sample_frames(all_frames, n_cols)
            selected = [ensure_rgb_256(f) for f in selected]
            rows.append(selected)
            print(f"  {task_cfg['name']:20s} | {label:12s} | "
                  f"{len(all_frames)} frames -> picked {n_cols}")
        task_frames.append(rows)

    # ── Build figure ─────────────────────────────────────────────────────
    # Layout: one block per task, stacked vertically.
    # Each block = 3 rows x 4 cols of images.
    # Between tasks: a thin gap.

    cell_size = 1.8  # inches per image cell
    hgap_task = 0.3  # vertical gap between tasks in inches
    title_height = 0.5  # space for task title

    fig_w = n_cols * cell_size + 1.6   # extra for row labels
    fig_h = n_tasks * (n_rows * cell_size + title_height) + (n_tasks - 1) * hgap_task + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # Outer gridspec: one row per task
    outer = GridSpec(n_tasks, 1, figure=fig,
                     hspace=hgap_task / fig_h * n_tasks,
                     top=0.97, bottom=0.02, left=0.12, right=0.98)

    for t_idx, (task_cfg, rows) in enumerate(zip(TASKS, task_frames)):
        # Inner gridspec: 3 rows x 4 cols
        inner = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer[t_idx],
                                        hspace=0.05, wspace=0.05)

        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                ax = fig.add_subplot(inner[r_idx, c_idx])
                ax.imshow(rows[r_idx][c_idx])
                ax.set_xticks([])
                ax.set_yticks([])

                # Row labels on leftmost column
                if c_idx == 0:
                    ax.set_ylabel(ROW_LABELS[r_idx], fontsize=10,
                                  fontweight='bold', rotation=90,
                                  labelpad=8)

                # Column headers (time indices) on top row
                if r_idx == 0:
                    frac = c_idx / max(n_cols - 1, 1)
                    ax.set_title(f't = {frac:.0%}' if frac > 0 else 't = 0',
                                 fontsize=9, pad=4)

        # Task title above the block
        # Use the first row's first axis position to place a text label
        top_ax = fig.add_subplot(inner[0, :])
        top_ax.set_visible(False)
        fig.text(0.55, outer[t_idx].get_position(fig).y1 + 0.005,
                 task_cfg['name'],
                 ha='center', va='bottom', fontsize=13,
                 fontweight='bold')

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'comparison_3x4.png')
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure: {out_path}")
    plt.close(fig)

    # Also save a PDF version
    out_pdf = os.path.join(OUTPUT_DIR, 'comparison_3x4.pdf')
    fig2 = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    outer2 = GridSpec(n_tasks, 1, figure=fig2,
                      hspace=hgap_task / fig_h * n_tasks,
                      top=0.97, bottom=0.02, left=0.12, right=0.98)

    for t_idx, (task_cfg, rows) in enumerate(zip(TASKS, task_frames)):
        inner2 = GridSpecFromSubplotSpec(n_rows, n_cols,
                                         subplot_spec=outer2[t_idx],
                                         hspace=0.05, wspace=0.05)
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                ax = fig2.add_subplot(inner2[r_idx, c_idx])
                ax.imshow(rows[r_idx][c_idx])
                ax.set_xticks([])
                ax.set_yticks([])
                if c_idx == 0:
                    ax.set_ylabel(ROW_LABELS[r_idx], fontsize=10,
                                  fontweight='bold', rotation=90,
                                  labelpad=8)
                if r_idx == 0:
                    frac = c_idx / max(n_cols - 1, 1)
                    ax.set_title(f't = {frac:.0%}' if frac > 0 else 't = 0',
                                 fontsize=9, pad=4)
        fig2.text(0.55, outer2[t_idx].get_position(fig2).y1 + 0.005,
                  task_cfg['name'],
                  ha='center', va='bottom', fontsize=13,
                  fontweight='bold')

    fig2.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {out_pdf}")
    plt.close(fig2)


if __name__ == '__main__':
    make_figure()
