#!/usr/bin/env python3
"""
Generate ablation comparison plots from train.log files.

Produces:
  1. Eval score curve (mean ± std) over epochs for all 4 variants
  2. Best score bar chart
  3. Training loss curve

Usage:
    python plot_ablation.py --task dmc_proprio_dmc_walker_walk
"""
import argparse
import pathlib
import re
import sys

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / 'runs'

VARIANTS = ['baseline', 'A', 'B']
COLORS = {'baseline': '#1f77b4', 'A': '#ff7f0e', 'B': '#2ca02c'}
LABELS = {'baseline': 'Baseline', 'A': 'A (NLL Distill)', 'B': 'B (Value-Wt)'}


def parse_log(log_path):
    """Extract eval scores and training losses from train.log."""
    evals = []
    train_losses = []
    best_score = None

    eval_pat = re.compile(
        r'Epoch\s+(\d+)/\d+\s+\|.*eval\s+=\s+([\d.]+)\s+\+/-\s+([\d.]+)')
    loss_pat = re.compile(
        r'Epoch\s+(\d+)/\d+\s+\|\s+loss\s+=\s+([0-9eE.+-]+)')
    best_pat = re.compile(r'Best eval score\s*:\s*([\d.]+)')

    with open(log_path) as f:
        for line in f:
            m = eval_pat.search(line)
            if m:
                evals.append({
                    'epoch': int(m.group(1)),
                    'mean': float(m.group(2)),
                    'std': float(m.group(3)),
                })
            m2 = loss_pat.search(line)
            if m2:
                train_losses.append({
                    'epoch': int(m2.group(1)),
                    'loss': float(m2.group(2)),
                })
            m3 = best_pat.search(line)
            if m3:
                best_score = float(m3.group(1))

    return {'evals': evals, 'losses': train_losses, 'best_score': best_score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    args = ap.parse_args()

    # Lazy import matplotlib (may not be available in all envs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    task_dir = RUNS_DIR / args.task
    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}")
        sys.exit(1)

    # Parse all logs
    results = {}
    for v in VARIANTS:
        log_path = task_dir / v / 'train.log'
        if log_path.exists():
            results[v] = parse_log(log_path)
        else:
            print(f"Warning: {log_path} not found")

    if not results:
        print("No results found.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────
    # Plot 1: Eval score curve (mean ± std)
    # ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for v in VARIANTS:
        r = results.get(v)
        if not r:
            continue
        epochs = [e['epoch'] for e in r['evals']]
        means = [e['mean'] for e in r['evals']]
        stds = [e['std'] for e in r['evals']]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(epochs, means, color=COLORS[v], label=LABELS[v],
                linewidth=2, marker='o', markersize=3)
        ax.fill_between(epochs, means_arr - stds_arr, means_arr + stds_arr,
                        color=COLORS[v], alpha=0.15)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Eval Score', fontsize=13)
    ax.set_title('DMC Walker Walk — Loss Ablation (Eval Score)', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = task_dir / 'eval_score_curve.png'
    fig.savefig(str(out1), dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # ──────────────────────────────────────────────────────────────────────
    # Plot 2: Best score bar chart
    # ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    names = []
    scores = []
    colors = []
    for v in VARIANTS:
        r = results.get(v)
        if r and r['best_score'] is not None:
            names.append(LABELS[v])
            scores.append(r['best_score'])
            colors.append(COLORS[v])

    bars = ax.bar(names, scores, color=colors, edgecolor='black', linewidth=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{score:.1f}', ha='center', va='bottom', fontsize=12,
                fontweight='bold')

    ax.set_ylabel('Best Eval Score', fontsize=13)
    ax.set_title('DMC Walker Walk — Best Score Comparison', fontsize=14)
    # Set y-axis to start slightly below the min score
    if scores:
        ymin = min(scores) - 50
        ymax = max(scores) + 50
        ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out2 = task_dir / 'best_score_bar.png'
    fig.savefig(str(out2), dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    # ──────────────────────────────────────────────────────────────────────
    # Plot 3: Training loss curve
    # ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for v in VARIANTS:
        r = results.get(v)
        if not r:
            continue
        epochs = [e['epoch'] for e in r['losses']]
        losses = [e['loss'] for e in r['losses']]
        ax.plot(epochs, losses, color=COLORS[v], label=LABELS[v],
                linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Training Loss', fontsize=13)
    ax.set_title('DMC Walker Walk — Training Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out3 = task_dir / 'training_loss_curve.png'
    fig.savefig(str(out3), dpi=150)
    plt.close(fig)
    print(f"Saved: {out3}")

    # ──────────────────────────────────────────────────────────────────────
    # Plot 4: Combined figure (2x2)
    # ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Eval score curve
    ax = axes[0, 0]
    for v in VARIANTS:
        r = results.get(v)
        if not r:
            continue
        epochs = [e['epoch'] for e in r['evals']]
        means = [e['mean'] for e in r['evals']]
        stds = [e['std'] for e in r['evals']]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(epochs, means, color=COLORS[v], label=LABELS[v],
                linewidth=2, marker='o', markersize=2)
        ax.fill_between(epochs, means_arr - stds_arr, means_arr + stds_arr,
                        color=COLORS[v], alpha=0.12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eval Score')
    ax.set_title('Evaluation Score')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) Bar chart
    ax = axes[0, 1]
    if names:
        bars = ax.bar(names, scores, color=colors, edgecolor='black', linewidth=0.5)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')
        if scores:
            ax.set_ylim(min(scores) - 50, max(scores) + 50)
    ax.set_ylabel('Best Score')
    ax.set_title('Best Score Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # (1,0) Training loss
    ax = axes[1, 0]
    for v in VARIANTS:
        r = results.get(v)
        if not r:
            continue
        epochs = [e['epoch'] for e in r['losses']]
        losses = [e['loss'] for e in r['losses']]
        ax.plot(epochs, losses, color=COLORS[v], label=LABELS[v],
                linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) Score table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [['Variant', 'Best Score']]
    for v in VARIANTS:
        r = results.get(v)
        if r and r['best_score'] is not None:
            table_data.append([LABELS[v], f'{r["best_score"]:.1f}'])
        else:
            table_data.append([LABELS[v], 'N/A'])
    # Check ordering
    all_scores = []
    for v in VARIANTS:
        r = results.get(v)
        if r and r['best_score'] is not None:
            all_scores.append(r['best_score'])
    if len(all_scores) == 3:
        ordering_ok = all_scores[0] < all_scores[1] and all_scores[0] < all_scores[2]
        table_data.append(['', ''])
        table_data.append(['baseline < A,B?', 'YES' if ordering_ok else 'NO'])

    tbl = ax.table(cellText=table_data, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)
    # Style header row
    for j in range(2):
        tbl[0, j].set_facecolor('#cccccc')
        tbl[0, j].set_text_props(fontweight='bold')
    ax.set_title('Summary', fontsize=12)

    fig.suptitle('DMC Walker Walk — Loss Ablation Study', fontsize=15,
                 fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out4 = task_dir / 'ablation_summary.png'
    fig.savefig(str(out4), dpi=150)
    plt.close(fig)
    print(f"Saved: {out4}")

    # ──────────────────────────────────────────────────────────────────────
    # Print summary
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Best Score Summary")
    print("=" * 50)
    for v in VARIANTS:
        r = results.get(v)
        if r and r['best_score'] is not None:
            print(f"  {LABELS[v]:<20s}  {r['best_score']:>8.1f}")
    if len(all_scores) == 3:
        print(f"\n  Target:  baseline < A  AND  baseline < B")
        print(f"  Actual:  baseline={all_scores[0]:.1f}  A={all_scores[1]:.1f}  B={all_scores[2]:.1f}")
        print(f"  Result:  {'PASS' if ordering_ok else 'FAIL'}")
    print("=" * 50)


if __name__ == '__main__':
    main()
