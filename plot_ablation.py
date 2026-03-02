"""Plot ablation experiment comparison for dmc_proprio_dmc_walker_walk."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_scores(logdir):
    """Load episode scores from scores.jsonl or metrics.jsonl."""
    scores_file = os.path.join(logdir, 'scores.jsonl')
    metrics_file = os.path.join(logdir, 'metrics.jsonl')

    steps, scores = [], []

    # Try scores.jsonl first
    if os.path.exists(scores_file):
        with open(scores_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if 'episode/score' in data and 'step' in data:
                    steps.append(data['step'])
                    scores.append(data['episode/score'])

    # Fallback to metrics.jsonl
    if not steps and os.path.exists(metrics_file):
        with open(metrics_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if 'episode/score' in data and 'step' in data:
                    steps.append(data['step'])
                    scores.append(data['episode/score'])

    return np.array(steps), np.array(scores)


def smooth(values, window=20):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def main():
    base_dir = '/home/manager/wjc/dreamerv3-main/dmc_proprio_dmc_walker_walk'
    groups = {
        'Baseline': os.path.join(base_dir, 'baseline'),
        'A (Distillation)': os.path.join(base_dir, 'A'),
        'B (Value-Weighted)': os.path.join(base_dir, 'B'),
        'A+B (Combined)': os.path.join(base_dir, 'A+B'),
    }
    colors = {
        'Baseline': '#888888',
        'A (Distillation)': '#2196F3',
        'B (Value-Weighted)': '#FF9800',
        'A+B (Combined)': '#4CAF50',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Training curves
    ax1 = axes[0]
    window = 20
    final_scores = {}

    for name, logdir in groups.items():
        steps, scores = load_scores(logdir)
        if len(steps) == 0:
            print(f"Warning: No data found for {name} in {logdir}")
            continue

        # Raw data (transparent)
        ax1.scatter(steps, scores, alpha=0.05, s=1, color=colors[name])

        # Smoothed
        if len(scores) >= window:
            smoothed = smooth(scores, window)
            smooth_steps = steps[window - 1:][:len(smoothed)]
            ax1.plot(smooth_steps, smoothed, label=name, color=colors[name],
                     linewidth=2)
        else:
            ax1.plot(steps, scores, label=name, color=colors[name], linewidth=2)

        # Final score: mean of last 10% episodes
        n_tail = max(1, len(scores) // 10)
        final_scores[name] = np.mean(scores[-n_tail:])

    ax1.set_xlabel('Environment Steps', fontsize=12)
    ax1.set_ylabel('Episode Score', fontsize=12)
    ax1.set_title('DMC Walker Walk - Loss Ablation Training Curves', fontsize=14)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart of final scores
    ax2 = axes[1]
    if final_scores:
        names = list(final_scores.keys())
        vals = [final_scores[n] for n in names]
        bar_colors = [colors[n] for n in names]
        bars = ax2.bar(range(len(names)), vals, color=bar_colors, edgecolor='black',
                       linewidth=0.5)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
        ax2.set_ylabel('Final Score (mean of last 10%)', fontsize=12)
        ax2.set_title('Final Performance Comparison', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=11,
                     fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'ablation_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

    # Print summary
    print("\n=== Final Score Summary ===")
    for name, score in sorted(final_scores.items(),
                               key=lambda x: x[1]):
        print(f"  {name}: {score:.1f}")


if __name__ == '__main__':
    main()
