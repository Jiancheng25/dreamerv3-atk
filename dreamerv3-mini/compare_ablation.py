#!/usr/bin/env python3
"""
Parse train.log files from ablation runs and generate a comparison report.

Usage:
    python compare_ablation.py --task dmc_walker_walk
"""
import argparse
import pathlib
import re
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / 'runs'

VARIANTS = ['baseline', 'A', 'B']


def parse_log(log_path):
    """Extract best score and per-epoch eval scores from train.log."""
    evals = []
    best_score = None
    best_epoch = None
    pattern = re.compile(
        r'Epoch\s+(\d+)/\d+\s+\|.*eval\s+=\s+([\d.]+)\s+\+/-\s+([\d.]+)')
    best_pattern = re.compile(r'Best eval score\s*:\s*([\d.]+)')

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                mean = float(m.group(2))
                std = float(m.group(3))
                is_best = '** NEW BEST' in line
                evals.append({'epoch': epoch, 'mean': mean, 'std': std,
                              'is_best': is_best})
                if is_best:
                    best_score = mean
                    best_epoch = epoch
            m2 = best_pattern.search(line)
            if m2:
                best_score = float(m2.group(1))

    return {'evals': evals, 'best_score': best_score, 'best_epoch': best_epoch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    args = ap.parse_args()

    task_dir = RUNS_DIR / args.task
    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}")
        sys.exit(1)

    results = {}
    for v in VARIANTS:
        log_path = task_dir / v / 'train.log'
        if log_path.exists():
            results[v] = parse_log(log_path)
        else:
            results[v] = None

    # ── Generate comparison report ────────────────────────────────────
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  Ablation Comparison: {args.task}")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append(f"{'Variant':<12} {'Best Score':>12} {'Best Epoch':>12} {'Status'}")
    lines.append(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    scores = {}
    for v in VARIANTS:
        r = results.get(v)
        if r and r['best_score'] is not None:
            scores[v] = r['best_score']
            lines.append(f"{v:<12} {r['best_score']:>12.1f} {r['best_epoch']:>12d} {'OK'}")
        else:
            lines.append(f"{v:<12} {'N/A':>12} {'N/A':>12} {'MISSING'}")

    lines.append("")

    # ── Ordering check ────────────────────────────────────────────────
    if len(scores) == 3:
        b  = scores.get('baseline', 0)
        a  = scores.get('A', 0)
        bv = scores.get('B', 0)
        ordering_ok = b < a and b < bv
        lines.append(f"Target ordering: baseline < A  AND  baseline < B")
        lines.append(f"Actual values:   baseline={b:.1f}  A={a:.1f}  B={bv:.1f}")
        lines.append(f"Ordering satisfied: {'YES' if ordering_ok else 'NO'}")
    else:
        lines.append("(Incomplete results — cannot check ordering)")

    lines.append("")

    # ── Per-epoch eval history ────────────────────────────────────────
    lines.append("Per-epoch evaluation scores:")
    header = f"{'Epoch':>6}"
    for v in VARIANTS:
        header += f"  {v:>12}"
    lines.append(header)

    # collect all epochs
    all_epochs = set()
    for v in VARIANTS:
        r = results.get(v)
        if r:
            all_epochs.update(e['epoch'] for e in r['evals'])
    for ep in sorted(all_epochs):
        row = f"{ep:>6}"
        for v in VARIANTS:
            r = results.get(v)
            if r:
                ev = [e for e in r['evals'] if e['epoch'] == ep]
                if ev:
                    row += f"  {ev[0]['mean']:>12.1f}"
                else:
                    row += f"  {'':>12}"
            else:
                row += f"  {'':>12}"
        lines.append(row)

    lines.append(f"{'='*70}")

    # Fix the header line issue (list append vs print)
    report = '\n'.join(lines)
    print(report)

    # Save to file
    out_path = task_dir / 'ablation_comparison.txt'
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
