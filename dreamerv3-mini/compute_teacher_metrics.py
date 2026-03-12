#!/usr/bin/env python3
"""Compute 3-layer evaluation metrics for DreamerV3 TEACHER models."""
import os, sys, pathlib, argparse, numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
GAMMA = 0.997
TEST_FRACTION = 0.2

TEACHER_SCORES = {
    'crafter_reward_25m':  9.82,
    'crafter_reward_100m': 8.10,
    'crafter_reward':      8.92,
}

def detect_episodes(reward, discount_return, gamma=GAMMA):
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

def compute_1step_error_discrete(logits, actions):
    predicted = logits.argmax(axis=-1)
    return 1.0 - float((predicted == actions).mean())

def compute_kstep_rollout_error(logits, act, episodes, is_discrete, k):
    window_starts = []
    for ep_s, ep_e in episodes:
        if ep_e - ep_s >= k:
            for t in range(ep_s, ep_e - k + 1):
                window_starts.append(t)
    if not window_starts:
        return None, None
    window_starts = np.array(window_starts)
    n_w = len(window_starts)
    indices = window_starts[:, None] + np.arange(k)[None, :]
    flat_idx = indices.ravel()
    flat_logits = logits[flat_idx]
    flat_act = act[flat_idx]
    if is_discrete:
        pred = flat_logits.argmax(axis=-1).reshape(n_w, k)
        teacher = flat_act.reshape(n_w, k)
        per_step = (pred != teacher).astype(np.float32)
    else:
        pred = flat_logits.reshape(n_w, k, -1)
        teacher = flat_act.reshape(n_w, k, -1)
        per_step = ((pred - teacher) ** 2).mean(axis=-1)
    return float(per_step.mean(axis=1).mean()), float(per_step.mean(axis=1).std())

def compute_ndcg(teacher_logits, student_logits):
    t_max = teacher_logits.max(axis=-1, keepdims=True)
    t_exp = np.exp(teacher_logits - t_max)
    t_probs = t_exp / t_exp.sum(axis=-1, keepdims=True)
    _n, k = teacher_logits.shape
    discount = 1.0 / np.log2(np.arange(k) + 2)
    s_rank = np.argsort(-student_logits, axis=-1)
    t_rank = np.argsort(-t_probs, axis=-1)
    s_rel = np.take_along_axis(t_probs, s_rank, axis=-1)
    t_rel = np.take_along_axis(t_probs, t_rank, axis=-1)
    dcg  = (s_rel * discount).sum(axis=-1)
    idcg = (t_rel * discount).sum(axis=-1)
    return float((dcg / (idcg + 1e-10)).mean())

def compute_top1_hit(logits_a, logits_b):
    return float((logits_a.argmax(axis=-1) == logits_b.argmax(axis=-1)).mean())

def compute_kendall_tau(teacher_logits, student_logits):
    _n, k = teacher_logits.shape
    idx_i, idx_j = np.triu_indices(k, k=1)
    t_diff = teacher_logits[:, idx_i] - teacher_logits[:, idx_j]
    s_diff = student_logits[:, idx_i] - student_logits[:, idx_j]
    product = t_diff * s_diff
    concordant = (product > 0).sum(axis=1).astype(float)
    discordant = (product < 0).sum(axis=1).astype(float)
    total = concordant + discordant
    return float(np.where(total > 0, (concordant - discordant) / total, 0.0).mean())

def evaluate_teacher(task_name):
    data_path = DATA_DIR / f'teacher_data_{task_name}.npz'
    if not data_path.exists():
        print(f"  [SKIP] {data_path} not found"); return None
    d = np.load(str(data_path), allow_pickle=True)
    act = d['act']; reward = d['reward']; discount_return = d['discount_return']
    act_logits = d['act_logits']; act_type = str(d['act_type'])
    is_discrete = (act_type == 'discrete')
    n = len(act)
    print(f"\n{'='*70}")
    print(f"  TEACHER: {task_name}  ({n} transitions, {act_type})")
    print(f"{'='*70}")
    episodes = detect_episodes(reward, discount_return)
    print(f"  Episodes: {len(episodes)}")
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(episodes))
    n_test = max(1, int(len(episodes) * TEST_FRACTION))
    test_ep_idx = set(perm[:n_test].tolist())
    test_indices = []
    test_episodes_raw = []
    for i, (s, e) in enumerate(episodes):
        if i in test_ep_idx:
            new_s = len(test_indices)
            test_indices.extend(range(s, e))
            new_e = len(test_indices)
            test_episodes_raw.append((new_s, new_e))
    test_indices = np.array(test_indices)
    t_act = act[test_indices]; t_logits = act_logits[test_indices]
    t_rew = reward[test_indices]; t_dr = discount_return[test_indices]
    test_episodes = test_episodes_raw
    print(f"  Test set: {len(test_indices)} trans ({n_test} eps)")

    results = {}
    # Layer A
    k1 = compute_1step_error_discrete(t_logits, t_act) if is_discrete else float(np.mean((t_logits - t_act)**2))
    results['k1_error'] = k1
    k2m, _ = compute_kstep_rollout_error(t_logits, t_act, test_episodes, is_discrete, 2)
    results['k2_error'] = k2m
    k5m, _ = compute_kstep_rollout_error(t_logits, t_act, test_episodes, is_discrete, 5)
    results['k5_error'] = k5m
    print(f"  Layer A: k1={k1:.4f}  k2={k2m:.4f}  k5={k5m:.4f}")

    # Layer B
    if is_discrete:
        ndcg = compute_ndcg(t_logits, t_logits)
        top1 = compute_top1_hit(t_logits, t_logits)
        tau  = compute_kendall_tau(t_logits, t_logits)
        results['ndcg'] = ndcg; results['top1_hit'] = top1; results['kendall_tau'] = tau
        print(f"  Layer B: NDCG={ndcg:.4f}  Top1={top1:.4f}  tau={tau:.4f}")

    score = TEACHER_SCORES.get(task_name)
    results['score'] = score
    print(f"  Layer C: Score={score}")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', nargs='+', required=True)
    args = ap.parse_args()
    all_r = {}
    for t in args.task:
        r = evaluate_teacher(t)
        if r: all_r[t] = r
    if all_r:
        print(f"\n\n{'='*95}")
        print(f"  SUMMARY")
        print(f"{'='*95}")
        print(f"  {'Task':<25s} {'Score':>6s} {'k1 Err':>8s} {'k2 Err':>8s} {'k5 Err':>8s} {'NDCG':>8s} {'Top-1':>8s} {'tau':>8s}")
        print(f"  {'-'*90}")
        for t, r in all_r.items():
            s = f"{r['score']:.2f}" if r.get('score') else "N/A"
            print(f"  {t:<25s} {s:>6s} {r['k1_error']:>8.4f} {r['k2_error']:>8.4f} {r['k5_error']:>8.4f} {r.get('ndcg',0):>8.4f} {r.get('top1_hit',0):>8.4f} {r.get('kendall_tau',0):>8.4f}")

if __name__ == '__main__':
    main()
