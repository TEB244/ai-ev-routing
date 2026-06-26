#!/usr/bin/env python3
"""
Find the best 4xxx base model per decision-maker, for the 10xxx inference batch.

Scans the 4xxx experiments' training metrics and ranks them, per DM, by their
converged performance (mean reward over the last N episodes). The winner of each
DM is what you should set as the source model in generate_inference_experiments.py
(SOURCE_EXP) so the inference runs load a genuinely good, fully-trained model.

Pure stdlib + pyyaml (matches check_progress.py) -- no pandas needed.

Metrics layout (same as check_progress.py):
    <metrics-root>/Exp_<N>/train/metrics_agent_episode_level.csv
    columns include: aggregation, episode, agent_index, reward, distance

On the huron server the postfix-code metrics live under metrics_postfix, e.g.:
    python find_best_base_models.py --metrics-root /storage_1/metrics_postfix

Usage:
    python find_best_base_models.py
    python find_best_base_models.py --metrics-root /storage_1/metrics_postfix
    python find_best_base_models.py --range 4000 4143 --last-n 100 --min-complete 0.8
    python find_best_base_models.py --csv-out base_model_ranking.csv

"Best" = highest mean reward over the last --last-n episodes (reward is the
training objective; higher/less-negative is better). Mean distance (lower is
better) and completion % are shown as cross-checks. Experiments below
--min-complete of their configured episode budget are flagged and de-prioritised
so an undertrained run can't win on a lucky tail.
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. On DRAC/huron: source ~/envs/merl_env/bin/activate",
          file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent
EXP_DIR = REPO_ROOT / "experiments"

DM_ORDER = ["DQN", "REINFORCE", "CMA", "ODT"]


def metrics_root_candidates(override):
    if override:
        return [Path(override)]
    return [
        Path("/storage_1/metrics_postfix"),   # huron, post-fix code (this study)
        Path("/storage_1/metrics"),           # huron
        Path.home() / "scratch" / "metrics",  # DRAC ($USER)
        REPO_ROOT / "_local_metrics",         # local testing
    ]


def resolve_root(override):
    for r in metrics_root_candidates(override):
        if r.is_dir():
            return r
    return None


def expected_episodes(cfg):
    """Configured total episodes/generations (for completion %)."""
    algo = cfg["algorithm_settings"]["algorithm"]
    agg = cfg.get("federated_learning_settings", {}).get("aggregation_count", 1)
    if algo in ("DQN", "REINFORCE", "PPO", "DDPG", "ODT"):
        per = cfg.get("nn_hyperparameters", {}).get("num_episodes")
    elif algo in ("CMA", "DENSER", "NEAT"):
        per = cfg.get("cma_parameters", {}).get("max_generations")
    else:
        per = None
    return (agg * per) if per else None


def find_agent_csv(root, exp_num):
    """Locate the per-episode agent metrics CSV for an experiment, if present."""
    base = root / f"Exp_{exp_num}" / "train"
    preferred = base / "metrics_agent_episode_level.csv"
    if preferred.exists():
        return preferred
    if base.is_dir():
        hits = sorted(base.glob("metrics_agent_*.csv"))
        if hits:
            return hits[0]
    return None


def scan_agent_csv(csv_path):
    """
    Single-pass scan. Returns:
        pairs            : sorted distinct (agg, ep) tuples
        reward_per_pair  : dict (agg, ep) -> [reward, ...]   (per agent)
        dist_per_pair    : dict (agg, ep) -> [distance, ...]
    """
    pairs = set()
    reward_per_pair = {}
    dist_per_pair = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            i_agg = header.index("aggregation")
            i_ep = header.index("episode")
            i_rew = header.index("reward")
        except ValueError as e:
            raise RuntimeError(f"Missing expected column in {csv_path}: {e}")
        i_dist = header.index("distance") if "distance" in header else None

        for row in reader:
            if len(row) <= max(i_agg, i_ep, i_rew):
                continue
            try:
                agg = int(row[i_agg])
                ep = int(row[i_ep])
                rew = float(row[i_rew])
            except ValueError:
                continue
            pair = (agg, ep)
            pairs.add(pair)
            reward_per_pair.setdefault(pair, []).append(rew)
            if i_dist is not None:
                try:
                    dist_per_pair.setdefault(pair, []).append(float(row[i_dist]))
                except ValueError:
                    pass
    return sorted(pairs), reward_per_pair, dist_per_pair


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def evaluate_exp(root, exp_num, last_n):
    """Return a result dict for one experiment, or None if no usable metrics."""
    cfg_path = EXP_DIR / f"Exp_{exp_num}" / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    algo = cfg["algorithm_settings"]["algorithm"]
    seed = cfg["environment_settings"]["seed"]
    exp_total = expected_episodes(cfg)

    csv_path = find_agent_csv(root, exp_num)
    if csv_path is None:
        return {"exp": exp_num, "dm": algo, "seed": seed, "status": "no-metrics",
                "actual": 0, "expected": exp_total, "complete": 0.0,
                "reward": float("nan"), "distance": float("nan")}
    try:
        pairs, rew_pp, dist_pp = scan_agent_csv(csv_path)
    except Exception as e:
        return {"exp": exp_num, "dm": algo, "seed": seed, "status": f"error:{e}",
                "actual": 0, "expected": exp_total, "complete": 0.0,
                "reward": float("nan"), "distance": float("nan")}

    actual = len(pairs)
    complete = (actual / exp_total) if (exp_total and exp_total > 0) else float("nan")
    tail = pairs[-min(last_n, actual):] if actual else []
    tail_rewards = [r for p in tail for r in rew_pp.get(p, [])]
    tail_dists = [d for p in tail for d in dist_pp.get(p, [])]
    return {"exp": exp_num, "dm": algo, "seed": seed, "status": "ok",
            "actual": actual, "expected": exp_total,
            "complete": complete,
            "reward": mean(tail_rewards), "distance": mean(tail_dists)}


def main():
    p = argparse.ArgumentParser(
        description="Rank 4xxx base models per DM by converged reward.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--range", nargs=2, type=int, default=[4000, 4143],
                   help="Inclusive experiment range (default: 4000 4143).")
    p.add_argument("--metrics-root", help="Metrics root (default: auto, prefers /storage_1/metrics_postfix).")
    p.add_argument("--last-n", type=int, default=100,
                   help="Episodes from the end to average for converged performance (default: 100).")
    p.add_argument("--min-complete", type=float, default=0.8,
                   help="Min fraction of configured episodes to be eligible to win (default: 0.8).")
    p.add_argument("--top", type=int, default=5, help="Rows to show per DM (default: 5).")
    p.add_argument("--csv-out", help="Optional path to write all rows as CSV.")
    args = p.parse_args()

    root = resolve_root(args.metrics_root)
    if root is None:
        print("ERROR: no metrics root found. Tried:", file=sys.stderr)
        for c in metrics_root_candidates(args.metrics_root):
            print(f"  {c}", file=sys.stderr)
        return 1
    print(f"Metrics root: {root}")
    print(f"Ranking by mean reward over last {args.last_n} episodes "
          f"(eligible if >= {args.min_complete:.0%} complete)\n")

    results = []
    for n in range(args.range[0], args.range[1] + 1):
        r = evaluate_exp(root, n, args.last_n)
        if r is not None:
            results.append(r)

    by_dm = {}
    for r in results:
        by_dm.setdefault(r["dm"], []).append(r)

    recommended = {}
    for dm in DM_ORDER + sorted(set(by_dm) - set(DM_ORDER)):
        rows = by_dm.get(dm)
        if not rows:
            continue
        # Eligible = has metrics and meets completion bar (NaN completion -> treat as eligible).
        def eligible(r):
            return r["status"] == "ok" and r["actual"] > 0 and (
                r["complete"] != r["complete"] or r["complete"] >= args.min_complete)
        # Sort by reward desc (higher is better), NaN rewards last.
        rows_sorted = sorted(
            rows, key=lambda r: (-r["reward"] if r["reward"] == r["reward"] else float("inf")))

        print(f"=== {dm} ===")
        print(f"  {'exp':>5}  {'seed':>5}  {'reward':>10}  {'distance':>9}  "
              f"{'eps':>6}  {'complete':>8}")
        for r in rows_sorted[:args.top]:
            comp = "n/a" if r["complete"] != r["complete"] else f"{r['complete']*100:5.1f}%"
            flag = "" if eligible(r) else "  (low-complete)"
            rew = f"{r['reward']:+.2f}" if r["reward"] == r["reward"] else "  n/a"
            dist = f"{r['distance']:.3f}" if r["distance"] == r["distance"] else "n/a"
            print(f"  {r['exp']:>5}  {r['seed']:>5}  {rew:>10}  {dist:>9}  "
                  f"{r['actual']:>6}  {comp:>8}{flag}")

        winner = next((r for r in rows_sorted if eligible(r)), None)
        if winner is None:
            winner = next((r for r in rows_sorted if r["reward"] == r["reward"]), None)
            note = "  (no run met completion bar; best available)"
        else:
            note = ""
        if winner:
            recommended[dm] = winner["exp"]
            print(f"  -> best: Exp_{winner['exp']} (reward {winner['reward']:+.2f}){note}\n")
        else:
            print(f"  -> no usable metrics for {dm}\n")

    if recommended:
        print("Recommended SOURCE_EXP (paste into generate_inference_experiments.py):")
        print("SOURCE_EXP = {")
        for dm in DM_ORDER:
            if dm in recommended:
                print(f'    "{dm}":{" " * (10 - len(dm))}{recommended[dm]},')
        print("}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["exp", "dm", "seed", "status", "actual_eps",
                        "expected_eps", "complete_frac", "mean_reward", "mean_distance"])
            for r in sorted(results, key=lambda x: x["exp"]):
                w.writerow([r["exp"], r["dm"], r["seed"], r["status"], r["actual"],
                            r["expected"], f"{r['complete']:.4f}" if r["complete"] == r["complete"] else "",
                            f"{r['reward']:.6f}" if r["reward"] == r["reward"] else "",
                            f"{r['distance']:.6f}" if r["distance"] == r["distance"] else ""])
        print(f"\nAll rows written to {args.csv_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
