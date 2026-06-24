#!/usr/bin/env python3
"""
Sanity-check a batch of finished experiments before committing to the next one.

For every experiment in the range this reports, per run:
  * completeness   - episodes recorded vs expected
  * learning       - first-N vs last-N mean reward (did it improve?)
  * final reward   - last-N mean (for cross-seed comparison)
  * health flags   - NaN/inf, didn't-learn, incomplete

and a per-(algorithm, season, aggregations) cross-seed summary that flags
high variance or non-learners - i.e. runs that "finished" but look wrong.

Pure stdlib + pyyaml (no pandas) so it runs on huron or a login node.

Usage:
    python validate_batch.py 4000-4179 --metrics-root /storage_1/metrics_postfix
    python validate_batch.py 4000-4179 --metrics-root ... --verbose
    python validate_batch.py 4000-4179 --metrics-root ... --exclude-algo CMA

Reward convention: rewards are negative; HIGHER (less negative) = better, so a
positive first->last delta means it learned.
"""

import argparse
import csv
import sys
from collections import defaultdict
from math import isnan, isinf
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. On DRAC: source ~/envs/merl_env/bin/activate", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent


def parse_ranges(specs):
    out = []
    for s in specs:
        s = s.strip()
        if s.lower().endswith("xxx"):
            base = int(s[:-3]) * 1000
            out.append((base, base + 999))
        elif "-" in s:
            a, b = s.split("-")
            out.append((int(a), int(b)))
        else:
            out.append((int(s), int(s)))
    return out


def resolve_root(override):
    cands = ([Path(override)] if override else
             [Path("/storage_1/metrics_postfix"), Path("/storage_1/metrics"),
              Path.home() / "scratch" / "metrics"])
    for r in cands:
        if r.is_dir():
            return r
    return None


def expected_episodes(cfg):
    algo = cfg["algorithm_settings"]["algorithm"]
    aggs = cfg.get("federated_learning_settings", {}).get("aggregation_count", 1)
    if algo in ("DQN", "REINFORCE", "PPO", "DDPG", "ODT"):
        per = cfg.get("nn_hyperparameters", {}).get("num_episodes")
    elif algo in ("CMA", "DENSER", "NEAT"):
        per = cfg.get("cma_parameters", {}).get("max_generations")
    else:
        per = None
    return (aggs * per if per else None), aggs


def scan(csv_path):
    """Return ordered list of per-(agg,ep) mean rewards, and a nan/inf flag."""
    rewards = {}
    order = []
    bad = False
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        try:
            ia, ie, ir = header.index("aggregation"), header.index("episode"), header.index("reward")
        except ValueError:
            return [], False
        for row in r:
            if len(row) <= max(ia, ie, ir):
                continue
            try:
                key = (int(row[ia]), int(row[ie]))
                val = float(row[ir])
            except ValueError:
                continue
            if isnan(val) or isinf(val):
                bad = True
                continue
            if key not in rewards:
                rewards[key] = [0.0, 0]
                order.append(key)
            rewards[key][0] += val
            rewards[key][1] += 1
    means = [rewards[k][0] / rewards[k][1] for k in order]   # CSV order = time order
    return means, bad


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Health/sanity check for a finished experiment batch.")
    p.add_argument("ranges", nargs="+", help="e.g. 4000-4179")
    p.add_argument("--metrics-root")
    p.add_argument("--verbose", action="store_true", help="one row per experiment (not just flagged ones)")
    p.add_argument("--exclude-algo", nargs="*", default=[], help="skip algos, e.g. --exclude-algo CMA")
    p.add_argument("--learn-min", type=float, default=2.0, help="min first->last reward gain to count as learning")
    p.add_argument("--spread-max", type=float, default=25.0, help="flag a seed-group whose final-reward range exceeds this")
    p.add_argument("--complete-pct", type=float, default=95.0, help="%% of expected episodes to count as complete")
    args = p.parse_args()

    root = resolve_root(args.metrics_root)
    if root is None:
        print("ERROR: no metrics root found.", file=sys.stderr)
        return 1
    print(f"Metrics root: {root}\n")
    exclude = {a.upper() for a in args.exclude_algo}

    rows = []           # per-experiment dicts
    groups = defaultdict(list)   # (algo, season, aggs) -> [(seed, final, flags)]

    for a, b in parse_ranges(args.ranges):
        for n in range(a, b + 1):
            cfgp = REPO / "experiments" / f"Exp_{n}" / "config.yaml"
            if not cfgp.exists():
                continue
            cfg = yaml.safe_load(open(cfgp))
            algo = cfg["algorithm_settings"]["algorithm"]
            if algo.upper() in exclude:
                continue
            env = cfg["environment_settings"]
            season, seed = env.get("season", "?"), env.get("seed", "?")
            expected, aggs = expected_episodes(cfg)
            csvp = root / f"Exp_{n}" / "train" / "metrics_agent_episode_level.csv"

            row = {"n": n, "algo": algo, "season": season, "seed": seed, "aggs": aggs, "flags": []}
            if not csvp.exists():
                row["flags"].append("NO-DATA")
                rows.append(row)
                continue

            means, bad = scan(csvp)
            nep = len(means)
            row["nep"] = nep
            row["pct"] = (nep / expected * 100) if expected else 0.0
            if bad:
                row["flags"].append("NaN/inf")
            if nep == 0:
                row["flags"].append("EMPTY")
                rows.append(row)
                continue

            win = max(1, min(200, nep // 4))
            first = sum(means[:win]) / win
            last = sum(means[-win:]) / win
            row["first"], row["last"], row["delta"] = first, last, last - first
            row["min"], row["max"] = min(means), max(means)

            if row["pct"] < args.complete_pct:
                row["flags"].append(f"INCOMPLETE({row['pct']:.0f}%)")
            if row["delta"] < args.learn_min:
                row["flags"].append(f"NO-LEARN(gain{row['delta']:+.1f})")

            rows.append(row)
            groups[(algo, season, aggs)].append((seed, last, bool(row["flags"])))

    # ---- per-experiment output ----
    flagged = [r for r in rows if r["flags"]]
    if args.verbose:
        print(f"{'Exp':>6} {'algo':>9} {'season':>7} {'seed':>5} {'aggs':>4} "
              f"{'%':>5} {'first':>8} {'last':>8} {'gain':>7}  flags")
        print("-" * 86)
        for r in rows:
            if "first" in r:
                print(f"{r['n']:>6} {r['algo']:>9} {r['season']:>7} {str(r['seed']):>5} {r['aggs']:>4} "
                      f"{r['pct']:>4.0f}% {r['first']:>8.1f} {r['last']:>8.1f} {r['delta']:>+7.1f}  "
                      f"{', '.join(r['flags']) or 'ok'}")
            else:
                print(f"{r['n']:>6} {r['algo']:>9} {r['season']:>7} {str(r['seed']):>5} {r['aggs']:>4} "
                      f"{'--':>5} {'--':>8} {'--':>8} {'--':>7}  {', '.join(r['flags'])}")
        print()

    # ---- cross-seed consistency ----
    print("CROSS-SEED CHECK  (same algo/season/aggs across seeds - final reward should agree)")
    print("-" * 78)
    inconsistent = 0
    for key in sorted(groups):
        finals = [f for _, f, _ in groups[key]]
        spread = max(finals) - min(finals)
        flag = ""
        if spread > args.spread_max:
            flag = f"  <-- HIGH SPREAD ({spread:.0f})"
            inconsistent += 1
        algo, season, aggs = key
        vals = ", ".join(f"{f:.1f}" for f in finals)
        print(f"  {algo:>9} {season:>7} {aggs:>3}agg  finals=[{vals}]{flag}")

    # ---- summary ----
    print("\n" + "=" * 60)
    n_total = len(rows)
    n_ok = sum(1 for r in rows if not r["flags"])
    print(f"  experiments checked: {n_total}")
    print(f"  healthy (no flags):  {n_ok}")
    print(f"  flagged:             {len(flagged)}")
    print("=" * 60)
    if flagged:
        print("\nFLAGGED (review these before the next batch):")
        for r in sorted(flagged, key=lambda x: x["n"]):
            extra = ""
            if "last" in r:
                extra = f"  (final={r['last']:.1f}, gain={r['delta']:+.1f}, {r['pct']:.0f}%)"
            print(f"  Exp_{r['n']:<5} {r['algo']:>9} {r['season']:>7} seed={r['seed']}: "
                  f"{', '.join(r['flags'])}{extra}")
    if inconsistent:
        print(f"\n{inconsistent} seed-group(s) disagree by > {args.spread_max:.0f} reward - "
              f"a 'finished' run there may be a bad seed/unstable; eyeball its curve with plot_reward_curve.py.")
    if not flagged and not inconsistent:
        print("\nAll runs complete, learned, and seed-consistent. Reasonable to proceed to 5xxx/6xxx.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
