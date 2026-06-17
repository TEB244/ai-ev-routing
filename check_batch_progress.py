#!/usr/bin/env python3
"""
Batch progress + wall-time-need estimate across experiment ranges.

For every experiment in the given ranges, reports how far it got (episodes
& aggregations vs expected) and, for runs that did NOT finish, estimates the
wall time it would have needed = configured_wall / fraction_completed. Use
the per-DM recommendation at the bottom to size 5xxx/6xxx wall times.

Pure stdlib + pyyaml (no pandas) so it runs on huron or a login node.

Usage:
    python check_batch_progress.py 4000-4179 --metrics-root /storage_1/metrics_postfix
    python check_batch_progress.py 4000-4035 4108-4179 --metrics-root /storage_1/metrics_postfix
    python check_batch_progress.py 4000-4179 --metrics-root ... --verbose

Caveats:
  * "needed wall" assumes a partial run consumed ~its full wall (true for a
    SLURM TIME LIMIT) and progressed roughly linearly. A run that died early
    (watchdog "did not finish within N s", or a crash) shows a tiny fraction,
    so its estimate is bogus -- those are flagged with (!) and EXCLUDED from
    the per-DM recommendation. Fix those via the watchdog/crash, not wall time.
"""

import argparse
import csv
import re
import sys
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


def metrics_root_candidates(override):
    if override:
        return [Path(override)]
    return [Path("/storage_1/metrics_postfix"), Path("/storage_1/metrics"),
            Path.home() / "scratch" / "metrics"]


def resolve_root(override):
    for r in metrics_root_candidates(override):
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
    total = aggs * per if per else None
    return total, aggs, per


def wall_seconds(exp_dir):
    tj = exp_dir / "train_job.sh"
    if not tj.exists():
        return None
    m = re.search(r"--time=(\d+):(\d\d):(\d\d)", tj.read_text())
    if not m:
        return None
    h, mm, ss = (int(x) for x in m.groups())
    return h * 3600 + mm * 60 + ss


def scan_episodes(csv_path):
    """Return (n_distinct_episodes, max_aggregation_index_seen)."""
    seen = set()
    max_agg = -1
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            i_agg = header.index("aggregation")
            i_ep = header.index("episode")
        except ValueError:
            return 0, -1
        for row in reader:
            if len(row) <= max(i_agg, i_ep):
                continue
            try:
                agg = int(row[i_agg]); ep = int(row[i_ep])
            except ValueError:
                continue
            seen.add((agg, ep))
            if agg > max_agg:
                max_agg = agg
    return len(seen), max_agg


def fmt_h(seconds):
    return "?" if seconds is None else f"{seconds / 3600:.1f}h"


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Batch progress + wall-need estimate.")
    p.add_argument("ranges", nargs="+", help="e.g. 4000-4179  or  4xxx")
    p.add_argument("--metrics-root")
    p.add_argument("--verbose", action="store_true", help="one row per experiment")
    p.add_argument("--done-pct", type=float, default=99.0, help="%% complete to call DONE")
    p.add_argument("--margin", type=float, default=1.3, help="safety multiplier on recommended wall")
    p.add_argument("--incomplete-only", action="store_true",
                   help="print ONLY the experiment numbers that aren't DONE (one per line, "
                        "for piping into an sbatch loop); table/summary go to stderr")
    p.add_argument("--exclude-algo", nargs="*", default=[],
                   help="skip experiments for these algorithms, e.g. --exclude-algo CMA "
                        "(CMA runs on another user's scratch and won't be on huron)")
    args = p.parse_args()

    root = resolve_root(args.metrics_root)
    if root is None:
        print("ERROR: no metrics root found.", file=sys.stderr)
        return 1

    # In --incomplete-only mode, all human-readable output goes to stderr so
    # stdout is a clean list of experiment numbers to pipe into sbatch.
    out = sys.stderr if args.incomplete_only else sys.stdout
    exclude = {a.upper() for a in (args.exclude_algo or [])}
    print(f"Metrics root: {root}\n", file=out)

    ranges = parse_ranges(args.ranges)
    need = {}          # algo -> max needed seconds
    counts = {}        # algo -> {done, partial, early, nodata}
    incomplete = []    # exp numbers that are not DONE (need running)

    if args.verbose:
        print(f"{'Exp':>6} {'algo':>9} {'season':>7} {'aggs':>9} {'episodes':>15} "
              f"{'%':>6} {'wall':>6} {'need~':>7} status", file=out)
        print("-" * 86, file=out)

    for a, b in ranges:
        for n in range(a, b + 1):
            cfg_p = REPO / "experiments" / f"Exp_{n}" / "config.yaml"
            if not cfg_p.exists():
                continue
            cfg = yaml.safe_load(open(cfg_p))
            algo = cfg["algorithm_settings"]["algorithm"]
            if algo.upper() in exclude:
                continue
            season = cfg["environment_settings"].get("season", "?")
            total, aggs, per = expected_episodes(cfg)
            wall = wall_seconds(REPO / "experiments" / f"Exp_{n}")
            counts.setdefault(algo, {"done": 0, "partial": 0, "early": 0, "nodata": 0})

            csv_p = root / f"Exp_{n}" / "train" / "metrics_agent_episode_level.csv"
            if not csv_p.exists():
                counts[algo]["nodata"] += 1
                incomplete.append(n)
                if args.verbose:
                    print(f"{n:>6} {algo:>9} {season:>7} {'-':>9} {'no csv':>15} "
                          f"{'-':>6} {fmt_h(wall):>6} {'-':>7} NO DATA", file=out)
                continue

            actual, max_agg = scan_episodes(csv_p)
            pct = (actual / total * 100) if total else 0.0
            frac = (actual / total) if total else 0.0
            aggs_str = f"{max_agg + 1}/{aggs}"
            eps_str = f"{actual}/{total}" if total else str(actual)

            if pct >= args.done_pct:
                status = "DONE"
                counts[algo]["done"] += 1
                need_s = None
            elif frac < 0.05:
                status = "EARLY-EXIT(!)"     # likely watchdog/crash, not wall
                counts[algo]["early"] += 1
                need_s = None
                incomplete.append(n)
            else:
                status = "PARTIAL"
                counts[algo]["partial"] += 1
                need_s = (wall / frac) if (wall and frac > 0) else None
                if need_s:
                    need[algo] = max(need.get(algo, 0), need_s)
                incomplete.append(n)

            if args.verbose:
                print(f"{n:>6} {algo:>9} {season:>7} {aggs_str:>9} {eps_str:>15} "
                      f"{pct:>5.0f}% {fmt_h(wall):>6} {fmt_h(need_s):>7} {status}", file=out)

    print("\n" + "=" * 60, file=out)
    print(f"{'algo':>9}  {'done':>5} {'partial':>8} {'early(!)':>9} {'nodata':>7}", file=out)
    print("-" * 60, file=out)
    for algo, c in sorted(counts.items()):
        print(f"{algo:>9}  {c['done']:>5} {c['partial']:>8} {c['early']:>9} {c['nodata']:>7}", file=out)
    print("=" * 60, file=out)

    if not args.incomplete_only:
        print("\nRECOMMENDED WALL TIME for the next batch (from partial-run estimates):")
        if not need:
            print("  (no usable partial runs — every run either finished or exited early.)")
        for algo, secs in sorted(need.items()):
            rec_h = int((secs * args.margin) / 3600) + 1   # round up to whole hours
            print(f"  {algo:>9}: worst partial needed ~{secs/3600:.1f}h  ->  set ~{rec_h}h  "
                  f"(x{args.margin} margin)")
        if any(c["early"] for c in counts.values()):
            print("\n(!) EARLY-EXIT runs are NOT wall-time failures (tiny progress) -- likely the")
            print("    per-aggregation watchdog or a crash. Check their error.log; fix the cause,")
            print("    not the wall. (The watchdog scaling fix in main.py addresses the 1-agg case.)")

    if args.incomplete_only:
        incs = sorted(set(incomplete))
        print(f"\n{len(incs)} experiment(s) not DONE -> need running:", file=out)
        for n in incs:
            print(n)   # stdout: clean list for piping
    return 0


if __name__ == "__main__":
    sys.exit(main())
