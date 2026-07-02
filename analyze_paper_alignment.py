#!/usr/bin/env python3
"""
analyze_paper_alignment.py

Rough pre-6xxx check: does the finished 4xxx + 5xxx data look CORRECT, and does it
AGREE with the SURE-DM paper's expected findings -- so you can commit to the 6xxx
batch without discovering afterwards that something needs a rerun?

It deliberately separates two very different kinds of problem:

  * DATA problems  -> FIX + RERUN before 6xxx
      missing / incomplete / NaN / flat-degenerate runs, or 4xxx-vs-5xxx seed
      instability. These are bugs; adding 6xxx on top won't rescue them.

  * NARRATIVE problems  -> revise the paper, do NOT rerun
      a ranking that is stable across seeds but differs from the paper's claim.
      The runs are sound; the finding just changed.

Paper reference (paper_figure_generators/PAPER_REVISION_HANDOFF.md). Reward is a
negated cost, so it is NEGATIVE and HIGHER (less negative) = BETTER:
    DM ranking, best -> worst:  DQN (-79.4)  >  REINFORCE (-89)  >  CMA (-115.3)
    ODT's rank is intentionally NOT asserted (the action-space discretisation left
    it open), so ODT is reported descriptively, never as pass/fail.

Layout (per 1000-block): DQN x000-x035, REINFORCE x036-x071, CMA x072-x107,
ODT x108-x179. 4xxx / 5xxx / 6xxx differ only by seed (1234 / 2468 / 3702), so
4xxx and 5xxx are genuine replicates of the same configs.

Pure stdlib + pyyaml, so it runs on huron or a login node. ASCII-only output.

Usage:
    python analyze_paper_alignment.py                                  # defaults to 4xxx 5xxx
    python analyze_paper_alignment.py 4xxx 5xxx --metrics-root /storage_1/metrics_postfix
    python analyze_paper_alignment.py 4xxx 5xxx --metrics-root ... --verbose
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

# ---- paper's expected findings (PAPER_REVISION_HANDOFF.md) ----
EXPECTED_ORDER = ["DQN", "REINFORCE", "CMA"]                        # best -> worst (higher reward better)
REF_REWARD = {"DQN": -79.39, "REINFORCE": -89.0, "CMA": -115.30}   # loose magnitude anchors
DM_ORDER = ["CMA", "DQN", "ODT", "REINFORCE"]                      # alphabetical (paper figure convention)
RL_DMS = {"DQN", "REINFORCE", "ODT"}                               # expected to improve over training


def parse_ranges(specs):
    out = []
    for s in specs:
        s = s.strip()
        if s.lower().endswith("xxx"):
            base = int(s[:-3]) * 1000
            out.append((base, base + 999))
        elif "-" in s:
            a, b = s.split("-"); out.append((int(a), int(b)))
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
    """Per-(aggregation, episode) mean reward in time order, plus a NaN/inf flag."""
    acc, order, bad = {}, [], False
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
                key = (int(row[ia]), int(row[ie])); val = float(row[ir])
            except ValueError:
                continue
            if isnan(val) or isinf(val):
                bad = True; continue
            if key not in acc:
                acc[key] = [0.0, 0]; order.append(key)
            acc[key][0] += val; acc[key][1] += 1
    return [acc[k][0] / acc[k][1] for k in order], bad


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x):
    return "   --  " if x is None or (isinstance(x, float) and isnan(x)) else f"{x:7.1f}"


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Pre-6xxx sanity + paper-alignment check.")
    p.add_argument("ranges", nargs="*", default=["4xxx", "5xxx"],
                   help="batches to check (default: 4xxx 5xxx)")
    p.add_argument("--metrics-root")
    p.add_argument("--verbose", action="store_true", help="list every experiment")
    p.add_argument("--final-window", type=int, default=200,
                   help="episodes at the end to average for 'final reward' (paper uses 200)")
    p.add_argument("--learn-min", type=float, default=2.0,
                   help="min first->last reward gain for an RL run to count as learning")
    p.add_argument("--spread-max", type=float, default=30.0,
                   help="flag a config whose 4xxx-vs-5xxx final reward differs by more than this")
    p.add_argument("--mag-tol", type=float, default=3.0,
                   help="flag a DM whose pooled reward is >this-x off the paper reference")
    args = p.parse_args()

    root = resolve_root(args.metrics_root)
    if root is None:
        print("ERROR: no metrics root found (pass --metrics-root).", file=sys.stderr)
        return 1
    print(f"Metrics root: {root}")
    print(f"Batches: {', '.join(args.ranges)}\n")

    rows = []
    for a, b in parse_ranges(args.ranges):
        for n in range(a, b + 1):
            cfgp = REPO / "experiments" / f"Exp_{n}" / "config.yaml"
            if not cfgp.exists():
                continue
            cfg = yaml.safe_load(open(cfgp))
            algo = cfg["algorithm_settings"]["algorithm"]
            env = cfg["environment_settings"]
            season, seed = env.get("season", "?"), env.get("seed", "?")
            expected, aggs = expected_episodes(cfg)
            row = {"n": n, "batch": n // 1000, "algo": algo, "season": season,
                   "seed": seed, "aggs": aggs, "flags": [], "final": None, "gain": None}
            csvp = root / f"Exp_{n}" / "train" / "metrics_agent_episode_level.csv"
            if not csvp.exists():
                row["flags"].append("NO-DATA"); rows.append(row); continue
            means, bad = scan(csvp)
            nep = len(means)
            row["pct"] = (nep / expected * 100) if expected else 0.0
            if bad:
                row["flags"].append("NaN/inf")
            if nep == 0:
                row["flags"].append("EMPTY"); rows.append(row); continue
            w = max(1, min(args.final_window, nep // 4 or 1))
            first, last = mean(means[:w]), mean(means[-w:])
            row["first"], row["final"], row["gain"] = first, last, last - first
            row["rmin"], row["rmax"] = min(means), max(means)
            if expected and row["pct"] < 95.0:
                row["flags"].append(f"INCOMPLETE({row['pct']:.0f}%)")
            if row["rmax"] - row["rmin"] < 1e-6:
                row["flags"].append("DEGENERATE(flat)")
            elif algo in RL_DMS and row["gain"] < args.learn_min:
                row["flags"].append(f"NO-LEARN(gain{row['gain']:+.1f})")
            rows.append(row)

    if not rows:
        print("No experiments found for those ranges (are the configs in this repo?).", file=sys.stderr)
        return 1

    # ---------- [1] DATA HEALTH ----------
    print("=" * 78)
    print("[1] DATA HEALTH  (hard flags here mean fix + rerun before 6xxx)")
    print("-" * 78)
    flagged = [r for r in rows if r["flags"]]
    have = [r for r in rows if r["final"] is not None]
    if args.verbose:
        for r in sorted(rows, key=lambda x: x["n"]):
            print(f"  Exp_{r['n']}  {r['algo']:>9} {str(r['season']):>7} seed={str(r['seed']):>4} "
                  f"{r['aggs']}agg  final={fmt(r['final'])}  {', '.join(r['flags']) or 'ok'}")
        print()
    print(f"  experiments found : {len(rows)}")
    print(f"  usable (have data): {len(have)}")
    print(f"  flagged           : {len(flagged)}")
    for r in sorted(flagged, key=lambda x: x["n"]):
        print(f"    Exp_{r['n']:<5} {r['algo']:>9} {str(r['season']):>7} seed={r['seed']}: {', '.join(r['flags'])}")

    if not have:
        print("\nNo usable reward data -- every run is missing or empty (see [1]).")
        print("Sort the data out before the alignment checks can say anything.")
        return 1

    # ---------- [2] pooled DM ranking ----------
    by_dm = defaultdict(list)
    for r in have:
        by_dm[r["algo"]].append(r["final"])
    dm_mean = {dm: mean(v) for dm, v in by_dm.items()}
    ranked = sorted(dm_mean, key=lambda d: dm_mean[d], reverse=True)   # higher = better

    print("\n" + "=" * 78)
    print(f"[2] DECISION-MAKER RANKING  (pooled final-{args.final_window}-ep mean reward; higher = better)")
    print("-" * 78)
    for dm in ranked:
        if dm == "ODT":
            tag = "  (paper leaves ODT's rank open)"
        elif dm == EXPECTED_ORDER[0]:
            tag = "  <- paper says best"
        elif dm == EXPECTED_ORDER[-1]:
            tag = "  <- paper says worst"
        else:
            tag = ""
        print(f"    {dm:>9}  {dm_mean[dm]:8.1f}   (n={len(by_dm[dm])}){tag}")
    computed_core = [d for d in ranked if d in EXPECTED_ORDER]
    match = computed_core == EXPECTED_ORDER
    print(f"\n  paper order (core): {' > '.join(EXPECTED_ORDER)}")
    print(f"  computed  (core)  : {' > '.join(computed_core)}   [{'MATCH' if match else 'MISMATCH'}]")
    if "ODT" in dm_mean:
        print(f"  ODT lands at position {ranked.index('ODT') + 1}/{len(ranked)} overall "
              f"(reward {dm_mean['ODT']:.1f}).")

    # ---------- [3] ranking stability per batch ----------
    print("\n" + "=" * 78)
    print("[3] RANKING STABILITY across batches (each replicate batch should give the same core order)")
    print("-" * 78)
    batch_orders = {}
    for batch in sorted({r["batch"] for r in have}):
        bdm = defaultdict(list)
        for r in have:
            if r["batch"] == batch:
                bdm[r["algo"]].append(r["final"])
        bmean = {dm: mean(v) for dm, v in bdm.items()}
        border = [d for d in sorted(bmean, key=lambda d: bmean[d], reverse=True) if d in EXPECTED_ORDER]
        batch_orders[batch] = tuple(border)
        detail = ", ".join(f"{d}={bmean[d]:.1f}" for d in EXPECTED_ORDER if d in bmean)
        print(f"    {batch}xxx: {' > '.join(border)}   ({detail})")
    stable = len(set(batch_orders.values())) <= 1
    print(f"  -> {'STABLE (same core order in every batch)' if stable else 'UNSTABLE (core order changes between batches!)'}")

    # ---------- [4] per-cell preservation ----------
    print("\n" + "=" * 78)
    print("[4] PER-CELL ranking preservation  (DQN > REINFORCE > CMA within each season/agg cell)")
    print("-" * 78)
    cell = defaultdict(lambda: defaultdict(list))     # (season,agg) -> dm -> [finals]
    for r in have:
        cell[(r["season"], r["aggs"])][r["algo"]].append(r["final"])
    ncell = 0
    violations = []
    for key in sorted(cell, key=lambda k: (str(k[0]), k[1])):
        dms = cell[key]
        if not all(d in dms for d in EXPECTED_ORDER):
            continue
        ncell += 1
        m = {d: mean(dms[d]) for d in EXPECTED_ORDER}
        order = sorted(m, key=lambda d: m[d], reverse=True)
        if order != EXPECTED_ORDER:
            violations.append((key, order, m))
    print(f"  cells with DQN/REINFORCE/CMA all present   : {ncell}")
    print(f"  cells preserving DQN > REINFORCE > CMA      : {ncell - len(violations)}/{ncell}")
    for key, order, m in violations:
        print(f"    VIOLATION {key[0]}/{key[1]}agg: {' > '.join(order)}  (" +
              ", ".join(f"{d}={m[d]:.1f}" for d in EXPECTED_ORDER) + ")")

    # ---------- [5] replicate consistency ----------
    print("\n" + "=" * 78)
    print("[5] REPLICATE CONSISTENCY  (same config across batches; final reward should agree)")
    print("-" * 78)
    conf = defaultdict(lambda: defaultdict(list))     # (algo,season,agg) -> batch -> [finals]
    for r in have:
        conf[(r["algo"], r["season"], r["aggs"])][r["batch"]].append(r["final"])
    diffs, big = [], []
    for key, bmap in conf.items():
        if len(bmap) < 2:
            continue
        bmeans = {b: mean(v) for b, v in bmap.items()}
        d = max(bmeans.values()) - min(bmeans.values())
        diffs.append(d)
        if d > args.spread_max:
            big.append((key, bmeans, d))
    if diffs:
        diffs_sorted = sorted(diffs)
        med = diffs_sorted[len(diffs_sorted) // 2]
        print(f"  configs comparable across >=2 batches : {len(diffs)}")
        print(f"  median batch-to-batch reward diff     : {med:.1f}")
        print(f"  configs exceeding spread-max ({args.spread_max:.0f})     : {len(big)}")
        for key, bmeans, d in sorted(big, key=lambda x: -x[2])[:15]:
            detail = ", ".join(f"{b}xxx={v:.1f}" for b, v in sorted(bmeans.items()))
            print(f"    {key[0]:>9} {str(key[1]):>7} {key[2]}agg: diff={d:.1f}  ({detail})")
    else:
        print("  (only one batch has data -- need both 4xxx and 5xxx to compare replicates.)")

    # ---------- [6] aggregation-frequency trend ----------
    print("\n" + "=" * 78)
    print("[6] AGGREGATION-FREQUENCY trend  (descriptive; the paper makes no 50-vs-10-vs-1 claim)")
    print("-" * 78)
    agg_vals = defaultdict(lambda: defaultdict(list))   # dm -> agg -> finals
    for r in have:
        agg_vals[r["algo"]][r["aggs"]].append(r["final"])
    aggs_present = sorted({r["aggs"] for r in have})
    if len(aggs_present) > 1:
        print("    {:>9}".format("DM") + "".join(f"  {a:>3}agg" for a in aggs_present))
        for dm in DM_ORDER:
            if dm in agg_vals:
                cells = "".join(
                    (f"  {mean(agg_vals[dm][a]):6.1f}" if a in agg_vals[dm] else f"  {'--':>6}")
                    for a in aggs_present)
                print(f"    {dm:>9}" + cells)
    else:
        print(f"  only one aggregation setting present ({aggs_present}); nothing to compare.")

    # ---------- [7] magnitude sanity ----------
    print("\n" + "=" * 78)
    print("[7] MAGNITUDE sanity vs handoff reference  (loose; catches sign flip / way-off scale)")
    print("-" * 78)
    mag_off = []
    for dm, ref in REF_REWARD.items():
        if dm not in dm_mean:
            continue
        got = dm_mean[dm]
        ratio = (got / ref) if ref else float("inf")
        ok = (got < 0) and (1.0 / args.mag_tol) <= ratio <= args.mag_tol
        if not ok:
            mag_off.append(dm)
        print(f"    {dm:>9}: computed {got:8.1f}   paper ref {ref:8.1f}   [{'ok' if ok else 'OFF'}]")

    # ---------- VERDICT ----------
    hard = [r for r in rows if any(f.startswith(("NO-DATA", "EMPTY", "INCOMPLETE", "NaN", "DEGENERATE"))
                                   for f in r["flags"])]
    nolearn = [r for r in rows if any(f.startswith("NO-LEARN") for f in r["flags"])]
    unstable_replicates = diffs and len(big) > max(2, len(diffs) // 5)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("-" * 78)
    if hard:
        print(f"  NOT READY: {len(hard)} run(s) have hard data problems (see [1]) -- missing,")
        print("             incomplete, NaN, or flat. Fix + rerun those before 6xxx, else the")
        print("             pooled 4/5/6 set will have holes or bad cells.")
    elif unstable_replicates:
        print(f"  CAUTION: {len(big)} configs disagree by > {args.spread_max:.0f} reward between 4xxx and 5xxx (see [5]).")
        print("           That's seed instability -- eyeball those curves (plot_reward_curve.py)")
        print("           before trusting pooled means. 6xxx would add a 3rd seed but won't fix a")
        print("           genuinely unstable config.")
    elif not stable:
        print("  CAUTION: the DM order flips between 4xxx and 5xxx (see [3]). With only 2 seeds")
        print("           that may be noise near a tie -- confirm before leaning on the ranking.")
    else:
        print("  DATA OK: complete, learned, no NaN/flat, and 4xxx vs 5xxx are consistent.")
        if nolearn:
            print(f"           ({len(nolearn)} RL run(s) show little learning -- see [1]; worth a glance,")
            print("            but a weak learner can be a real result, not necessarily a bug.)")
        if match and not mag_off:
            print(f"  PAPER-ALIGNED: core order {' > '.join(computed_core)} matches the paper, magnitudes sane.")
            print("  -> Safe to submit 6xxx (it just adds the 3rd seed to the pool).")
        else:
            note = f"computed {' > '.join(computed_core)}"
            if mag_off:
                note += f"; off-scale vs ref: {', '.join(mag_off)}"
            print(f"  RANKING/SCALE DIFFERS FROM PAPER ({note}).")
            print("  -> The runs look SOUND (complete + stable), so this is a NARRATIVE matter:")
            print("     revise the paper's claim, don't rerun. A stable ranking won't change when")
            print("     6xxx adds a 3rd seed, so 6xxx is still safe -- but flag this to your advisor.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
