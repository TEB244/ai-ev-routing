#!/usr/bin/env python3
"""
Sum total kWh and CO2e across experiments' CarbonTracker output.

Each experiment writes <root>/Exp_<N>/train/metrics_sustainability_episode.csv
with one row per (episode, zone, aggregation) and columns:
    kwh  -- kWh consumed that episode
    co2  -- grams CO2e that episode   (may be blank/None on DRAC nodes,
            which often can't resolve grid carbon intensity)

This sums kwh and co2 over every row of every experiment in the requested
ranges and prints per-range subtotals plus one grand total. Stdlib only, so
it runs anywhere (no pandas needed).

Usage:
    # Default: 7xxx + 9xxx, auto-detect metrics root
    python sum_emissions.py

    # Explicit ranges (mix of NNNN-NNNN, single Exp, or "Nxxx" shorthand)
    python sum_emissions.py --ranges 7000-7179 9000-9251

    # Point at a specific download dir
    python sum_emissions.py --metrics-root /storage_1/metrics_emissions

    # Per-experiment breakdown, not just totals
    python sum_emissions.py --verbose

    # If co2 is missing/zero, estimate CO2e = kwh * intensity (gCO2e/kWh).
    # Quebec hydro grid (Narval/Rorqual) is ~1.5; pass your own if known.
    python sum_emissions.py --intensity 1.5
"""

import argparse
import csv
import sys
from pathlib import Path

CSV_REL = Path("train") / "metrics_sustainability_episode.csv"


def metrics_root_candidates(override):
    if override:
        return [Path(override)]
    return [
        Path("/storage_1/metrics_emissions"),   # suggested download dir
        Path("/storage_1/metrics"),             # huron lab server
        Path.home() / "scratch" / "metrics",    # on a DRAC login node
        Path("/home/hartman/scratch/metrics"),
    ]


def resolve_root(override):
    for r in metrics_root_candidates(override):
        if r.is_dir():
            return r
    return None


def parse_ranges(specs):
    """'7xxx' -> (7000,7999); '7000-7179' -> (7000,7179); '7042' -> (7042,7042)."""
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


def in_ranges(n, ranges):
    return any(a <= n <= b for a, b in ranges)


def range_label(a, b):
    if a // 1000 == b // 1000 and a % 1000 == 0 and b % 1000 == 999:
        return f"{a // 1000}xxx"
    return f"{a}-{b}"


def sum_file(path):
    """Return (kwh_sum, co2_sum, n_rows, n_null_kwh, n_null_co2)."""
    kwh = co2 = 0.0
    rows = null_kwh = null_co2 = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            try:
                kwh += float(row.get("kwh", ""))
            except (TypeError, ValueError):
                null_kwh += 1
            try:
                co2 += float(row.get("co2", ""))
            except (TypeError, ValueError):
                null_co2 += 1
    return kwh, co2, rows, null_kwh, null_co2


def main():
    p = argparse.ArgumentParser(
        description="Sum total kWh and CO2e across experiment ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ranges", nargs="+", default=["7xxx", "9xxx"],
                   help="Experiment ranges (default: 7xxx 9xxx).")
    p.add_argument("--metrics-root", help="Metrics root dir (auto-detected if omitted).")
    p.add_argument("--verbose", action="store_true", help="Print per-experiment rows.")
    p.add_argument("--intensity", type=float, default=None,
                   help="gCO2e/kWh used to ESTIMATE CO2e from kWh (e.g. 1.5 for QC hydro). "
                        "Shown alongside the measured co2 sum.")
    args = p.parse_args()

    root = resolve_root(args.metrics_root)
    if root is None:
        print("ERROR: no metrics root found. Tried:", file=sys.stderr)
        for c in metrics_root_candidates(args.metrics_root):
            print(f"  {c}", file=sys.stderr)
        return 1
    print(f"Metrics root: {root}")

    ranges = parse_ranges(args.ranges)

    found = []   # (n, path)
    missing = []
    for d in sorted(root.glob("Exp_*")):
        try:
            n = int(d.name.replace("Exp_", ""))
        except ValueError:
            continue
        if not in_ranges(n, ranges):
            continue
        csvp = d / CSV_REL
        if csvp.exists():
            found.append((n, csvp))
        else:
            missing.append(n)

    # Accumulate, tagged by which range each experiment falls in
    per_range = {range_label(a, b): {"kwh": 0.0, "co2": 0.0, "exps": 0, "rows": 0, "null_co2": 0}
                 for a, b in ranges}
    grand = {"kwh": 0.0, "co2": 0.0, "rows": 0, "null_co2": 0, "null_kwh": 0}

    if args.verbose and found:
        print(f"\n{'Exp':>6}  {'kWh':>12}  {'CO2e (g)':>12}  {'rows':>6}  null_co2")
        print("  " + "-" * 52)

    for n, path in found:
        kwh, co2, rows, nk, nc = sum_file(path)
        lab = next(range_label(a, b) for a, b in ranges if a <= n <= b)
        per_range[lab]["kwh"] += kwh
        per_range[lab]["co2"] += co2
        per_range[lab]["exps"] += 1
        per_range[lab]["rows"] += rows
        per_range[lab]["null_co2"] += nc
        grand["kwh"] += kwh
        grand["co2"] += co2
        grand["rows"] += rows
        grand["null_co2"] += nc
        grand["null_kwh"] += nk
        if args.verbose:
            print(f"{n:>6}  {kwh:>12.4f}  {co2:>12.2f}  {rows:>6}  {nc}")

    print("\n" + "=" * 58)
    print(f"{'Range':>8}  {'experiments':>11}  {'kWh':>12}  {'CO2e (g)':>12}")
    print("-" * 58)
    for lab, v in per_range.items():
        print(f"{lab:>8}  {v['exps']:>11}  {v['kwh']:>12.4f}  {v['co2']:>12.2f}")
    print("-" * 58)
    print(f"{'TOTAL':>8}  {sum(v['exps'] for v in per_range.values()):>11}  "
          f"{grand['kwh']:>12.4f}  {grand['co2']:>12.2f}")
    print("=" * 58)

    print(f"\nGRAND TOTAL")
    print(f"  Energy:  {grand['kwh']:.4f} kWh")
    print(f"  CO2e:    {grand['co2']:.2f} g   ({grand['co2'] / 1000:.4f} kg)   [from stored co2 column]")
    if args.intensity is not None:
        est = grand["kwh"] * args.intensity
        print(f"  CO2e:    {est:.2f} g   ({est / 1000:.4f} kg)   "
              f"[ESTIMATE = {grand['kwh']:.2f} kWh x {args.intensity} gCO2e/kWh]")

    # Health report
    print(f"\nExperiments found:   {len(found)}")
    if missing:
        print(f"Experiments MISSING the csv: {len(missing)}")
        # compact ranges
        miss = sorted(missing)
        runs, s, prev = [], miss[0], miss[0]
        for x in miss[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((s, prev)); s = x; prev = x
        runs.append((s, prev))
        print("  " + ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs))
    if grand["null_co2"]:
        print(f"NOTE: {grand['null_co2']}/{grand['rows']} rows had blank/None co2 "
              f"(DRAC nodes often can't resolve grid intensity). Use --intensity to estimate.")
    if grand["null_kwh"]:
        print(f"NOTE: {grand['null_kwh']}/{grand['rows']} rows had blank/None kwh.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
