#!/usr/bin/env python3
"""
Sum total kWh and CO2e across experiments from DRAC portal power data.

Reads <root>/Exp_<N>/train/power_and_co2_metrics.csv (written by
drac/get_drac_power_usage.py) with columns:
    time   -- timestamp of each power sample (parseable by pandas)
    power  -- node power draw in Watts at that sample
    co2    -- total CO2 for the WHOLE job in kg (same value every row)

Per experiment:
    energy (kWh) = trapezoidal integral of power(W) over time(s) / 3.6e6
    CO2e   (kg)  = the single co2 value (it's already a whole-job total)
Then summed over every experiment in the requested ranges.

(This reads the DRAC-portal file, NOT the CarbonTracker
metrics_sustainability_episode.csv, which is missing/disabled for several
algorithms on DRAC.)

Usage:
    python sum_emissions.py                                  # 7xxx + 9xxx
    python sum_emissions.py --ranges 7000-7179 9000-9251
    python sum_emissions.py --metrics-root /storage_1/metrics --verbose
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV_REL = Path("train") / "power_and_co2_metrics.csv"


def metrics_root_candidates(override):
    if override:
        return [Path(override)]
    return [
        Path("/storage_1/metrics"),
        Path("/storage_1/metrics_emissions"),
        Path.home() / "scratch" / "metrics",
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


def energy_and_co2(path):
    """Return (kWh, co2_kg, n_samples) for one experiment, or None if unusable."""
    df = pd.read_csv(path)
    if df.empty or "power" not in df.columns or "time" not in df.columns:
        return None

    # Time -> seconds-from-start. Prefer datetime parsing (portal writes
    # timestamps); fall back to numeric epoch (s or ms) if that fails.
    t = pd.to_datetime(df["time"], errors="coerce")
    if t.notna().sum() >= 2:
        secs = (t - t.min()).dt.total_seconds().to_numpy()
    else:
        tn = pd.to_numeric(df["time"], errors="coerce").to_numpy()
        secs = tn - np.nanmin(tn)
        if np.nanmax(secs) > 1e7:        # looks like milliseconds
            secs = secs / 1000.0

    power = pd.to_numeric(df["power"], errors="coerce").to_numpy()
    mask = ~(np.isnan(secs) | np.isnan(power))
    secs, power = secs[mask], power[mask]

    if len(secs) < 2:
        kwh = 0.0
    else:
        order = np.argsort(secs)
        secs, power = secs[order], power[order]
        trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed it
        joules = trapezoid(power, secs)   # W * s = J
        kwh = joules / 3.6e6

    co2 = pd.to_numeric(df["co2"], errors="coerce").dropna()
    co2_kg = float(co2.iloc[0]) if len(co2) else float("nan")  # whole-job total
    return kwh, co2_kg, len(df)


def main():
    p = argparse.ArgumentParser(
        description="Sum total kWh and CO2e across experiment ranges (DRAC portal data).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ranges", nargs="+", default=["7xxx", "9xxx"],
                   help="Experiment ranges (default: 7xxx 9xxx).")
    p.add_argument("--metrics-root", help="Metrics root dir (auto-detected if omitted).")
    p.add_argument("--verbose", action="store_true", help="Print per-experiment rows.")
    args = p.parse_args()

    root = resolve_root(args.metrics_root)
    if root is None:
        print("ERROR: no metrics root found. Tried:", file=sys.stderr)
        for c in metrics_root_candidates(args.metrics_root):
            print(f"  {c}", file=sys.stderr)
        return 1
    print(f"Metrics root: {root}")

    ranges = parse_ranges(args.ranges)

    found, missing = [], []
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

    per_range = {range_label(a, b): {"kwh": 0.0, "co2": 0.0, "exps": 0, "no_co2": 0}
                 for a, b in ranges}
    grand = {"kwh": 0.0, "co2": 0.0, "no_co2": 0}

    if args.verbose and found:
        print(f"\n{'Exp':>6}  {'kWh':>10}  {'CO2e (kg)':>10}  {'samples':>8}")
        print("  " + "-" * 42)

    for n, path in found:
        res = energy_and_co2(path)
        if res is None:
            missing.append(n)
            continue
        kwh, co2_kg, nsamp = res
        lab = next(range_label(a, b) for a, b in ranges if a <= n <= b)
        per_range[lab]["kwh"] += kwh
        per_range[lab]["exps"] += 1
        grand["kwh"] += kwh
        if np.isnan(co2_kg):
            per_range[lab]["no_co2"] += 1
            grand["no_co2"] += 1
        else:
            per_range[lab]["co2"] += co2_kg
            grand["co2"] += co2_kg
        if args.verbose:
            co2_str = "nan" if np.isnan(co2_kg) else f"{co2_kg:.4f}"
            print(f"{n:>6}  {kwh:>10.4f}  {co2_str:>10}  {nsamp:>8}")

    print("\n" + "=" * 52)
    print(f"{'Range':>8}  {'exps':>5}  {'kWh':>12}  {'CO2e (kg)':>12}")
    print("-" * 52)
    for lab, v in per_range.items():
        print(f"{lab:>8}  {v['exps']:>5}  {v['kwh']:>12.4f}  {v['co2']:>12.4f}")
    print("-" * 52)
    print(f"{'TOTAL':>8}  {sum(v['exps'] for v in per_range.values()):>5}  "
          f"{grand['kwh']:>12.4f}  {grand['co2']:>12.4f}")
    print("=" * 52)

    print(f"\nGRAND TOTAL")
    print(f"  Energy:  {grand['kwh']:.4f} kWh")
    print(f"  CO2e:    {grand['co2']:.4f} kg   ({grand['co2'] * 1000:.2f} g)")

    print(f"\nExperiments found: {len(found)}")
    if missing:
        miss = sorted(set(missing))
        print(f"Experiments with no/unusable power csv: {len(miss)}")
        runs, s, prev = [], miss[0], miss[0]
        for x in miss[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((s, prev)); s = x; prev = x
        runs.append((s, prev))
        print("  " + ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs))
    if grand["no_co2"]:
        print(f"NOTE: {grand['no_co2']} experiment(s) had power but no co2 value.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
