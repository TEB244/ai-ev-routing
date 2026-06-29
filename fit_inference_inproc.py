#!/usr/bin/env python3
"""
Fit the per-car inference-energy regression from IN-PROCESS energy measurements
(EnergyMeter, written by main.py on huron / -server Local), since the DRAC portal
can't resolve these short jobs.

Reads, per experiment:
    <metrics-root>/Exp_<N>/eval/inference_energy.csv
    cols: exp,dm,num_cars,n_zones,num_episodes,seconds,n_samples,gpu_measured,
          cpu_kwh,gpu_kwh,total_kwh,co2_g
(one row per run; the most recent row wins if a job was re-run).

Energy attribution per DM:
    CPU DMs (DQN/REINFORCE/CMA): cpu_kwh  -- process-level (psutil x TDP), marginal.
                                 (gpu_kwh would just be idle-GPU baseline.)
    ODT:                         total_kwh -- it runs on the GPU (NVML-measured).

Regression: energy(cars) = a + b * cars
    a (intercept) = fixed load (env build + base setup)
    b (slope)     = per-car cost. NOTE: in this framework a network is
                    instantiated PER car, so b is dominated by per-car model
                    setup; the forward pass itself is ~0.3 s/episode (negligible).

Usage:
    python fit_inference_inproc.py --metrics-root ~/scratch/metrics
    python fit_inference_inproc.py --range 10000 10047 --csv-out inproc.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_inference_regression import ols_fit  # noqa: E402


def metrics_root_candidates(override):
    if override:
        return [Path(override)]
    return [Path.home() / "scratch" / "metrics", Path("/storage_1/metrics_postfix"),
            Path("/storage_1/metrics"), Path(__file__).resolve().parent / "_local_metrics"]


def resolve_root(override):
    for r in metrics_root_candidates(override):
        if r.is_dir():
            return r
    return Path(override) if override else None


def read_last_row(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main():
    p = argparse.ArgumentParser(description="Fit per-car inference energy from in-process meter data.")
    p.add_argument("--range", nargs=2, type=int, default=[10000, 10047])
    p.add_argument("--metrics-root")
    p.add_argument("--csv-out")
    p.add_argument("--gpu-dms", nargs="*", default=[],
                   help="DMs that were run on GPU (-g 0): use total_kwh (cpu+gpu) for "
                        "them; everyone else uses cpu_kwh. Default: all cpu_kwh, since "
                        "the runs were CPU and the idle-GPU baseline isn't a DM's cost.")
    args = p.parse_args()
    gpu_dms = set(args.gpu_dms)

    root = resolve_root(args.metrics_root)
    if root is None or not root.is_dir():
        print("ERROR: no metrics root found; pass --metrics-root.", file=sys.stderr)
        return 1
    print(f"Metrics root: {root}\n")

    per_dm = {}
    rows_out = []
    missing = []
    for n in range(args.range[0], args.range[1] + 1):
        csvp = root / f"Exp_{n}" / "eval" / "inference_energy.csv"
        if not csvp.exists():
            missing.append(n)
            continue
        r = read_last_row(csvp)
        if r is None:
            missing.append(n)
            continue
        dm = r["dm"]
        cars = int(r["num_cars"])
        cpu_kwh = float(r["cpu_kwh"])
        gpu_kwh = float(r["gpu_kwh"])
        total_kwh = float(r["total_kwh"])
        energy = total_kwh if dm in gpu_dms else cpu_kwh
        per_dm.setdefault(dm, []).append((cars, energy, cpu_kwh, gpu_kwh, float(r["seconds"])))
        rows_out.append((n, dm, cars, r["seconds"], r["gpu_measured"], cpu_kwh, gpu_kwh, total_kwh))

    if not per_dm:
        print("No inference_energy.csv files found. Run the jobs on huron with "
              "`-server Local` first.", file=sys.stderr)
        return 1

    print(f"{'DM':<10} {'n':>3}  {'energy used':>11}  {'load kWh':>11}  {'kWh/car':>12}  {'R^2':>6}")
    print("-" * 64)
    for dm in sorted(per_dm):
        pts = sorted(per_dm[dm])
        cars = [c for c, _, _, _, _ in pts]
        energy = [e for _, e, _, _, _ in pts]
        a, b, r2 = ols_fit(cars, energy)
        used = "total_kwh" if dm in gpu_dms else "cpu_kwh"
        print(f"{dm:<10} {len(pts):>3}  {used:>11}  {a:>11.6f}  {b:>12.6f}  {r2:>6.3f}")
    print("-" * 64)
    print("load = fixed env+base setup (intercept); kWh/car = per-car cost (slope).")
    print("Per-car is dominated by per-car model instantiation; the forward pass")
    print("itself is ~0.3 s/episode (negligible) -- see run timings.\n")

    # Show the raw points so the scaling is visible.
    print(f"{'exp':>6} {'dm':<10} {'cars':>5} {'sec':>7} {'cpu_kWh':>11} {'gpu_kWh':>11}")
    for n, dm, cars, secs, gpum, cpu, gpu, tot in sorted(rows_out):
        print(f"{n:>6} {dm:<10} {cars:>5} {secs:>7} {cpu:>11.6f} {gpu:>11.6f}")

    if missing:
        print(f"\nNote: {len(missing)} experiment(s) missing inference_energy.csv: {missing}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["exp", "dm", "num_cars", "seconds", "gpu_measured",
                        "cpu_kwh", "gpu_kwh", "total_kwh"])
            w.writerows(sorted(rows_out))
        print(f"\nPer-experiment rows written to {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
