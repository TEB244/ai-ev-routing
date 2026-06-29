#!/usr/bin/env python3
"""
Fit the per-car inference-energy regression for the 10xxx inference batch.

For each decision-maker, regress per-job energy (and CO2e) against the number of
cars to recover:

    energy(n_cars) = a + b * n_cars
        a (intercept) = fixed cost to spin up + load the model  ("load" cost)
        b (slope)     = marginal inference energy per car

Reads the DRAC-portal power CSVs written by get_drac_power_usage.py --mode eval:
    <metrics-root>/Exp_<N>/eval/power_and_co2_metrics.csv   (columns: time,power,co2)
and the experiment configs in experiments/Exp_<N>/config.yaml (for DM, car count,
and zone count). Energy integration matches sum_emissions.py exactly.

Usage:
    python fit_inference_regression.py
    python fit_inference_regression.py --range 10000 10047 --metrics-root /storage_1/metrics
    python fit_inference_regression.py --csv-out inference_regression.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Reuse the exact energy/CO2 integration used for the training-phase numbers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sum_emissions import energy_and_co2, metrics_root_candidates  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
EXP_DIR = REPO_ROOT / "experiments"


def resolve_root(override):
    for r in ([Path(override)] if override else metrics_root_candidates(None)):
        if r.is_dir():
            return r
    return Path(override) if override else None


def read_config(exp_num):
    path = EXP_DIR / f"Exp_{exp_num}" / "config.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def ols_fit(x, y):
    """Return (intercept a, slope b, R^2) for y = a + b*x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def main():
    p = argparse.ArgumentParser(description="Fit per-car inference-energy regression (10xxx).")
    p.add_argument("--range", nargs=2, type=int, default=[10000, 10047],
                   help="Inclusive experiment range (default: 10000 10047).")
    p.add_argument("--metrics-root", help="Metrics root dir (auto-detected if omitted).")
    p.add_argument("--csv-out", help="Optional path to write per-experiment rows as CSV.")
    args = p.parse_args()

    root = resolve_root(args.metrics_root)
    if root is None or not root.is_dir():
        print("ERROR: no metrics root found; pass --metrics-root.", file=sys.stderr)
        return 1
    print(f"Metrics root: {root}\n")

    # Gather points per DM: {dm: [(total_cars, cars_per_zone, kwh, co2_kg, exp), ...]}
    per_dm = {}
    rows = []
    missing = []
    for n in range(args.range[0], args.range[1] + 1):
        cfg = read_config(n)
        if cfg is None:
            continue
        dm = cfg["algorithm_settings"]["algorithm"]
        cars = cfg["environment_settings"]["num_of_cars"]
        n_zones = len(cfg["environment_settings"]["coords"])
        csvp = root / f"Exp_{n}" / "eval" / "power_and_co2_metrics.csv"
        if not csvp.exists():
            missing.append(n)
            continue
        res = energy_and_co2(csvp)
        if res is None:
            missing.append(n)
            continue
        kwh, co2_kg, _ = res
        num_eps = int(cfg.get("nn_hyperparameters", {}).get("num_episodes", 1) or 1)
        per_dm.setdefault(dm, []).append((cars * n_zones, cars, kwh, co2_kg, num_eps))
        rows.append((n, dm, cars, n_zones, cars * n_zones, kwh, co2_kg, num_eps))

    if not rows:
        print("No usable eval power CSVs found. Run the jobs, then scrape with:\n"
              "  python drac/get_drac_power_usage.py -e 10000 10047 -u <user> "
              "-p <metrics-root> --mode eval --host <portal-host>", file=sys.stderr)
        return 1

    # Per-DM regression. The inference is looped N_ep times per job, so
    # energy = load + (N_ep * per_car) * cars and the fitted slope is N_ep*per_car;
    # divide by N_ep to report energy per car for ONE inference episode. The
    # intercept (load) is one-time per job and is NOT divided.
    print(f"{'DM':<10} {'n':>3} {'N_ep':>5}  {'load kWh':>10}  {'kWh/car':>12}  "
          f"{'kWh/total-car':>14}  {'R^2':>6}")
    print("-" * 74)
    summary = []
    for dm in sorted(per_dm):
        pts = per_dm[dm]
        cars_cfg = [c for _, c, _, _, _ in pts]
        cars_tot = [t for t, _, _, _, _ in pts]
        kwh = [k for _, _, k, _, _ in pts]
        co2 = [c for _, _, _, c, _ in pts]
        N = max(p[4] for p in pts)   # inference loop count (constant within a batch)
        a, b_cfg, r2 = ols_fit(cars_cfg, kwh)
        _, b_tot, _ = ols_fit(cars_tot, kwh)
        print(f"{dm:<10} {len(pts):>3} {N:>5}  {a:>10.6f}  {b_cfg / N:>12.6f}  "
              f"{b_tot / N:>14.6f}  {r2:>6.3f}")
        a_c, b_c_cfg, r2_c = ols_fit(cars_cfg, co2)
        _, b_c_tot, _ = ols_fit(cars_tot, co2)
        summary.append({
            "dm": dm, "n_points": len(pts), "num_episodes": N,
            "load_kwh": a, "kwh_per_car": b_cfg / N, "kwh_per_total_car": b_tot / N, "r2_kwh": r2,
            "load_co2_kg": a_c, "co2kg_per_car": b_c_cfg / N,
            "co2kg_per_total_car": b_c_tot / N, "r2_co2": r2_c,
        })
    print("-" * 74)
    print("load = fixed model-load + spin-up energy (intercept, one-time per job).")
    print("kWh/car = per-car energy for ONE inference episode (fitted slope / N_ep).")
    print("'/car' uses the per-zone config value; '/total-car' divides by zones.\n")

    print("CO2e regression (kg, per-car already divided by N_ep):")
    print(f"{'DM':<10}  {'load kg':>13}  {'kg/car':>13}  {'R^2':>6}")
    print("-" * 50)
    for s in summary:
        print(f"{s['dm']:<10}  {s['load_co2_kg']:>13.6f}  "
              f"{s['co2kg_per_car']:>13.6f}  {s['r2_co2']:>6.3f}")

    if missing:
        print(f"\nNote: {len(missing)} experiment(s) had no usable eval power CSV "
              f"(not run / not scraped): {missing}")

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["exp", "dm", "cars_per_zone", "n_zones", "total_cars", "kwh", "co2_kg", "num_episodes"])
            w.writerows(sorted(rows))
        print(f"\nPer-experiment rows written to {args.csv_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
