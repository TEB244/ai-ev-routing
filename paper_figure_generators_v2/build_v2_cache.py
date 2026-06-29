#!/usr/bin/env python3
"""
Build the v2 figure/table cache from the revised 4xxx metrics.

Run this ONCE on Huron (where /storage_1/metrics_postfix lives) before opening
the v2 notebooks. It does a single pass over the raw post-fix per-experiment
metrics and writes small intermediate CSVs to table_data/_v2_cache/ that the
notebooks read.

Usage (from inside paper_figure_generators_v2/):
    python build_v2_cache.py
    python build_v2_cache.py --metrics-root /storage_1/metrics_postfix
    python build_v2_cache.py --experiments 4000-4179 --last-n 100

Notes:
  * Only the 4xxx batch is used (3 seeds, all four seasons, all aggregation
    levels). 5xxx / 6xxx are intentionally excluded -- they are the 6 seeds
    still running.
  * Reward and in-simulation behaviour come from this new data. Training time,
    power, CO2 and model size are NOT touched here; those stay on the old data
    inside the Experiment 3 notebook.
"""

import argparse
import sys
import os

sys.path.append(os.path.abspath(".."))  # repo root, for environment.data_loader

import v2_data  # noqa: E402


def parse_experiments(spec):
    if not spec:
        return v2_data.EXPERIMENTS_4XXX
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-root", default=None,
                   help="Override metrics_postfix root (default: auto-detect).")
    p.add_argument("--experiments", default=None,
                   help="Ranges/ids, e.g. '4000-4179' (default: 4000-4179).")
    p.add_argument("--last-n", type=int, default=v2_data.DEFAULT_LAST_N_EPISODES,
                   help="Trailing episodes treated as steady-state (default: 100).")
    p.add_argument("--out", default=str(v2_data.CACHE_DIR),
                   help="Cache output directory (default: table_data/_v2_cache).")
    args = p.parse_args()

    v2_data.build_cache(
        new_root=args.metrics_root,
        experiments=parse_experiments(args.experiments),
        last_n=args.last_n,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
