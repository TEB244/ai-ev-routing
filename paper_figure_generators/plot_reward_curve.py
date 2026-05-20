#!/usr/bin/env python3
"""
Plot a single experiment's training reward curve.

Useful for quickly verifying an experiment finished and learned. Reads
the same metrics CSV that check_progress.py uses and produces a PNG with
per-episode reward + rolling mean + first/last summary lines.

Usage:
    python paper_figure_generators/plot_reward_curve.py <exp_num> [options]

Examples:
    # Save figure to default location (figures/exp_<N>_reward_curve.png)
    python paper_figure_generators/plot_reward_curve.py 9000

    # Use a specific metrics root (e.g. Santiago's scratch)
    python paper_figure_generators/plot_reward_curve.py 9090 \\
        --metrics-root /home/sgomezro/scratch/metrics

    # Custom rolling-mean window and output path
    python paper_figure_generators/plot_reward_curve.py 7045 \\
        --window 250 --out my_check.png

    # Open the figure window interactively (local machine only)
    python paper_figure_generators/plot_reward_curve.py 7000 --show

Output:
    PNG file at paper_figure_generators/figures/exp_<N>_reward_curve.png
    (or --out path), plus a brief summary printed to stdout.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. On DRAC: source ~/envs/merl_env/bin/activate",
          file=sys.stderr)
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")  # headless by default
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]

ALGO_COLORS = {
    "DQN":       "darkorange",
    "REINFORCE": "forestgreen",
    "CMA":       "turquoise",
    "CMA-ES":    "turquoise",
    "ODT":       "blueviolet",
}


# ----------------------------------------------------------- metrics resolution
def metrics_root_candidates(override=None):
    if override:
        return [Path(override)]
    return [
        Path.home() / "scratch" / "metrics",
        Path("/storage_1/metrics"),
        Path("/home/hartman/scratch/metrics"),
        Path("/home/sgomezro/scratch/metrics"),
        Path("/home/epigou/scratch/metrics"),
        REPO_ROOT / "_local_metrics",
    ]


def find_metrics_dir(exp_num, override=None):
    for base in metrics_root_candidates(override):
        d = base / f"Exp_{exp_num}"
        if d.exists():
            return d
    return None


# ----------------------------------------------------------- CSV scan (stdlib)
def scan_csv(csv_path):
    """
    Single-pass scan of metrics_agent_episode_level.csv.
    Returns sorted list of (agg, ep) pairs and parallel list of mean reward
    across all cars for that (agg, ep).
    """
    rewards = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            i_agg = header.index("aggregation")
            i_ep  = header.index("episode")
            i_rew = header.index("reward")
        except ValueError as e:
            raise RuntimeError(f"Missing expected column in {csv_path}: {e}")
        for row in reader:
            if len(row) <= max(i_agg, i_ep, i_rew):
                continue
            try:
                agg = int(row[i_agg])
                ep  = int(row[i_ep])
                rew = float(row[i_rew])
            except ValueError:
                continue
            rewards.setdefault((agg, ep), []).append(rew)
    pairs = sorted(rewards.keys())
    means = [sum(rewards[p]) / len(rewards[p]) for p in pairs]
    return pairs, means


def rolling_mean(values, window):
    """Simple rolling mean over a list using a running-sum approach."""
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    s = sum(values[:window])
    out.append(s / window)
    for i in range(window, len(values)):
        s += values[i] - values[i - window]
        out.append(s / window)
    # Pad the start so the rolling line aligns with the per-episode line
    pad = [out[0]] * (window - 1)
    return pad + out


# -------------------------------------------------------- expected episode count
def get_expected_episodes(cfg):
    algo = cfg["algorithm_settings"]["algorithm"]
    fed = cfg.get("federated_learning_settings", {})
    aggregation_count = fed.get("aggregation_count", 1)
    if algo in ("DQN", "REINFORCE", "PPO", "DDPG", "ODT"):
        eps_per_agg = cfg.get("nn_hyperparameters", {}).get("num_episodes")
    elif algo in ("CMA", "DENSER", "NEAT"):
        eps_per_agg = cfg.get("cma_parameters", {}).get("max_generations")
    else:
        eps_per_agg = None
    if eps_per_agg is None:
        return None
    return aggregation_count * eps_per_agg


# ----------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(
        description="Plot training reward curve for one experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("exp_num", type=int, help="Experiment number")
    p.add_argument("--metrics-root", help="Override metrics root directory")
    p.add_argument("--out", help="Output PNG path (default: paper_figure_generators/figures/exp_<N>_reward_curve.png)")
    p.add_argument("--window", type=int, help="Rolling-mean window in episodes (default: ~2%% of episodes)")
    p.add_argument("--show", action="store_true", help="Open figure interactively (needs a display)")
    p.add_argument("--dpi", type=int, default=180, help="Output DPI (default 180)")
    args = p.parse_args()

    if args.show:
        # Switch to a real backend for interactive display
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass

    # Load config for title and reference values
    cfg_path = REPO_ROOT / "experiments" / f"Exp_{args.exp_num}" / "config.yaml"
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    algo = cfg["algorithm_settings"]["algorithm"]
    seed = cfg["environment_settings"]["seed"]
    num_zones = len(cfg["environment_settings"]["coords"])
    expected_eps = get_expected_episodes(cfg)

    # Locate CSV
    metrics_dir = find_metrics_dir(args.exp_num, args.metrics_root)
    if metrics_dir is None:
        print(f"ERROR: metrics dir not found for Exp_{args.exp_num}", file=sys.stderr)
        print(f"  Tried:", file=sys.stderr)
        for c in metrics_root_candidates(args.metrics_root):
            print(f"    {c}", file=sys.stderr)
        return 1
    csv_path = metrics_dir / "train" / "metrics_agent_episode_level.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        return 1

    print(f"Scanning {csv_path} ...")
    pairs, means = scan_csv(csv_path)
    n = len(pairs)
    if n == 0:
        print(f"ERROR: no episode data in CSV", file=sys.stderr)
        return 1

    pct = (n / expected_eps * 100) if expected_eps else None

    # Compute summary stats
    n_summary = max(1, min(200, n // 4))   # first/last window for summary lines
    first_mean = sum(means[:n_summary]) / n_summary
    last_mean  = sum(means[-n_summary:]) / n_summary
    delta = last_mean - first_mean

    # Rolling mean
    window = args.window or max(10, n // 50)
    rolling = rolling_mean(means, window)

    # Plot
    color = ALGO_COLORS.get(algo, "gray")
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = list(range(n))
    ax.plot(xs, means, lw=0.5, alpha=0.4, color=color, label="per-episode mean")
    ax.plot(xs, rolling, lw=2.5, color=color, label=f"{window}-ep rolling mean")

    # Aggregation boundary indicators
    for i in range(1, n):
        if pairs[i][0] != pairs[i - 1][0]:
            ax.axvline(i, color="red", alpha=0.10, lw=1)

    # First/last reference lines
    ax.axhline(first_mean, color="gray",  linestyle=":",  lw=1, alpha=0.7)
    ax.axhline(last_mean,  color="black", linestyle="--", lw=1, alpha=0.7)

    # Annotate first/last means
    ax.annotate(f"first {n_summary} eps: {first_mean:+.2f}",
                xy=(0.01, first_mean), xycoords=("axes fraction", "data"),
                xytext=(6, -10), textcoords="offset points",
                fontsize=9, color="gray", fontweight="bold")
    ax.annotate(f"last {n_summary} eps: {last_mean:+.2f}",
                xy=(0.99, last_mean), xycoords=("axes fraction", "data"),
                xytext=(-6, 8), textcoords="offset points", ha="right",
                fontsize=9, color="black", fontweight="bold")

    # Improvement arrow
    if abs(delta) > 0.5:
        ax.annotate("",
                    xy=(n * 0.05, last_mean),
                    xytext=(n * 0.05, first_mean),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
        midpoint = (first_mean + last_mean) / 2
        ax.text(n * 0.07, midpoint, f"{delta:+.1f}\nreward",
                fontsize=11, color="black", fontweight="bold",
                ha="left", va="center")

    # Title
    title_parts = [f"Exp_{args.exp_num}", f"{algo}", f"seed={seed}", f"zones={num_zones}"]
    if expected_eps:
        title_parts.append(f"{n}/{expected_eps} eps ({pct:.0f}%)")
    else:
        title_parts.append(f"{n} eps")
    ax.set_title("  |  ".join(title_parts), fontsize=12)
    ax.set_xlabel("Global episode")
    ax.set_ylabel("Mean reward across cars")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "paper_figure_generators" / "figures"
                     / f"exp_{args.exp_num}_reward_curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out_path}")
    print()
    print(f"  algorithm:           {algo}")
    print(f"  seed:                {seed}")
    print(f"  zones:               {num_zones}")
    print(f"  episodes recorded:   {n}" + (f" / {expected_eps} ({pct:.1f}%)" if expected_eps else ""))
    print(f"  first {n_summary}-ep mean:    {first_mean:+.2f}")
    print(f"  last  {n_summary}-ep mean:    {last_mean:+.2f}")
    print(f"  delta (last-first):  {delta:+.2f}")
    print(f"  rolling-mean window: {window} eps")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
