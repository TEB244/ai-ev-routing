#!/usr/bin/env python3
"""
Check progress of an experiment based on its saved metrics.

Pure stdlib + pyyaml — no pandas required.

Usage:
    python check_progress.py <experiment_number> [--metrics-root PATH] [--user USER]

Examples:
    python check_progress.py 9000
    python check_progress.py 7045 --metrics-root /storage_1/metrics
    python check_progress.py 9090 --user sgomezro

On DRAC the default metrics root is ~/scratch/metrics/. If the experiment was
run by another user (e.g. CMA by sgomezro, ODT by epigou), point at their
scratch with --metrics-root if you can read it.

Output includes: expected vs actual episode count, percentage complete,
current aggregation, last-100-episode mean reward, CSV last modification
time, SLURM job state if running, and an ETA derived from SLURM elapsed
time when available.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed in this Python environment.", file=sys.stderr)
    print("On DRAC: source ~/envs/merl_env/bin/activate", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------- algorithm -> owner
# Each DM is run by a specific lab member (their scratch holds the metrics
# and their SLURM account submits the job). This lets the script auto-fill
# --user and --metrics-root when not explicitly given.
ALGO_OWNER = {
    "DQN":       "hartman",
    "REINFORCE": "hartman",
    "CMA":       "sgomezro",
    "ODT":       "epigou",
}


# ----------------------------------------------------------- metrics resolution
def metrics_root_candidates(override=None, owner=None):
    """Where to look for an experiment's metrics directory, in priority order.

    If `owner` is provided (derived from the config's algorithm), the
    owner-specific scratch path is tried first.
    """
    if override:
        return [Path(override)]
    candidates = []
    if owner:
        candidates.append(Path(f"/home/{owner}/scratch/metrics"))
    candidates.extend([
        Path.home() / "scratch" / "metrics",          # DRAC / cluster ($USER)
        Path("/storage_1/metrics"),                    # huron lab server
        Path("/home/hartman/scratch/metrics"),         # DRAC hartman (DQN/REINFORCE)
        Path("/home/sgomezro/scratch/metrics"),        # DRAC sgomezro (CMA)
        Path("/home/epigou/scratch/metrics"),          # DRAC epigou (ODT)
        REPO_ROOT / "_local_metrics",                  # local testing
    ])
    # de-dup while preserving order
    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def find_metrics_dir(exp_num, override=None, owner=None):
    for base in metrics_root_candidates(override, owner=owner):
        d = base / f"Exp_{exp_num}"
        if d.exists():
            return d
    return None


# ------------------------------------------------------ expected-episode counts
def get_expected_episodes(cfg):
    """Total number of episodes/generations the experiment is configured to run."""
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
        return None, aggregation_count, None
    return aggregation_count * eps_per_agg, aggregation_count, eps_per_agg


# ------------------------------------------------------------------ formatting
def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0:
        return f"{d}d {h:02d}h {m:02d}m"
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def parse_squeue_elapsed(elapsed):
    """SLURM %M format: D-HH:MM:SS | HH:MM:SS | MM:SS | SS  -> seconds."""
    days = 0
    if "-" in elapsed:
        d, elapsed = elapsed.split("-", 1)
        days = int(d)
    parts = elapsed.split(":")
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


# ---------------------------------------------------------------- slurm lookup
def check_slurm_status(exp_num, user=None):
    """Return (state, job_id, elapsed_secs) for a matching job, or None."""
    try:
        cmd = ["squeue", "-h", "-o", "%i %j %T %M"]
        if user:
            cmd += ["-u", user]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        job_id, name, state, elapsed = parts
        if f"Exp_{exp_num}_train" in name or f"Exp_{exp_num}_eval" in name:
            try:
                elapsed_sec = parse_squeue_elapsed(elapsed)
            except Exception:
                elapsed_sec = None
            return (state, job_id, elapsed_sec)
    return None


# --------------------------------------------------------------- CSV scanning
def scan_agent_csv(csv_path):
    """
    Single-pass scan of metrics_agent_episode_level.csv. Returns:
        distinct_pairs : sorted list of (agg, ep) tuples actually seen
        rewards_per_pair : dict (agg, ep) -> [reward, ...]
    Pure stdlib; handles files with >1M rows in a few seconds.
    """
    distinct = set()
    rewards_per_pair = {}
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
            pair = (agg, ep)
            distinct.add(pair)
            rewards_per_pair.setdefault(pair, []).append(rew)

    return sorted(distinct), rewards_per_pair


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(
        description="Check progress of an experiment from its metrics CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("exp_num", type=int, help="Experiment number, e.g. 9000")
    p.add_argument("--metrics-root", help="Override metrics root directory (default: inferred from algorithm owner)")
    p.add_argument(
        "--user",
        default=None,
        help="SLURM user to query (default: inferred from algorithm owner -- "
             "hartman for DQN/REINFORCE, sgomezro for CMA, epigou for ODT; "
             "falls back to $USER)",
    )
    args = p.parse_args()

    # Load config
    cfg_path = REPO_ROOT / "experiments" / f"Exp_{args.exp_num}" / "config.yaml"
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    algo = cfg["algorithm_settings"]["algorithm"]
    seed = cfg["environment_settings"]["seed"]
    num_zones = len(cfg["environment_settings"]["coords"])
    expected_eps, agg_count, eps_per_agg = get_expected_episodes(cfg)

    # Auto-fill user from the algorithm's owner (override with --user)
    owner = ALGO_OWNER.get(algo)
    if args.user is None:
        args.user = owner or os.environ.get("USER")

    owner_tag = f", owner={owner}" if owner else ""
    print(f"Exp_{args.exp_num}  ({algo}, seed={seed}, zones={num_zones}{owner_tag})")
    if expected_eps is None:
        print(f"  Expected: unknown (algorithm '{algo}' not handled by this script)")
    else:
        per_zone = " per zone" if num_zones > 1 else ""
        print(f"  Expected: {agg_count} aggs * {eps_per_agg} eps "
              f"= {expected_eps} episodes{per_zone}")

    # Find metrics (owner-aware: try the algorithm owner's scratch first)
    metrics_dir = find_metrics_dir(args.exp_num, args.metrics_root, owner=owner)
    slurm = check_slurm_status(args.exp_num, args.user)

    if metrics_dir is None:
        print(f"  Metrics:  NOT FOUND in any of:")
        for c in metrics_root_candidates(args.metrics_root, owner=owner):
            print(f"              {c}")
        if slurm:
            elapsed = format_duration(slurm[2])
            print(f"  SLURM:    {slurm[0]} (job {slurm[1]}, elapsed {elapsed})")
        else:
            print(f"  SLURM:    no matching job (not queued / not running)")
        return 0

    print(f"  Metrics:  {metrics_dir}")
    agent_csv = metrics_dir / "train" / "metrics_agent_episode_level.csv"
    if not agent_csv.exists():
        print(f"  Status:   metrics dir exists but agent CSV not yet written")
        if slurm:
            elapsed = format_duration(slurm[2])
            print(f"  SLURM:    {slurm[0]} (job {slurm[1]}, elapsed {elapsed})")
        return 0

    # Scan CSV with stdlib only (no pandas)
    try:
        pairs, rewards_per_pair = scan_agent_csv(agent_csv)
    except Exception as e:
        print(f"  ERROR reading {agent_csv}: {e}", file=sys.stderr)
        return 1

    actual_eps = len(pairs)

    if expected_eps:
        pct = actual_eps / expected_eps * 100
        print(f"  Actual:   {actual_eps} episodes  ({pct:.1f}%)")
    else:
        print(f"  Actual:   {actual_eps} episodes")

    if actual_eps > 0:
        last_agg, last_ep = pairs[-1]
        print(f"  Current:  aggregation {last_agg + 1}/{agg_count}, "
              f"episode {last_ep + 1}/{eps_per_agg}")

        # Last-N-episode mean reward
        n = min(100, actual_eps)
        tail_pairs = pairs[-n:]
        tail_rewards = []
        for pair in tail_pairs:
            tail_rewards.extend(rewards_per_pair[pair])
        print(f"  Last {n}-ep mean reward: {mean(tail_rewards):+.2f}")

    # CSV freshness
    mtime = agent_csv.stat().st_mtime
    mtime_dt = datetime.fromtimestamp(mtime)
    age = time.time() - mtime
    print(f"  CSV mtime: {mtime_dt:%Y-%m-%d %H:%M:%S}  ({format_duration(age)} ago)")

    # SLURM info + ETA
    if slurm:
        state, job_id, elapsed_sec = slurm
        elapsed_str = format_duration(elapsed_sec)
        print(f"  SLURM:    {state} (job {job_id}, elapsed {elapsed_str})")
        if expected_eps and actual_eps > 0 and elapsed_sec and elapsed_sec > 0:
            rate = actual_eps / elapsed_sec        # eps per second
            remaining_eps = expected_eps - actual_eps
            eta_sec = remaining_eps / rate if rate > 0 else None
            print(f"  Rate:     {rate * 60:.1f} eps/min "
                  f"({rate * 3600:.0f} eps/hour)")
            if remaining_eps > 0:
                print(f"  ETA:      {format_duration(eta_sec)} remaining")
            else:
                print(f"  ETA:      done (waiting for job to wrap up)")
    else:
        if actual_eps and expected_eps and actual_eps >= expected_eps:
            print(f"  SLURM:    no active job -- looks finished")
        elif age > 600:
            print(f"  SLURM:    no active job -- possibly stopped / crashed "
                  f"(CSV idle {format_duration(age)})")
        else:
            print(f"  SLURM:    no active job (CSV updated recently -- "
                  f"reading just after job ended?)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
