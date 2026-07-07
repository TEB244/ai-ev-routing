"""
v2_data -- data layer for paper_figure_generators_v2.

Regenerates the reward / in-simulation behaviour inputs of the SURE-DM paper
figures and tables from the **revised 4xxx experiments only** (3 seeds, all
four seasons, all aggregation levels), reading the raw per-experiment metrics
that the post-fix training runs wrote to ``/storage_1/metrics_postfix`` on
Huron.

Why a separate module + cache
-----------------------------
The original notebooks read pre-aggregated CSVs from
``/storage_1/metrics/formatted_experiment_data/{part_1,part_4}/``. Those files
only exist for the *original* (pre-rerun) data, so the v2 notebooks instead
aggregate the raw post-fix metrics. The raw ``metrics_agent_episode_level.csv``
files are large (10k episodes x cars x zones), so we do a single pass with
``build_v2_cache.py`` and write small intermediate CSVs to
``table_data/_v2_cache/``. The notebooks then read those (fast, low-memory).

What comes from the NEW data vs the OLD data
--------------------------------------------
NEW (this module / metrics_postfix):
  * cumulative average reward curves        -> Fig 3 (episode_plateau_and_seasons)
  * per-(algorithm, season) distance / peak  -> Table 2 (tab:combined_metrics)
    traffic / battery-use / reward
  * per-algorithm final reward (mean +/- std)-> Table 3 reward (tab:resource_analysis)
                                                and the reward input to Fig 6
OLD (unchanged, kept in the Experiment 3 notebook's existing cells):
  * training duration, power, CO2 emissions, model size  -> Table 3 duration/
    emissions, Fig 5 (ridgeline), Fig 6 energy + model-size terms.

The cumulative-average-reward definition mirrors
``environment/evaluation.evaluate_by_agent``: per (aggregation, episode) mean
reward across cars/zones, recalculated to a global episode index
(``aggregation * episodes_per_aggregation + episode``), then an expanding
(running) mean over episodes.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from environment.data_loader import load_config_file

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Revised main batch: 4xxx is a *complete* grid on its own
# (4 seasons x 4 decision-makers x 3 aggregation levels x 3 seeds = 180 exps).
# 4xxx alone previews trends with 3 seeds; the published figures pool
# 4xxx+5xxx+6xxx = 9 seeds (see EXPERIMENTS_ALL).
EXPERIMENTS_4XXX = list(range(4000, 4180))

# Full published set: 4xxx + 5xxx + 6xxx = 9 seeds (3 per batch), all four
# seasons and all aggregation levels. 5xxx/6xxx have finished running; build the
# cache over this range to move the reward / in-simulation-behaviour figures and
# tables from the 3-seed preview to the full 9-seed results. This is the default.
EXPERIMENTS_ALL = (list(range(4000, 4180))
                   + list(range(5000, 5180))
                   + list(range(6000, 6180)))

# NEW post-fix raw metrics (the data that was just rerun). Tried in order.
NEW_METRICS_ROOT_CANDIDATES = [
    "/storage_1/metrics_postfix",
    "../../../../storage_1/metrics_postfix",
    "../metrics_postfix",
    "../_local_metrics",  # local-dev fallback
]

# Repo-internal per-experiment configs (version controlled, same for old/new).
REPO_EXPERIMENTS = "../experiments"

# Where the one-pass cache is written / read.
CACHE_DIR = Path("table_data") / "_v2_cache"

# How many trailing global episodes count as "steady-state" behaviour for the
# table metrics and the final reward. The paper describes the last 100 episodes
# as steady behaviour (Fig 3 caption); change here if you want a different
# window.
DEFAULT_LAST_N_EPISODES = 100

# Canonical display names.
ALGO_DISPLAY = {"CMA": "CMA-ES", "DQN": "DQN", "ODT": "ODT", "REINFORCE": "REINFORCE"}


# --------------------------------------------------------------------------
# Roots / config helpers
# --------------------------------------------------------------------------

def _resolve(candidates):
    for r in candidates:
        if r and os.path.isdir(r):
            return r
    return None


def resolve_new_root(override=None):
    """Resolve the metrics_postfix root (new post-fix data)."""
    if override:
        return override if os.path.isdir(override) else None
    return _resolve(NEW_METRICS_ROOT_CANDIDATES)


def _episodes_per_aggregation(cfg):
    algo = cfg["algorithm_settings"]["algorithm"]
    if algo in ("DQN", "REINFORCE", "PPO", "DDPG", "ODT"):
        return cfg["nn_hyperparameters"]["num_episodes"]
    if algo in ("CMA", "DENSER", "NEAT"):
        return cfg["cma_parameters"]["max_generations"]
    return None


def _exp_meta(exp_num, repo_experiments=REPO_EXPERIMENTS):
    """Return per-experiment metadata from its version-controlled config."""
    cfg_p = Path(repo_experiments) / f"Exp_{exp_num}" / "config.yaml"
    if not cfg_p.exists():
        return None
    cfg = load_config_file(str(cfg_p))
    env_c = cfg["environment_settings"]
    return {
        "exp_num": exp_num,
        "algorithm": cfg["algorithm_settings"]["algorithm"],
        "season": env_c["season"],
        "seed": env_c["seed"],
        "num_aggs": cfg["federated_learning_settings"]["aggregation_count"],
        "eps_per_agg": _episodes_per_aggregation(cfg),
    }


def _exp_train_dir(exp_num, new_root):
    return Path(new_root) / f"Exp_{exp_num}" / "train"


# --------------------------------------------------------------------------
# Per-experiment aggregation (single file -> small result)
# --------------------------------------------------------------------------

def _reward_curve_one(agent_path, meta):
    """Cumulative (expanding) average reward per global episode for one run.

    Mirrors environment/evaluation.evaluate_by_agent, which weights each *zone*
    equally: mean reward over agents within each (aggregation, episode, zone),
    then an equal-weight average across zones, then an expanding mean over global
    episodes. For the 4xxx runs zones are balanced (same car count per zone), so
    this equals a flat pooled mean; doing it zone-aware keeps it correct if a
    zone ever has a different agent count. Reads only the needed columns.
    """
    eps = meta["eps_per_agg"] or 0
    df = pd.read_csv(agent_path, usecols=["aggregation", "episode", "zone", "reward"])
    # mean over agents within each zone, then equal-weight the zones
    per_zone = df.groupby(["aggregation", "episode", "zone"], as_index=False)["reward"].mean()
    per_ep = per_zone.groupby(["aggregation", "episode"], as_index=False)["reward"].mean()
    per_ep["episode"] = per_ep["aggregation"] * eps + per_ep["episode"]
    per_ep = per_ep.sort_values("episode").reset_index(drop=True)
    per_ep["cumulative_reward"] = per_ep["reward"].expanding().mean()
    out = per_ep[["episode", "cumulative_reward"]].copy()
    out["algorithm"] = meta["algorithm"]
    out["season"] = meta["season"]
    out["seed"] = meta["seed"]
    out["num_aggs"] = meta["num_aggs"]
    return out


def _tail_global_episodes(aggregation, episode, eps_per_agg, last_n):
    """Boolean mask selecting rows in the last ``last_n`` distinct global eps."""
    global_ep = aggregation * (eps_per_agg or 0) + episode
    keep = np.sort(pd.unique(global_ep))[-last_n:]
    return global_ep.isin(keep), global_ep


def _env_metrics_one(agent_path, station_path, meta, last_n):
    """Per-car-episode behaviour over the last ``last_n`` episodes of one run.

    Returns (agent_rows, station_rows) with columns matching the old
    formatted_experiment_data/part_4 CSVs so the Experiment 2 table cell is
    unchanged: agent -> distance_traveled, battery_used, reward; station ->
    traffic. Both tagged with algorithm/season/seed.
    """
    a = pd.read_csv(
        agent_path,
        usecols=["aggregation", "episode", "distance", "reward",
                 "starting_battery", "ending_battery"],
    )
    mask, _ = _tail_global_episodes(a["aggregation"], a["episode"], meta["eps_per_agg"], last_n)
    a = a[mask]
    agent_rows = pd.DataFrame({
        "algorithm": meta["algorithm"],
        "season": meta["season"],
        "seed": meta["seed"],
        "distance_traveled": a["distance"].to_numpy(),
        "battery_used": (a["starting_battery"] - a["ending_battery"]).to_numpy(),
        "reward": a["reward"].to_numpy(),
    })

    station_rows = None
    if station_path is not None and Path(station_path).exists():
        s = pd.read_csv(station_path, usecols=["aggregation", "episode", "traffic"])
        smask, _ = _tail_global_episodes(s["aggregation"], s["episode"], meta["eps_per_agg"], last_n)
        s = s[smask]
        station_rows = pd.DataFrame({
            "algorithm": meta["algorithm"],
            "season": meta["season"],
            "seed": meta["seed"],
            "traffic": s["traffic"].to_numpy(),
        })
    return agent_rows, station_rows


def _final_reward_one(agent_path, meta, last_n):
    """Mean reward over the last ``last_n`` episodes of one run (one scalar)."""
    a = pd.read_csv(agent_path, usecols=["aggregation", "episode", "reward"])
    mask, _ = _tail_global_episodes(a["aggregation"], a["episode"], meta["eps_per_agg"], last_n)
    return float(a.loc[mask, "reward"].mean())


# --------------------------------------------------------------------------
# One-pass cache build (used by build_v2_cache.py)
# --------------------------------------------------------------------------

def build_cache(new_root=None, experiments=EXPERIMENTS_ALL,
                last_n=DEFAULT_LAST_N_EPISODES, out_dir=CACHE_DIR, verbose=True):
    """Single pass over the 4xxx runs: read each raw CSV once, write the small
    intermediates the notebooks consume. Returns the list of missing exps."""
    new_root = resolve_new_root(new_root)
    if new_root is None:
        raise SystemExit(
            "No metrics_postfix root found. Pass --metrics-root, or run where "
            f"one of these exists: {NEW_METRICS_ROOT_CANDIDATES}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"NEW metrics root : {new_root}")
        print(f"Experiments      : {experiments[0]}-{experiments[-1]} ({len(experiments)})")
        print(f"Steady-state win : last {last_n} episodes")
        print(f"Cache out        : {out_dir}\n")

    curve_frames, agent_frames, station_frames, reward_rows = [], [], [], []
    loaded, missing = 0, []

    for n in experiments:
        meta = _exp_meta(n)
        if meta is None:
            missing.append(n)
            continue
        agent_p = _exp_train_dir(n, new_root) / "metrics_agent_episode_level.csv"
        station_p = _exp_train_dir(n, new_root) / "metrics_station_episode_level.csv"
        if not agent_p.exists():
            missing.append(n)
            continue
        try:
            curve_frames.append(_reward_curve_one(agent_p, meta))
            ar, sr = _env_metrics_one(agent_p, station_p, meta, last_n)
            agent_frames.append(ar)
            if sr is not None:
                station_frames.append(sr)
            reward_rows.append({**{k: meta[k] for k in
                                   ("exp_num", "algorithm", "season", "seed", "num_aggs")},
                                "reward_final": _final_reward_one(agent_p, meta, last_n)})
            loaded += 1
            if verbose and loaded % 20 == 0:
                print(f"  ... {loaded} experiments aggregated")
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            print(f"  [WARN] Exp_{n}: {type(e).__name__}: {e}")
            missing.append(n)

    if loaded == 0:
        raise SystemExit("No experiments loaded -- check the metrics root and ranges.")

    pd.concat(curve_frames, ignore_index=True).to_csv(out_dir / "reward_curves.csv", index=False)
    pd.concat(agent_frames, ignore_index=True).to_csv(out_dir / "env_agent.csv", index=False)
    if station_frames:
        pd.concat(station_frames, ignore_index=True).to_csv(out_dir / "env_station.csv", index=False)
    pd.DataFrame(reward_rows).to_csv(out_dir / "reward_per_exp.csv", index=False)

    meta_txt = out_dir / "_BUILD_INFO.txt"
    meta_txt.write_text(
        f"metrics_root={new_root}\n"
        f"experiments={experiments[0]}-{experiments[-1]}\n"
        f"loaded={loaded}\nmissing={len(missing)}\n"
        f"last_n_episodes={last_n}\n"
        f"missing_list={missing}\n"
    )
    if verbose:
        print(f"\nDone. Loaded {loaded}/{len(experiments)}; {len(missing)} missing.")
        if missing:
            print(f"Missing: {missing}")
        print(f"Wrote cache to {out_dir}/")
    return missing


# --------------------------------------------------------------------------
# Cache loaders (used by the notebooks)
# --------------------------------------------------------------------------

def _require(path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Build the v2 cache first:\n"
            "    python build_v2_cache.py --metrics-root /storage_1/metrics_postfix\n"
            "(run from inside paper_figure_generators_v2/).")
    return path


def load_reward_curves(cache_dir=CACHE_DIR):
    """Fig 3 input: cumulative reward per global episode, per algorithm/season/
    seed/num_aggs."""
    return pd.read_csv(_require(Path(cache_dir) / "reward_curves.csv"))


def load_env_metrics(cache_dir=CACHE_DIR):
    """Table 2 input: (agent_df, station_df) with the part_4 column names."""
    cache_dir = Path(cache_dir)
    agent_df = pd.read_csv(_require(cache_dir / "env_agent.csv"))
    station_p = cache_dir / "env_station.csv"
    station_df = pd.read_csv(station_p) if station_p.exists() else pd.DataFrame()
    return agent_df, station_df


def reward_summary(cache_dir=CACHE_DIR):
    """Table 3 / Fig 6 input.

    Returns (per_algo, per_algo_byseed):
      per_algo        -- mean/std/count of the per-experiment final reward,
                         pooled across all 4xxx experiments of each DM.
      per_algo_byseed -- pooled to a per-seed final reward first (mean over each
                         seed's seasons/aggregation levels), then mean/std across
                         seeds. Matches the paper's "per-seed final reward"
                         language used for the Wilcoxon test.
    """
    per_exp = pd.read_csv(_require(Path(cache_dir) / "reward_per_exp.csv"))
    per_algo = (per_exp.groupby("algorithm")["reward_final"]
                .agg(["mean", "std", "count"]).reset_index())
    per_seed = (per_exp.groupby(["algorithm", "seed"])["reward_final"]
                .mean().reset_index())
    per_algo_byseed = (per_seed.groupby("algorithm")["reward_final"]
                       .agg(["mean", "std", "count"]).reset_index())
    return per_algo, per_algo_byseed
