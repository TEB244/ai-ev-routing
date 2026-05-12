"""
Build paper_figure_generators/Sensitivity Analysis.ipynb.

Run from the repo root:
    python _scripts/build_sensitivity_notebook.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper_figure_generators" / "Sensitivity Analysis.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(lines),
    }


def split_lines(text):
    """Convert a triple-quoted block into nbformat-style source lines.

    Every line except the last ends in '\\n'; the last has no trailing newline.
    """
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = []

# -------------------------------------------------------------------- header
cells.append(md(*split_lines("""# Sensitivity Analysis

Reward-weight sensitivity sweep for the SURE-DM paper.

The reward function is parameterised as

```
R = -(w_d * D * f_d + w_T * max_tau * f_t + w_e * E * f_e)
```

where the `f_*` scales are unit-conversion factors held fixed at baseline
(`distance_scale=100`, `traffic_scale=1`, `energy_scale=0.001`) and the
`w_*` weights are tunable coefficients defaulting to 1.0. This notebook
sweeps each weight over `{0, 1, 5, 7, 10}` for each of the four
decision-makers (DQN, REINFORCE, CMA, ODT), with three seeds per cell.

The 180 experiments are laid out as

| Range | Model | distance_weight | traffic_weight | energy_weight |
|---|---|---|---|---|
| 7000-7044 | DQN       | 7000-7014 | 7015-7029 | 7030-7044 |
| 7045-7089 | REINFORCE | 7045-7059 | 7060-7074 | 7075-7089 |
| 7090-7134 | CMA       | 7090-7104 | 7105-7119 | 7120-7134 |
| 7135-7179 | ODT       | 7135-7149 | 7150-7164 | 7165-7179 |

Within each 15-experiment block, ordering is **value-major, seed-minor**:
offsets +0..+2 use weight=0 (ablation), +3..+5 use weight=1 (baseline),
+6..+8 use weight=5, +9..+11 use weight=7, +12..+14 use weight=10. Seeds
within each triplet are `[1234, 5555, 2020]`.

The notebook has two parts:

1. **Validate Exp_7000** — confirm the data has been transferred onto this
   server and the training run produced sensible numbers.
2. **Sensitivity figures** — load whatever subset of the 180 experiments
   is available locally and produce the figures needed for the paper.
""")))

# -------------------------------------------------------------------- imports
cells.append(md("### Imports"))
cells.append(code(*split_lines("""import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from environment.data_loader import load_config_file""")))

# -------------------------------------------------------------------- config
cells.append(md("### Config"))
cells.append(code(*split_lines("""# Per-DM colors, matching the rest of the paper
colors = {
    "DQN":       'darkorange',
    "REINFORCE": 'forestgreen',
    "CMA":       'turquoise',
    "CMA-ES":    'turquoise',
    "ODT":       'blueviolet',
}

# Sensitivity sweep layout (mirrors generate_sensitivity_experiments.py and
# experiments/experiment_list.txt). Numbers are exp-number offsets within
# each model's 45-experiment block.
SWEEP_VALUES = [0, 1, 5, 7, 10]
SEEDS = [1234, 5555, 2020]
SWEPT_VARS = ['distance_weight', 'traffic_weight', 'energy_weight']
MODELS = ['DQN', 'REINFORCE', 'CMA', 'ODT']

# Block layout: each model gets 45 consecutive experiments
MODEL_START = {
    'DQN':       7000,
    'REINFORCE': 7045,
    'CMA':       7090,
    'ODT':       7135,
}

# Candidate metrics roots. The notebook tries these in order; first hit wins.
METRICS_ROOT_CANDIDATES = [
    '../../../../storage_1/metrics',
    '/storage_1/metrics',
    '../metrics',
    '../_sensitivity_metrics',
]""")))

# -------------------------------------------------------------------- helpers
cells.append(md("### Helpers"))
cells.append(code(*split_lines("""def exp_metadata(exp_num):
    \"\"\"Return (model, swept_var, swept_value, seed) for a sensitivity exp number.\"\"\"
    for model, start in MODEL_START.items():
        if start <= exp_num < start + 45:
            offset = exp_num - start
            swept_var = SWEPT_VARS[offset // 15]                  # 0..2
            within_block = offset % 15
            value = SWEEP_VALUES[within_block // 3]               # 0..4
            seed = SEEDS[within_block % 3]                        # 0..2
            return model, swept_var, value, seed
    raise ValueError(f"Exp_{exp_num} is outside the 7000-7179 sensitivity range")


def resolve_metrics_root():
    for root in METRICS_ROOT_CANDIDATES:
        if os.path.isdir(root):
            return root
    return None


def exp_paths(exp_num, root=None):
    \"\"\"Return dict of expected file paths for a given experiment.\"\"\"
    if root is None:
        root = resolve_metrics_root()
    if root is None:
        return None
    base = Path(root) / f"Exp_{exp_num}" / "train"
    return {
        'base':    base,
        'agent':   base / 'metrics_agent_episode_level.csv',
        'station': base / 'metrics_station_episode_level.csv',
        'config':  Path('..') / 'experiments' / f'Exp_{exp_num}' / 'config.yaml',
    }


def load_exp(exp_num, root=None):
    \"\"\"Load one experiment's agent + station CSVs and config. Returns None if missing.\"\"\"
    paths = exp_paths(exp_num, root)
    if paths is None or not paths['agent'].exists() or not paths['station'].exists():
        return None
    model, swept_var, value, seed = exp_metadata(exp_num)
    agent = pd.read_csv(paths['agent'])
    station = pd.read_csv(paths['station'])
    agent['exp_num']    = exp_num
    agent['model']      = model
    agent['swept_var']  = swept_var
    agent['weight']     = value
    agent['seed']       = seed
    station['exp_num']  = exp_num
    station['model']    = model
    station['swept_var']= swept_var
    station['weight']   = value
    station['seed']     = seed
    return {'agent': agent, 'station': station, 'config_path': paths['config']}


METRICS_ROOT = resolve_metrics_root()
print(f"Metrics root resolved to: {METRICS_ROOT}")""")))

# -------------------------------------------------------------------- Part 1 header
cells.append(md(*split_lines("""# Part 1 - Validate Exp_7000

Confirm the transfer worked and the training run looks healthy before
relying on the data. Exp_7000 is DQN with `distance_weight = 0`
(ablation of the distance term), seed 1234.""")))

# -------------------------------------------------------------------- 1A files
cells.append(md("### 1A. File presence"))
cells.append(code(*split_lines("""TARGET_EXP = 7000

meta = exp_metadata(TARGET_EXP)
print(f"Exp_{TARGET_EXP}: model={meta[0]}, swept_var={meta[1]}, weight={meta[2]}, seed={meta[3]}")

paths = exp_paths(TARGET_EXP)
print(f"\\nExpected metrics base: {paths['base']}")
print(f"  exists:        {paths['base'].exists()}")

for key in ('agent', 'station', 'config'):
    p = paths[key]
    if p.exists():
        size_kb = p.stat().st_size / 1024
        print(f"  {key:8s} {p}  ({size_kb:.1f} kB)")
    else:
        print(f"  {key:8s} MISSING -> {p}")

# Surface every file under the metrics dir so the user can confirm what's there
if paths['base'].exists():
    print("\\nAll files in metrics base:")
    for f in sorted(paths['base'].glob('*')):
        print(f"  {f.name}  ({f.stat().st_size/1024:.1f} kB)")""")))

# -------------------------------------------------------------------- 1B load
cells.append(md("### 1B. Load Exp_7000"))
cells.append(code(*split_lines("""data = load_exp(TARGET_EXP)
assert data is not None, (
    f"Exp_{TARGET_EXP} not found under any of:\\n  "
    + "\\n  ".join(METRICS_ROOT_CANDIDATES)
    + "\\n\\nIf you just rsync'd from narval, check that the destination matches one of these paths."
)

agent = data['agent']
station = data['station']

print(f"Agent CSV:   {len(agent):>6} rows, columns = {list(agent.columns)}")
print(f"Station CSV: {len(station):>6} rows, columns = {list(station.columns)}")
print()
print("Agent head:")
display(agent.head())
print("\\nStation head:")
display(station.head())""")))

# -------------------------------------------------------------------- 1C sanity
cells.append(md("### 1C. Sanity checks"))
cells.append(code(*split_lines("""cfg = load_config_file(str(paths['config']))
env_c = cfg['environment_settings']
num_aggs = cfg['federated_learning_settings']['aggregation_count']

# Episode count is per-aggregation for RL, per-generation for CMA, etc.
if cfg['algorithm_settings']['algorithm'] in ['DQN', 'REINFORCE', 'PPO', 'DDPG', 'ODT']:
    eps_per_agg = cfg['nn_hyperparameters']['num_episodes']
elif cfg['algorithm_settings']['algorithm'] in ['CMA', 'DENSER', 'NEAT']:
    eps_per_agg = cfg['cma_parameters']['max_generations']
else:
    eps_per_agg = None

expected_episodes = (num_aggs * eps_per_agg) if eps_per_agg else None

print(f"Algorithm:              {cfg['algorithm_settings']['algorithm']}")
print(f"Aggregations:           {num_aggs}")
print(f"Episodes per agg:       {eps_per_agg}")
print(f"Expected total eps:     {expected_episodes}")
print(f"Distinct (agg, ep):     {agent[['aggregation','episode']].drop_duplicates().shape[0]}")
print(f"Num cars:               {env_c['num_of_cars']}")
print(f"Distinct agents/ep:     {agent.groupby(['aggregation','episode'])['agent_index'].nunique().mean():.1f} (expect {env_c['num_of_cars']})")
print()
print("Reward weights in this experiment's config:")
print(f"  distance_weight = {env_c.get('distance_weight', 1.0)}")
print(f"  traffic_weight  = {env_c.get('traffic_weight',  1.0)}")
print(f"  energy_weight   = {env_c.get('energy_weight',   1.0)}")
print()
print("Reward summary (per car-episode):")
print(agent['reward'].describe())
print()
print("Distance summary:")
print(agent['distance'].describe())
print()
print("Station traffic summary (per episode peak):")
print(station['traffic'].describe())""")))

# -------------------------------------------------------------------- 1D training curve
cells.append(md("### 1D. Training curve"))
cells.append(code(*split_lines("""# Mean reward across cars per (aggregation, episode), then plotted in time order.
curve = (agent
         .groupby(['aggregation', 'episode'], as_index=False)['reward']
         .mean()
         .sort_values(['aggregation', 'episode'])
         .reset_index(drop=True))
curve['global_episode'] = curve.index

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(curve['global_episode'], curve['reward'], lw=0.8, alpha=0.7, label='per-episode mean')
# Rolling mean to make the trend visible
window = max(5, len(curve) // 50)
ax.plot(curve['global_episode'], curve['reward'].rolling(window, min_periods=1).mean(),
        lw=2, label=f'{window}-ep rolling mean', color='black')
# Aggregation boundaries
for agg in sorted(curve['aggregation'].unique())[1:]:
    boundary = curve[curve['aggregation'] == agg]['global_episode'].iloc[0]
    ax.axvline(boundary, color='red', alpha=0.15, lw=1)
ax.set_xlabel('Episode (global)')
ax.set_ylabel('Mean reward across cars')
ax.set_title(f"Exp_{TARGET_EXP} training curve  ({meta[0]}, {meta[1]}={meta[2]}, seed={meta[3]})")
ax.legend(loc='best')
plt.tight_layout()
plt.show()""")))

# -------------------------------------------------------------------- 1E validation summary
cells.append(md("### 1E. Validation summary"))
cells.append(code(*split_lines("""checks = []

# 1. Files present
checks.append(('Agent CSV exists',   paths['agent'].exists()))
checks.append(('Station CSV exists', paths['station'].exists()))

# 2. Row counts match expectations
if expected_episodes:
    actual_eps = agent[['aggregation', 'episode']].drop_duplicates().shape[0]
    checks.append((f'Episode count ~ expected ({actual_eps} vs {expected_episodes})',
                   abs(actual_eps - expected_episodes) <= max(2, 0.05 * expected_episodes)))
checks.append(('Agent rows > 0', len(agent) > 0))
checks.append(('Station rows > 0', len(station) > 0))

# 3. Rewards are negative (sanity: this is a minimisation cast as -sum)
checks.append(('Rewards are negative on average', agent['reward'].mean() < 0))

# 4. With distance_weight=0 (Exp_7000), the distance term is ablated. We can't
#    test the agent's behaviour directly, but we can confirm the config did
#    set distance_weight=0 as expected.
if TARGET_EXP == 7000:
    checks.append(('Config distance_weight == 0 (ablation)',
                   abs(env_c.get('distance_weight', 1.0)) < 1e-9))

for desc, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {desc}")

n_pass = sum(1 for _, ok in checks if ok)
print(f"\\n{n_pass} / {len(checks)} checks passed.")""")))

# -------------------------------------------------------------------- Part 2 header
cells.append(md(*split_lines("""# Part 2 - Sensitivity figures

Loads every Exp_7xxx available locally and produces:

- **Fig A** — Final reward as a function of the swept weight, per DM. One
  subplot per swept variable. Demonstrates whether DM ranking is preserved.
- **Fig B** — Underlying-metric response: how each of (distance traveled,
  peak traffic, battery delta) shifts as its own weight is varied. Tests
  the hypothesis that the agent re-balances behaviour to favour the
  up-weighted term.
- **Fig C** — DM ranking stability: which DM "wins" on reward at each
  weight setting.
- **Summary table** — per-(model, swept_var, weight) means and stds.""")))

# -------------------------------------------------------------------- 2A bulk load
cells.append(md("### 2A. Bulk-load all available sensitivity experiments"))
cells.append(code(*split_lines("""ALL_EXPS = list(range(7000, 7180))
loaded = {}
missing = []

for n in ALL_EXPS:
    d = load_exp(n)
    if d is None:
        missing.append(n)
    else:
        loaded[n] = d

print(f"Loaded:  {len(loaded):>3} / {len(ALL_EXPS)} experiments")
print(f"Missing: {len(missing):>3}")

if missing:
    # Group missing into contiguous ranges for compact display
    runs = []
    s = missing[0]; prev = s
    for n in missing[1:]:
        if n == prev + 1:
            prev = n
        else:
            runs.append((s, prev)); s = n; prev = n
    runs.append((s, prev))
    print("  ranges: " + ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs))

# Build flat dataframes for downstream analysis
if loaded:
    agent_all   = pd.concat([d['agent']   for d in loaded.values()], ignore_index=True)
    station_all = pd.concat([d['station'] for d in loaded.values()], ignore_index=True)
    print(f"\\nCombined: {len(agent_all):,} agent rows, {len(station_all):,} station rows")
else:
    agent_all = pd.DataFrame()
    station_all = pd.DataFrame()
    print("\\nNo data loaded - the figures below will be empty.")""")))

# -------------------------------------------------------------------- 2B per-exp summary
cells.append(md("### 2B. Per-experiment summary (one row per Exp_7xxx)"))
cells.append(code(*split_lines("""# Use the last N episodes per experiment as "final" performance so we
# average over converged behaviour rather than early exploration.
LAST_N_EPISODES = 50  # adjust if your runs are shorter

def summarise_one(exp_num, d):
    a = d['agent']; s = d['station']
    n_ep = a[['aggregation', 'episode']].drop_duplicates().shape[0]
    # Pick the last N (aggregation, episode) pairs by their order of appearance
    last_pairs = (a[['aggregation', 'episode']]
                  .drop_duplicates()
                  .tail(min(LAST_N_EPISODES, n_ep)))
    a_tail = a.merge(last_pairs, on=['aggregation', 'episode'])
    s_tail = s.merge(last_pairs, on=['aggregation', 'episode'])
    battery_delta = a_tail['starting_battery'] - a_tail['ending_battery']
    model, swept_var, weight, seed = exp_metadata(exp_num)
    return {
        'exp_num':         exp_num,
        'model':           model,
        'swept_var':       swept_var,
        'weight':          weight,
        'seed':            seed,
        'reward_mean':     a_tail['reward'].mean(),
        'reward_std':      a_tail['reward'].std(),
        'distance_mean':   a_tail['distance'].mean(),
        'distance_std':    a_tail['distance'].std(),
        'peak_traffic':    s_tail['traffic'].max() if len(s_tail) else np.nan,
        'mean_peak_traf':  s_tail['traffic'].mean() if len(s_tail) else np.nan,
        'battery_delta':   battery_delta.mean(),
        'n_episodes_used': len(last_pairs),
    }

summary = pd.DataFrame([summarise_one(n, d) for n, d in loaded.items()])
summary = summary.sort_values(['model', 'swept_var', 'weight', 'seed']).reset_index(drop=True)
print(f"Summary table: {len(summary)} experiments")
display(summary.head(20))""")))

# -------------------------------------------------------------------- 2C Fig A reward vs weight
cells.append(md("### 2C. Figure A - Final reward vs swept weight"))
cells.append(code(*split_lines("""def aggregate_over_seeds(df, value_col):
    g = df.groupby(['model', 'swept_var', 'weight'])[value_col]
    return g.agg(['mean', 'std', 'count']).reset_index()


def plot_metric_grid(df, value_col, ylabel, title, fname=None):
    \"\"\"One row of 3 subplots (one per swept variable), all DMs overlaid.\"\"\"
    if df.empty:
        print(f"No data for {title}")
        return
    agg = aggregate_over_seeds(df, value_col)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, sv in zip(axes, SWEPT_VARS):
        sub = agg[agg['swept_var'] == sv]
        for model in MODELS:
            line = sub[sub['model'] == model].sort_values('weight')
            if line.empty:
                continue
            ax.errorbar(line['weight'], line['mean'], yerr=line['std'],
                        marker='o', lw=1.8, capsize=3,
                        color=colors.get(model, 'gray'), label=model)
        ax.set_title(f'Sweep: {sv}')
        ax.set_xlabel('weight value')
        ax.set_xticks(SWEEP_VALUES)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc='best', fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    if fname:
        out = Path('figures') / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f'Saved {out}')
    plt.show()


plot_metric_grid(
    summary, 'reward_mean',
    ylabel='Mean reward (last episodes)',
    title='Figure A - Final reward vs swept weight, per decision-maker',
    fname='sensitivity_fig_a_reward.png',
)""")))

# -------------------------------------------------------------------- 2D Fig B underlying metrics
cells.append(md(*split_lines("""### 2D. Figure B - Underlying-metric response

Three rows (one per underlying metric: distance, peak traffic, battery
delta), three columns (one per swept variable). The diagonal where the
swept variable matches the metric is the most informative cell -
e.g. row `distance` × col `distance_weight` answers \"does up-weighting
distance actually reduce distance travelled?\"""")))
cells.append(code(*split_lines("""metrics_to_plot = [
    ('distance_mean',  'Mean distance (km, approx)',      'distance_weight'),
    ('mean_peak_traf', 'Mean peak traffic (cars)',        'traffic_weight'),
    ('battery_delta',  'Mean battery delta (start-end)',  'energy_weight'),
]

if summary.empty:
    print('No data - skipping Figure B.')
else:
    fig, axes = plt.subplots(len(metrics_to_plot), 3,
                              figsize=(15, 4 * len(metrics_to_plot)),
                              sharex='col')
    for r, (col, ylabel, expected_sv) in enumerate(metrics_to_plot):
        agg = aggregate_over_seeds(summary, col)
        for c, sv in enumerate(SWEPT_VARS):
            ax = axes[r, c]
            sub = agg[agg['swept_var'] == sv]
            for model in MODELS:
                line = sub[sub['model'] == model].sort_values('weight')
                if line.empty:
                    continue
                ax.errorbar(line['weight'], line['mean'], yerr=line['std'],
                            marker='o', lw=1.6, capsize=3,
                            color=colors.get(model, 'gray'), label=model)
            ax.grid(alpha=0.3)
            ax.set_xticks(SWEEP_VALUES)
            if r == 0:
                ax.set_title(f'Sweep: {sv}')
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == len(metrics_to_plot) - 1:
                ax.set_xlabel('weight value')
            # Highlight the diagonal cell (matched metric/weight pair)
            if sv == expected_sv:
                for spine in ax.spines.values():
                    spine.set_edgecolor('crimson'); spine.set_linewidth(1.6)
    axes[0, -1].legend(loc='best', fontsize=8)
    fig.suptitle('Figure B - Underlying-metric response per swept weight\\n'
                 '(red border = swept variable matches the metric on that row)',
                 fontsize=12)
    plt.tight_layout()
    out = Path('figures') / 'sensitivity_fig_b_metric_response.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show()""")))

# -------------------------------------------------------------------- 2E Fig C ranking stability
cells.append(md(*split_lines("""### 2E. Figure C - DM ranking stability

For each (swept variable, weight value), rank the DMs by their mean
reward across seeds. The headline claim of the sensitivity analysis is
that this ranking is preserved across the sweep.""")))
cells.append(code(*split_lines("""if summary.empty:
    print('No data - skipping Figure C.')
else:
    ranked = (summary
              .groupby(['swept_var', 'weight', 'model'])['reward_mean']
              .mean()
              .reset_index())
    ranked['rank'] = (ranked
                      .groupby(['swept_var', 'weight'])['reward_mean']
                      .rank(method='min', ascending=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, sv in zip(axes, SWEPT_VARS):
        sub = ranked[ranked['swept_var'] == sv]
        for model in MODELS:
            line = sub[sub['model'] == model].sort_values('weight')
            if line.empty:
                continue
            ax.plot(line['weight'], line['rank'], marker='o', lw=2,
                    color=colors.get(model, 'gray'), label=model)
        ax.set_title(f'Sweep: {sv}')
        ax.set_xticks(SWEEP_VALUES)
        ax.set_yticks([1, 2, 3, 4])
        ax.invert_yaxis()      # rank 1 at top
        ax.set_xlabel('weight value')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Rank (1 = best reward)')
    axes[-1].legend(loc='best', fontsize=9)
    fig.suptitle('Figure C - DM ranking by reward at each weight setting')
    plt.tight_layout()
    out = Path('figures') / 'sensitivity_fig_c_ranking.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show()""")))

# -------------------------------------------------------------------- 2F summary table
cells.append(md("### 2F. Summary table for paper appendix"))
cells.append(code(*split_lines("""if summary.empty:
    print('No data - skipping summary table.')
else:
    # Collapse across seeds: mean ± std of reward per (model, swept_var, weight)
    tbl = (summary
           .groupby(['model', 'swept_var', 'weight'])
           .agg(reward_mean=('reward_mean', 'mean'),
                reward_std =('reward_mean', 'std'),
                n_seeds    =('seed', 'nunique'))
           .reset_index())
    tbl['reward'] = tbl.apply(
        lambda r: f"{r['reward_mean']:.2f} +/- {r['reward_std']:.2f}"
                  if r['n_seeds'] >= 2 else f"{r['reward_mean']:.2f}",
        axis=1)
    pivot = (tbl
             .pivot_table(index=['model', 'swept_var'],
                          columns='weight',
                          values='reward',
                          aggfunc='first'))
    display(pivot)
    out = Path('table_data') / 'sensitivity_summary.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out)
    print(f'Saved {out}')""")))

# -------------------------------------------------------------------- assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({len(cells)} cells)")
