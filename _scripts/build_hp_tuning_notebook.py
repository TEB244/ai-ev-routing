"""
Build paper_figure_generators/Hyperparameter Tuning.ipynb.

This notebook loads results from the 9000-9179 hyperparameter-tuning
experiments and emits a per-algorithm recommended HP configuration.

Run from the repo root:
    python _scripts/build_hp_tuning_notebook.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper_figure_generators" / "Hyperparameter Tuning.ipynb"


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
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = []

# ============================================================== header
cells.append(md(*split_lines("""# Hyperparameter Tuning (9000-9179)

A reviewer asked for systematic HP tuning that was missing in the
original paper. This notebook loads the 180 HP-tuning experiments,
identifies the best value per HP per algorithm, and emits the
recommended HP configurations to apply to the rerun of the 4xxx /
5xxx / 6xxx main experiments.

**Design:**

- One-at-a-time (OAT) sweeps over 3 hyperparameters per algorithm,
  5 values per HP, 3 seeds per cell. Total = 4 x 3 x 5 x 3 = 180.
- Reduced compute scope per experiment (1 zone, spring only, 25 RL
  aggregations) - rankings are robust to this kind of reduction.
- Reward weights pinned at the recommended baseline (1, 1, 1) so the
  HP tuning is not confounded by reward shape.

**Selection criterion:** mean reward over the last N episodes (default
N=200) per (model, swept_HP, value), averaged across the 3 seeds. We
pick the HP value with the highest mean reward, breaking ties by
preferring values closer to existing defaults (Occam) and lower
across-seed std (stability).

**Output:** the final cell prints the recommended config snippet to
apply to each 4xxx template, and saves a CSV/JSON with the
recommendations under `paper_figure_generators/table_data/`.""")))

# ============================================================== imports
cells.append(md("### Imports"))
cells.append(code(*split_lines("""import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment.data_loader import load_config_file""")))

# ============================================================== config
cells.append(md("### Config"))
cells.append(code(*split_lines("""colors = {
    'DQN':       'darkorange',
    'REINFORCE': 'forestgreen',
    'CMA':       'turquoise',
    'CMA-ES':    'turquoise',
    'ODT':       'blueviolet',
}

# 9xxx HP tuning layout - kept in lock-step with
# generate_hp_tuning_experiments.py
SEEDS = [1234, 5555, 2020]
N_SEEDS  = 3

# Explicit experiment-number layout for both v1 (9000-9179) and v2
# (9180-9251) batches. Each tuple is (start_exp, algorithm, hp_name,
# values_list). Within each sweep, ordering is value-major / seed-minor
# (3 seeds per value).
EXP_LAYOUT = [
    # v1 - 9000-9179
    (9000, 'DQN',       'learning_rate',     [1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]),
    (9015, 'DQN',       'discount_factor',   [0.90, 0.95, 0.99, 0.995, 0.999]),
    (9030, 'DQN',       'buffer_limit',      [150, 500, 1500, 5000, 15000]),
    (9045, 'REINFORCE', 'learning_rate',     [1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]),
    (9060, 'REINFORCE', 'discount_factor',   [0.90, 0.95, 0.99, 0.995, 0.999]),
    (9075, 'REINFORCE', 'layers_arch',       [[32, 32], [64, 64], [128, 64, 64], [256, 128, 64], [512, 256, 128, 64]]),
    (9090, 'CMA',       'initial_sigma',     [0.01, 0.05, 0.10, 0.30, 1.00]),
    (9105, 'CMA',       'population_dimension', [10, 20, 40, 80, 160]),
    (9120, 'CMA',       'max_generations',   [50, 100, 200, 400, 800]),
    (9135, 'ODT',       'learning_rate',     [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]),
    (9150, 'ODT',       'embed_dim',         [64, 128, 256, 512, 1024]),
    (9165, 'ODT',       'n_layer',           [1, 2, 4, 6, 8]),
    # v2 - 9180-9251 (extension batch covering literature-impactful HPs the v1 sweep missed)
    (9180, 'DQN',       'target_network_update_frequency', [5, 10, 25, 50, 100]),
    (9195, 'DQN',       'target_episode_epsilon_frac',     [0.1, 0.2, 0.3, 0.5, 0.7]),
    (9210, 'DQN',       'learning_rate',     [3.0e-2, 1.0e-1]),    # extension of v1 lr sweep
    (9216, 'REINFORCE', 'learning_rate',     [3.0e-2, 1.0e-1]),    # extension of v1 lr sweep
    (9222, 'ODT',       'K',                 [5, 10, 20, 40, 80]),
    (9237, 'ODT',       'rtg',               [-30, -50, -75, -100, -150]),
]

# (algorithm, [(hp_name, hp_path_in_config, values, centre)])
# Each HP's `values` is the union of v1 (9000-9179) and v2 (9180-9251)
# sweep values when they cover the same HP - the metadata function knows
# which experiment number maps to which value, but the analysis treats
# them as one combined sweep.
HP_BLOCKS = [
    ('DQN', [
        ('learning_rate',
         ('nn_hyperparameters', 'learning_rate'),
         [1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1],  # v1 + v2 ext
         1.0e-3),
        ('discount_factor',
         ('nn_hyperparameters', 'discount_factor'),
         [0.90, 0.95, 0.99, 0.995, 0.999],
         0.99),
        ('buffer_limit',
         ('nn_hyperparameters', 'buffer_limit'),
         [150, 500, 1500, 5000, 15000],
         1500),
        ('target_network_update_frequency',
         ('nn_hyperparameters', 'target_network_update_frequency'),
         [5, 10, 25, 50, 100],
         25),
        ('target_episode_epsilon_frac',
         ('nn_hyperparameters', 'target_episode_epsilon_frac'),
         [0.1, 0.2, 0.3, 0.5, 0.7],
         0.3),
    ]),
    ('REINFORCE', [
        ('learning_rate',
         ('nn_hyperparameters', 'learning_rate'),
         [1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1],  # v1 + v2 ext
         1.0e-3),
        ('discount_factor',
         ('nn_hyperparameters', 'discount_factor'),
         [0.90, 0.95, 0.99, 0.995, 0.999],
         0.99),
        ('layers_arch',
         ('nn_hyperparameters', 'layers'),
         [[32, 32], [64, 64], [128, 64, 64], [256, 128, 64], [512, 256, 128, 64]],
         [128, 64, 64]),
    ]),
    ('CMA', [
        ('initial_sigma',
         ('cma_parameters', 'initial_sigma'),
         [0.01, 0.05, 0.10, 0.30, 1.00],
         0.10),
        ('population_dimension',
         ('cma_parameters', 'population_dimension'),
         [10, 20, 40, 80, 160],
         20),
        ('max_generations',
         ('cma_parameters', 'max_generations'),
         [50, 100, 200, 400, 800],
         200),
    ]),
    ('ODT', [
        ('learning_rate',
         ('odt_hyperparameters', 'learning_rate'),
         [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1],
         1.0e-4),
        ('embed_dim',
         ('odt_hyperparameters', 'embed_dim'),
         [64, 128, 256, 512, 1024],
         512),
        ('n_layer',
         ('odt_hyperparameters', 'n_layer'),
         [1, 2, 4, 6, 8],
         4),
        ('K',
         ('odt_hyperparameters', 'K'),
         [5, 10, 20, 40, 80],
         10),
        ('rtg',
         ('odt_hyperparameters', 'online_rtg'),
         [-30, -50, -75, -100, -150],
         -60),
    ]),
]

START_EXP = 9000

METRICS_ROOT_CANDIDATES = [
    '../../../../storage_1/metrics',
    '/storage_1/metrics',
    '../metrics',
    '../_local_metrics',
]
def resolve_metrics_root():
    for r in METRICS_ROOT_CANDIDATES:
        if os.path.isdir(r):
            return r
    return None
METRICS_ROOT = resolve_metrics_root()
print(f'Metrics root: {METRICS_ROOT}')""")))

# ============================================================== helpers
cells.append(md("### Helpers"))
cells.append(code(*split_lines("""def hp_exp_metadata(exp_num):
    \"\"\"Return (algorithm, hp_name, value, seed) for any 9xxx experiment.

    Handles both the v1 batch (9000-9179) and the v2 extension batch
    (9180-9251). Each entry in EXP_LAYOUT is a sweep starting at
    `start_exp`, containing `len(values) * 3` experiments (value-major,
    seed-minor ordering).
    \"\"\"
    for start_exp, algo, hp_name, values in EXP_LAYOUT:
        n_exps = len(values) * N_SEEDS
        if start_exp <= exp_num < start_exp + n_exps:
            offset = exp_num - start_exp
            v_idx = offset // N_SEEDS
            s_idx = offset % N_SEEDS
            return algo, hp_name, values[v_idx], SEEDS[s_idx]
    raise ValueError(f'Exp_{exp_num} not in any HP-tuning sweep range')


def exp_paths(exp_num):
    if METRICS_ROOT is None:
        return None
    base = Path(METRICS_ROOT) / f'Exp_{exp_num}' / 'train'
    return {
        'base':    base,
        'agent':   base / 'metrics_agent_episode_level.csv',
        'station': base / 'metrics_station_episode_level.csv',
        'config':  Path('..') / 'experiments' / f'Exp_{exp_num}' / 'config.yaml',
    }


def load_exp(exp_num):
    p = exp_paths(exp_num)
    if p is None or not p['agent'].exists():
        return None
    a = pd.read_csv(p['agent'])
    algo, hp_name, value, seed = hp_exp_metadata(exp_num)
    a['exp_num']   = exp_num
    a['algorithm'] = algo
    a['hp_name']   = hp_name
    # value may be a list (REINFORCE layers); store as string for grouping
    a['hp_value']  = str(value) if isinstance(value, list) else value
    a['seed']      = seed
    return {'agent': a, 'meta': (algo, hp_name, value, seed)}""")))

# ============================================================== load
cells.append(md("### Load all 9xxx HP-tuning experiments"))
cells.append(code(*split_lines("""ALL_EXPS = list(range(9000, 9252))   # v1 (9000-9179) + v2 (9180-9251)
loaded = {}
missing = []
for n in ALL_EXPS:
    d = load_exp(n)
    if d is None:
        missing.append(n)
    else:
        loaded[n] = d
print(f'Loaded:  {len(loaded):>3} / {len(ALL_EXPS)} experiments')
print(f'Missing: {len(missing):>3}')
if missing:
    runs = []
    s = missing[0]; prev = s
    for n in missing[1:]:
        if n == prev + 1:
            prev = n
        else:
            runs.append((s, prev)); s = n; prev = n
    runs.append((s, prev))
    print('  ranges:', ', '.join(f'{a}-{b}' if a != b else str(a) for a, b in runs))""")))

# ============================================================== summary table
cells.append(md(*split_lines("""### Per-experiment summary

For each loaded experiment, compute the mean reward over the last
N=200 episodes (the converged-region average) and per-seed std. This
is the metric we use to rank HP values.""")))
cells.append(code(*split_lines("""LAST_N = 200

def summarise(exp_num, d):
    a = d['agent']
    algo, hp_name, value, seed = d['meta']
    pairs = a[['aggregation', 'episode']].drop_duplicates()
    last_pairs = pairs.tail(min(LAST_N, len(pairs)))
    a_tail = a.merge(last_pairs, on=['aggregation', 'episode'])
    return {
        'exp_num':         exp_num,
        'algorithm':       algo,
        'hp_name':         hp_name,
        'hp_value':        str(value) if isinstance(value, list) else value,
        'hp_value_sortkey': _sortkey(value),
        'seed':            seed,
        'reward_mean':     a_tail['reward'].mean(),
        'reward_std_within': a_tail['reward'].std(),
        'n_eps_used':      len(last_pairs),
    }

def _sortkey(v):
    \"\"\"Stable numeric sort key for any HP value (list or scalar).\"\"\"
    if isinstance(v, list):
        return float(sum(v))   # total network capacity proxy
    return float(v)

if loaded:
    rows = [summarise(n, d) for n, d in loaded.items()]
    summary = pd.DataFrame(rows).sort_values(['algorithm', 'hp_name', 'hp_value_sortkey', 'seed']).reset_index(drop=True)
    print(f'Summary table: {len(summary)} experiments')
    display(summary.head(20))
else:
    summary = pd.DataFrame()
    print('No data loaded - downstream cells will be skipped until the 9xxx batch is rsync-ed onto this server.')""")))

# ============================================================== plots
cells.append(md(*split_lines("""### Performance vs HP value (per algorithm, per HP)

Four rows (one per algorithm), three columns (one per swept HP).
Each subplot shows mean reward vs HP value with error bars across seeds.""")))
cells.append(code(*split_lines("""if not summary.empty:
    algos = [b[0] for b in HP_BLOCKS]
    hp_dict = dict(HP_BLOCKS)
    # Number of columns = max HPs across algorithms (some have more than others
    # after the v2 batch: DQN and ODT now have 5, REINFORCE and CMA have 3).
    max_hps = max(len(hp_dict[a]) for a in algos)
    fig, axes = plt.subplots(len(algos), max_hps,
                              figsize=(3.2 * max_hps + 2, 3.6 * len(algos)),
                              squeeze=False)
    for r, algo in enumerate(algos):
        hps_for_algo = hp_dict[algo]
        for c, hp_tuple in enumerate(hps_for_algo):
            hp_name = hp_tuple[0]
            ax = axes[r, c]
            sub = summary[(summary['algorithm'] == algo) & (summary['hp_name'] == hp_name)]
            if sub.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'{algo}: {hp_name}', fontsize=10)
                continue
            agg = (sub.groupby(['hp_value', 'hp_value_sortkey'])['reward_mean']
                      .agg(['mean', 'std', 'count'])
                      .reset_index()
                      .sort_values('hp_value_sortkey'))
            xs = np.arange(len(agg))
            ax.errorbar(xs, agg['mean'], yerr=agg['std'],
                        marker='o', lw=2, capsize=4, color=colors[algo])
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v in agg['hp_value']], rotation=25, fontsize=8)
            best_idx = int(agg['mean'].idxmax()) if not agg['mean'].isna().all() else None
            if best_idx is not None:
                best_pos = list(agg.index).index(best_idx)
                ax.axvline(best_pos, color='red', alpha=0.25, lw=2)
            ax.set_title(f'{algo}: {hp_name}', fontsize=10)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel('mean reward (last 200 eps)', fontsize=9)
        # Hide unused subplots when this algorithm has fewer HPs than the max
        for c in range(len(hps_for_algo), max_hps):
            axes[r, c].axis('off')
    fig.suptitle('Hyperparameter sweeps - reward vs swept value (3 seeds, error bars = std)\\n'
                 'v1 (9000-9179) + v2 (9180-9251) combined; red line = best value per HP',
                 fontsize=12)
    plt.tight_layout()
    out = Path('figures') / 'hp_tuning_sweeps.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show()""")))

# ============================================================== best per HP
cells.append(md(*split_lines("""### Recommendations: best value per HP per algorithm

For each (algorithm, swept HP), pick the value with the highest
seed-averaged final reward. Ties broken by preferring the value
closer to the existing default (Occam) and then lower across-seed std.""")))
cells.append(code(*split_lines("""def recommend(summary, hp_blocks):
    rows = []
    for algo, hps in hp_blocks:
        for hp_name, hp_path, values, centre in hps:
            sub = summary[(summary['algorithm'] == algo) & (summary['hp_name'] == hp_name)]
            if sub.empty:
                rows.append({
                    'algorithm': algo, 'hp_name': hp_name,
                    'best_value': None, 'best_reward': np.nan, 'best_std': np.nan,
                    'centre_value': str(centre) if isinstance(centre, list) else centre,
                    'note': 'no data',
                })
                continue
            agg = (sub.groupby(['hp_value','hp_value_sortkey'])['reward_mean']
                      .agg(mean='mean', std='std', count='count')
                      .reset_index())
            best_mean = agg['mean'].max()
            # Tolerance ties: anything within 1 std of best
            tol = agg.loc[agg['mean'].idxmax(), 'std']
            tol = tol if not np.isnan(tol) else 0
            tie_mask = agg['mean'] >= (best_mean - tol)
            ties = agg[tie_mask].copy()
            # Tie-break: prefer closer to centre, then lower std
            centre_sk = float(sum(centre)) if isinstance(centre, list) else float(centre)
            ties['dist_to_centre'] = (ties['hp_value_sortkey'] - centre_sk).abs()
            ties = ties.sort_values(['dist_to_centre', 'std', 'mean'], ascending=[True, True, False])
            best = ties.iloc[0]
            rows.append({
                'algorithm':   algo,
                'hp_name':     hp_name,
                'best_value':  best['hp_value'],
                'best_reward': float(best['mean']),
                'best_std':    float(best['std']) if not np.isnan(best['std']) else None,
                'centre_value': str(centre) if isinstance(centre, list) else centre,
                'note':        'unique best' if tie_mask.sum() == 1 else f'{tie_mask.sum()} tied (Occam-broken)',
            })
    return pd.DataFrame(rows)

if not summary.empty:
    recs = recommend(summary, HP_BLOCKS)
    print('Per-HP recommendations:')
    display(recs)""")))

# ============================================================== consolidated configs
cells.append(md(*split_lines("""### Consolidated config snippets

Apply the recommendations to the four template configs (Exp_4000,
Exp_4036, Exp_4072, Exp_4108). These snippets can be patched into all
4xxx, 5xxx, and 6xxx configs before re-running the main experiments.""")))
cells.append(code(*split_lines("""def _coerce(v, original):
    \"\"\"Cast back to the original Python type so YAML serialises clean.\"\"\"
    if isinstance(original, list):
        try:
            return json.loads(v.replace(\"'\", '\"'))
        except Exception:
            return original
    if isinstance(original, bool):
        return bool(v)
    if isinstance(original, int):
        try:
            return int(float(v))
        except Exception:
            return original
    try:
        return float(v)
    except Exception:
        return v

if not summary.empty:
    output = {}
    for algo, hps in HP_BLOCKS:
        rec_algo = recs[recs['algorithm'] == algo]
        block_name = {
            'DQN':       'nn_hyperparameters',
            'REINFORCE': 'nn_hyperparameters',
            'CMA':       'cma_parameters',
            'ODT':       'odt_hyperparameters',
        }[algo]
        snippet = {block_name: {}}
        for hp_name, hp_path, values, centre in hps:
            row = rec_algo[rec_algo['hp_name'] == hp_name]
            if row.empty or row.iloc[0]['best_value'] is None:
                continue
            v = row.iloc[0]['best_value']
            snippet[block_name][hp_path[-1]] = _coerce(v, centre)
        output[algo] = snippet
    print(json.dumps(output, indent=2, default=str))
    out = Path('table_data') / 'hp_tuning_recommendations.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, default=str))
    print(f'\\nSaved {out}')
    # Also save the per-HP table as CSV
    recs.to_csv(Path('table_data') / 'hp_tuning_recommendations.csv', index=False)
    print(f'Saved table_data/hp_tuning_recommendations.csv')""")))

# ============================================================== apply script
cells.append(md(*split_lines("""### Apply recommendations to 4xxx / 5xxx / 6xxx configs

After running this notebook end-to-end, the recommendations are stored
in `table_data/hp_tuning_recommendations.json`. The block below patches
those values into every 4xxx, 5xxx, and 6xxx config (and 7xxx, 8xxx
too, since those are derived from the same templates). Run with
`apply = True` to actually write the configs. By default it dry-runs
and shows you what would change.""")))
cells.append(code(*split_lines("""apply = False   # set to True to actually patch the configs

if not summary.empty and Path('table_data/hp_tuning_recommendations.json').exists():
    recs_data = json.loads(Path('table_data/hp_tuning_recommendations.json').read_text())

    # Determine each experiment's algorithm by reading its config
    import glob
    target_dirs = sorted(set(
        glob.glob('../experiments/Exp_4*') +
        glob.glob('../experiments/Exp_5*') +
        glob.glob('../experiments/Exp_6*') +
        glob.glob('../experiments/Exp_7*') +
        glob.glob('../experiments/Exp_8*')
    ))
    changes = 0
    for exp_dir in target_dirs:
        cfg_path = Path(exp_dir) / 'config.yaml'
        if not cfg_path.exists():
            continue
        import yaml
        cfg = yaml.safe_load(open(cfg_path))
        algo = cfg.get('algorithm_settings', {}).get('algorithm')
        if algo not in recs_data:
            continue
        snippet = recs_data[algo]
        local_changes = []
        for block, fields in snippet.items():
            cfg.setdefault(block, {})
            for k, v in fields.items():
                if cfg[block].get(k) != v:
                    local_changes.append(f'{block}.{k}: {cfg[block].get(k)!r} -> {v!r}')
                    cfg[block][k] = v
        if local_changes and apply:
            yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=True)
            changes += 1
        elif local_changes:
            print(f'{Path(exp_dir).name}: would change {local_changes[:3]}')
    if apply:
        print(f'\\nPatched {changes} config files.')
    else:
        print(f'\\n(dry run - set apply=True and re-run this cell to write)')""")))

# ============================================================== assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({len(cells)} cells)")
