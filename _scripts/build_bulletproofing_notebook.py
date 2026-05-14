"""
Build paper_figure_generators/Bulletproofing.ipynb.

This is the companion notebook to Sensitivity Analysis.ipynb. It defends
the sensitivity result against the "your agents aren't really optimising
the reward" critique by running two layers of analysis:

  Part 1 - Free analyses on existing 7xxx data (no new runs):
     1A. Initial vs final reward improvement (proves learning happens)
     1B. Training curves at w_d in {0, 1, 10} side by side
     1C. Random-policy baseline derived from DQN's epsilon=1 phase

  Part 2 - Single-term reward experiments (Exp_8000-8026):
     2A. Load 8xxx
     2B. Per-experiment summary
     2C. Killer figure: per-metric bars by reward shape (diagonal-dominance check)
     2D. Heatmap version of the diagonal-dominance check
     2E. Statistical test that matched-metric is lowest
     2F. Verdict

Run from the repo root:
    python _scripts/build_bulletproofing_notebook.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper_figure_generators" / "Bulletproofing.ipynb"


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
cells.append(md(*split_lines("""# Bulletproofing the Sensitivity Analysis

The 7xxx sweep showed that agent behaviour is essentially invariant to
the choice of reward weights within `{0, 1, 5, 7, 10}`. A skeptical
reviewer might read that as "the agents are not actually optimising the
reward." This notebook addresses that critique in two layers.

**Part 1** uses existing 7xxx data (no new training runs needed):

- **1A.** Initial vs final reward across training - confirms that
  learning is happening at all.
- **1B.** Training curves at distance-weight values {0, 1, 10} for the
  same model and seed, side by side - shows the learning dynamics
  differ even when the converged behaviour does not.
- **1C.** Random-policy baseline derived from DQN's `epsilon=1` phase
  in the first aggregation - quantifies how much better trained agents
  are than purely random ones.

**Part 2** uses the 8xxx single-term-reward experiments (Exp_8000-8026):

- **2A.** Load the 8xxx batch.
- **2B.** Per-experiment summary.
- **2C/2D.** The killer figure: when trained on a single-term reward,
  does the agent's behaviour preferentially reduce that term's metric?
  If yes, the optimisation machinery works and the 7xxx invariance is
  fully explained by the narrow {0..10} weight ratio range. If no, the
  critique stands.
- **2E.** Paired test for diagonal dominance (matched metric lowest).
- **2F.** Verdict.""")))

# ============================================================== imports
cells.append(md("### Imports"))
cells.append(code(*split_lines("""import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from environment.data_loader import load_config_file""")))

# ============================================================== config
cells.append(md("### Config"))
cells.append(code(*split_lines("""colors = {
    "DQN":       'darkorange',
    "REINFORCE": 'forestgreen',
    "CMA":       'turquoise',
    "CMA-ES":    'turquoise',
    "ODT":       'blueviolet',
}

# 7xxx sensitivity layout
SWEEP_VALUES = [0, 1, 5, 7, 10]
SWEPT_VARS = ['distance_weight', 'traffic_weight', 'energy_weight']
SENS_MODEL_START = {
    'DQN':       7000,
    'REINFORCE': 7045,
    'CMA':       7090,
    'ODT':       7135,
}

# 8xxx bulletproofing layout (3 DMs * 3 reward shapes * 3 seeds)
REWARD_SHAPES = ['distance_only', 'traffic_only', 'energy_only']
SHAPE_TO_METRIC = {
    'distance_only': 'distance_mean',
    'traffic_only':  'mean_peak_traf',
    'energy_only':   'energy_proxy',
}
METRIC_LABEL = {
    'distance_mean':  'Mean distance (km, approx)',
    'mean_peak_traf': 'Mean peak traffic (cars)',
    'energy_proxy':   'Energy proxy (1 / avg battery)',
}
BP_MODEL_START = {
    'DQN':       8000,
    'REINFORCE': 8009,
    'CMA':       8018,
}

SEEDS = [1234, 5555, 2020]

METRICS_ROOT_CANDIDATES = [
    '../../../../storage_1/metrics',
    '/storage_1/metrics',
    '../metrics',
    '../_sensitivity_metrics',
]""")))

# ============================================================== helpers
cells.append(md("### Helpers"))
cells.append(code(*split_lines("""def resolve_metrics_root():
    for root in METRICS_ROOT_CANDIDATES:
        if os.path.isdir(root):
            return root
    return None


METRICS_ROOT = resolve_metrics_root()
print(f"Metrics root resolved to: {METRICS_ROOT}")


def sens_exp_metadata(exp_num):
    \"\"\"7xxx metadata: (model, swept_var, weight, seed).\"\"\"
    for model, start in SENS_MODEL_START.items():
        if start <= exp_num < start + 45:
            offset = exp_num - start
            swept_var = SWEPT_VARS[offset // 15]
            within_block = offset % 15
            value = SWEEP_VALUES[within_block // 3]
            seed = SEEDS[within_block % 3]
            return model, swept_var, value, seed
    raise ValueError(f"Exp_{exp_num} not in 7xxx range")


def bp_exp_metadata(exp_num):
    \"\"\"8xxx metadata: (model, reward_shape, seed). 9 exps per model,
    3 reward shapes (3 seeds each).\"\"\"
    for model, start in BP_MODEL_START.items():
        if start <= exp_num < start + 9:
            offset = exp_num - start
            shape = REWARD_SHAPES[offset // 3]
            seed = SEEDS[offset % 3]
            return model, shape, seed
    raise ValueError(f"Exp_{exp_num} not in 8xxx range")


def exp_paths(exp_num, root=None):
    if root is None:
        root = METRICS_ROOT
    if root is None:
        return None
    base = Path(root) / f"Exp_{exp_num}" / "train"
    return {
        'base':    base,
        'agent':   base / 'metrics_agent_episode_level.csv',
        'station': base / 'metrics_station_episode_level.csv',
        'config':  Path('..') / 'experiments' / f'Exp_{exp_num}' / 'config.yaml',
    }


def load_exp_generic(exp_num, metadata_fn):
    paths = exp_paths(exp_num)
    if paths is None or not paths['agent'].exists() or not paths['station'].exists():
        return None
    a = pd.read_csv(paths['agent'])
    s = pd.read_csv(paths['station'])
    meta = metadata_fn(exp_num)
    a['exp_num'] = exp_num
    s['exp_num'] = exp_num
    return {'agent': a, 'station': s, 'meta': meta, 'config_path': paths['config']}""")))

# ============================================================== Part 1 header
cells.append(md(*split_lines("""# Part 1 - Free analyses on existing 7xxx data

These run today without new compute. They strengthen the same claim
as Part 2 at lower confidence, and serve as a first line of defense
even before the 8xxx experiments finish.""")))

# -------------------------------------------------------------- 1A initial vs final
cells.append(md(*split_lines("""## 1A. Did learning happen?

Compare mean reward in the first 200 episodes (when DQN's
epsilon is near 1.0 and the policy is essentially random) to mean
reward in the last 200 episodes. A substantial improvement means the
agent's behaviour did change in the direction the reward gradient
pushed it - the optimisation machinery is functional.

Restricted to **baseline-weight (w=1) experiments** so the reward
function is the same one the paper actually uses.""")))
cells.append(code(*split_lines("""# Baseline-weight experiments per model (the 3-cell triplets at weight=1
# within each sweep block). We pick distance_weight=1 representatives but
# any of the three baseline cells per (model, seed) would give identical
# data because the reward is the same.
BASELINE_EXPS = []
for model, start in SENS_MODEL_START.items():
    if model == 'ODT':
        continue
    for seed_offset in range(3):
        BASELINE_EXPS.append(start + 3 + seed_offset)  # +3 -> distance_weight=1

print(f"Baseline-weight experiments to check: {BASELINE_EXPS}")

INITIAL_EPISODES = 200
FINAL_EPISODES = 200

rows = []
for exp_num in BASELINE_EXPS:
    d = load_exp_generic(exp_num, sens_exp_metadata)
    if d is None:
        continue
    model, _, _, seed = d['meta']
    a = d['agent']
    # Order pairs by appearance and pick first/last N
    pairs = a[['aggregation', 'episode']].drop_duplicates().reset_index(drop=True)
    if len(pairs) < INITIAL_EPISODES + FINAL_EPISODES:
        continue
    first_pairs = pairs.head(INITIAL_EPISODES)
    last_pairs  = pairs.tail(FINAL_EPISODES)
    first = a.merge(first_pairs, on=['aggregation', 'episode'])['reward'].mean()
    last  = a.merge(last_pairs,  on=['aggregation', 'episode'])['reward'].mean()
    rows.append({'model': model, 'seed': seed, 'exp_num': exp_num,
                 'reward_first': first, 'reward_last': last,
                 'improvement': last - first})

learning_df = pd.DataFrame(rows)
if learning_df.empty:
    print('No baseline experiments loaded - 7xxx data missing.')
else:
    display(learning_df.sort_values(['model', 'seed']))
    summary = (learning_df
               .groupby('model')[['reward_first','reward_last','improvement']]
               .agg(['mean','std']))
    print('\\nPer-model summary:')
    display(summary)""")))

cells.append(code(*split_lines("""# Visual: paired before/after per (model, seed)
if not learning_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    grouped = learning_df.groupby('model')
    models = list(grouped.groups.keys())
    x = np.arange(len(models))
    means_first = [grouped.get_group(m)['reward_first'].mean() for m in models]
    means_last  = [grouped.get_group(m)['reward_last' ].mean() for m in models]
    stds_first  = [grouped.get_group(m)['reward_first'].std()  for m in models]
    stds_last   = [grouped.get_group(m)['reward_last' ].std()  for m in models]

    bar1 = ax.bar(x - width/2, means_first, width, yerr=stds_first,
                   capsize=4, color='lightgray', edgecolor='black',
                   label=f'First {INITIAL_EPISODES} eps (~random policy)')
    bar2 = ax.bar(x + width/2, means_last,  width, yerr=stds_last,
                   capsize=4,
                   color=[colors[m] for m in models],
                   edgecolor='black',
                   label=f'Last {FINAL_EPISODES} eps (trained policy)')
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel('Mean reward across cars')
    ax.set_title('1A. Initial vs final reward at baseline weight\\n(learning is happening if right bars are clearly higher)')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    for i, m in enumerate(models):
        delta = means_last[i] - means_first[i]
        ax.annotate(f'{delta:+.1f}', xy=(x[i], max(means_first[i], means_last[i])),
                    xytext=(0, 6), textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')
    plt.tight_layout()
    out = Path('figures') / 'bulletproof_1a_initial_vs_final.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show()""")))

# -------------------------------------------------------------- 1B side-by-side curves
cells.append(md(*split_lines("""## 1B. Training curves differ by weight value

Same model, same seed, three different distance-weight values
({0, 1, 10}). If the gradient signal is actually being processed, the
three curves should differ in shape even when the converged distance
metric is similar. We use DQN seed 1234 by convention; any (model,
seed) combination works.""")))
cells.append(code(*split_lines("""TARGET_MODEL = 'DQN'
TARGET_SEED  = 1234

# Locate (distance_weight=0, =1, =10) at TARGET_SEED for TARGET_MODEL
def find_exp(model, swept_var, weight, seed):
    start = SENS_MODEL_START[model]
    sweep_idx = SWEPT_VARS.index(swept_var)
    block_start = start + sweep_idx * 15
    value_idx = SWEEP_VALUES.index(weight)
    seed_idx = SEEDS.index(seed)
    return block_start + value_idx * 3 + seed_idx

exp_curves = {
    0:  find_exp(TARGET_MODEL, 'distance_weight', 0,  TARGET_SEED),
    1:  find_exp(TARGET_MODEL, 'distance_weight', 1,  TARGET_SEED),
    10: find_exp(TARGET_MODEL, 'distance_weight', 10, TARGET_SEED),
}
print(f'Comparing {TARGET_MODEL} seed={TARGET_SEED} at distance_weight in {list(exp_curves)}')
print(f'  Experiment numbers: {dict(exp_curves)}')

fig, ax = plt.subplots(figsize=(11, 4.5))
weight_colors = {0: '#d62728', 1: '#1f77b4', 10: '#2ca02c'}

any_loaded = False
for w, exp_num in exp_curves.items():
    d = load_exp_generic(exp_num, sens_exp_metadata)
    if d is None:
        print(f'  Exp_{exp_num} (w={w}): NOT LOADED')
        continue
    any_loaded = True
    a = d['agent']
    curve = (a.groupby(['aggregation', 'episode'], as_index=False)['reward']
              .mean()
              .sort_values(['aggregation', 'episode'])
              .reset_index(drop=True))
    curve['gep'] = curve.index
    # Smooth for visibility
    win = max(5, len(curve) // 50)
    ax.plot(curve['gep'], curve['reward'].rolling(win, min_periods=1).mean(),
            lw=2, color=weight_colors[w], label=f'distance_weight = {w}')

if any_loaded:
    ax.set_xlabel('Global episode')
    ax.set_ylabel('Mean reward across cars (smoothed)')
    ax.set_title(f'1B. Training curves for {TARGET_MODEL} seed={TARGET_SEED} at three distance-weight values')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path('figures') / 'bulletproof_1b_curves_by_weight.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show()
else:
    print('No data loaded - skipping figure 1B')""")))

# -------------------------------------------------------------- 1C random baseline
cells.append(md(*split_lines("""## 1C. Random-policy baseline from DQN's epsilon=1 phase

DQN's first ~100 episodes use `epsilon = 1.0` (decaying at 0.999), so
the policy is effectively random uniform over actions. The mean reward
in that window is a built-in random-policy baseline that doesn't need
its own experiments.

We compare to the final-aggregation mean reward of the same experiments.""")))
cells.append(code(*split_lines("""RANDOM_WINDOW = 100  # ~ first 100 episodes are epsilon ~ 1

dqn_baselines = [SENS_MODEL_START['DQN'] + 3 + s for s in range(3)]
rows = []
for exp_num in dqn_baselines:
    d = load_exp_generic(exp_num, sens_exp_metadata)
    if d is None:
        continue
    model, _, _, seed = d['meta']
    a = d['agent']
    pairs = a[['aggregation', 'episode']].drop_duplicates().reset_index(drop=True)
    if len(pairs) < RANDOM_WINDOW + 200:
        continue
    random_eps = pairs.head(RANDOM_WINDOW)
    final_eps  = pairs.tail(200)
    r_random = a.merge(random_eps, on=['aggregation', 'episode'])['reward'].mean()
    r_final  = a.merge(final_eps,  on=['aggregation', 'episode'])['reward'].mean()
    rows.append({'seed': seed, 'reward_random': r_random,
                 'reward_trained': r_final,
                 'gain': r_final - r_random,
                 'gain_pct': (r_final - r_random) / abs(r_random) * 100})

random_df = pd.DataFrame(rows)
if random_df.empty:
    print('No DQN baseline data loaded.')
else:
    display(random_df)
    print(f"\\nMean random-policy reward:   {random_df['reward_random'].mean():.2f}")
    print(f"Mean trained-policy reward:  {random_df['reward_trained'].mean():.2f}")
    print(f"Mean improvement:            {random_df['gain'].mean():+.2f} ({random_df['gain_pct'].mean():+.1f}%)")
    print()
    if random_df['gain'].mean() > 5:
        print('VERDICT: trained agents significantly outperform random policies -> learning is real.')
    elif random_df['gain'].mean() > 0:
        print('VERDICT: trained agents marginally outperform random policies. Weak but positive.')
    else:
        print('VERDICT: trained agents do NOT outperform random policies. Re-examine training.')""")))

# ============================================================== Part 2 header
cells.append(md(*split_lines("""# Part 2 - Single-term reward experiments (Exp_8000-8026)

Each 8xxx experiment trains an agent with a reward containing exactly
ONE term: distance-only, traffic-only, or energy-only. If the agent is
truly optimising the reward, the metric matching the active term
should be lowest in its own cell (the diagonal-dominance test).""")))

# -------------------------------------------------------------- 2A bulk load
cells.append(md("## 2A. Load 8xxx experiments"))
cells.append(code(*split_lines("""BP_EXPS = list(range(8000, 8027))
bp_loaded = {}
bp_missing = []
for n in BP_EXPS:
    d = load_exp_generic(n, bp_exp_metadata)
    if d is None:
        bp_missing.append(n)
    else:
        bp_loaded[n] = d

print(f'Loaded:  {len(bp_loaded):>2} / {len(BP_EXPS)} bulletproofing experiments')
print(f'Missing: {len(bp_missing):>2}')
if bp_missing:
    print(f'  experiments: {bp_missing}')""")))

# -------------------------------------------------------------- 2B summary
cells.append(md("## 2B. Per-experiment summary"))
cells.append(code(*split_lines("""LAST_N = 50  # last N episodes treated as 'converged'

def summarise_bp(exp_num, d):
    a = d['agent']; s = d['station']
    model, shape, seed = d['meta']
    pairs = a[['aggregation', 'episode']].drop_duplicates()
    last_pairs = pairs.tail(min(LAST_N, len(pairs)))
    a_tail = a.merge(last_pairs, on=['aggregation', 'episode'])
    s_tail = s.merge(last_pairs, on=['aggregation', 'episode'])
    avg_batt = a_tail['average_battery'].mean()
    return {
        'exp_num':         exp_num,
        'model':           model,
        'reward_shape':    shape,
        'seed':            seed,
        'reward_mean':     a_tail['reward'].mean(),
        'distance_mean':   a_tail['distance'].mean(),
        'mean_peak_traf':  s_tail['traffic'].mean() if len(s_tail) else np.nan,
        'avg_battery':     avg_batt,
        'energy_proxy':    1.0 / avg_batt if avg_batt else np.nan,
    }

if bp_loaded:
    bp_summary = pd.DataFrame([summarise_bp(n, d) for n, d in bp_loaded.items()])
    bp_summary = bp_summary.sort_values(['model','reward_shape','seed']).reset_index(drop=True)
    print(f'Summary table: {len(bp_summary)} experiments')
    display(bp_summary)
else:
    bp_summary = pd.DataFrame()
    print('No 8xxx data loaded - 2C/2D/2E will be skipped.')""")))

# -------------------------------------------------------------- 2C killer figure
cells.append(md(*split_lines("""## 2C. Killer figure - does behaviour match the active reward term?

Three subplots, one per physical metric. Within each subplot, three
groups (one per reward shape trained on). Within each group, one bar
per DM. If the optimisation machinery works, the group whose reward
shape matches the metric (the **highlighted** group on each subplot)
should produce the **lowest** bars.""")))
cells.append(code(*split_lines("""if not bp_summary.empty:
    metrics = ['distance_mean', 'mean_peak_traf', 'energy_proxy']
    matching_shape = {
        'distance_mean':  'distance_only',
        'mean_peak_traf': 'traffic_only',
        'energy_proxy':   'energy_only',
    }
    models = list(BP_MODEL_START.keys())
    shapes = REWARD_SHAPES
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(shapes))

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            means = []; stds = []
            for shape in shapes:
                sub = bp_summary[(bp_summary['model']==model) & (bp_summary['reward_shape']==shape)]
                means.append(sub[metric].mean() if len(sub) else np.nan)
                stds.append(sub[metric].std()  if len(sub) else np.nan)
            offset = (i - 1) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   color=colors[model], edgecolor='black', label=model)
        # Highlight matching column with a red dashed box
        mi = shapes.index(matching_shape[metric])
        ymin, ymax = ax.get_ylim()
        # Use post-data axis bounds for the rectangle - we'll set after data plotted
        ax.set_xticks(x); ax.set_xticklabels(shapes, rotation=15)
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_title(f'Metric: {METRIC_LABEL[metric]}')
        ax.grid(axis='y', alpha=0.3)
        # Now draw the highlight box
        ymin, ymax = ax.get_ylim()
        rect = plt.Rectangle((mi - 0.5, ymin), 1.0, ymax - ymin,
                             linewidth=2, linestyle='--', edgecolor='crimson',
                             facecolor='none', zorder=0)
        ax.add_patch(rect)
        ax.annotate('matching\\nreward', xy=(mi, ymax), xytext=(0, -18),
                    textcoords='offset points', ha='center', color='crimson',
                    fontsize=9, fontweight='bold')
    axes[-1].legend(loc='best', fontsize=9)
    fig.suptitle('2C. Behavioural response under single-term rewards\\n'
                 '(if optimising works, dashed columns should have the lowest bars)',
                 fontsize=12)
    plt.tight_layout()
    out = Path('figures') / 'bulletproof_2c_killer.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show()""")))

# -------------------------------------------------------------- 2D heatmap
cells.append(md(*split_lines("""## 2D. Diagonal-dominance heatmap

Same data as 2C in a compact 3-by-3 view: rows are the reward shape
trained on, columns are the metric measured. Each cell is normalised
to its column (so the smallest value in each column is 0.0 and the
largest is 1.0). The diagonal cells should all be near zero - the
matching reward shape produces the lowest metric value.""")))
cells.append(code(*split_lines("""if not bp_summary.empty:
    metrics = ['distance_mean', 'mean_peak_traf', 'energy_proxy']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, model in zip(axes, BP_MODEL_START.keys()):
        sub = bp_summary[bp_summary['model'] == model]
        if sub.empty:
            ax.set_title(f'{model}: no data')
            ax.axis('off')
            continue
        agg = sub.groupby('reward_shape')[metrics].mean()
        agg = agg.reindex(REWARD_SHAPES)
        # Column-wise normalise: 0 = lowest, 1 = highest within each metric
        normed = (agg - agg.min()) / (agg.max() - agg.min() + 1e-12)
        im = ax.imshow(normed.values, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
        for i in range(normed.shape[0]):
            for j in range(normed.shape[1]):
                ax.text(j, i, f'{normed.values[i,j]:.2f}', ha='center', va='center',
                        color='black', fontsize=10)
        # Draw the expected diagonal in dashed red
        for i in range(3):
            ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor='crimson', linewidth=2, linestyle='--'))
        ax.set_xticks(range(3)); ax.set_xticklabels([m.replace('_',chr(10)) for m in metrics],
                                                    fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(REWARD_SHAPES, fontsize=9)
        ax.set_title(f'{model}')
        ax.set_xlabel('Metric')
        if model == 'DQN':
            ax.set_ylabel('Reward shape trained on')
    fig.suptitle('2D. Diagonal-dominance heatmap (0=lowest metric value, 1=highest)\\n'
                 'Diagonal cells (red dashed) should be near 0 if optimisation works.',
                 fontsize=12)
    plt.tight_layout()
    out = Path('figures') / 'bulletproof_2d_heatmap.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show()""")))

# -------------------------------------------------------------- 2E paired test
cells.append(md(*split_lines("""## 2E. Paired test for diagonal dominance

For each (model, metric) pair, compare the matching-shape mean to the
two off-diagonal means. If the matching shape really is lowest, we
expect a negative `matching - non_matching` difference. Three seeds
per cell so this is descriptive rather than inferential, but the
direction and magnitude tell us whether the optimisation effect is
real.""")))
cells.append(code(*split_lines("""if not bp_summary.empty:
    rows = []
    metrics = ['distance_mean', 'mean_peak_traf', 'energy_proxy']
    matching_shape = {
        'distance_mean':  'distance_only',
        'mean_peak_traf': 'traffic_only',
        'energy_proxy':   'energy_only',
    }
    for model in BP_MODEL_START.keys():
        sub = bp_summary[bp_summary['model'] == model]
        if sub.empty:
            continue
        for metric in metrics:
            ms = matching_shape[metric]
            matched = sub[sub['reward_shape'] == ms][metric].values
            other   = sub[sub['reward_shape'] != ms][metric].values
            if len(matched) == 0 or len(other) == 0:
                continue
            rows.append({
                'model': model,
                'metric': metric,
                'matching_shape': ms,
                'matched_mean':  matched.mean(),
                'other_mean':    other.mean(),
                'difference':    matched.mean() - other.mean(),
                'diff_pct':      (matched.mean() - other.mean()) / abs(other.mean()) * 100,
                'matching_lowest': matched.mean() < other.mean(),
            })
    diag_df = pd.DataFrame(rows)
    display(diag_df)
    if not diag_df.empty:
        pass_count = diag_df['matching_lowest'].sum()
        total = len(diag_df)
        print(f'\\nDiagonal dominance: {pass_count} / {total} (model, metric) cells have matched-shape lowest.')
        print(f'Mean magnitude of effect: {diag_df[\"diff_pct\"].mean():.1f}%')""")))

# -------------------------------------------------------------- 2F verdict
cells.append(md("## 2F. Verdict"))
cells.append(code(*split_lines("""if bp_summary.empty:
    print('Awaiting 8xxx data. Re-run this notebook once Exp_8000-8026 are on the server.')
elif 'diag_df' in dir() and not diag_df.empty:
    pass_rate = diag_df['matching_lowest'].mean()
    effect = diag_df['diff_pct'].mean()
    print('==============================================================')
    if pass_rate >= 0.78 and effect < -5:
        print('VERDICT: STRONG - optimisation machinery is responsive.')
        print('  -> The 7xxx invariance is best explained as the {0..10}')
        print('     weight ratio range being too narrow to flip the')
        print('     dominant gradient signal, not as agents failing to')
        print('     optimise. Paper section is bulletproof.')
    elif pass_rate >= 0.55:
        print('VERDICT: MIXED - matching shape is lowest in a majority of')
        print('  cells but the effect size is small. Report honestly:')
        print('  some optimisation signal, action space constrains the rest.')
    else:
        print('VERDICT: WEAK - matched shape is not consistently lowest.')
        print('  The "agents not optimising" critique partially stands.')
        print('  Consider:')
        print('    - expanding action space (more candidate stations)')
        print('    - longer training')
        print('    - extreme-weight sweep (w=100, 1000)')
    print(f'  pass_rate={pass_rate:.0%}, mean_effect={effect:.1f}%')
    print('==============================================================')""")))

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
