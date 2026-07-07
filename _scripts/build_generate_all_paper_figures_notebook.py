"""
Build paper_figure_generators/Generate All Paper Figures.ipynb.

This is the single "run once, regenerate everything" notebook for the IEEE-TSC
revision. It consolidates ONLY the figure/table generators that the main paper
(VERDE/ieee_TSC_revisions/main.tex) actually uses, and writes every output to
paper_figure_generators/all_v2/ with the exact filenames the paper expects.

Workflow the notebook is built for:
    1. Run this notebook once on Huron (Run All).
    2. Upload all_v2/*.png  -> VERDE/ieee_TSC_revisions/figures/
       Upload all_v2/*.csv  -> VERDE/ieee_TSC_revisions/data/
    3. Refresh Overleaf. Figures + CSV-driven tables update automatically.

Paper outputs produced (5 figures + 4 table CSVs; Fig. 7 agg-level matrix is
intentionally omitted -- no generator exists for it):
    figures:
      fig_episode_plateau_and_seasons.png   (Exp 1, revised reward curves)
      fig_training_durations_Ethan4.png      (Exp 1, OLD power/time data)
      fig_ridgeline_withcma.png              (Exp 3, OLD power/time data)
      fig_sustainability_indicators.png      (Exp 3, reward NEW + energy/size OLD)
      fig_sensitivity_trends.png             (reward-weight sensitivity, from summary CSV)
    tables (pipe-separated CSVs, consumed by csvsimple \\csvreader in main.tex):
      table_hp_tuning.csv          -> tab:hp_tuning
      table_combined_metrics.csv   -> tab:combined_metrics
      table_resource_analysis.csv  -> tab:resource_analysis
      table_regional_emissions.csv -> tab:regional_emissions

Data provenance mirrors paper_figure_generators_v2 (see its README):
  * reward (curves, per-DM final), in-sim distance/peak-traffic = REVISED 4xxx
    (metrics_postfix, via v2_data + the build_v2_cache.py cache).
  * training duration, power, CO2 emissions, model size, Table-2 energy [kWh]
    = OLD /storage_1/metrics (unchanged).

Run from the repo root:
    python _scripts/build_generate_all_paper_figures_notebook.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper_figure_generators" / "Generate All Paper Figures.ipynb"


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
    """Triple-quoted block -> nbformat source lines (all but last end in '\\n')."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = []

# ====================================================================== header
cells.append(md(*split_lines(r"""# Generate All Paper Figures

**One notebook to regenerate every figure and table used in the main paper**
(`VERDE/ieee_TSC_revisions/main.tex`), for the IEEE-TSC revision.

Run this once on Huron (**Run All**), then:

1. Upload `all_v2/*.png` &rarr; `VERDE/ieee_TSC_revisions/figures/`
2. Upload `all_v2/*.csv` &rarr; `VERDE/ieee_TSC_revisions/data/`
3. Refresh Overleaf. Figures and the CSV-driven tables update automatically.

## What this produces (in `paper_figure_generators/all_v2/`)

| Output | Paper element | Data source |
|---|---|---|
| `fig_episode_plateau_and_seasons.png` | Fig. `episode_plateau` (Exp 1) | REVISED 4xxx reward curves |
| `fig_training_durations_Ethan4.png` | Fig. `training_durations` (Exp 1) | OLD power/time |
| `fig_ridgeline_withcma.png` | Fig. `power_plot` (Exp 3) | OLD power/time |
| `fig_sustainability_indicators.png` | Fig. `sustainability_indicator` (Exp 3) | reward NEW; energy/size OLD |
| `fig_sensitivity_trends.png` | Fig. `sensitivity` | `sensitivity_summary.csv` (7xxx sweep) |
| `table_hp_tuning.csv` | Table `tab:hp_tuning` | static tuning design + selections |
| `table_combined_metrics.csv` | Table `tab:combined_metrics` | distance/peak NEW; energy OLD |
| `table_resource_analysis.csv` | Table `tab:resource_analysis` | duration/emissions OLD; reward NEW |
| `table_regional_emissions.csv` | Table `tab:regional_emissions` | derived from resource-analysis emissions |

**Not produced:** Fig. `adaptability_comparison` (`fig_agg_level_train_eval_matrix.png`).
No generator for it exists in the repo (it needs eval-on-a-new-environment data).
Its file already lives in `ieee_TSC_revisions/figures/`; leave it as is.

## Prerequisites on Huron

* Revised post-fix metrics at `/storage_1/metrics_postfix` (for the NEW reward /
  behaviour data). The notebook builds the `paper_figure_generators_v2` cache
  automatically if it is missing.
* The OLD metrics tree at `/storage_1/metrics` (for power / time / CO2).
* `sensitivity_summary.csv` in `table_data/` (committed) for the sensitivity figure.

The CSVs are **pipe-separated** and hold pre-formatted LaTeX cell strings so the
paper's `\csvreader` calls render them verbatim (bolding, `$\pm$`, math)."""))
)

# ====================================================================== 0. setup
cells.append(md("## 0. Setup, paths, and the revised-data cache"))
cells.append(code(*split_lines(r"""import sys
import os
import json
import copy
import itertools
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Repo root (for `environment.*`) and the v2 package (for `v2_data`).
REPO_ROOT_ON_PATH = os.path.abspath('..')
V2_DIR = os.path.abspath(os.path.join('..', 'paper_figure_generators_v2'))
for p in (REPO_ROOT_ON_PATH, V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from environment.evaluation import *          # noqa: F401,F403
from environment.data_loader import *         # noqa: F401,F403

import v2_data                                 # revised-data layer (metrics_postfix)

# Output directory: everything the paper needs, in one place.
OUT = 'all_v2'
os.makedirs(OUT, exist_ok=True)

def outp(name):
    return os.path.join(OUT, name)

# The revised-data cache lives in the v2 package. Build it once if absent
# (reads /storage_1/metrics_postfix; see paper_figure_generators_v2/README.md).
V2_CACHE = os.path.join(V2_DIR, 'table_data', '_v2_cache')
if not os.path.exists(os.path.join(V2_CACHE, 'reward_curves.csv')):
    print('Revised-data cache not found -> building from metrics_postfix ...')
    v2_data.build_cache(out_dir=V2_CACHE)
else:
    print('Revised-data cache found:', V2_CACHE)

print('Outputs will be written to:', os.path.abspath(OUT))"""))
)

# ====================================================================== 1. config
cells.append(md("## 1. Shared configuration"))
cells.append(code(*split_lines(r"""# Per-DM colours, matching the rest of the paper.
colors = {
    "DQN": 'darkorange',
    "REINFORCE": 'forestgreen',
    "ODT": 'blueviolet',
    "CMA": 'turquoise',
    "CMA-ES": 'turquoise',
}
season_colors = {
    "winter": 'blue',
    "spring": 'forestgreen',
    "summer": 'gold',
    "autumn": 'darkorange',
}

# Raw algorithm name (in configs / caches) -> display name used in the paper.
ALGO_DISPLAY = {"CMA": "CMA-ES", "DQN": "DQN", "ODT": "ODT", "REINFORCE": "REINFORCE"}

# All 4xxx/5xxx/6xxx main experiments (OLD power/time data spans all three).
experiments = list(itertools.chain(
    range(4000, 4180),
    range(5000, 5180),
    range(6000, 6180),
))

def write_pipe_csv(path, header, rows):
    # Pipe-separated CSV (no quoting) for the paper's \csvreader. Cells may hold
    # LaTeX (commas, $, \textbf{}), so '|' is the separator -- no cell has a pipe.
    # No trailing newline so csvsimple does not read a phantom empty final row.
    lines = ['|'.join(str(c) for c in header)]
    lines += ['|'.join(str(c) for c in r) for r in rows]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    print('wrote', path, f'({len(rows)} rows)')

def pm(mean, std, nd=2):
    # '36.65 $\pm$ 1.24' style cell.
    return f"{mean:.{nd}f} " + r"$\pm$" + f" {std:.{nd}f}"

def bf(s):
    # Wrap a pre-formatted cell string in \textbf{...}.
    return r"\textbf{" + s + "}"
""")))

# ====================================================================== 2. OLD data
cells.append(md(*split_lines(r"""## 2. Load the OLD power / time metrics (`df_power`)

Reads `/storage_1/metrics/Exp_<n>/train/power_and_co2_metrics.csv` for every
4xxx/5xxx/6xxx experiment. This feeds the training-duration figure, the ridgeline
figure, and the duration + emissions columns of the resource-analysis table. This
is the OLD, unchanged data (training cost was not affected by the bug fixes)."""))
)
cells.append(code(*split_lines(r"""exp_power_data = []

def _load_csv_safe(filepath, columns_specific):
    try:
        return read_csv_data(filepath, columns=columns_specific)
    except json.JSONDecodeError as e:
        print(f"Error decoding {filepath}: {e}")
        return None

for exp_num in experiments:
    config_fname = f'../experiments/Exp_{exp_num}/config.yaml'
    if not os.path.exists(config_fname):
        continue
    c = load_config_file(config_fname)
    nn_c = c['nn_hyperparameters']
    federated_c = c['federated_learning_settings']
    algo_c = c['algorithm_settings']
    env_c = c['environment_settings']

    d_base = f"../../../../storage_1/metrics/Exp_{exp_num}"
    if not os.path.exists(d_base):
        d_base = f"../metrics/Exp_{exp_num}"
    base_path = f"{d_base}/train/"

    power_data = _load_csv_safe(f'{base_path}power_and_co2_metrics.csv',
                                ['time', 'power', 'co2'])
    if power_data is None:
        continue
    power_data['seed'] = env_c['seed']
    power_data['exp_num'] = exp_num
    power_data['algorithm'] = algo_c['algorithm']
    power_data['season'] = env_c['season']
    power_data['num_aggs'] = federated_c['aggregation_count']
    power_data['eps_per_agg'] = nn_c['num_episodes']
    exp_power_data.append(power_data)

df_power = pd.concat(exp_power_data, ignore_index=True)

# Elapsed time (s), cumulative power, CO2 -> grams.
df_power['cumulative_power'] = df_power.groupby('exp_num')['power'].cumsum()
df_power['time'] = pd.to_datetime(df_power['time'], errors='coerce')
df_power['time'] = df_power.groupby('exp_num')['time'].transform(
    lambda x: (x - x.min()).dt.total_seconds())
df_power['co2'] = df_power['co2'] * 1000  # to grams

# CMA logs 3 aggregation rounds; treat as the no-aggregation (1) setting.
df_power['num_aggs'] = np.where(
    (df_power['num_aggs'] == 3) & (df_power['algorithm'] == 'CMA'), 1, df_power['num_aggs'])

print('df_power:', df_power.shape[0], 'rows;',
      'algorithms =', sorted(df_power['algorithm'].unique()),
      '; num_aggs =', sorted(df_power['num_aggs'].unique()))"""))
)

# ====================================================================== 3. REVISED data
cells.append(md(*split_lines(r"""## 3. Load the REVISED reward / behaviour data (metrics_postfix)

From the `paper_figure_generators_v2` cache built by `build_v2_cache.py`:

* `cumulative_avg_reward_by_algorithm` &rarr; Fig. `episode_plateau`.
* `cumulative_agent_df` / `cumulative_station_df` &rarr; Table `combined_metrics`
  (distance, peak traffic).
* `per_algo` / `new_rewards` / `rew_stats` &rarr; reward column of Table
  `resource_analysis` and the reward term of the sustainability figure."""))
)
cells.append(code(*split_lines(r"""import importlib
importlib.reload(v2_data)

# Reward curves for Fig. episode_plateau.
cumulative_avg_reward_by_algorithm = v2_data.load_reward_curves(cache_dir=V2_CACHE)
cumulative_avg_reward_by_algorithm['num_aggs'] = np.where(
    (cumulative_avg_reward_by_algorithm['num_aggs'] == 3) &
    (cumulative_avg_reward_by_algorithm['algorithm'].isin(['CMA', 'DENSER'])),
    1, cumulative_avg_reward_by_algorithm['num_aggs'])

# In-simulation behaviour for Table combined_metrics.
cumulative_agent_df, cumulative_station_df = v2_data.load_env_metrics(cache_dir=V2_CACHE)

# Per-DM final reward (mean +/- std) for Table resource_analysis + Fig 6.
per_algo, per_algo_byseed = v2_data.reward_summary(cache_dir=V2_CACHE)
new_rewards = {ALGO_DISPLAY.get(a, a): m
               for a, m in zip(per_algo['algorithm'], per_algo['mean'])}
rew_stats = {ALGO_DISPLAY.get(r['algorithm'], r['algorithm']): (r['mean'], r['std'])
             for _, r in per_algo.iterrows()}

print('reward curves rows:', len(cumulative_avg_reward_by_algorithm))
print('agent rows:', len(cumulative_agent_df), '| station rows:', len(cumulative_station_df))
print('per-DM final reward (revised):')
for _, r in per_algo.sort_values('algorithm').iterrows():
    print(f"  {ALGO_DISPLAY.get(r['algorithm'], r['algorithm']):<10} "
          f"{r['mean']:8.2f} +/- {r['std']:.2f}  (n={int(r['count'])} exps)")"""))
)

# ====================================================================== FIG episode_plateau
cells.append(md("## 4. Figure &mdash; episodes-per-aggregation & seasonality (`fig_episode_plateau_and_seasons.png`)"))
cells.append(code(*split_lines(r"""avg_reward_by_algorithm = copy.deepcopy(cumulative_avg_reward_by_algorithm)

algorithm_handles, algorithm_labels, algorithms_added = [], [], set()

agg_levels_sorted = sorted(avg_reward_by_algorithm['num_aggs'].unique(), reverse=False)
num_plots = len(agg_levels_sorted) + 1

fig, axes = plt.subplots(1, num_plots, figsize=(18, 4), sharey=True, sharex=False)
plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 16, 'legend.fontsize': 16,
    'figure.titlesize': 18,
})
axes = np.atleast_1d(axes)

subplot_labels = ['(a)', '(b)', '(c)', '(d)']
def subplots_visual(ax, plot_ind):
    ax.text(-0.11, 1.02, subplot_labels[plot_ind], transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')

# Shift ODT episodes so its (collection + online) phase lines up after DQN's.
ep_shift = 6000
mask = avg_reward_by_algorithm['algorithm'] == 'ODT'
avg_reward_by_algorithm.loc[mask, 'episode'] += ep_shift

for plot_ind, agg_level in enumerate(agg_levels_sorted):
    ax = axes[plot_ind]
    subplots_visual(ax, plot_ind)
    agg = avg_reward_by_algorithm[avg_reward_by_algorithm['num_aggs'] == agg_level].copy()

    dqn = agg[agg['algorithm'] == 'DQN']
    if not dqn.empty:
        dqn_min = dqn.groupby('episode')['cumulative_reward'].min()
        dqn_max = dqn.groupby('episode')['cumulative_reward'].max()
        dqn_mean = dqn.groupby('episode')['cumulative_reward'].mean()
        ax.fill_between(dqn_min.index, dqn_min.values, dqn_max.values,
                        color=colors['DQN'], alpha=0.3)
        solid_line, = ax.plot(dqn_mean.index, dqn_mean.values, color=colors['DQN'])
        bridge = dqn_mean[dqn_mean.index <= ep_shift]
        dashed_line, = ax.plot(bridge.index, bridge.values, linestyle='--', color='purple')
        if 'dqn2odt' not in algorithms_added:
            algorithm_handles.extend([solid_line, dashed_line])
            algorithm_labels.extend(['DQN', 'ODT Collection Phase'])
            algorithms_added.update(['DQN', 'dqn2odt'])

    odt = agg[agg['algorithm'] == 'ODT']
    if not odt.empty:
        odt_mean = odt.groupby('episode')['cumulative_reward'].mean()
        odt_start_ep = odt_mean.index.min()
        y0 = dqn_mean.loc[ep_shift]
        y1 = odt_mean.loc[odt_start_ep]
        ax.plot([ep_shift, odt_start_ep], [y0, y1], linestyle='--', color='purple')

    total_eps = agg['episode'].max()
    eps_per_agg = np.ceil(total_eps / agg_level)

    for algo in agg['algorithm'].unique():
        algo_data = agg[agg['algorithm'] == algo]
        min_r = algo_data.groupby('episode')['cumulative_reward'].min()
        max_r = algo_data.groupby('episode')['cumulative_reward'].max()
        mean_r = algo_data.groupby('episode')['cumulative_reward'].mean()
        ax.fill_between(min_r.index, min_r.values, max_r.values, color=colors[algo], alpha=0.3)
        line, = ax.plot(mean_r.index, mean_r.values, color=colors[algo])
        name = 'CMA-ES' if algo == 'CMA' else algo
        if algo not in algorithms_added:
            algorithm_handles.append(line)
            algorithm_labels.append(f'{name}')
            algorithms_added.add(algo)

    for agg_i in range(1, agg_level + 1):
        ax.axvline(x=(agg_i * eps_per_agg), color='r', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Episodes')
    ax.grid(True)
    ax.set_xlim(0, 10000)

axes[0].set_ylabel('Cumulative Average Reward')
axes[0].set_ylim(-90, -57)

# Seasonality subplot over the last stretch of episodes.
range_bot = 9800
last_k = avg_reward_by_algorithm.loc[avg_reward_by_algorithm.episode > range_bot]
season_order = ['winter', 'spring', 'autumn', 'summer']
algorithm_order = ['CMA', 'DQN', 'ODT', 'REINFORCE']
ax = axes[3]
sns.boxplot(data=last_k, x="season", y="cumulative_reward", hue="algorithm",
            order=season_order, ax=ax, hue_order=algorithm_order, palette=colors)
ax.legend_.remove()
ax.set_xlabel('Seasons')
ax.yaxis.grid(True)
subplots_visual(ax, 3)

handle_map = dict(zip(algorithm_labels, algorithm_handles))
desired = ['DQN', 'REINFORCE', 'CMA-ES', 'ODT Collection Phase', 'ODT']
handles = [handle_map[label] for label in desired if label in handle_map]
labels = [label for label in desired if label in handle_map]
fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.68, 0.2))

fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(outp('fig_episode_plateau_and_seasons.png'), dpi=300, bbox_inches='tight')
plt.show()
print('wrote', outp('fig_episode_plateau_and_seasons.png'))"""))
)

# ====================================================================== FIG training_durations
cells.append(md(*split_lines(r"""## 5. Figure &mdash; training durations by aggregation frequency (`fig_training_durations_Ethan4.png`)

OLD power/time data. CMA-ES is excluded from this figure (matching the original
Experiment&nbsp;1 notebook), so a CMA-free copy of `df_power` is used here without
disturbing the shared frame."""))
)
cells.append(code(*split_lines(r"""df_td = df_power[df_power['algorithm'] != 'CMA'].copy()

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
})

final_times = (df_td.groupby(['exp_num', 'num_aggs', 'algorithm'])['time']
               .max().reset_index())
final_times['num_aggs_str'] = final_times['num_aggs'].astype(str)
final_times['time'] = final_times['time'] / 3600
final_times['algorithm'] = final_times['algorithm'].replace('CMA', 'CMA-ES')

# ODT's clock starts after its DQN-style collection phase.
mean_dqn = final_times.loc[final_times['algorithm'] == 'DQN', 'time'].mean()
final_times.loc[final_times['algorithm'] == 'ODT', 'time'] += mean_dqn

label_map = {'1': 'No Aggregation', '10': '1000', '50': '200'}
final_times['num_aggs_label'] = final_times['num_aggs_str'].map(label_map)
agg_labels_ordered = [label_map[k] for k in sorted(label_map.keys(), key=int)]
algorithms = sorted(final_times['algorithm'].unique())

g = sns.catplot(
    data=final_times, kind="bar", x="num_aggs_label", y="time", hue="algorithm",
    order=agg_labels_ordered, hue_order=algorithms, errorbar="sd", palette=colors,
    alpha=1, height=2.75, aspect=1.8, capsize=0.1, linewidth=2)
g.despine(left=True)
g.set_axis_labels("Number of Episodes per Aggregation", "Average Final Time (hours)")
if g._legend:
    g._legend.remove()

ax = g.axes.flatten()[0]
handles, labels = ax.get_legend_handles_labels()
if ax.get_legend() is not None:
    ax.get_legend().remove()
g.fig.legend(handles, labels, ncol=3, loc='upper center',
             bbox_to_anchor=(0.4, 1.05), frameon=False, fontsize=9)
g.fig.subplots_adjust(top=0.99, left=0.15)
plt.savefig(outp('fig_training_durations_Ethan4.png'), dpi=300, bbox_inches='tight')
plt.show()
print('wrote', outp('fig_training_durations_Ethan4.png'))"""))
)

# ====================================================================== FIG ridgeline
cells.append(md(*split_lines(r"""## 6. Figure &mdash; power & training-time ridgelines (`fig_ridgeline_withcma.png`)

OLD power/time data. Builds the `finals` frame (per-experiment final cumulative
power and wall-clock hours, with the ODT collection-phase adjustment) that the
resource-analysis table reuses for its Duration column."""))
)
cells.append(code(*split_lines(r"""df_10 = df_power[(df_power.num_aggs == 10) & (df_power.exp_num != 4129)].copy()
finals = (df_10.sort_values("time")
          .groupby(["algorithm", "exp_num"], as_index=False)
          .last()[["algorithm", "exp_num", "time", "cumulative_power"]])
finals["hours"] = finals["time"] / 3600.0
finals["cum_kW"] = finals["cumulative_power"] / 1000.0

mean_dqn_time = finals.loc[finals['algorithm'] == 'DQN', 'hours'].mean()
mean_reinforce_time = finals.loc[finals['algorithm'] == 'REINFORCE', 'hours'].mean()
mean_dqn_kw = finals.loc[finals['algorithm'] == 'DQN', 'cum_kW'].mean()
mean_reinforce_kw = finals.loc[finals['algorithm'] == 'REINFORCE', 'cum_kW'].mean()

def add_to_odt(row):
    if row.algorithm == "ODT":
        try:
            suffix = int(str(row.exp_num)[-3:])
        except ValueError:
            suffix = None
        if suffix is not None and 108 <= suffix <= 144:
            row.hours += mean_dqn_time
            row.cum_kW += mean_dqn_kw
        else:
            row.hours += mean_reinforce_time
            row.cum_kW += mean_reinforce_kw
    return row

finals = finals.apply(add_to_odt, axis=1)

sns.set_theme(style="white")
algos = finals.algorithm.unique()
n = len(algos)
fig, axes = plt.subplots(n, 2, figsize=(6, n * 0.65), sharex='col')
cum_min, cum_max = finals["cum_kW"].min(), finals["cum_kW"].max()
hr_min, hr_max = finals["hours"].min(), finals["hours"].max()

for i, algo in enumerate(algos):
    data = finals[finals.algorithm == algo]
    ax = axes[i, 0]
    sns.kdeplot(data.cum_kW, bw_adjust=0.5, fill=True, alpha=0.7, linewidth=1.5,
                common_norm=False, ax=ax, color=colors[algo])
    ax.set_xlim(cum_min, cum_max)
    ax.set_yticks([]); ax.tick_params(left=False)
    ax.axhline(0, color='black', linewidth=2, zorder=5)
    if i < n - 1:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    else:
        ax.set_xlabel("Cumulative Power (kW)", labelpad=10)
        ax.set_xticks(np.unique(np.concatenate(([0], ax.get_xticks()))))
        ax.tick_params(axis='x', pad=6)
    ax.set_ylabel(algo, rotation=0, ha='right', va='center', labelpad=10,
                  color=colors[algo], fontweight='bold')
    for s in ('top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(True); ax.spines['left'].set_linewidth(1)

    ax = axes[i, 1]
    sns.kdeplot(data.hours, bw_adjust=0.5, fill=True, alpha=0.7, linewidth=1.5,
                common_norm=False, ax=ax, color=colors[algo])
    ax.set_xlim(hr_min, hr_max)
    ax.set_ylabel(""); ax.set_yticks([]); ax.tick_params(left=False)
    ax.axhline(0, color='black', linewidth=2, zorder=5)
    if i < n - 1:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    else:
        ax.set_xlabel("Time (hours)", labelpad=10)
        ax.set_xticks(np.unique(np.concatenate(([0], ax.get_xticks()))))
        ax.tick_params(axis='x', pad=6)
    for s in ('top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(True); ax.spines['left'].set_linewidth(1)

fig.subplots_adjust(top=0.90, bottom=0.05, left=0.15, right=0.98, hspace=0.0, wspace=0.3)
fig.savefig(outp('fig_ridgeline_withcma.png'), dpi=150, bbox_inches="tight")
plt.show()
print('wrote', outp('fig_ridgeline_withcma.png'))"""))
)

# ====================================================================== FIG sustainability
cells.append(md(*split_lines(r"""## 7. Figure &mdash; sustainability indicator (`fig_sustainability_indicators.png`)

Reward is the REVISED per-DM final reward; energy-used and model-size stay on the
OLD published values (training cost is unchanged)."""))
)
cells.append(code(*split_lines(r"""# Reward = revised; Energy Used (kWh) and Model Size (MB) = OLD (unchanged).
_OLD_REWARD = {"CMA-ES": -73.73, "DQN": -64.13, "ODT": -62.22, "REINFORCE": -63.35}
raw_data_fig_5 = {
    "CMA-ES":    {"Reward": new_rewards.get("CMA-ES", _OLD_REWARD["CMA-ES"]),
                  "Energy Used (kWh)": 0.3654, "Model Size (MB)": 400},
    "DQN":       {"Reward": new_rewards.get("DQN", _OLD_REWARD["DQN"]),
                  "Energy Used (kWh)": 0.8796, "Model Size (MB)": 696},
    "ODT":       {"Reward": new_rewards.get("ODT", _OLD_REWARD["ODT"]),
                  "Energy Used (kWh)": 3.2067, "Model Size (MB)": 8000},
    "REINFORCE": {"Reward": new_rewards.get("REINFORCE", _OLD_REWARD["REINFORCE"]),
                  "Energy Used (kWh)": 0.6998, "Model Size (MB)": 696},
}

algorithms = list(raw_data_fig_5.keys())
rewards = np.array([raw_data_fig_5[a]["Reward"] for a in algorithms])
energy_used = np.array([raw_data_fig_5[a]["Energy Used (kWh)"] * 1000 for a in algorithms])
model_size = np.array([raw_data_fig_5[a]["Model Size (MB)"] for a in algorithms])

sigma_p, sigma_e, sigma_s = 1/2, 1/4, 1/4
S = ((1 - rewards) ** sigma_p) * ((1 + energy_used) ** sigma_e) * ((1 + model_size) ** sigma_s)

bar_colors = [colors[a] for a in algorithms]
fig_5 = plt.figure(figsize=(10, 6))
plt.bar(algorithms, S, color=bar_colors)
plt.xlabel('Algorithm')
plt.ylabel('Sustainability Score')
plt.title('Sustainability Score for Each Algorithm')
fig_5.savefig(outp('fig_sustainability_indicators.png'), dpi=150, bbox_inches="tight")
plt.show()
print('Sustainability scores:', dict(zip(algorithms, np.round(S, 3))))
print('wrote', outp('fig_sustainability_indicators.png'))"""))
)

# ====================================================================== FIG sensitivity
cells.append(md(*split_lines(r"""## 8. Figure &mdash; reward-weight sensitivity trends (`fig_sensitivity_trends.png`)

Reproduces `ieee_TSC_revisions/automation/generate_sensitivity_figure.py`: for each
reward weight ($w_d$, $w_t$, $w_e$), the final reward relative to the baseline
(all weights = 1), normalized within each DM and then averaged across the four DMs,
with a band showing the spread. Reads the committed `table_data/sensitivity_summary.csv`
(the 7xxx sweep). Set `REGEN_SENSITIVITY = True` to rebuild that summary from raw
7xxx metrics first (needs the 7xxx runs under a resolvable metrics root)."""))
)
cells.append(code(*split_lines(r"""# The paper's sensitivity figure (fig_sensitivity_trends.png) is the 2x2 per-DM
# FACETED view produced by plot_reward_by_dm() in the v2 Sensitivity notebook:
# seed-averaged final reward vs each swept weight, one panel per DM, one line per
# swept term. Reproduced from the committed sensitivity_summary.csv (no Huron
# needed); set REGEN_SENSITIVITY=True to rebuild that summary from raw 7xxx first.
import csv as _csv
from pathlib import Path

REGEN_SENSITIVITY = False
SENS_CSV = os.path.join('table_data', 'sensitivity_summary.csv')
SWEEP_VALUES = [0, 1, 5, 7, 10]
SWEPT_VARS = ['distance_weight', 'traffic_weight', 'energy_weight']

if REGEN_SENSITIVITY:
    # Rebuild per-experiment reward means from raw 7xxx (value-major, seed-minor).
    SEEDS = [1234, 5555, 2020]
    MODEL_START = {'DQN': 7000, 'REINFORCE': 7045, 'CMA': 7090, 'ODT': 7135}
    ROOTS = ['../../../../storage_1/metrics', '/storage_1/metrics', '../metrics']
    root = next((r for r in ROOTS if os.path.isdir(r)), None)
    def _meta(n):
        for m, start in MODEL_START.items():
            if start <= n < start + 45:
                off = n - start
                return m, SWEPT_VARS[off // 15], SWEEP_VALUES[(off % 15) // 3], SEEDS[(off % 15) % 3]
        return None
    LAST_N = 50
    recs = []
    if root is not None:
        for n in range(7000, 7180):
            ap = Path(root) / f'Exp_{n}' / 'train' / 'metrics_agent_episode_level.csv'
            if not ap.exists():
                continue
            m, var, val, seed = _meta(n)
            a = pd.read_csv(ap, usecols=['aggregation', 'episode', 'reward'])
            pairs = a[['aggregation', 'episode']].drop_duplicates().tail(LAST_N)
            a_t = a.merge(pairs, on=['aggregation', 'episode'])
            recs.append({'model': m, 'swept_var': var, 'weight': val,
                         'seed': seed, 'reward_mean': a_t['reward'].mean()})
    if recs:
        summary = pd.DataFrame(recs)
        print(f'Rebuilt sensitivity summary from {len(recs)} raw 7xxx runs.')
    else:
        REGEN_SENSITIVITY = False
        print('No raw 7xxx runs found; using committed sensitivity_summary.csv.')

if not REGEN_SENSITIVITY:
    # One row per (model, swept_var, weight); keep the mean from 'mean +/- std'.
    def _parse_mean(cell):
        return float(cell.split('+/-')[0].strip())
    rows = []
    with open(SENS_CSV, newline='') as f:
        rr = _csv.reader(f); next(rr)
        for line in rr:
            model, var = line[0], line[1]
            for w, cell in zip(SWEEP_VALUES, line[2:7]):
                rows.append({'model': model, 'swept_var': var, 'weight': w,
                             'reward_mean': _parse_mean(cell)})
    summary = pd.DataFrame(rows)

# --- Faceted per-DM plot (ports plot_reward_by_dm from the v2 notebook). ---
def aggregate_over_seeds(df, value_col):
    g = df.groupby(['model', 'swept_var', 'weight'])[value_col]
    return g.agg(['mean', 'std', 'count']).reset_index()

SINGLE_COL_IN = 3.5
PAPER_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'figure.labelsize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'grid.linewidth': 0.4,
}

def plot_reward_by_dm(df, value_col, ylabel, fname):
    agg = aggregate_over_seeds(df, value_col)
    dm_order = [m for m in ['CMA', 'DQN', 'ODT', 'REINFORCE'] if m in agg['model'].unique()]
    term_colors = {'distance_weight': '#1f77b4', 'traffic_weight': '#d62728', 'energy_weight': '#2ca02c'}
    term_label = {'distance_weight': 'distance', 'traffic_weight': 'traffic', 'energy_weight': 'energy'}
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COL_IN, SINGLE_COL_IN),
                                 sharex=True, sharey=True, constrained_layout=True)
        axes_flat = axes.flatten()
        handles = labels = None
        for ax, model in zip(axes_flat, dm_order):
            sub = agg[agg['model'] == model]
            for sv in SWEPT_VARS:
                line = sub[sub['swept_var'] == sv].sort_values('weight')
                if line.empty:
                    continue
                ax.plot(line['weight'], line['mean'], marker='o',
                        color=term_colors.get(sv, 'gray'), label=term_label.get(sv, sv))
            ax.set_title(model, color=colors.get(model, 'black'), fontweight='bold', pad=2)
            ax.set_xticks(SWEEP_VALUES); ax.grid(alpha=0.3)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
        for ax in axes_flat[len(dm_order):]:
            ax.set_visible(False)
        fig.supxlabel('weight value'); fig.supylabel(ylabel)
        if handles:
            fig.legend(handles, labels, ncol=3, loc='outside upper center', frameon=False,
                       columnspacing=1.0, handletextpad=0.4)
        plt.savefig(outp(fname), dpi=300)
        plt.show()

plot_reward_by_dm(summary, 'reward_mean', 'Mean reward (last episodes)', 'fig_sensitivity_trends.png')
print('wrote', outp('fig_sensitivity_trends.png'))"""))
)

# ====================================================================== TABLE combined_metrics
cells.append(md(*split_lines(r"""## 9. Table CSV &mdash; simulation metrics by DM & season (`table_combined_metrics.csv`)

`tab:combined_metrics`. Distance and Peak Traffic are REVISED (4xxx); Energy [kWh]
keeps the OLD published mean&plusmn;std (the revised battery columns are degenerate).
The global-minimum cell in each metric column is bolded automatically. Columns:
`alg | season | distance | peak | energy | endrule` (pre-formatted LaTeX)."""))
)
cells.append(code(*split_lines(r"""# OLD published Energy [kWh] (mean, std) by (display DM, season) -- unchanged.
OLD_ENERGY = {
    ("CMA-ES", "Winter"): (15.96, 1.31), ("CMA-ES", "Autumn"): (21.54, 3.04),
    ("CMA-ES", "Spring"): (17.95, 3.19), ("CMA-ES", "Summer"): (22.91, 2.96),
    ("DQN", "Winter"): (15.83, 3.29), ("DQN", "Autumn"): (21.33, 3.10),
    ("DQN", "Spring"): (17.57, 3.16), ("DQN", "Summer"): (22.79, 3.08),
    ("ODT", "Winter"): (15.29, 2.99), ("ODT", "Autumn"): (20.89, 2.55),
    ("ODT", "Spring"): (15.67, 3.70), ("ODT", "Summer"): (20.51, 1.24),
    ("REINFORCE", "Winter"): (16.44, 3.03), ("REINFORCE", "Autumn"): (21.20, 3.01),
    ("REINFORCE", "Spring"): (17.57, 3.15), ("REINFORCE", "Summer"): (22.54, 2.93),
}

DM_ORDER = ["CMA-ES", "DQN", "ODT", "REINFORCE"]          # paper row order
SEASON_ORDER = ["Winter", "Autumn", "Spring", "Summer"]   # paper season order

# Compute distance (mean/std) and peak traffic from the REVISED data.
recs = []
for disp in DM_ORDER:
    raw = {v: k for k, v in ALGO_DISPLAY.items()}[disp]
    for season in SEASON_ORDER:
        sl = season.lower()
        a = cumulative_agent_df[(cumulative_agent_df['algorithm'] == raw) &
                                (cumulative_agent_df['season'] == sl)]
        s = cumulative_station_df[(cumulative_station_df['algorithm'] == raw) &
                                  (cumulative_station_df['season'] == sl)]
        e_mean, e_std = OLD_ENERGY[(disp, season)]
        recs.append({
            'disp': disp, 'season': season,
            'dist_mean': float(a['distance_traveled'].mean()) if len(a) else float('nan'),
            'dist_std': float(a['distance_traveled'].std()) if len(a) else float('nan'),
            'peak': float(s['traffic'].max()) if len(s) else float('nan'),
            'e_mean': e_mean, 'e_std': e_std,
        })
cm = pd.DataFrame(recs)

# Global-minimum cells (lower is better for all three metrics).
imin_dist = cm['dist_mean'].idxmin()
imin_peak = cm['peak'].idxmin()
imin_energy = cm['e_mean'].idxmin()

rows = []
for i, r in cm.iterrows():
    dist_s = pm(r['dist_mean'], r['dist_std'])
    peak_s = f"{int(round(r['peak']))}"
    energy_s = pm(r['e_mean'], r['e_std'])
    if i == imin_dist:
        dist_s = bf(dist_s)
    if i == imin_peak:
        peak_s = bf(peak_s)
    if i == imin_energy:
        energy_s = bf(energy_s)
    first = (r['season'] == SEASON_ORDER[0])
    last = (r['season'] == SEASON_ORDER[-1])
    rows.append([r['disp'] if first else '', r['season'],
                 dist_s, peak_s, energy_s, 'hline' if last else ''])

write_pipe_csv(outp('table_combined_metrics.csv'),
               ['alg', 'season', 'distance', 'peak', 'energy', 'endrule'], rows)"""))
)

# ====================================================================== TABLE resource_analysis
cells.append(md(*split_lines(r"""## 10. Table CSV &mdash; training time / emissions / reward (`table_resource_analysis.csv`)

`tab:resource_analysis`. Duration and Emissions are OLD (`df_power`/`finals`);
Reward is the REVISED per-DM final reward. Emissions are converted from the OLD
recorded CO2 (grams) to kg. Best-in-column cells (min duration, min emissions,
max/least-negative reward) are bolded. Columns: `alg | duration | emissions | reward`."""))
)
cells.append(code(*split_lines(r"""# Duration [h] per DM from finals (moderate-aggregation subset, ODT-adjusted).
dur = finals.groupby('algorithm')['hours'].agg(['mean', 'std'])

# Emissions per DM: mean over experiments of each run's final (cumulative) CO2,
# grams -> kg. Also keep the mean-kg baseline for the regional-emissions table.
emis_kg = {}
for algo in df_power['algorithm'].unique():
    co2_totals = (df_power[df_power['algorithm'] == algo]
                  .groupby('exp_num')['co2'].last().to_numpy())
    emis_kg[algo] = (co2_totals.mean() / 1000.0, co2_totals.std() / 1000.0)

# Assemble raw numbers per display DM.
ra = []
for disp in ["CMA-ES", "DQN", "ODT", "REINFORCE"]:
    raw = {v: k for k, v in ALGO_DISPLAY.items()}[disp]
    d_mean, d_std = dur.loc[raw, 'mean'], dur.loc[raw, 'std']
    e_mean, e_std = emis_kg[raw]
    r_mean, r_std = rew_stats[disp]
    ra.append({'disp': disp, 'd_mean': d_mean, 'd_std': d_std,
               'e_mean': e_mean, 'e_std': e_std, 'r_mean': r_mean, 'r_std': r_std})
ra = pd.DataFrame(ra)

# Save the per-DM emissions baseline (kg at this study's CI) for §11.
EMISSIONS_BASE_KG = dict(zip(ra['disp'], ra['e_mean']))

i_dur = ra['d_mean'].idxmin()        # shortest training
i_emis = ra['e_mean'].idxmin()       # least emissions
i_rew = ra['r_mean'].idxmax()        # best (least negative) reward

rows = []
for i, r in ra.iterrows():
    d_s = pm(r['d_mean'], r['d_std'])
    e_s = pm(r['e_mean'], r['e_std'])
    r_s = pm(r['r_mean'], r['r_std'])
    if i == i_dur:
        d_s = bf(d_s)
    if i == i_emis:
        e_s = bf(e_s)
    if i == i_rew:
        r_s = bf(r_s)
    rows.append([r['disp'], d_s, e_s, r_s])

write_pipe_csv(outp('table_resource_analysis.csv'),
               ['alg', 'duration', 'emissions', 'reward'], rows)"""))
)

# ====================================================================== TABLE regional_emissions
cells.append(md(*split_lines(r"""## 11. Table CSV &mdash; regional emissions projection (`table_regional_emissions.csv`)

`tab:regional_emissions`. Pure arithmetic on the resource-analysis emissions: the
"This study" row is the measured per-DM kg at $CI=300$ g/kWh, and every other row
rescales it by that region's grid intensity ($\text{kg}\times CI/300$). Columns:
`region | ci | cma | dqn | reinforce | odt`."""))
)
cells.append(code(*split_lines(r"""BASE_CI = 300.0
REGIONS = [
    ("Quebec", 1.2),
    ("Ontario", 35),
    ("This study", 300),
    (r"U.S.\ average", 369),
    ("Global average", 480),
]
DM_COLS = ["CMA-ES", "DQN", "REINFORCE", "ODT"]   # paper column order

rows = []
for region, ci in REGIONS:
    scale = ci / BASE_CI
    vals = [f"{EMISSIONS_BASE_KG[dm] * scale:.3f}" for dm in DM_COLS]
    ci_str = f"{ci:g}"
    rows.append([region, ci_str] + vals)

write_pipe_csv(outp('table_regional_emissions.csv'),
               ['region', 'ci', 'cma', 'dqn', 'reinforce', 'odt'], rows)"""))
)

# ====================================================================== TABLE hp_tuning
cells.append(md(*split_lines(r"""## 12. Table CSV &mdash; hyperparameter tuning (`table_hp_tuning.csv`)

`tab:hp_tuning`. The search ranges are the fixed tuning design and the selected
values are the OAT-sweep recommendations (bolded where the paper bolds them). This
is static configuration; if a re-tune changes a selection, update it here. Columns:
`kind | hpname | hprange | hpsel` where `kind` is `section` (a per-DM header row)
or `hp` (a data row)."""))
)
cells.append(code(*split_lines(r"""# Search ranges match the CURRENT v2 sweep; selected values are the OAT
# recommendations (bolded where the paper bolds). endrule='hline' on the last
# row of each DM group so csvsimple draws an \hline before the next section.
LR7 = r"$10^{-5}$, $10^{-4}$, $10^{-3}$, $3\!\times\!10^{-3}$, $10^{-2}$, $3\!\times\!10^{-2}$, $10^{-1}$"
hp_rows = [
    ("section", "DQN", "", "", ""),
    ("hp", "learning rate", LR7, r"$\mathbf{10^{-2}}$", ""),
    ("hp", "discount factor", r"$0.90$, $0.95$, $0.99$, $0.995$, $0.999$", r"$0.99$", ""),
    ("hp", "replay buffer size", r"$150$, $500$, $1500$, $5000$, $15000$", r"$500$", ""),
    ("hp", r"target network update freq.\ (ep.)", r"$5$, $10$, $25$, $50$, $100$", r"$25$", ""),
    ("hp", r"$\epsilon$-decay episode fraction", r"$0.1$, $0.2$, $0.3$, $0.5$, $0.7$", r"$0.3$", "hline"),
    ("section", "REINFORCE", "", "", ""),
    ("hp", "learning rate", LR7, r"$\mathbf{10^{-3}}$", ""),
    ("hp", "discount factor", r"$0.90$, $0.95$, $0.99$, $0.995$, $0.999$", r"$0.99$", ""),
    ("hp", "hidden layers",
     r"$[32,32]$, $[64,64]$, $[128,64,64]$, $[256,128,64]$, $[512,256,128,64]$",
     r"$\mathbf{[64,64]}$", "hline"),
    ("section", "CMA-ES", "", "", ""),
    ("hp", r"initial $\sigma$", r"$0.01$, $0.05$, $0.10$, $0.30$, $1.00$", r"$0.10$", ""),
    ("hp", "population size", r"$10$, $20$, $40$, $80$, $160$", r"$20$", ""),
    ("hp", "max generations", r"$50$, $100$, $200$, $400$, $800$", r"$200$", "hline"),
    ("section", "ODT", "", "", ""),
    ("hp", "learning rate",
     r"$10^{-5}$, $10^{-4}$, $10^{-3}$, $10^{-2}$, $10^{-1}$", r"$\mathbf{10^{-4}}$", ""),
    ("hp", r"embedding dim.\ ($d$)", r"$64$, $128$, $256$, $512$, $1024$", r"$512$", ""),
    ("hp", "number of layers", r"$1$, $2$, $4$, $6$, $8$", r"$4$", ""),
    ("hp", r"context length ($K$)", r"$5$, $10$, $20$, $40$, $80$", r"$10$", ""),
    ("hp", "return-to-go target", r"$-30$, $-50$, $-75$, $-100$, $-150$", r"$\mathbf{-50}$", ""),
]

write_pipe_csv(outp('table_hp_tuning.csv'),
               ['kind', 'hpname', 'hprange', 'hpsel', 'endrule'], hp_rows)"""))
)

# ====================================================================== Wilcoxon diagnostic
cells.append(md(*split_lines(r"""## 13. Diagnostic &mdash; paired Wilcoxon on per-seed reward (optional)

Supports the `\pending{}` note in `tab:resource_analysis`'s caption. Collapses each
DM to one final reward per seed, then runs paired Wilcoxon signed-rank tests on the
principal pairs. With only the 4xxx seeds (n=3) the test is underpowered; it becomes
meaningful once the 5xxx+6xxx seeds land in the cache. Prints only."""))
)
cells.append(code(*split_lines(r"""from scipy.stats import wilcoxon

per_exp = pd.read_csv(os.path.join(V2_CACHE, 'reward_per_exp.csv'))
per_seed = (per_exp.groupby(['algorithm', 'seed'])['reward_final'].mean().unstack('algorithm'))
n = per_seed.shape[0]
print(f"Per-seed final reward ({n} seeds):\n", per_seed.round(2), "\n")
min_p = 2.0 ** -(n - 1) if n >= 1 else float('nan')
print(f"Paired Wilcoxon (two-sided); n={n}; smallest achievable p = {min_p:.3f}")
if n < 6:
    print(f"  [UNDERPOWERED] n={n} < 6 -> cannot reach alpha=0.05; rerun once 9 seeds land.\n")

PAIRS = [('ODT', 'REINFORCE'), ('ODT', 'DQN'), ('DQN', 'REINFORCE'),
         ('CMA', 'ODT'), ('CMA', 'DQN'), ('CMA', 'REINFORCE')]
for a, b in PAIRS:
    if a in per_seed.columns and b in per_seed.columns:
        d = per_seed[[a, b]].dropna()
        try:
            stat, pval = wilcoxon(d[a], d[b])
            flag = 'significant' if pval < 0.05 else 'n.s.'
            print(f"  {a:<10} vs {b:<10}: p={pval:.4f} ({flag})  "
                  f"means {d[a].mean():.2f} / {d[b].mean():.2f}  (n={len(d)})")
        except ValueError as e:
            print(f"  {a:<10} vs {b:<10}: {e}")"""))
)

# ====================================================================== summary
cells.append(md("## 14. Done &mdash; upload checklist"))
cells.append(code(*split_lines(r"""produced = sorted(os.listdir(OUT))
print('Files in', os.path.abspath(OUT), ':')
for f in produced:
    print('   ', f)
print()
print('Upload steps:')
print('  1. all_v2/*.png  ->  VERDE/ieee_TSC_revisions/figures/')
print('  2. all_v2/*.csv  ->  VERDE/ieee_TSC_revisions/data/')
print('  3. Refresh Overleaf.')"""))
)

# ====================================================================== assemble
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
