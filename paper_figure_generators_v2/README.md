# Paper Figure Generators — v2 (revised 4xxx preview)

A copy of `paper_figure_generators/` rewired to regenerate a subset of the SURE-DM
paper figures/tables from the **revised 4xxx experiments only**, so you can see how
the trends move while the 5xxx and 6xxx reruns (the other 6 seeds) finish.

## What changed vs the original

Each main batch (4xxx / 5xxx / 6xxx) is a *complete* grid on its own —
4 seasons × 4 decision-makers × 3 aggregation levels × 3 seeds = 180 experiments.
The published figures pool all three batches ("9 seeds"). **v2 uses 4xxx alone
(3 seeds, all four seasons), read from the new post-fix data.**

| Data | Source in v2 |
|---|---|
| Reward (curves + final), distance, peak traffic | **NEW** — raw `/storage_1/metrics_postfix/Exp_<n>/train/metrics_*.csv`, 4xxx only |
| Training duration, power, CO₂ emissions, model size | **OLD** — unchanged (the Experiment 3 notebook's existing cells still read `/storage_1/metrics`) |
| Table 2 `Energy [kWh]` (a "kWh metric") | **OLD** — published values kept; see note below |

The original notebooks read pre-aggregated CSVs from
`/storage_1/metrics/formatted_experiment_data/{part_1,part_4}/`, which only exist for
the original data. v2 instead aggregates the raw post-fix metrics itself, via a
one-pass cache builder.

## How to run on Huron

From inside `paper_figure_generators_v2/` (so relative paths resolve — keep it
four directories deep, exactly like the original):

```sh
# 1) One-pass aggregation of the revised 4xxx raw metrics into a small cache.
#    (Reads /storage_1/metrics_postfix/Exp_4000..4179/train/*.csv once.)
python build_v2_cache.py --metrics-root /storage_1/metrics_postfix

# 2) Launch Jupyter and Run All on each notebook below.
jupyter lab
```

`build_v2_cache.py` writes `table_data/_v2_cache/{reward_curves.csv, env_agent.csv,
env_station.csv, reward_per_exp.csv}`. Re-run it whenever more 4xxx seeds land. It
prints how many of the 180 experiments it found (so you can see partial-batch
coverage). Useful flags: `--experiments 4000-4179`, `--last-n 100`.

## What each notebook produces (paper mapping)

| Notebook | Paper output | File |
|---|---|---|
| `Experiment 1 Figures.ipynb` (first figure cell) | **Fig 3** `fig:episode_plateau` | `figures/episode_plateau_and_seasons.png` |
| `Experiment 2 Figures.ipynb` | **Table 2** `tab:combined_metrics` (Distance / Peak Traffic / Reward = new; Energy = old) | `table_data/table_1.csv` |
| `Experiment 3 Figures.ipynb` (reward cell) | **Table 3** `tab:resource_analysis` *reward column* + per-algorithm reward deliverable | printed + `table_data/reward_4xxx_revised.csv` |
| `Experiment 3 Figures.ipynb` (Fig-5 cell) | **Fig 6** `fig:sustainability_indicator` (reward = new; energy + model size = old) | `figures/sustainability_indicators.png` |

The Experiment 3 reward cell also prints a **LaTeX-ready Reward column** for Table 3,
in two poolings: across all 4xxx experiments, and per-seed-then-across-seeds (the
paper's "per-seed final reward" convention used for the Wilcoxon test).

## Notes / caveats

- **Figure 7** (`fig_agg_level_train_eval_matrix.png`) is **not** regenerated here.
  It needs evaluation-on-a-new-environment data (a separate `eval/`-mode run), and no
  generator for it exists in the repo. Revisit once that data/generator is available.
- **Table 2 Energy [kWh].** The raw `metrics_agent_episode_level.csv` battery columns
  are degenerate (`starting_battery == ending_battery`), so simulation energy can't be
  derived from them. Per the "kWh metrics → old data" instruction, the Energy column
  keeps the old published values. If your new runs log a usable energy signal, swap it
  into the Table 2 cell.
- **3 seeds, not 9.** Bands/std are over 3 seeds × 4 seasons of 4xxx, so they'll be
  wider/noisier than the final 9-seed figures. This is a preview, not the final figure.
- The other notebooks (Sensitivity, Hyperparameter Tuning, Bulletproofing, Discussion)
  are carried over unchanged from the original directory and are not part of this
  preview.

All new data plumbing lives in `v2_data.py`; `build_v2_cache.py` is its CLI.
