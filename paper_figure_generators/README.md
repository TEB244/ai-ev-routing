# Paper Figure Generators

This folder contains the Jupyter notebooks that produce every figure and table
used in the SURE-DM paper. The notebooks are set up to run **as-is on the
Huron lab server**. If you run them anywhere else you will need to mirror
Huron's data layout (see [Running off Huron](#running-off-huron) below).

## Quick start (Huron)

1. SSH into Huron and clone this repo somewhere under your home directory, so
   that `ai-ev-routing/paper_figure_generators/` sits **four directories deep**
   from `/` (e.g. `/home/<user>/<group>/ai-ev-routing/paper_figure_generators/`).
   The notebooks use the relative path `../../../../storage_1/metrics/` to
   reach the shared data, so the clone depth matters.
2. Install the repo's dependencies into whatever Python env you use on Huron:
   ```sh
   pip install -r ../requirements.txt
   ```
3. Launch JupyterLab from inside `paper_figure_generators/` (so relative paths
   resolve correctly):
   ```sh
   cd paper_figure_generators
   jupyter lab
   ```
4. Open a notebook and `Run All`. The figures and tables are written next to
   the notebooks under [`figures/`](figures/) and [`table_data/`](table_data/).

## What each notebook produces

| Notebook | Paper output | Saved to |
|---|---|---|
| `Experiment 1 Figures.ipynb` | Episode plateau / seasonal reward curves | `figures/episode_plateau_and_seasons.png` |
| `Experiment 1 Figures.ipynb` | Training duration comparison | `figures/training_durations.png` |
| `Experiment 2 Figures.ipynb` | Table 1 | `table_data/table_1.csv` |
| `Experiment 3 Figures.ipynb` | Per-DM reward ridgelines | `figures/ridgeline.png` |
| `Experiment 3 Figures.ipynb` | Sustainability indicators | `figures/sustainability_indicators.png` |
| `Experiment 3 Figures.ipynb` | Table 2 | `table_data/table_2.csv` |
| `Discussion Figures.ipynb` | DM behaviour profiles | `figures/dm_profiles.png` |

The `figures/` and `table_data/` directories are committed with the last set of
rendered outputs, so you can diff your re-runs against what went into the
paper.

## Where the data comes from

Everything the notebooks read lives under `/storage_1/metrics/` on Huron. The
notebooks reach it via the relative path `../../../../storage_1/metrics/`.

Two kinds of inputs are loaded:

- **Raw per-experiment training metrics** —
  `/storage_1/metrics/Exp_<NUM>/train/power_and_co2_metrics.csv`
  (plus other files in the same folder).
  Experiment numbers span ranges like `4000–4179`, `5000–5179`, and
  `6000–6179`; the exact IDs each notebook pulls are defined at the top of
  that notebook.
- **Pre-formatted aggregate data** —
  `/storage_1/metrics/formatted_experiment_data/part_1/` (Experiment 1) and
  `/storage_1/metrics/formatted_experiment_data/part_4/` (Experiment 3).
  These are the CSVs produced once by the eval pipeline so notebooks don't
  have to re-aggregate every run.

The notebooks also read per-experiment YAML configs from **inside this repo**,
at `../experiments/Exp_<NUM>/config.yaml`. Those are version-controlled, so
nothing to set up there.

If you hit a permission-denied error on a file under `/storage_1/metrics/`,
ask whoever owns that experiment to `chmod` it or re-run it under a shared
group. A few of the raw `Exp_*` directories have historically been
user-locked.

## Running off Huron

The notebooks have a fallback: if `../../../../storage_1/metrics/Exp_<NUM>`
doesn't exist, they look for `../metrics/Exp_<NUM>` instead. So if you want
to run locally, either:

- symlink `../../../../storage_1/metrics` to a local copy of the metrics
  tree, or
- drop the `Exp_<NUM>` folders under `<repo>/metrics/` to use the fallback
  path.

Either way you still need the `formatted_experiment_data/part_1` and
`part_4` folders for Experiments 1 and 3 — there is no fallback for those,
so copy them from Huron if you're working offline.
