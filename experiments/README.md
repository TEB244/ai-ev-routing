# Experiments

Every VERDE run is defined by a numbered directory in here. `main.py -e <NNNN>`
resolves to `experiments/Exp_<NNNN>/config.yaml`, so the experiment number is the
only handle needed to run, track, or analyse a run.

Each directory contains:

| File | Contents |
|---|---|
| `config.yaml` | The full experiment definition. See the Configuration section of the [root README](../README.md#configuration). |
| `description.txt` | Human-readable summary: seed, DM, season, fleet size, episode and aggregation counts. |
| `train_job.sh` | SLURM submission script for training. |
| `eval_job.sh` | SLURM submission script for evaluation. |
| `train_job_gpu.sh` | GPU variant, present where the DM benefits from one. |

Experiment *outputs* are not stored here. Runs write their metrics to the path given
by `main.py -d` — on the cluster a scratch directory, on the lab server a path under
`/storage_1/`. This directory holds definitions only.

> `config.yaml` is authoritative. A few `description.txt` files in the ODT blocks
> carry a stale `Model:` line inherited from the template they were copied from;
> trust the `algorithm_settings.algorithm` field in the config.

## Numbering scheme

There are 1,525 experiment directories. The leading digit identifies the batch, and
the paper's results come from the **4xxx and above** ranges — about 1,100 runs. The
rest is exploratory work kept for provenance.

| Range | Count | Purpose |
|---|---|---|
| `Exp_0001`–`Exp_0479` | 198 | Early exploratory runs, used to settle episode and aggregation counts. Sparse — the range has gaps. Pre-dates the paper's experimental design. |
| `Exp_1000`–`Exp_1066` | 52 | Superseded exploratory batch. Sparse. |
| `Exp_2000`–`Exp_2143` | 144 | Superseded exploratory batch. |
| `Exp_3000`–`Exp_3002` | 3 | Offline trajectory generation for ODT. Produces the `.h5` datasets ODT experiments pretrain on. |
| `Exp_4000`–`Exp_4179` | 180 | **Main experiments, seed batch 1.** |
| `Exp_5000`–`Exp_5179` | 180 | **Main experiments, seed batch 2.** |
| `Exp_6000`–`Exp_6179` | 180 | **Main experiments, seed batch 3.** |
| `Exp_7000`–`Exp_7179` | 180 | **Reward-weight sensitivity analysis.** |
| `Exp_8000`–`Exp_8026` | 27 | **Single-term reward ablation** (diagonal-dominance check). |
| `Exp_9000`–`Exp_9326` | 327 | **Hyperparameter tuning**, v1 and v2 sweeps. |
| `Exp_9994`–`Exp_9999` | 6 | Ad-hoc debug slots. No descriptions, not part of any result. |
| `Exp_10000`–`Exp_10047` | 48 | **Inference-energy profiling**, for the deployment-cost analysis. |

The 0xxx/1xxx/2xxx batches are kept for provenance. No paper figure or table reads
them; the figure notebooks only touch 4xxx and above.

> **The experiment-number → (DM, seed, season, aggregation level) mapping is
> load-bearing.** The analysis notebooks index results by experiment number, so
> renumbering a batch silently invalidates every figure derived from it.

## Main experiments — 4xxx / 5xxx / 6xxx

The three batches are **seed batches, not season batches**. Each one is a complete
180-experiment grid spanning all four seasons, and the published figures pool all
three to give nine seeds per DM:

| Batch | Seeds |
|---|---|
| 4xxx | 1234, 5555, 2020 |
| 5xxx | 2468, 4040, 11110 |
| 6xxx | 3702, 6060, 16665 |

All three share the same internal layout. Taking 4xxx as the example:

| Range | DM | Count |
|---|---|---|
| 4000–4035 | DQN | 36 |
| 4036–4071 | REINFORCE | 36 |
| 4072–4107 | CMA | 36 |
| 4108–4179 | ODT | 72 |

Within a DM block the ordering is season-major, then federated aggregation level,
then seed:

- **Season** (4): spring, winter, summer, autumn
- **Aggregation level** (3): `aggregation_count` of 50, 10, then 1
- **Seed** (3): the batch's three seeds, in the order listed above

So 4000–4002 are spring / 50 aggregations / the three seeds; 4003–4005 are spring /
10 aggregations; 4006–4008 are spring / 1 aggregation; 4009 starts winter. ODT
occupies twice the span because it is run in two configurations.

These batches produce the main results table, the convergence and seasonal reward
figures, the sustainability indicator, and the cross-zone adaptation results.

## Sensitivity analysis — 7xxx

Sweeps each of the three reward weights over `{0, 1, 5, 7, 10}` per DM, three seeds
per cell, to show the DM ranking is preserved under perturbation of the `(1, 1, 1)`
baseline.

| Range | DM | `distance_weight` | `traffic_weight` | `energy_weight` |
|---|---|---|---|---|
| 7000–7044 | DQN | 7000–7014 | 7015–7029 | 7030–7044 |
| 7045–7089 | REINFORCE | 7045–7059 | 7060–7074 | 7075–7089 |
| 7090–7134 | CMA | 7090–7104 | 7105–7119 | 7120–7134 |
| 7135–7179 | ODT | 7135–7149 | 7150–7164 | 7165–7179 |

Within each 15-experiment sweep the ordering is value-major / seed-minor: offsets
+0…+2 use weight 0, +3…+5 use weight 1 (the baseline), +6…+8 use 5, +9…+11 use 7,
+12…+14 use 10. Seeds within each triplet are 1234, 5555, 2020.

## Single-term reward ablation — 8xxx

Each run trains against a reward containing exactly one term, testing whether the
agent genuinely optimises the reward it is given: the metric matching the active
term should be lowest in its own cell.

| Range | DM | Reward |
|---|---|---|
| 8000–8002 | DQN | distance only `(1, 0, 0)` |
| 8003–8005 | DQN | traffic only `(0, 1, 0)` |
| 8006–8008 | DQN | energy only `(0, 0, 1)` |
| 8009–8017 | REINFORCE | the same three shapes, in the same order |
| 8018–8026 | CMA | the same three shapes, in the same order |

Seeds 1234, 5555, 2020 within each triplet. ODT is excluded on compute cost.

## Hyperparameter tuning — 9xxx

One-at-a-time sweeps, five values per hyperparameter, three seeds per cell, at
reduced scope (one zone, spring only, 25 RL aggregations, reward weights pinned at
`(1, 1, 1)`). The centre value of each sweep matches the default, so the centre
cells are functionally identical across sweeps for a given DM and seed.

**v1 — `Exp_9000`–`Exp_9179`**, 45 per DM as three sweeps of 15:

| Range | DM | Sweep 1 | Sweep 2 | Sweep 3 |
|---|---|---|---|---|
| 9000–9044 | DQN | `learning_rate` | `discount_factor` | `buffer_limit` |
| 9045–9089 | REINFORCE | `learning_rate` | `discount_factor` | `layers_arch` |
| 9090–9134 | CMA | `initial_sigma` | `population_dimension` | `max_generations` |
| 9135–9179 | ODT | `learning_rate` | `embed_dim` | `n_layer` |

**v2 — `Exp_9180`–`Exp_9251`**, covering hyperparameters v1 missed and extending the
learning-rate range past v1's upper edge. Every v2 config pins the v1-tuned values
for non-swept hyperparameters so the operating point matches.

| Range | DM | Swept | Values |
|---|---|---|---|
| 9180–9194 | DQN | `target_network_update_frequency` | 5, 10, 25, 50, 100 |
| 9195–9209 | DQN | `target_episode_epsilon_frac` | 0.1, 0.2, 0.3, 0.5, 0.7 |
| 9210–9215 | DQN | `learning_rate` (extension) | 3e-2, 1e-1 |
| 9216–9221 | REINFORCE | `learning_rate` (extension) | 3e-2, 1e-1 |
| 9222–9236 | ODT | `K` (context length) | 5, 10, 20, 40, 80 |
| 9237–9251 | ODT | `rtg` (online and eval, paired) | -30, -50, -75, -100, -150 |

CMA has no v2 sweeps — its three canonical hyperparameters were fully covered by v1.

**ODT re-run — `Exp_9252`–`Exp_9326`**, repeating all five ODT sweeps (v1
`learning_rate` / `embed_dim` / `n_layer`, then v2 `K` / `rtg`) under
`forward_mode: sequential`:

| Range | Swept |
|---|---|
| 9252–9266 | `learning_rate` |
| 9267–9281 | `embed_dim` |
| 9282–9296 | `n_layer` |
| 9297–9311 | `K` |
| 9312–9326 | `rtg` |

## Inference-energy profiling — 10xxx

Inference-only runs: forward passes from pretrained weights with no gradient step,
used to measure deployment energy against training energy. Four DMs × four fleet
sizes × three seeds:

| Range | DM |
|---|---|
| 10000–10011 | DQN |
| 10012–10023 | REINFORCE |
| 10024–10035 | CMA |
| 10036–10047 | ODT |

Within each block, fleet size varies 50 → 100 → 150 → 200 cars, three seeds per
size. Each run loads pretrained weights from the corresponding 4xxx experiment,
named in its `description.txt`.

These feed [`fit_inference_regression.py`](../fit_inference_regression.py) and
[`fit_inference_inproc.py`](../fit_inference_inproc.py), which fit the inference
energy model used in the deployment analysis.

## Regenerating a batch

The 7xxx, 8xxx and 9xxx batches are generated, not hand-written. The generators are
idempotent, so re-running one rewrites its configs and job scripts in place:

```sh
python generate_sensitivity_experiments.py      # 7000-7179
python generate_bulletproofing_experiments.py   # 8000-8026
python generate_hp_tuning_experiments.py        # 9000-9179
python generate_hp_tuning_v2_experiments.py     # 9180-9251
```

Run [`validate_batch.py`](../validate_batch.py) over a batch before submitting it.

## Which notebook reads which range

All notebooks live in [`../paper_figure_generators/`](../paper_figure_generators/).
`Generate All Paper Figures.ipynb` is the single entry point that regenerates every
paper output; the per-experiment notebooks below remain the authoritative source for
each individual figure or table.

| Notebook | Reads |
|---|---|
| `Experiment 1 Figures.ipynb` | 4xxx, 5xxx, 6xxx — convergence and training curves |
| `Experiment 2 Figures.ipynb` | 4xxx, 5xxx, 6xxx — per-DM per-season metrics table |
| `Experiment 3 Figures.ipynb` | 4xxx, 5xxx, 6xxx — sustainability indicator |
| `Discussion Figures.ipynb` | derived data — DM behaviour profiles |
| `Sensitivity Analysis.ipynb` | 7xxx |
| `Bulletproofing.ipynb` | 8xxx, plus free analyses over 7xxx |
| `Hyperparameter Tuning.ipynb` | 9xxx (understands the v1 and v2 ranges) |

`plot_reward_curve.py <exp_num>` saves the training reward curve for a single
experiment, and [`check_progress.py`](../check_progress.py) reports progress and
SLURM state for an in-flight run.

The notebooks are built from scripts in [`../_scripts/`](../_scripts/) — edit the
build script and re-run it rather than editing a notebook by hand.

## Other directories

| Directory | Contents |
|---|---|
| `Episode_Plateau_Eval/` | Exploratory notebooks over the 0xxx batch that set the episode and aggregation counts, plus early tuning comparisons. |
| `Emissions_Eval/` | Early emissions-accounting notebook. |
| `Archived/` | Superseded analysis notebook, kept for provenance. |
| `odt_test/` | Standalone ODT smoke-test scripts. |
| `initial_config.yaml` | Template for the run-mode settings a new experiment starts from. |
| `experiment_list.txt` | Historical log of the 0xxx/1xxx/2xxx batches and the global changes applied across later ones. |
