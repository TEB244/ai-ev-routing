# Experiments — reference

This file documents every active experiment batch in the paper-revision
workflow. For older batches (Exp_0xxx, 1xxx, 2xxx) see
`experiment_list.txt` — those pre-date the current revision and are not
part of the rerun.

## Workflow overview

The paper revision involves rerunning a substantial fraction of the
experiments after fixing several bugs (see "Global changes" below). The
dependency order is:

```
  9xxx HP tuning ─┬─→ HP analysis → recommended HP values
                  │                      │
                  │                      ▼
                  └────→ apply to 4xxx/5xxx/6xxx/7xxx/8xxx configs
                                         │
                ┌────────────────────────┼────────────────────────┐
                ▼                        ▼                        ▼
        4xxx/5xxx/6xxx              7xxx                       8xxx
        (main experiments)       (sensitivity)             (bulletproofing)
                │                        │                        │
                └────────────────────────┼────────────────────────┘
                                         ▼
                         Re-run paper figure-generator notebooks
                                         │
                                         ▼
                                   Final paper data
```

## Algorithm → owner mapping

| Algorithm | Owner | SLURM user | Scratch path |
|---|---|---|---|
| DQN | Lucas Hartman | `hartman` (cluster) / `lhartma8` (mail) | `/home/hartman/scratch/metrics/` |
| REINFORCE | Lucas Hartman | `hartman` | `/home/hartman/scratch/metrics/` |
| CMA | Santiago Gomez-Rosero | `sgomezro` | `/home/sgomezro/scratch/metrics/` |
| ODT | Ethan Pigou | `epigou` | `/home/epigou/scratch/metrics/` |

`check_progress.py` and `plot_reward_curve.py` auto-detect the owner
from the config's algorithm and look in the right scratch path.

## Global changes affecting all 4xxx–9xxx batches

These were applied across every config in the rerun ranges as part of
the revision:

1. **Action space discretised to 5 mix profiles.**
   `environment_settings.action_dim = 5`. The agent argmax-selects one of
   5 distance/traffic mix profiles per state, with mix coefficient
   `w_i = i/(N-1)` for `i ∈ {0, 1, 2, 3, 4}`:
   - 0: `(distance_mult, traffic_mult) = (1.0, 0.0)`  pure-distance
   - 1: `(0.75, 0.25)`  lean-distance
   - 2: `(0.5, 0.5)`    balanced
   - 3: `(0.25, 0.75)`  lean-traffic
   - 4: `(0.0, 1.0)`    pure-traffic

2. **Reward weights pinned at `(1, 1, 1)`** with unit-conversion scales
   `(distance_scale=100, traffic_scale=1, energy_scale=0.001)`. See
   `paper_figure_generators/Sensitivity Analysis.ipynb` Part 3.

3. **Bug fixes baked in** (cannot be turned off): DQN loss now gathers
   the chosen action's Q-value, REINFORCE uses a cross-episode EMA
   baseline, state normalisation uses fixed per-feature scaling, and the
   env's `generate_paths` consumes argmax(action) rather than continuous
   weights. See the commit log for `952c22fe` and `6e17ed41`.

4. **Hyperparameter bumps** applied across all configs:
   `learning_rate: 1e-5 → 1e-3`, `buffer_limit: 150 → 1500`,
   `discount_factor: 0.999 → 0.99`. Will be re-applied after 9xxx HP
   tuning completes with the empirically tuned values.

5. **SLURM wall times bumped by +5h** on every DQN/REINFORCE
   `train_job.sh`. DQN now 13h, REINFORCE 21h (for the longer 10000-ep
   configs in 4xxx/5xxx/6xxx).

---

## Exp_3xxx — Offline data generation for ODT (planned)

| Status | Range | Algorithm | Count | Purpose |
|---|---|---|---|---|
| **PLANNED** | TBD (likely `Exp_3001`) | DQN | 1 | Single high-quality DQN run with `save_offline_data: true` to produce the offline `.h5` trajectories ODT pretrains on. |

To be created after HP tuning completes, using the tuned HPs from the
9xxx analysis. Required before ODT (4xxx/5xxx/6xxx/7xxx/8xxx/9xxx ODT
runs) can produce meaningful results, because the existing
`Exp_3000` data was generated with pre-fix code and a 3-element action
space.

**Action item for after HP analysis:** create `Exp_3001` config (DQN,
tuned HPs, all 4 zones, all 10000 episodes, spring season,
`save_offline_data: true`) and point every ODT config's
`odt_hyperparameters.offline_dataset_path` at it.

---

## Exp_4xxx, Exp_5xxx, Exp_6xxx — Main paper experiments (rerun pending)

| Status | Range | Algorithms | Count | Purpose |
|---|---|---|---|---|
| **RERUN PENDING** (waiting on HP tuning) | 4000–4179 | DQN(36) + REINFORCE(36) + CMA(36) + ODT(72) | 180 | Original Experiments 1/2 (spring), per seed × per car_model_zone combination. |
| **RERUN PENDING** | 5000–5179 | same split | 180 | Same as 4xxx for season 2 (winter). |
| **RERUN PENDING** | 6000–6179 | same split | 180 | Same as 4xxx for season 3 (summer). |

These produce the data behind Table I, Figures 2/3/4, the sustainability
analysis (Figure 5), and the cross-zone adaptation results in the
original paper. All three batches need to be rerun with the post-fix
code, the tuned HPs from 9xxx, and the new 5-profile action space.

**Total: 540 experiments across the three seasons.**

### Layout per batch (4xxx as the example)

Within 4000–4179 the original layout is one experiment per
`(model, seed, car_model_zone)` combination. The exact mapping is in
`experiment_list.txt` under the old "Set Name" entries. **Do not touch
the experiment-number → (model, seed) mapping** during the rerun — the
paper's analysis notebooks index by that mapping.

Wall times (per job): DQN 13h, REINFORCE 21h, CMA 10h40m, ODT 30h.

### Submission (after HP analysis is complete)

```bash
# DQN + REINFORCE (Lucas)
for n in $(seq 4000 4071); do sbatch experiments/Exp_${n}/train_job.sh; done

# CMA (Santiago)
for n in $(seq 4072 4107); do sbatch experiments/Exp_${n}/train_job.sh; done

# ODT (Ethan)
for n in $(seq 4108 4179); do sbatch experiments/Exp_${n}/train_job.sh; done
```

Same pattern for 5xxx and 6xxx (winter and summer).

---

## Exp_7000–7179 — Sensitivity analysis (rerun pending)

| Status | Range | Algorithms | Count | Purpose |
|---|---|---|---|---|
| **RERUN PENDING** | 7000–7179 | 45 per algo | 180 | Reward-weight sensitivity sweep. |

Sweeps each of the three reward weights (`distance_weight`,
`traffic_weight`, `energy_weight`) over `{0, 1, 5, 7, 10}` for each
algorithm, with 3 seeds per cell. Justifies the chosen
`(distance_weight=1, traffic_weight=1, energy_weight=1)` baseline by
showing DM ranking is preserved under perturbation.

### Layout

| Range | Model | distance_weight sweep | traffic_weight sweep | energy_weight sweep |
|---|---|---|---|---|
| 7000–7044 | DQN | 7000–7014 | 7015–7029 | 7030–7044 |
| 7045–7089 | REINFORCE | 7045–7059 | 7060–7074 | 7075–7089 |
| 7090–7134 | CMA | 7090–7104 | 7105–7119 | 7120–7134 |
| 7135–7179 | ODT | 7135–7149 | 7150–7164 | 7165–7179 |

Within each 15-experiment sweep, ordering is value-major / seed-minor:
offsets +0..+2 use weight=0 (ablation), +3..+5 use weight=1 (baseline),
+6..+8 use weight=5, +9..+11 use weight=7, +12..+14 use weight=10.
Seeds within each triplet: 1234, 5555, 2020.

Generated by `generate_sensitivity_experiments.py`. Analysis in
`paper_figure_generators/Sensitivity Analysis.ipynb` (Parts 1, 2, 3).

---

## Exp_8000–8026 — Bulletproofing / single-term reward (rerun pending)

| Status | Range | Algorithms | Count | Purpose |
|---|---|---|---|---|
| **RERUN PENDING** | 8000–8026 | DQN(9) + REINFORCE(9) + CMA(9) | 27 | Defends the sensitivity result against the "agents aren't really optimising the reward" critique. |

Each experiment trains an agent with a reward that contains exactly ONE
term (`distance_only` = `(1, 0, 0)`, `traffic_only` = `(0, 1, 0)`,
`energy_only` = `(0, 0, 1)`). If the agent actually optimises the
reward, the metric matching the active term should be lowest in its own
cell — the diagonal-dominance test.

### Layout

| Range | Model | Reward shape |
|---|---|---|
| 8000–8002 | DQN | distance-only |
| 8003–8005 | DQN | traffic-only |
| 8006–8008 | DQN | energy-only |
| 8009–8011 | REINFORCE | distance-only |
| 8012–8014 | REINFORCE | traffic-only |
| 8015–8017 | REINFORCE | energy-only |
| 8018–8020 | CMA | distance-only |
| 8021–8023 | CMA | traffic-only |
| 8024–8026 | CMA | energy-only |

Seeds within each triplet: 1234, 5555, 2020. ODT excluded due to
compute cost; could be added later if 3-DM result is inconclusive.

Generated by `generate_bulletproofing_experiments.py`. Analysis in
`paper_figure_generators/Bulletproofing.ipynb`.

---

## Exp_9000–9179 — Hyperparameter tuning v1 (DQN + REINFORCE completed; CMA + ODT in progress)

| Status | Range | Algorithm | Count | Purpose |
|---|---|---|---|---|
| **DONE** | 9000–9044 | DQN | 45 | HP tuning: lr / discount_factor / buffer_limit |
| **DONE** | 9045–9089 | REINFORCE | 45 | HP tuning: lr / discount_factor / layers_arch |
| **PENDING (Santiago)** | 9090–9134 | CMA | 45 | HP tuning: initial_sigma / population_dimension / max_generations |
| **PENDING (Ethan)** | 9135–9179 | ODT | 45 | HP tuning: lr / embed_dim / n_layer |

180 experiments total. One-at-a-time (OAT) sweeps over 3 HPs per
algorithm, 5 values per HP, 3 seeds per cell.

### Layout per algorithm

Each algorithm's 45 experiments split into 3 sweeps of 15 each
(5 values × 3 seeds, value-major / seed-minor).

| Range | Algorithm | HP 1 (15 exps each) | HP 2 | HP 3 |
|---|---|---|---|---|
| 9000–9044 | DQN | `learning_rate` ∈ {1e-5, 1e-4, 1e-3, 3e-3, 1e-2} | `discount_factor` ∈ {0.90, 0.95, 0.99, 0.995, 0.999} | `buffer_limit` ∈ {150, 500, 1500, 5000, 15000} |
| 9045–9089 | REINFORCE | same as DQN | same as DQN | `layers_arch` (5 preset architectures) |
| 9090–9134 | CMA | `initial_sigma` ∈ {0.01, 0.05, 0.10, 0.30, 1.00} | `population_dimension` ∈ {10, 20, 40, 80, 160} | `max_generations` ∈ {50, 100, 200, 400, 800} |
| 9135–9179 | ODT | `learning_rate` ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} | `embed_dim` ∈ {64, 128, 256, 512, 1024} | `n_layer` ∈ {1, 2, 4, 6, 8} |

The centre value of each sweep (offset +6,+7,+8 within the 15-block)
matches the post-fix default, so three centre cells per (algorithm,
seed) are functionally identical.

Reduced compute scope per experiment: 1 zone, spring only, 25 RL
aggregations (12.5k transitions total per car), reward weights pinned
at `(1, 1, 1)`.

Generated by `generate_hp_tuning_experiments.py`. Analysis in
`paper_figure_generators/Hyperparameter Tuning.ipynb`.

### Submission

```bash
# DQN (Lucas) — DONE
for n in $(seq 9000 9044); do sbatch experiments/Exp_${n}/train_job.sh; done

# REINFORCE (Lucas) — DONE
for n in $(seq 9045 9089); do sbatch experiments/Exp_${n}/train_job.sh; done

# CMA (Santiago)
for n in $(seq 9090 9134); do sbatch experiments/Exp_${n}/train_job.sh; done

# ODT (Ethan) — wait for offline data (Exp_3001) before submitting
for n in $(seq 9135 9179); do sbatch experiments/Exp_${n}/train_job.sh; done
```

---

## Exp_9180–9251 — Hyperparameter tuning v2 / extension (pending)

| Status | Range | Algorithm | Count | Purpose |
|---|---|---|---|---|
| **PENDING (Lucas)** | 9180–9215 | DQN | 36 | Adds literature-impactful HPs the v1 sweep missed + extends lr range above v1's upper edge. |
| **PENDING (Lucas)** | 9216–9221 | REINFORCE | 6 | Extends lr range above v1's upper edge. |
| **PENDING (Ethan)** | 9222–9251 | ODT | 30 | Adds context length K and return-to-go (RTG) conditioning — the two HPs every DT paper tunes. |

72 experiments total. Same reduced compute scope as v1. Every v2 config
explicitly sets the v1-tuned values for non-swept HPs so the operating
point matches v1.

### Layout

| Range | Algorithm | Swept HP | Values |
|---|---|---|---|
| 9180–9194 | DQN | `target_network_update_frequency` | 5, 10, 25, 50, 100 |
| 9195–9209 | DQN | `target_episode_epsilon_frac` | 0.1, 0.2, 0.3, 0.5, 0.7 |
| 9210–9215 | DQN | `learning_rate` EXTENSION | 3e-2, 1e-1 |
| 9216–9221 | REINFORCE | `learning_rate` EXTENSION | 3e-2, 1e-1 |
| 9222–9236 | ODT | `K` (context length) | 5, 10, 20, 40, 80 |
| 9237–9251 | ODT | `rtg` (online_rtg + eval_rtg paired) | -30, -50, -75, -100, -150 |

Within each block: 3 seeds per value (1234, 5555, 2020), value-major /
seed-minor ordering.

CMA has no v2 additions — its three canonical HPs were fully covered
by v1.

Generated by `generate_hp_tuning_v2_experiments.py`. Shares analysis
notebook with v1 (`Hyperparameter Tuning.ipynb` understands both
ranges).

### Submission

```bash
# DQN (Lucas)
for n in $(seq 9180 9215); do sbatch experiments/Exp_${n}/train_job.sh; done

# REINFORCE (Lucas)
for n in $(seq 9216 9221); do sbatch experiments/Exp_${n}/train_job.sh; done

# ODT (Ethan)
for n in $(seq 9222 9251); do sbatch experiments/Exp_${n}/train_job.sh; done
```

---

## Tooling reference

| Script | Purpose |
|---|---|
| `check_progress.py <exp_num>` | Print %-complete, current aggregation, last-100-ep mean reward, SLURM state, ETA. Auto-detects owner from config. |
| `paper_figure_generators/plot_reward_curve.py <exp_num>` | Save a PNG of the training reward curve for one experiment. Auto-detects metrics path. |
| `generate_sensitivity_experiments.py` | Regenerate configs and job files for 7xxx (idempotent). |
| `generate_bulletproofing_experiments.py` | Regenerate configs and job files for 8xxx. |
| `generate_hp_tuning_experiments.py` | Regenerate configs and job files for 9000–9179 (v1). |
| `generate_hp_tuning_v2_experiments.py` | Regenerate configs and job files for 9180–9251 (v2). |
| `_scripts/build_sensitivity_notebook.py` | Rebuild `Sensitivity Analysis.ipynb`. |
| `_scripts/build_bulletproofing_notebook.py` | Rebuild `Bulletproofing.ipynb`. |
| `_scripts/build_hp_tuning_notebook.py` | Rebuild `Hyperparameter Tuning.ipynb`. |
| `_scripts/patch_sensitivity_notebook.py` | Add or refresh Part 3 of `Sensitivity Analysis.ipynb` (reward-weight recommendation). |

## Analysis notebooks (in `paper_figure_generators/`)

| Notebook | Reads from | What it produces |
|---|---|---|
| `Hyperparameter Tuning.ipynb` | Exp_9000–9251 | Per-HP reward curves + recommended HP values + apply-to-all-configs cell |
| `Sensitivity Analysis.ipynb` | Exp_7000–7179 | Figures A/B/C + DM-ranking table + reward-weight recommendation (Part 3) |
| `Bulletproofing.ipynb` | Exp_8000–8026 | Diagonal-dominance test + Part 1 free analyses from 7xxx data |
| `Experiment 1 Figures.ipynb` | Exp_4xxx, 5xxx, 6xxx | Main paper Figures 2/3 (convergence, training curves) |
| `Experiment 2 Figures.ipynb` | Exp_4xxx, 5xxx, 6xxx | Table I (per-DM per-season metrics) |
| `Experiment 3 Figures.ipynb` | Exp_4xxx, 5xxx, 6xxx | Sustainability indicator (Figure 5) |
| `Discussion Figures.ipynb` | derived data | Discussion-section figures |

## Total experiment count (rerun ranges only)

| Range | Count | Status |
|---|---|---|
| 3xxx (data gen, planned) | 1 | TBD |
| 4xxx (season 1) | 180 | pending HP analysis |
| 5xxx (season 2) | 180 | pending HP analysis |
| 6xxx (season 3) | 180 | pending HP analysis |
| 7xxx (sensitivity) | 180 | pending HP analysis |
| 8xxx (bulletproofing) | 27 | pending HP analysis |
| 9000–9179 (HP tuning v1) | 180 | DQN+REINFORCE done, CMA+ODT pending |
| 9180–9251 (HP tuning v2) | 72 | pending submission |
| **Total** | **1000** | |
