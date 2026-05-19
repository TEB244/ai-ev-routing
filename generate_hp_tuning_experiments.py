"""
Generate the 180 hyperparameter-tuning experiments (Exp_9000-9179).

Motivation
----------
A reviewer flagged that the original paper never performed systematic
hyperparameter tuning. After fixing the DQN/REINFORCE learning bugs we
also need defensible HP values for the rerun of the 4xxx/5xxx/6xxx
main experiments. This batch tunes each algorithm using a standard
one-at-a-time (OAT) sweep over three hyperparameters per algorithm,
with five values per HP and three seeds per cell.

Design choices (academic best practice + compute manageability)
---------------------------------------------------------------
* One-at-a-time sweeps rather than full grid: 15 experiments per HP
  instead of 5^3 = 125. Justified because we already have the original
  paper's defaults as the centre point, so we only need to characterise
  local sensitivity along each axis.
* 5 log-spaced values per HP (or domain-appropriate spacing) covering
  roughly two orders of magnitude around the centre, so we can see both
  "too low" and "too high" failure modes.
* 3 seeds per cell so we can report mean +/- std and reject ties.
* Reduced compute scope per experiment relative to the main 4xxx
  experiments: 1 zone instead of 4, spring season only, 5000 episodes
  for RL agents (vs 10000 in the main experiments). HP rankings are
  robust to this kind of reduction because we only need the relative
  ordering of HP values, not the absolute final performance.
* Reward weights pinned at the recommended (1, 1, 1) baseline so the
  HP tuning measures the algorithm, not its interaction with reward
  shape.

Layout
------
180 = 4 algorithms x 3 swept HPs x 5 values x 3 seeds.

Block | Range       | Algorithm | Sweep over
------|-------------|-----------|-----------
A     | 9000-9044   | DQN       | learning_rate(0-14), discount_factor(15-29), buffer_limit(30-44)
B     | 9045-9089   | REINFORCE | learning_rate(0-14), discount_factor(15-29), layers_arch(30-44)
C     | 9090-9134   | CMA       | initial_sigma(0-14), population_dimension(15-29), max_generations(30-44)
D     | 9135-9179   | ODT       | learning_rate(0-14), embed_dim(15-29), n_layer(30-44)

Within each 15-experiment block, ordering is value-major then seed-minor:
  offset +0,1,2     value index 0   seeds 1234, 5555, 2020
  offset +3,4,5     value index 1   seeds 1234, 5555, 2020
  offset +6,7,8     value index 2 (centre / current default)
  offset +9,10,11   value index 3
  offset +12,13,14  value index 4

The centre value (offset +6,+7,+8) of each sweep corresponds to the
post-fix default we are calibrating against. Three configs per
(algorithm, seed) tuple are therefore functionally identical (they
all reduce to the algorithm's centre config), so when analysing
results you can collapse the centre cells across HP sweeps.
"""

import copy
import os
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXP_DIR = REPO_ROOT / "experiments"

# Re-use per-algorithm SLURM templates so paths, accounts, MPS, mail
# users, and wall times stay consistent with the rest of the project.
from generate_sensitivity_experiments import (  # noqa: E402
    make_train_job,
    make_eval_job,
)

# -------------------------------------------------------------- layout
SEEDS = [1234, 5555, 2020]
START_EXP = 9000
BLOCK_SIZE = 45  # 3 HPs * 5 values * 3 seeds

# Template configs to clone from (gives us the rest of the experiment
# settings: env, federated, hardware paths, etc.).
TEMPLATE_NUM = {
    "DQN":       4000,
    "REINFORCE": 4036,
    "CMA":       4072,
    "ODT":       4108,
}

# Reward weights and unit-conversion scales are pinned at the
# recommended baseline values (see Sensitivity Analysis notebook).
PINNED_REWARD = {
    "distance_weight": 1.0,
    "traffic_weight":  1.0,
    "energy_weight":   1.0,
    "distance_scale":  100.0,
    "traffic_scale":   1.0,
    "energy_scale":    0.001,
}

# Reduced compute scope: HP tuning is about relative ranking, not
# absolute performance, so shorter runs with one zone suffice.
REDUCED_SCOPE = {
    "num_zones":         1,        # only first zone
    "season":            "spring", # only one season
    # RL runs: 25 aggregations x 200 eps/agg = 5000 episodes (vs 10k)
    "rl_num_aggs":       25,
    # ODT: keep its existing aggregation count but reduce online iters
    # CMA's max_generations is itself a swept HP - leave it alone in
    # the centre case
}

# Network-architecture preset list for REINFORCE layers sweep.
LAYERS_PRESETS = [
    [32, 32],
    [64, 64],
    [128, 64, 64],   # CENTRE (matches existing default)
    [256, 128, 64],
    [512, 256, 128, 64],
]

# ---------------------------------------------------------- sweep table
# (block_name, algorithm, [(hp_name, hp_path, values, centre), ...])
HP_BLOCKS = [
    ("DQN", "DQN", [
        ("learning_rate",
         ("nn_hyperparameters", "learning_rate"),
         [1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2],
         1.0e-3),
        ("discount_factor",
         ("nn_hyperparameters", "discount_factor"),
         [0.90, 0.95, 0.99, 0.995, 0.999],
         0.99),
        ("buffer_limit",
         ("nn_hyperparameters", "buffer_limit"),
         [150, 500, 1500, 5000, 15000],
         1500),
    ]),
    ("REINFORCE", "REINFORCE", [
        ("learning_rate",
         ("nn_hyperparameters", "learning_rate"),
         [1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2],
         1.0e-3),
        ("discount_factor",
         ("nn_hyperparameters", "discount_factor"),
         [0.90, 0.95, 0.99, 0.995, 0.999],
         0.99),
        ("layers_arch",
         ("nn_hyperparameters", "layers"),
         LAYERS_PRESETS,
         LAYERS_PRESETS[2]),
    ]),
    ("CMA", "CMA", [
        ("initial_sigma",
         ("cma_parameters", "initial_sigma"),
         [0.01, 0.05, 0.10, 0.30, 1.00],
         0.10),
        ("population_dimension",
         ("cma_parameters", "population_dimension"),
         [10, 20, 40, 80, 160],
         20),
        ("max_generations",
         ("cma_parameters", "max_generations"),
         [50, 100, 200, 400, 800],
         200),
    ]),
    ("ODT", "ODT", [
        ("learning_rate",
         ("odt_hyperparameters", "learning_rate"),
         [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1],
         1.0e-4),
        ("embed_dim",
         ("odt_hyperparameters", "embed_dim"),
         [64, 128, 256, 512, 1024],
         512),
        ("n_layer",
         ("odt_hyperparameters", "n_layer"),
         [1, 2, 4, 6, 8],
         4),
    ]),
]


def load_template(algo: str) -> dict:
    with open(EXP_DIR / f"Exp_{TEMPLATE_NUM[algo]:04d}" / "config.yaml") as f:
        return yaml.safe_load(f)


def set_nested(cfg: dict, path: tuple, value):
    """cfg["a"]["b"] = value when path = ("a", "b")."""
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def apply_reduced_scope(cfg: dict, algo: str):
    """Trim the cloned template to a smaller-but-still-meaningful run."""
    es = cfg["environment_settings"]
    # 1 zone only
    es["coords"] = es["coords"][:REDUCED_SCOPE["num_zones"]]
    es["season"] = REDUCED_SCOPE["season"]
    # Pin reward weights and scales at baseline (1, 1, 1) so the HP
    # tuning is not confounded by reward-weight choice.
    for k, v in PINNED_REWARD.items():
        es[k] = v
    # Reduce RL training length but keep CMA / ODT as configured
    if algo in ("DQN", "REINFORCE"):
        cfg["federated_learning_settings"]["aggregation_count"] = REDUCED_SCOPE["rl_num_aggs"]
    return cfg


def make_description(exp_num, algo, hp_name, hp_value, centre, seed,
                     reduced_scope: dict) -> str:
    return f"""
### Experiment {exp_num}: HP tuning ###
--------------------------------------
Algorithm:  {algo}
Swept HP:   {hp_name}
HP value:   {hp_value}
HP centre:  {centre}  (post-fix default we are calibrating against)
Seed:       {seed}

Computational scope:
  zones:    {reduced_scope['num_zones']}   (template had 4)
  season:   {reduced_scope['season']}
  RL aggregation count: {reduced_scope['rl_num_aggs']}   (template had 50)

Reward weights pinned at recommended baseline:
  distance_weight: 1.0   traffic_weight: 1.0   energy_weight: 1.0
Scales pinned at unit-conversion defaults:
  distance_scale: 100    traffic_scale: 1      energy_scale: 0.001

Purpose: identify the best value for {hp_name} for {algo}. Combined
with the other two HP sweeps for {algo}, this batch determines the
HP configuration to use when re-running the main 4xxx/5xxx/6xxx
experiments.
"""


def main():
    summary = []
    for block_idx, (block_name, algo, hps) in enumerate(HP_BLOCKS):
        template = load_template(algo)
        block_start = START_EXP + block_idx * BLOCK_SIZE

        for hp_idx, (hp_name, hp_path, values, centre) in enumerate(hps):
            sweep_start = block_start + hp_idx * (len(values) * len(SEEDS))

            for v_idx, value in enumerate(values):
                for s_idx, seed in enumerate(SEEDS):
                    exp_num = sweep_start + v_idx * len(SEEDS) + s_idx

                    cfg = copy.deepcopy(template)
                    cfg = apply_reduced_scope(cfg, algo)

                    # Seed
                    cfg["environment_settings"]["seed"] = seed

                    # For the other two HPs in this algorithm, set them
                    # at their centre value (so each sweep is OAT)
                    for other_name, other_path, other_values, other_centre in hps:
                        set_nested(cfg, other_path, _materialise(other_centre))
                    # Now override the swept HP with this sample value
                    set_nested(cfg, hp_path, _materialise(value))

                    # Bookkeeping for ODT exp_name (so internal ODT IDs
                    # do not point at the cloned template's number)
                    if "odt_hyperparameters" in cfg:
                        cfg["odt_hyperparameters"]["exp_name"] = exp_num
                        if "experiment_number" in cfg["odt_hyperparameters"]:
                            cfg["odt_hyperparameters"]["experiment_number"] = exp_num

                    out_dir = EXP_DIR / f"Exp_{exp_num}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    with open(out_dir / "config.yaml", "w") as f:
                        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)

                    with open(out_dir / "description.txt", "w") as f:
                        f.write(make_description(exp_num, algo, hp_name, value,
                                                  centre, seed, REDUCED_SCOPE))

                    with open(out_dir / "train_job.sh", "w", newline="\n") as f:
                        f.write(make_train_job(exp_num, algo))
                    with open(out_dir / "eval_job.sh", "w", newline="\n") as f:
                        f.write(make_eval_job(exp_num, algo))

                    summary.append((exp_num, algo, hp_name, value, seed))

    n = len(summary)
    print(f"Generated {n} experiments: Exp_{summary[0][0]} - Exp_{summary[-1][0]}")
    print()
    last_pair = (None, None)
    for exp_num, algo, hp_name, value, seed in summary:
        if (algo, hp_name) != last_pair:
            print(f"  {algo:10s} {hp_name:24s}  starts at Exp_{exp_num}")
            last_pair = (algo, hp_name)


def _materialise(v):
    """Make sure values get serialised as concrete YAML scalars (no
    weird tuple objects or numpy types)."""
    if isinstance(v, (list, tuple)):
        return list(v)
    return v


if __name__ == "__main__":
    main()
