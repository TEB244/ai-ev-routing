"""
Generate the v2 / extension HP-tuning experiments (Exp_9180-9251).

Motivation
----------
The v1 HP-tuning batch (9000-9179, three HPs per algorithm) was a
reasonable starting point but did NOT cover the hyperparameters that
the RL literature most commonly identifies as impactful for each
algorithm class. Specifically:

  * DQN: target_network_update_frequency is cited by Mnih et al. 2015
    as central to DQN stability; the v1 sweep omitted it. Likewise the
    exploration schedule (target_episode_epsilon_frac) is a primary
    knob and was omitted. Also, lr in the v1 sweep ended best at the
    upper edge of the range (1e-2), so the range needs extending.

  * REINFORCE: lr was tied at the upper edge of v1; extending the
    range to confirm whether higher lr remains best.

  * ODT: the v1 sweep tuned learning_rate / embed_dim / n_layer but
    NOT the context length K or the return-to-go (RTG) conditioning,
    which are the two HPs every Decision Transformer paper actually
    tunes (Chen et al. 2021, Zheng et al. 2022).

  * CMA: the v1 sweep already covered the canonical CMA HPs
    (initial_sigma, population_dimension, max_generations). No
    additions needed.

This v2 batch only sweeps HPs that are already config-driven (no
code changes required).

Layout
------
72 experiments split across DQN (42), REINFORCE (6), and ODT (30):

| Range       | Algorithm | Swept HP                          | n values |
|-------------|-----------|-----------------------------------|---------:|
| 9180-9194   | DQN       | target_network_update_frequency   | 5        |
| 9195-9209   | DQN       | target_episode_epsilon_frac       | 5        |
| 9210-9215   | DQN       | learning_rate (extension)         | 2        |
| 9216-9221   | REINFORCE | learning_rate (extension)         | 2        |
| 9222-9236   | ODT       | K (context length)                | 5        |
| 9237-9251   | ODT       | rtg (online_rtg + eval_rtg paired)| 5        |

Within each block: 3 seeds per value, ordering value-major / seed-minor
(matching v1):
   offset +0,1,2     value index 0   seeds 1234, 5555, 2020
   offset +3,4,5     value index 1   seeds 1234, 5555, 2020
   etc

Reduced compute scope per experiment (same as v1): 1 zone, spring only,
25 RL aggregations. Reward weights pinned at (1, 1, 1) baseline.

Owner split for cluster submission
----------------------------------
DQN + REINFORCE (9180-9221, 42 exps)   -> Lucas (hartman scratch)
ODT (9222-9251, 30 exps)               -> Ethan (epigou scratch)
CMA -> nothing here, no v2 sweep needed.
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
START_EXP = 9180

# Template configs to clone from (same as v1).
TEMPLATE_NUM = {
    "DQN":       4000,
    "REINFORCE": 4036,
    "ODT":       4108,
}

# Reward weights and unit-conversion scales pinned at baseline so HP
# tuning is not confounded by reward-weight choice.
PINNED_REWARD = {
    "distance_weight": 1.0,
    "traffic_weight":  1.0,
    "energy_weight":   1.0,
    "distance_scale":  100.0,
    "traffic_scale":   1.0,
    "energy_scale":    0.001,
}

REDUCED_SCOPE = {
    "num_zones":   1,
    "season":      "spring",
    "rl_num_aggs": 25,
}

# (block_name, algorithm, hp_name, hp_path, values, centre)
# A flat list so the layout is explicit; the generator just iterates.
V2_BLOCKS = [
    ("DQN", "DQN",
     "target_network_update_frequency",
     ("nn_hyperparameters", "target_network_update_frequency"),
     [5, 10, 25, 50, 100],
     25),
    ("DQN", "DQN",
     "target_episode_epsilon_frac",
     ("nn_hyperparameters", "target_episode_epsilon_frac"),
     [0.1, 0.2, 0.3, 0.5, 0.7],
     0.3),
    ("DQN", "DQN",
     "learning_rate_ext",
     ("nn_hyperparameters", "learning_rate"),
     [3.0e-2, 1.0e-1],
     1.0e-3),  # centre is the v1 default; we are extending beyond v1's max
    ("REINFORCE", "REINFORCE",
     "learning_rate_ext",
     ("nn_hyperparameters", "learning_rate"),
     [3.0e-2, 1.0e-1],
     1.0e-3),
    ("ODT", "ODT",
     "K",
     ("odt_hyperparameters", "K"),
     [5, 10, 20, 40, 80],
     10),
    ("ODT", "ODT",
     "rtg",
     ("odt_hyperparameters", "online_rtg"),
     [-30, -50, -75, -100, -150],
     -60),
]


def load_template(algo: str) -> dict:
    with open(EXP_DIR / f"Exp_{TEMPLATE_NUM[algo]:04d}" / "config.yaml") as f:
        return yaml.safe_load(f)


def set_nested(cfg: dict, path: tuple, value):
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def apply_reduced_scope(cfg: dict, algo: str):
    es = cfg["environment_settings"]
    es["coords"] = es["coords"][:REDUCED_SCOPE["num_zones"]]
    es["season"] = REDUCED_SCOPE["season"]
    for k, v in PINNED_REWARD.items():
        es[k] = v
    if algo in ("DQN", "REINFORCE"):
        cfg["federated_learning_settings"]["aggregation_count"] = REDUCED_SCOPE["rl_num_aggs"]
    return cfg


# v1 centre values for all HPs per algorithm. Every v2 config sets all of
# these (so non-swept HPs sit at the same operating point as v1) and then
# overrides the swept HP with its sweep value. Without this the 4xxx
# template's lr=1e-5 default would silently apply to non-lr v2 sweeps and
# nothing would learn.
ALGO_CENTRES = {
    "DQN": {
        ("nn_hyperparameters", "learning_rate"):                  1.0e-3,
        ("nn_hyperparameters", "discount_factor"):                0.99,
        ("nn_hyperparameters", "buffer_limit"):                   1500,
        ("nn_hyperparameters", "target_network_update_frequency"): 25,
        ("nn_hyperparameters", "target_episode_epsilon_frac"):    0.3,
    },
    "REINFORCE": {
        ("nn_hyperparameters", "learning_rate"):     1.0e-3,
        ("nn_hyperparameters", "discount_factor"):   0.99,
        ("nn_hyperparameters", "layers"):            [128, 64, 64],
    },
    "ODT": {
        ("odt_hyperparameters", "learning_rate"):  1.0e-4,
        ("odt_hyperparameters", "embed_dim"):      512,
        ("odt_hyperparameters", "n_layer"):        4,
        ("odt_hyperparameters", "n_head"):         4,
        ("odt_hyperparameters", "K"):              10,
        ("odt_hyperparameters", "online_rtg"):     -60,
        ("odt_hyperparameters", "eval_rtg"):       -60,
    },
}


def make_description(exp_num, algo, hp_name, hp_value, centre, seed,
                     reduced_scope: dict) -> str:
    return f"""
### Experiment {exp_num}: HP tuning (v2) ###
-------------------------------------------
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

Purpose (v2 batch): cover hyperparameters that the v1 batch
(Exp_9000-9179) missed but that the RL literature identifies as
impactful for {algo}. Combined with the v1 results, the v2 batch
determines the full set of HP values to apply when re-running the main
4xxx/5xxx/6xxx experiments.
"""


def _materialise(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    return v


def main():
    summary = []
    exp_num = START_EXP

    for block_name, algo, hp_name, hp_path, values, centre in V2_BLOCKS:
        template = load_template(algo)
        for value in values:
            for seed in SEEDS:
                cfg = copy.deepcopy(template)
                cfg = apply_reduced_scope(cfg, algo)
                cfg["environment_settings"]["seed"] = seed

                # Set every HP to its v1 centre, then override the
                # swept HP with the sweep value (CRITICAL: without this,
                # the 4xxx template's lr=1e-5 would silently apply).
                for centre_path, centre_val in ALGO_CENTRES[algo].items():
                    set_nested(cfg, centre_path, _materialise(centre_val))
                set_nested(cfg, hp_path, _materialise(value))

                # ODT RTG is set in two places: online_rtg (training)
                # and eval_rtg (evaluation). Sweep them paired so the
                # conditioning signal is consistent.
                if algo == "ODT" and hp_name == "rtg":
                    set_nested(cfg, ("odt_hyperparameters", "eval_rtg"),
                                _materialise(value))

                # ODT exp_name bookkeeping
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
                exp_num += 1

    n = len(summary)
    print(f"Generated {n} experiments: Exp_{summary[0][0]} - Exp_{summary[-1][0]}")
    print()
    last_pair = (None, None)
    for exp_num, algo, hp_name, value, seed in summary:
        if (algo, hp_name) != last_pair:
            print(f"  {algo:10s} {hp_name:34s}  starts at Exp_{exp_num}")
            last_pair = (algo, hp_name)


if __name__ == "__main__":
    main()
