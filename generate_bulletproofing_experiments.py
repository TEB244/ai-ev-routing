"""
Generate single-term-reward bulletproofing experiments (Exp_8000-8026).

Purpose
-------
The 7xxx sensitivity sweep showed that agent behaviour is invariant to
reward-weight choice within {0, 1, 5, 7, 10}. A skeptical reviewer might
read that as "the agents are not actually optimising the reward."

These 27 experiments test that critique directly: each experiment trains
an agent with a reward that contains exactly ONE term (distance only,
traffic only, or energy only). If the agent's emergent behaviour shifts
to favour whichever term is active, the optimisation machinery works
and the 7xxx invariance is best explained by the narrow {0..10} weight
ratio range rather than by an unresponsive agent.

Reward parameterisation
-----------------------
Same as 7xxx: R = -(w_d * D * f_d + w_T * T * f_t + w_e * E * f_e)
with the unit-conversion scales held at baseline (100, 1, 0.001) and
two of the three weights set to 0:

    distance-only:  (w_d, w_T, w_e) = (1, 0, 0)
    traffic-only:   (w_d, w_T, w_e) = (0, 1, 0)
    energy-only:    (w_d, w_T, w_e) = (0, 0, 1)

Layout (27 experiments, 3 DMs * 3 reward shapes * 3 seeds)
----------------------------------------------------------
| Range       | Model     | Reward |
|-------------|-----------|--------|
| 8000-8002   | DQN       | distance-only |
| 8003-8005   | DQN       | traffic-only  |
| 8006-8008   | DQN       | energy-only   |
| 8009-8011   | REINFORCE | distance-only |
| 8012-8014   | REINFORCE | traffic-only  |
| 8015-8017   | REINFORCE | energy-only   |
| 8018-8020   | CMA       | distance-only |
| 8021-8023   | CMA       | traffic-only  |
| 8024-8026   | CMA       | energy-only   |

Seeds within each triplet: 1234, 5555, 2020 (matching 7xxx).

ODT is intentionally excluded here because its 30h wall time makes the
9-run batch expensive; if the 3-DM result shows behavioural response,
ODT becomes an optional extension rather than a required experiment.
"""

import os
import shutil
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXP_DIR = REPO_ROOT / "experiments"

# Re-use the per-DM templates from the sensitivity generator (paths, SLURM
# resources, GPU/MPS setup, etc.) so the two sets stay in sync.
from generate_sensitivity_experiments import (  # noqa: E402
    make_train_job,
    make_eval_job,
)

# ---------------------------------------------------------------------- layout
MODELS = ["DQN", "REINFORCE", "CMA"]   # ODT deliberately excluded
SEEDS = [1234, 5555, 2020]

# Each reward shape sets exactly one weight to 1.0; the other two to 0.0.
REWARD_SHAPES = [
    ("distance_only", {"distance_weight": 1.0, "traffic_weight": 0.0, "energy_weight": 0.0}),
    ("traffic_only",  {"distance_weight": 0.0, "traffic_weight": 1.0, "energy_weight": 0.0}),
    ("energy_only",   {"distance_weight": 0.0, "traffic_weight": 0.0, "energy_weight": 1.0}),
]

# Unit-conversion scales (held at baseline)
BASELINE_SCALES = {
    "distance_scale": 100.0,
    "traffic_scale":  1.0,
    "energy_scale":   0.001,
}

# See generate_sensitivity_experiments.NN_HYPERPARAM_OVERRIDES.
NN_HYPERPARAM_OVERRIDES = {
    "learning_rate":   1.0e-03,
    "buffer_limit":    1500,
    "discount_factor": 0.99,
}

# Block sizing: 3 reward-shapes * 3 seeds = 9 experiments per model
BLOCK_SIZE = 9

# Template config to clone from. Use 4xxx baselines so episode counts and
# hyperparameters match each DM's main experiments.
TEMPLATE_CONFIG = {
    "DQN":       EXP_DIR / "Exp_4000" / "config.yaml",
    "REINFORCE": EXP_DIR / "Exp_4036" / "config.yaml",
    "CMA":       EXP_DIR / "Exp_4072" / "config.yaml",
}


def make_description(exp_num, model, reward_shape, weights, seed,
                     num_episodes, num_aggregations) -> str:
    return f"""
### Experiment {exp_num}: Bulletproofing - single-term reward ###
-----------------------------------------------------------------
Seed: {seed}
Model: {model}
Season: spring
Number of cars: 100
Number of chargers: 1
Number of episodes: {num_episodes}
Number of aggregations: {num_aggregations}
Average rewards when training: False

Reward shape: {reward_shape}
  distance_weight: {weights['distance_weight']}
  traffic_weight:  {weights['traffic_weight']}
  energy_weight:   {weights['energy_weight']}

Unit-conversion scales (baseline):
  distance_scale: 100.0
  traffic_scale:  1.0
  energy_scale:   0.001

Purpose: test whether agent behaviour shifts under single-term rewards.
A distance-only-trained agent should drive less; a traffic-only-trained
agent should pick less congested stations; etc. Confirming this rules
out the "agent does not optimise the reward" critique that the 7xxx
invariance result could otherwise invite.
"""


def main():
    base_exp = 8000
    summary = []

    for model_idx, model in enumerate(MODELS):
        model_start = base_exp + model_idx * BLOCK_SIZE

        for shape_idx, (shape_name, weights) in enumerate(REWARD_SHAPES):
            shape_start = model_start + shape_idx * len(SEEDS)

            for seed_idx, seed in enumerate(SEEDS):
                exp_num = shape_start + seed_idx
                out_dir = EXP_DIR / f"Exp_{exp_num}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Clone the per-DM template config
                with open(TEMPLATE_CONFIG[model]) as f:
                    cfg = yaml.safe_load(f)

                # Override: seed, weights (single-term), scales (baseline)
                cfg["environment_settings"]["seed"] = seed
                # Apply learning-rate / buffer-limit / discount overrides.
                nn_block = cfg.setdefault("nn_hyperparameters", {})
                for k, v in NN_HYPERPARAM_OVERRIDES.items():
                    nn_block[k] = v
                for k, v in BASELINE_SCALES.items():
                    cfg["environment_settings"][k] = float(v)
                for k, v in weights.items():
                    cfg["environment_settings"][k] = float(v)

                with open(out_dir / "config.yaml", "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=True)

                # Episode counts derived from the cloned config
                if model in ("DQN", "REINFORCE", "ODT"):
                    num_eps = cfg["nn_hyperparameters"]["num_episodes"]
                else:
                    num_eps = cfg["cma_parameters"]["max_generations"]
                num_aggs = cfg["federated_learning_settings"]["aggregation_count"]

                # Description
                desc = make_description(
                    exp_num=exp_num,
                    model=model,
                    reward_shape=shape_name,
                    weights=weights,
                    seed=seed,
                    num_episodes=num_eps,
                    num_aggregations=num_aggs,
                )
                with open(out_dir / "description.txt", "w") as f:
                    f.write(desc)

                # Per-DM SLURM job files (paths, GPU/MPS, mail user, etc.)
                with open(out_dir / "train_job.sh", "w", newline="\n") as f:
                    f.write(make_train_job(exp_num, model))
                with open(out_dir / "eval_job.sh", "w", newline="\n") as f:
                    f.write(make_eval_job(exp_num, model))

                summary.append((exp_num, model, shape_name, seed))

    # Compact summary print
    last_seen = (None, None)
    print(f"Generated {len(summary)} experiments: Exp_{summary[0][0]} - Exp_{summary[-1][0]}")
    print()
    for exp_num, model, shape, seed in summary:
        if (model, shape) != last_seen:
            print(f"  {model:10s} {shape:14s}  starts at Exp_{exp_num}")
            last_seen = (model, shape)


if __name__ == "__main__":
    main()
