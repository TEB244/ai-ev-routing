"""
Generate the 180 reward-weight sensitivity-analysis experiments (Exp_7000-7179).

Reward parameterisation
-----------------------
Reward = -(w_d * D * f_d + w_T * max_tau * f_t + w_e * E * f_e)

  f_d, f_t, f_e -- unit-conversion scales (held at baseline 100, 1, 0.001).
                   They convert simulator units (degrees, raw counts, Wh)
                   to natural physical units (~km, cars, kWh).

  w_d, w_T, w_e -- tunable sensitivity weights (default 1.0). These are
                   what we sweep here.

Layout
------
180 = 4 models x 3 swept weights x 5 values x 3 seeds (one-at-a-time sweep).

Block | Range       | Model     | Sweep over (the *_weight)
------|-------------|-----------|--------------------------
A     | 7000-7044   | DQN       | distance(0-14), traffic(15-29), energy(30-44)
B     | 7045-7089   | REINFORCE | distance(0-14), traffic(15-29), energy(30-44)
C     | 7090-7134   | CMA       | distance(0-14), traffic(15-29), energy(30-44)
D     | 7135-7179   | ODT       | distance(0-14), traffic(15-29), energy(30-44)

Within each 15-experiment sweep block, ordering is value-major, seed-minor:
  offset   value index     seed
  ------   -----------     --------
  +0,1,2   v0 = 0          1234, 5555, 2020   (ablation: term removed)
  +3,4,5   v1 = 1          1234, 5555, 2020   (baseline)
  +6,7,8   v2 = 5          1234, 5555, 2020
  +9,10,11 v3 = 7          1234, 5555, 2020
  +12,13,14 v4 = 10        1234, 5555, 2020

Each config also writes the (baseline) scales explicitly so the full reward
parameterisation is visible:
  distance_scale = 100, traffic_scale = 1, energy_scale = 0.001
"""

import copy
import os
from pathlib import Path

import yaml

# Resolve paths relative to this script
REPO_ROOT = Path(__file__).resolve().parent
EXP_DIR = REPO_ROOT / "experiments"

# Per-model template experiment numbers (matching aggregation=50, season=spring, seed=1234)
TEMPLATE_FOR_MODEL = {
    "DQN": 4000,
    "REINFORCE": 4036,
    "CMA": 4072,
    "ODT": 4108,
}

# Fixed schedule: model order, sweep order, value order, seed order
MODEL_ORDER = ["DQN", "REINFORCE", "CMA", "ODT"]
SWEEP_ORDER = ["distance_weight", "traffic_weight", "energy_weight"]
SEEDS = [1234, 5555, 2020]

# Same set of values for every weight. 0 ablates the term, 1 is baseline.
SWEEP_VALUES_PER_WEIGHT = [0, 1, 5, 7, 10]
SWEEP_VALUES = {w: SWEEP_VALUES_PER_WEIGHT for w in SWEEP_ORDER}

# Baselines: weights default to 1.0, scales are the unit-conversion factors
# established in the calibration write-up.
BASELINE_WEIGHTS = {
    "distance_weight": 1.0,
    "traffic_weight": 1.0,
    "energy_weight": 1.0,
}
BASELINE_SCALES = {
    "distance_scale": 100.0,
    "traffic_scale": 1.0,
    "energy_scale": 0.001,
}

# Hyperparameter overrides applied on top of each cloned 4xxx template.
# The 4xxx baselines use lr=1e-5 + buffer_limit=150 + discount=0.999, which
# verified locally does NOT learn under the post-fix code path. Bumped to
# values that DO learn at 100-car scale.
NN_HYPERPARAM_OVERRIDES = {
    "learning_rate":   1.0e-03,
    "buffer_limit":    1500,
    "discount_factor": 0.99,
}

START_EXP = 7000


def load_template(template_num: int) -> dict:
    path = EXP_DIR / f"Exp_{template_num:04d}" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Per-model SLURM job templates.
#
# These mirror the existing 4xxx job files for each DM, with two intentional
# adjustments per the supervisor's instruction:
#   * REINFORCE training uses the hartman scratch path (the existing 4036
#     train currently points at sgomezro; we override to hartman so DQN and
#     REINFORCE are consistent).
#   * The existing Exp_4108 eval is a stale copy of the DQN eval (it
#     references Exp_4000 and uses the hartman path). We synthesise a clean
#     ODT eval from the ODT training template with -eval True instead.

# Scratch paths per model
SCRATCH_PATH = {
    "DQN":       "/home/hartman/scratch/metrics/Exp",
    "REINFORCE": "/home/hartman/scratch/metrics/Exp",
    "CMA":       "/home/sgomezro/scratch/metrics/Exp",
    "ODT":       "/home/hartman/scratch/metrics/Exp",   # Lucas took over ODT from Ethan
}


def _dqn_train(exp_num: int, model: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_train
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=13:00:00
#SBATCH --mem=6G

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment {exp_num}"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e {exp_num} -d "{SCRATCH_PATH[model]}" -verb True

"""


def _dqn_eval(exp_num: int, model: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_eval
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --mem=6G


#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting evaluation for experiment {exp_num}"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e {exp_num} -d "{SCRATCH_PATH[model]}" -eval True
    """


def _reinforce_train(exp_num: int, model: str) -> str:
    # Wall time progression: 4036 template was 02:00:00, bumped to 16:00:00
    # after TIMEOUTs at 2h on narval, then bumped again to 21:00:00 after
    # the post-fix code path added overhead that left REINFORCE close to
    # the 16h limit.
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_train
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=21:00:00
#SBATCH --mem=6G


#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment {exp_num}"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e {exp_num} -d "{SCRATCH_PATH[model]}" -verb True

"""


def _cma_train(exp_num: int, model: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_train
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:40:00
#SBATCH --mem=6G


echo "Starting training for experiment {exp_num}"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e {exp_num} -d "{SCRATCH_PATH[model]}"

"""


def _cma_eval(exp_num: int, model: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_eval
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:56:10
#SBATCH --mem=6G


echo "Starting evaluation for experiment {exp_num}"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e {exp_num} -d "{SCRATCH_PATH[model]}" -eval True
    """


def _odt_train(exp_num: int, model: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_train
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=30:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca   # Lucas took over ODT from Ethan

echo "Starting training for experiment {exp_num}"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4

# Activate Nvidia MPS:
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d


python main.py -g 0 -e {exp_num} -d "{SCRATCH_PATH[model]}"
"""


def _odt_eval(exp_num: int, model: str) -> str:
    # Synthesised from the ODT training template since the existing Exp_4108
    # eval is a stale copy of the DQN eval. Same SLURM resources as ODT
    # training, but with -eval True and a shorter wall time.
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_eval
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca   # Lucas took over ODT from Ethan

echo "Starting evaluation for experiment {exp_num}"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4

# Activate Nvidia MPS:
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d


python main.py -g 0 -e {exp_num} -d "{SCRATCH_PATH[model]}" -eval True
"""


TRAIN_JOB_BUILDER = {
    "DQN":       _dqn_train,
    "REINFORCE": _reinforce_train,
    "CMA":       _cma_train,
    "ODT":       _odt_train,
}

EVAL_JOB_BUILDER = {
    "DQN":       _dqn_eval,
    "REINFORCE": _dqn_eval,   # REINFORCE eval matches DQN's format (existing 4036 eval is identical structure)
    "CMA":       _cma_eval,
    "ODT":       _odt_eval,
}


def make_train_job(exp_num: int, model: str) -> str:
    return TRAIN_JOB_BUILDER[model](exp_num, model)


def make_eval_job(exp_num: int, model: str) -> str:
    return EVAL_JOB_BUILDER[model](exp_num, model)


def make_description(
    exp_num: int,
    model: str,
    seed: int,
    num_episodes: int,
    num_aggregations: int,
    swept_var: str,
    swept_value,
    distance_weight,
    traffic_weight,
    energy_weight,
    distance_scale,
    traffic_scale,
    energy_scale,
) -> str:
    return f"""
### Experiment {exp_num}: ###
--------------------------------
Seed: {seed}
Model: {model}
Season: spring
Number of cars: 100
Number of chargers: 1
Number of episodes: {num_episodes}
Number of aggregations: {num_aggregations}
Average rewards when training: False

Sensitivity sweep variable: {swept_var}
Sensitivity sweep value:    {swept_value}

Weights (tunable, swept):
  distance_weight: {distance_weight}
  traffic_weight:  {traffic_weight}
  energy_weight:   {energy_weight}

Unit-conversion scales (held at baseline):
  distance_scale: {distance_scale}
  traffic_scale:  {traffic_scale}
  energy_scale:   {energy_scale}
"""


def main():
    if not EXP_DIR.exists():
        raise SystemExit(f"experiments dir not found at {EXP_DIR}")

    # Cache templates so we only read each once
    templates = {m: load_template(t) for m, t in TEMPLATE_FOR_MODEL.items()}

    manifest_rows = []  # for printing/summary at end

    exp_num = START_EXP
    for model in MODEL_ORDER:
        template = templates[model]
        num_eps = template["nn_hyperparameters"]["num_episodes"]
        num_aggs = template["federated_learning_settings"]["aggregation_count"]

        for swept_var in SWEEP_ORDER:
            for value in SWEEP_VALUES[swept_var]:
                for seed in SEEDS:
                    cfg = copy.deepcopy(template)

                    # Inject seed
                    cfg["environment_settings"]["seed"] = seed

                    # Override the nn_hyperparameters that govern actual
                    # learning behaviour (the 4xxx templates use values that
                    # do not learn under the post-fix code path).
                    nn_block = cfg.setdefault("nn_hyperparameters", {})
                    for k, v in NN_HYPERPARAM_OVERRIDES.items():
                        nn_block[k] = v

                    # Always pin the unit-conversion scales at baseline.
                    for k, v in BASELINE_SCALES.items():
                        cfg["environment_settings"][k] = float(v)

                    # Set weights: baseline 1.0 for everything, then
                    # override the swept weight with this sample value.
                    for k, v in BASELINE_WEIGHTS.items():
                        cfg["environment_settings"][k] = float(v)
                    cfg["environment_settings"][swept_var] = float(value)

                    # Update odt exp_name/experiment_number so any internal
                    # ODT bookkeeping points at this experiment, not the
                    # template's number.
                    if "odt_hyperparameters" in cfg:
                        cfg["odt_hyperparameters"]["exp_name"] = exp_num
                        if "experiment_number" in cfg["odt_hyperparameters"]:
                            cfg["odt_hyperparameters"]["experiment_number"] = exp_num

                    out_dir = EXP_DIR / f"Exp_{exp_num}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Write config
                    with open(out_dir / "config.yaml", "w") as f:
                        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)

                    # Write description
                    desc = make_description(
                        exp_num=exp_num,
                        model=model,
                        seed=seed,
                        num_episodes=num_eps,
                        num_aggregations=num_aggs,
                        swept_var=swept_var,
                        swept_value=value,
                        distance_weight=cfg["environment_settings"]["distance_weight"],
                        traffic_weight=cfg["environment_settings"]["traffic_weight"],
                        energy_weight=cfg["environment_settings"]["energy_weight"],
                        distance_scale=cfg["environment_settings"]["distance_scale"],
                        traffic_scale=cfg["environment_settings"]["traffic_scale"],
                        energy_scale=cfg["environment_settings"]["energy_scale"],
                    )
                    with open(out_dir / "description.txt", "w") as f:
                        f.write(desc)

                    # Write SLURM job files (per-DM templates: paths, account,
                    # GPU/MPS, mail user, time/mem differ by model)
                    with open(out_dir / "train_job.sh", "w", newline="\n") as f:
                        f.write(make_train_job(exp_num, model))
                    with open(out_dir / "eval_job.sh", "w", newline="\n") as f:
                        f.write(make_eval_job(exp_num, model))

                    manifest_rows.append((exp_num, model, swept_var, value, seed))
                    exp_num += 1

    # Sanity: should have generated 180 experiments
    expected = len(MODEL_ORDER) * len(SWEEP_ORDER) * 5 * len(SEEDS)
    assert len(manifest_rows) == expected, (
        f"expected {expected}, got {len(manifest_rows)}"
    )

    # Print compact manifest
    print(f"Generated {len(manifest_rows)} experiments: "
          f"Exp_{START_EXP} - Exp_{exp_num - 1}")
    last_block = (None, None)
    for n, m, var, val, seed in manifest_rows:
        block = (m, var)
        if block != last_block:
            print(f"\n  {m:10s}  {var:15s}  -->  starts at Exp_{n}")
            last_block = block


if __name__ == "__main__":
    main()
