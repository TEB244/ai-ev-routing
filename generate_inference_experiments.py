"""
Generate the 10xxx inference-energy experiments (Exp_10000-10047).

Purpose
-------
These are *inference-only* eval runs: each loads a pretrained model and runs a
single episode, with the number of cars swept over {10, 50, 100, 200}. Running
each car count as its own SLURM job (= its own DRAC-portal power measurement)
lets us regress per-job energy against car count and recover:

    energy(n_cars) = a + b * n_cars
        a (intercept) = fixed cost to spin up + load the model (the "load" cost)
        b (slope)     = marginal inference energy per car

Layout (48 = 4 DMs x 4 car counts x 3 seeds), DM-major / car-major / seed-minor:

    Block | Range         | Model     | car counts (3 seeds each)
    ------|---------------|-----------|---------------------------
    A     | 10000-10011   | DQN       | 10,50,100,200
    B     | 10012-10023   | REINFORCE | 10,50,100,200
    C     | 10024-10035   | CMA       | 10,50,100,200
    D     | 10036-10047   | ODT       | 10,50,100,200

How inference is made "pure"
----------------------------
Each config sets `eval_config.inference_only: true`, which the (patched) code
path honours so eval does forward passes only:
  * main.py        -> passes train_model=False (no gradient step) and loads
                      weights from `eval_config.weights_from_exp`.
  * cma.py         -> runs the trained solution once (no population, no tell()).
  * train_odt.py   -> loads the pretrained model.pt from the source experiment
                      and rolls out with 0 updates / 0 pretrain iters.

Weights are loaded from a *source* experiment (the pretrained model) via
`eval_config.weights_from_exp` -- nothing is copied. Set SOURCE_EXP below to the
trained experiments you want to measure; their saved models must exist under
`saved_networks/Exp_<src>/` (on DRAC, in your scratch saved_networks).

NOTE ON ZONES: configs keep the source's multi-zone `coords` (4 zones for the
4xxx baselines). `num_of_cars` is therefore PER ZONE, so total cars = 4 * value.
The regression is still valid; `fit_inference_regression.py` reports per-config
and per-total-car slopes. ODT additionally *requires* the multi-zone setup
because it loads weights from the "next" zone.

Usage:
    python generate_inference_experiments.py            # generate all DMs found
    python generate_inference_experiments.py --dms DQN  # just one DM
"""

import argparse
import copy
from pathlib import Path

import yaml

# Resolve paths relative to this script
REPO_ROOT = Path(__file__).resolve().parent
EXP_DIR = REPO_ROOT / "experiments"

# ----------------------------------------------------------------------------
# EDIT THESE: the pretrained experiment to load weights from, per DM.
# Each must have a usable saved model under saved_networks/Exp_<src>/.
# Defaults are the per-DM baselines used as sensitivity templates (4xxx); ODT's
# baseline (4108) has no saved global model locally, so we default it to 9999.
# ----------------------------------------------------------------------------
# Best 4xxx base model per DM, selected by find_best_base_models.py on huron
# (metrics_postfix) -- highest mean reward over the last 100 episodes, all
# fully trained (10000 eps; ODT 8000). Re-run that script to refresh.
SOURCE_EXP = {
    "DQN":       4028,   # reward -76.52, seed 5555
    "REINFORCE": 4064,   # reward -85.27, seed 5555
    "CMA":       4103,   # reward -94.38, seed 5555
    "ODT":       4140,   # reward -75.99, seed 2020
}

# Sweep definition
MODEL_ORDER = ["DQN", "REINFORCE", "CMA", "ODT"]
CAR_COUNTS = [10, 50, 100, 200]
SEEDS = [1234, 5555, 2020]
START_EXP = 10000

# Scratch path for the eval-metrics output (-d). All inference jobs run on Narval
# as hartman, so everything writes to the same scratch. (The sensitivity script
# split these by training owner/cluster -- CMA->sgomezro, ODT->links/scratch on
# Rorqual -- but that's irrelevant for these Narval inference runs.)
SCRATCH_PATH = {
    "DQN":       "/home/hartman/scratch/metrics/Exp",
    "REINFORCE": "/home/hartman/scratch/metrics/Exp",
    "CMA":       "/home/hartman/scratch/metrics/Exp",
    "ODT":       "/home/hartman/scratch/metrics/Exp",
}


def load_template(template_num: int) -> dict:
    path = EXP_DIR / f"Exp_{template_num:04d}" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_inference_overrides(cfg: dict, model: str, car_count: int, seed: int,
                              exp_num: int, source_exp: int) -> dict:
    """Turn a cloned training config into a single-episode inference config."""
    env = cfg.setdefault("environment_settings", {})
    env["num_of_cars"] = car_count
    env["seed"] = seed
    env["saving_data_deepness"] = "episode_level"

    cfg.setdefault("algorithm_settings", {})["algorithm"] = model

    # Exactly one aggregation, one episode.
    cfg.setdefault("federated_learning_settings", {})["aggregation_count"] = 1
    nn = cfg.setdefault("nn_hyperparameters", {})
    nn["num_episodes"] = 1
    nn["eps_per_save"] = 1

    # CMA: one generation (the inference path skips the population search anyway).
    if "cma_parameters" in cfg:
        cfg["cma_parameters"]["max_generations"] = 1

    # ODT: no offline pretrain, one online iter, zero gradient updates.
    if "odt_hyperparameters" in cfg:
        odt = cfg["odt_hyperparameters"]
        odt["max_pretrain_iters"] = 0
        odt["max_online_iters"] = 1
        odt["num_updates_per_online_iter"] = 0
        odt["num_online_rollouts"] = 1
        odt["num_eval_episodes"] = 1
        odt["eval_interval"] = 1
        odt["exp_name"] = exp_num
        if "experiment_number" in odt:
            odt["experiment_number"] = exp_num

    # Eval / inference behaviour.
    ev = cfg.setdefault("eval_config", {})
    ev["inference_only"] = True
    ev["weights_from_exp"] = source_exp
    ev["continue_training"] = False
    ev["evaluate_on_diff_seed"] = False
    ev["evaluate_on_diff_zone"] = False
    ev["save_offline_data"] = False
    ev["save_data"] = True
    return cfg


def make_description(exp_num, model, source_exp, car_count, seed, n_zones) -> str:
    return f"""
### Experiment {exp_num}: ###  (inference-energy batch)
--------------------------------
Model: {model}
Mode: inference-only eval (forward passes, no gradient step)
Pretrained weights from: Exp_{source_exp}
Seed: {seed}
Number of cars (per zone): {car_count}
Number of zones: {n_zones}
Total cars: {car_count * n_zones}
Number of episodes: 1
Number of aggregations: 1

Part of the per-car inference-energy regression:
    energy(n_cars) = a + b * n_cars
    a = fixed model-load + spin-up cost, b = energy per car.
"""


def _cpu_eval_job(exp_num: int, model: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_eval
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --mem=6G

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting inference (eval) for experiment {exp_num}"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=2

python main.py -e {exp_num} -d "{SCRATCH_PATH[model]}" -eval True
"""


def _odt_eval_job(exp_num: int, model: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=Exp_{exp_num}_eval
#SBATCH --output=experiments/Exp_{exp_num}/output.log
#SBATCH --error=experiments/Exp_{exp_num}/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting inference (eval) for experiment {exp_num}"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

# Activate Nvidia MPS:
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

python main.py -g 0 -e {exp_num} -d "{SCRATCH_PATH[model]}" -eval True
"""


def make_eval_job(exp_num: int, model: str) -> str:
    return _odt_eval_job(exp_num, model) if model == "ODT" else _cpu_eval_job(exp_num, model)


def main():
    p = argparse.ArgumentParser(description="Generate 10xxx inference-energy experiments.")
    p.add_argument("--dms", nargs="+", default=MODEL_ORDER,
                   help=f"Subset of DMs to generate (default: {MODEL_ORDER}).")
    args = p.parse_args()

    if not EXP_DIR.exists():
        raise SystemExit(f"experiments dir not found at {EXP_DIR}")

    manifest = []
    skipped = []
    exp_num = START_EXP
    for model in MODEL_ORDER:
        source_exp = SOURCE_EXP[model]
        # Reserve this DM's 12-experiment block regardless, so numbering is stable.
        block_start = exp_num
        if model not in args.dms or not (EXP_DIR / f"Exp_{source_exp:04d}" / "config.yaml").exists():
            if model in args.dms:
                skipped.append((model, source_exp))
            exp_num += len(CAR_COUNTS) * len(SEEDS)
            continue

        template = load_template(source_exp)
        n_zones = len(template["environment_settings"]["coords"])

        for car_count in CAR_COUNTS:
            for seed in SEEDS:
                cfg = apply_inference_overrides(
                    copy.deepcopy(template), model, car_count, seed, exp_num, source_exp)

                out_dir = EXP_DIR / f"Exp_{exp_num}"
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "config.yaml", "w") as f:
                    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
                with open(out_dir / "description.txt", "w") as f:
                    f.write(make_description(exp_num, model, source_exp, car_count, seed, n_zones))
                with open(out_dir / "eval_job.sh", "w", newline="\n") as f:
                    f.write(make_eval_job(exp_num, model))

                manifest.append((exp_num, model, source_exp, car_count, seed))
                exp_num += 1
        print(f"  {model:10s} (src Exp_{source_exp}) -> Exp_{block_start}-Exp_{exp_num - 1}")

    print(f"\nGenerated {len(manifest)} experiments.")
    if skipped:
        print("Skipped (source config not found locally -- run on DRAC, or fix SOURCE_EXP):")
        for m, s in skipped:
            print(f"  {m}: source Exp_{s}")


if __name__ == "__main__":
    main()
