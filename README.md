# VERDE — Vehicle Evaluation for Routing Decisions and Emissions

VERDE is a simulation and evaluation framework for measuring **both** the routing
performance **and** the end-to-end computational sustainability (training *and*
inference) of AI-based decision-makers (DMs) for electric-vehicle (EV) routing and
charging.

Most EV-routing work reports routing quality in isolation. VERDE instead runs
several AI paradigms inside one high-fidelity environment — real Ontario charging
infrastructure, seasonal battery degradation, dynamic traffic, and heterogeneous EV
energy profiles — and reports routing reward alongside the kWh and kg CO₂e spent to
obtain it.

This repository is the source-code artifact for the paper below.

## Paper

> **Evaluating the Environmental Cost and Performance Trade-offs in AI-Based Electric Vehicle Routing**
> Lucas Hartman, Ethan Pigou, Santiago Gomez-Rosero, Miriam A. M. Capretz
> Department of Electrical and Computer Engineering, Western University,
> London, Ontario, Canada, N6A 5B9

| Author | Contact |
|---|---|
| Lucas Hartman | [lhartma8@uwo.ca](mailto:lhartma8@uwo.ca) |
| Ethan Pigou | [epigou@uwo.ca](mailto:epigou@uwo.ca) |
| Santiago Gomez-Rosero | [sgomezro@uwo.ca](mailto:sgomezro@uwo.ca) |
| Miriam A. M. Capretz | [mcapretz@uwo.ca](mailto:mcapretz@uwo.ca) |

## Decision-makers

Four DMs are compared in the paper:

| DM | Paradigm | Implementation |
|---|---|---|
| **CMA** | Evolution strategies (CMA-ES) | [`decision_makers/cma_agent.py`](decision_makers/cma_agent.py) |
| **DQN** | Value-based deep RL | [`decision_makers/dqn_agent.py`](decision_makers/dqn_agent.py) |
| **REINFORCE** | Policy-gradient RL | [`decision_makers/reinforce_agent.py`](decision_makers/reinforce_agent.py) |
| **ODT** | Online Decision Transformer | [`decision_makers/agent_odt.py`](decision_makers/agent_odt.py) |

The framework is extensible, and three further DMs ship as additional baselines but
are **not** part of the paper's evaluation: `PYVRP`, `BAYESIAN`, and `MPC`. See
[Adding a decision-maker](#adding-a-decision-maker).

## Repository layout

| Path | Contents |
|---|---|
| [`main.py`](main.py) | Entry point. Runs one experiment in training or evaluation mode. |
| [`environment/`](environment/) | Simulation environment: routing, pathfinding, charger data, energy metering, evaluation. |
| [`decision_makers/`](decision_makers/) | The DM policies/agents themselves. |
| [`training_processes/`](training_processes/) | Per-DM training loops, federated aggregation, and the DM dispatch layer. |
| [`experiments/`](experiments/) | One directory per experiment, each holding its `config.yaml`, a description, and SLURM job scripts. See [`experiments/README.md`](experiments/README.md) for the numbering scheme. |
| [`paper_figure_generators/`](paper_figure_generators/) | Notebooks that regenerate every figure and table in the paper. See its [README](paper_figure_generators/README.md). |
| [`paper_figure_generators_v2/`](paper_figure_generators_v2/) | Data-access layer (`v2_data.py`, `build_v2_cache.py`) that the figure notebooks read the revised metrics through. |
| [`drac/`](drac/) | Setup and job-submission helpers for the Digital Research Alliance of Canada (DRAC) clusters. |
| [`_scripts/`](_scripts/) | Build scripts that (re)generate the analysis notebooks from source. |

Top-level `*.py` files not listed above are analysis and batch-management utilities
(progress checking, config generation, statistics). They are described under
[Utilities](#utilities).

## Installation

```sh
git clone https://github.com/TEB244/ai-ev-routing.git
cd ai-ev-routing
```

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins exact versions matching the DRAC cluster environment the
paper's results were produced on. For cluster installs, use the wheel lists in
[`drac/`](drac/) instead and follow [`drac/DRAC_Setup.md`](drac/DRAC_Setup.md).

## Running an experiment

Every run is identified by an experiment number, which resolves to
`experiments/Exp_<NNNN>/config.yaml`. To train:

```sh
python main.py -e 4000 -d /path/to/metrics/Exp -verb True -server Local
```

To evaluate a trained model:

```sh
python main.py -e 4000 -d /path/to/metrics/Exp -eval True -server Local
```

| Flag | Meaning |
|---|---|
| `-e`, `--experiments_list` | Experiment number to run (resolves to `experiments/Exp_<NNNN>/config.yaml`). |
| `-d`, `--data_dir` | Output prefix for metrics. The run writes to `<data_dir>_<NNNN>/`. |
| `-c`, `--number_processors` | Number of worker processes. |
| `-g`, `--list_gpus` | GPU indices to use, e.g. `-g 0 1`. |
| `-eval` | Evaluation mode instead of training. |
| `-verb`, `--verbose` | Verbose logging. |
| `-profile` | Run under cProfile, writing `profile.out`. |
| `-server` | `DRAC` (default) or `Local`. Selects which `save_path_metrics` entry is used. |

Each experiment directory also contains ready-made SLURM scripts:

```sh
sbatch experiments/Exp_4000/train_job.sh
sbatch experiments/Exp_4000/eval_job.sh
```

> **Note.** Training metrics are written to the path given by `-d` (on the cluster,
> a scratch directory), **not** into this repository. The repository holds
> experiment *definitions*; the raw metrics they produce are far too large to
> version and are not distributed here.

## Configuration

Each experiment is defined by a single `experiments/Exp_<NNNN>/config.yaml`, grouped
into these blocks:

| Block | Controls |
|---|---|
| `algorithm_settings` | Which DM to use (`algorithm`), and whether one agent serves a whole zone or each car. |
| `environment_settings` | Zone coordinates and radius, season, EV models and their charge/consumption profiles, fleet size, charger count, traffic noise, action dimension, seed. |
| `nn_hyperparameters` | Shared neural-network hyperparameters: learning rate, discount factor, layer sizes, buffer limit, epsilon schedule, episode counts. |
| `cma_parameters` | CMA-ES specific: initial sigma, population dimension, max generations. |
| `odt_hyperparameters` | ODT specific: context length `K`, embedding dimension, layers/heads, return-to-go targets, pretrain/online iteration counts. |
| `federated_learning_settings` | Aggregation count and the city / zone / model mixing weights. |
| `eval_config` | Evaluation-mode behaviour, what gets saved, and the metrics output paths per server. |

To create a new experiment, copy an existing directory and edit its `config.yaml`,
or use [`create_config.py`](create_config.py). Whole batches are produced by the
`generate_*_experiments.py` scripts — see [`experiments/README.md`](experiments/README.md).

## Adding a decision-maker

VERDE dispatches on the `algorithm_settings.algorithm` string, so a new DM needs
three pieces:

1. **The agent.** Add `decision_makers/<name>_agent.py` holding the policy — the
   network or optimizer, and its action selection. Follow an existing agent
   (`dqn_agent.py` is the simplest) for the expected interface.

2. **The training loop.** Add `training_processes/<name>.py` exporting
   `train_<name>(...)`. It must accept the same positional arguments as the existing
   loops (see [`training_processes/reinforce_v2.py`](training_processes/reinforce_v2.py))
   and return the 4-tuple:

   ```python
   local_weights_per_agent, avg_rewards, avg_output_values, new_buffers
   ```

   Returning weights in the same structure as the other DMs is what lets federated
   aggregation in [`training_processes/federated_learning.py`](training_processes/federated_learning.py)
   work unchanged. A DM with no weights to share (for example a classical solver) can
   return an empty structure and run unaggregated.

3. **Registration.** Add a branch in
   [`training_processes/train_selector.py`](training_processes/train_selector.py):

   ```python
   elif algorithm_dm == 'MYDM':
       from training_processes.mydm import train_mydm as train
   ```

Then set `algorithm_settings.algorithm: MYDM` in an experiment config and run it.
Energy and CO₂ accounting is handled by the environment
([`environment/_energy_meter.py`](environment/_energy_meter.py)), so a new DM is
measured on the same footing as the existing ones with no extra work.

## Reproducing the paper's figures and tables

All figures and tables are regenerated by the notebooks in
[`paper_figure_generators/`](paper_figure_generators/). The single entry point is
**`Generate All Paper Figures.ipynb`**, which writes every paper output to `all_v2/`.

The notebooks read raw per-experiment metrics from the lab's shared storage, which
is not distributed with this repository.
[`paper_figure_generators/README.md`](paper_figure_generators/README.md) documents
the expected data layout and how to point the notebooks at a local copy. The
per-experiment YAML configs the notebooks also read *are* version-controlled here.

## Utilities

| Script | Purpose |
|---|---|
| [`create_config.py`](create_config.py) | Create a new experiment config. |
| [`update_config.py`](update_config.py) | Apply a field update across existing configs. |
| [`find_experiments.py`](find_experiments.py) | Find experiments matching an aggregation count and reward type. |
| [`check_progress.py`](check_progress.py) / [`check_batch_progress.py`](check_batch_progress.py) | Report progress, SLURM state, and ETA for a run or a batch. |
| [`validate_batch.py`](validate_batch.py) | Sanity-check a batch of generated configs before submission. |
| [`find_best_base_models.py`](find_best_base_models.py) | Select the best-performing trained models to transfer from. |
| [`compute_wilcoxon.py`](compute_wilcoxon.py) | Wilcoxon signed-rank tests over per-seed final rewards. |
| [`sum_emissions.py`](sum_emissions.py) | Aggregate CO₂/power metrics across experiments. |
| [`fit_inference_regression.py`](fit_inference_regression.py) / [`fit_inference_inproc.py`](fit_inference_inproc.py) | Fit the inference-energy models used in the deployment analysis. |
| [`analyze_paper_alignment.py`](analyze_paper_alignment.py) | Cross-check generated figures/tables against the values claimed in the paper. |
| `generate_*_experiments.py` | Generate whole experiment batches (sensitivity, bulletproofing, HP tuning). |

## Framework overview

![Process Diagram](images/process_diagram.png)
