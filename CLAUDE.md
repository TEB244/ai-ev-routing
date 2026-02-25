# CLAUDE.md - AI-EV-Routing Project Guide

## What This Project Is

A federated reinforcement learning research framework for sustainable electric vehicle (EV) routing. Multiple decision-making algorithms are compared within a simulation environment that models real-world traffic, seasonal temperature effects on batteries, and authentic Ontario charging station distributions.

**Paper:** SURE-DM: Sustainable Urban Routing Evaluation of Decision Makers

## How to Run

```bash
# Train experiment on CPU
python main.py -e <exp_number> -verb True

# Train on GPU 0
python main.py -e <exp_number> -g 0 -verb True

# Evaluate a trained model
python main.py -e <exp_number> -g 0 -eval True

# Specify metrics output directory
python main.py -e <exp_number> -d "/path/to/metrics/Exp" -verb True
```

### CLI Arguments
| Flag | Type | Description |
|------|------|-------------|
| `-e` | int | Experiment number (required) |
| `-g` | list[int] | GPU indices (default: CPU) |
| `-d` | str | Metrics output directory |
| `-verb` | bool | Verbose logging |
| `-eval` | bool | Evaluation mode (vs training) |
| `-c` | int | Number of processors |
| `-server` | str | `DRAC` or `Local` |

## Repository Structure

```
ai-ev-routing/
├── main.py                           # Entry point: parses args, launches training/eval
├── create_config.py                  # Generates experiment config permutations
├── requirements.txt                  # Python dependencies
│
├── decision_makers/                  # Neural network / agent implementations
│   ├── dqn_agent.py                 # DQN (Deep Q-Network)
│   ├── reinforce_agent.py           # REINFORCE (policy gradient)
│   ├── rwa_agent.py                 # RWA (RL with Attention) - transformer-based
│   ├── agent_odt.py                 # ODT (Offline Decision Transformer)
│   ├── cma_agent.py                 # CMA-ES (evolutionary)
│   ├── bayesian_agent.py            # Bayesian optimization
│   ├── mpc_agent.py                 # Model Predictive Control
│   ├── pyvrp_agent.py              # Vehicle Routing Problem solver
│   └── _transformer_backbone.py     # GPT-2 backbone (used by ODT)
│
├── training_processes/               # Training loops per algorithm
│   ├── train_selector.py            # Routes to correct trainer based on config
│   ├── dqn_v2.py                    # DQN training (experience replay + target net)
│   ├── reinforce_v2.py              # REINFORCE training (on-policy)
│   ├── rwa.py                       # RWA training (on-policy, attention-based)
│   ├── cma.py, bayesian.py, mpc.py  # Other algorithm trainers
│   ├── federated_learning.py        # Weight aggregation across zones
│   ├── writer_proccess.py           # Multiprocessing logger
│   └── odt/                         # Offline Decision Transformer trainer
│
├── environment/                      # Simulation environment
│   ├── environment_main.py          # Core EnvironmentClass (~1200 lines)
│   ├── _pathfinding.py              # Dijkstra + haversine distance
│   ├── evaluation.py                # Metrics computation + plotting
│   ├── data_loader.py               # YAML config loader
│   └── data/                        # Ontario charger dataset, temperature data
│
├── experiments/                      # One directory per experiment
│   └── Exp_XXXX/
│       ├── config.yaml              # Full experiment configuration
│       ├── description.txt          # Human-readable summary
│       ├── train_job.sh             # SLURM job script (CPU)
│       ├── train_job_gpu.sh         # SLURM job script (GPU)
│       └── eval_job.sh              # SLURM evaluation script
│
└── drac/                            # Digital Research Alliance cluster tools
    ├── generate_jobs.py             # Auto-generate SLURM scripts
    └── run_exp_on_drac.py           # Submit batch jobs
```

## Decision Maker Interface

Every agent in `decision_makers/` exports these functions:

```python
def initialize(state_dim, action_dim, layers, device_agents, **kwargs):
    """Create and return the neural network(s), moved to device."""

def compute_loss(experiences, gamma, network):
    """Compute loss from (states, actions, rewards, dones) tuple."""

def agent_learn(experiences, gamma, network, optimizer, device):
    """One gradient update step."""

def get_actions(state, networks, episode_idx, agent_idx, device, epsilon, random_threshold, nn_by_zone):
    """Select actions using epsilon-greedy policy."""

def save_model(network, filename):
    """Save network state_dict to file."""
```

The training loop in `training_processes/` calls these functions. The `train_selector.py` dispatcher maps `algorithm_settings.algorithm` to the correct trainer module.

## State / Action / Reward Spaces

### State Vector
Constructed in `environment_main.py:1052`. Formula: `state_dim = (num_chargers * 3 * 2) + 6`

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Charger features | `num_chargers * 3 * 2` | Contiguous `[traffic_0, traffic_1, ..., dist_0, dist_1, ...]` (all traffic then all distances) |
| Global context | 6 | `[num_chargers*3, route_dist, num_cars, model_index, temperature, timestep]` |

With `num_chargers=1`: state_dim = 12 (6 charger + 6 global)

States are z-score normalized and rounded to 3 decimal places.

### Action Space
Dimension: `action_dim = num_chargers * 3` (typically 3)

Network outputs go through `sigmoid()` to produce [0,1] weights used by `generate_paths()` to reweight the routing graph: `graph *= distance_weight + traffic * (1 - distance_weight)`. Dijkstra then finds the optimal path.

### Reward
`reward = -(distance_cost + peak_traffic_cost + energy_cost)` (always negative; closer to 0 is better)

## Training Pipeline Flow

```
main.py
  └─ main_loop()
       ├─ Initialize GPU/CPU devices
       ├─ Load experiment config from experiments/Exp_XXXX/config.yaml
       ├─ Create EnvironmentClass per zone
       └─ For each aggregation step:
            ├─ Spawn zone processes (multiprocessing)
            │   └─ train_selector.train_route()
            │        └─ train_<algorithm>()    # e.g., train_rwa()
            │             ├─ Initialize network(s) from global_weights
            │             ├─ Episode loop:
            │             │   ├─ environment.reset_episode()
            │             │   ├─ For each car: reset_agent → get_actions → generate_paths
            │             │   ├─ environment.simulate_routes() → rewards
            │             │   └─ agent_learn(experiences)
            │             └─ Return local weights
            ├─ federated_learning.get_global_weights()  # Aggregate across zones
            └─ Save global weights for next aggregation
```

## Experiment Config Structure

```yaml
algorithm_settings:
  algorithm: RWA          # DQN | REINFORCE | RWA | ODT | CMA | MPC | PYVRP | BAYESIAN
  agent_by_zone: false    # true=1 network/zone, false=1 network/car

environment_settings:
  num_of_cars: 100
  num_of_chargers: 1
  action_dim: 3           # num_chargers * 3
  coords: [[lat,lon],...] # Zone center coordinates (4 zones typical)
  seed: 1234
  season: spring          # Affects temperature → battery efficiency
  max_sim_steps: 50

nn_hyperparameters:
  num_episodes: 200       # Episodes per aggregation step
  learning_rate: 1.0e-05
  discount_factor: 0.999
  epsilon: 1.0            # Initial exploration rate
  layers: [128, 64, 64]   # MLP layer sizes
  batch_size: 75          # DQN replay batch size
  buffer_limit: 150       # DQN replay buffer capacity
  average_rewards_when_training: false

federated_learning_settings:
  aggregation_count: 50   # Total aggregation steps
  zone_multiplier: 0.35
  model_multiplier: 0.50
  city_multiplier: 0.15

# Optional (RWA-specific, defaults used if absent):
attention_hyperparameters:
  embed_dim: 32
  num_heads: 4
  num_layers: 1
  attention_dropout: 0.0
```

## Key Algorithms

### DQN (`dqn_v2.py`)
- Off-policy, experience replay buffer (size=150, batch=75)
- Target network with soft updates (tau=0.001), updated every 25 episodes
- MSE loss on Q-values
- Epsilon-greedy exploration

### REINFORCE (`reinforce_v2.py`)
- On-policy, policy gradient with discounted returns
- Loss: `-log(pi(a|s)) * G_t`
- RMSprop optimizer, gradient clipping max_norm=1.0
- No baseline subtraction

### RWA (`rwa.py` + `rwa_agent.py`)
- Hybrid MLP backbone + cross-attention policy network
- MLP backbone processes full state (same as REINFORCE for guaranteed learning floor)
- Cross-attention: backbone queries per-charger tokens for structured reasoning
- On-policy like REINFORCE, same loss function
- Advantage: attention benefit grows with more chargers; never worse than MLP alone

## Federated Learning

Weights aggregated across zones at three levels:
- **City level** (15%): Global average across all zones
- **Zone level** (35%): Average within each geographic zone
- **Model level** (50%): Average by EV model type (Tesla Y, BYD Song, Tesla 3)

Combined: `w = 0.15 * city + 0.35 * zone + 0.50 * model`

## Common Pitfalls

- `num_chargers=1` means only 3 charger-leg tokens + 1 context = 4 tokens for attention. Attention benefits grow with more chargers.
- DQN config values (`batch_size`, `buffer_limit`) exist in shared `nn_hyperparameters` but are ignored by REINFORCE/RWA.
- Learning rate matters: REINFORCE/RWA use 1e-5, DQN can handle 1e-4.
- The `target_episode_epsilon_frac` field controls epsilon decay schedule; defaults to 0.3 if absent.
- States are z-score normalized in `environment_main.py:1068`, so raw feature magnitudes don't matter.

## DRAC Cluster

Jobs submitted via SLURM. Scripts in `experiments/Exp_XXXX/train_job.sh`:
```bash
sbatch experiments/Exp_4180/train_job_gpu.sh   # GPU training
sbatch experiments/Exp_4180/eval_job.sh         # Evaluation
```

Generate job scripts for batch experiments: `python drac/generate_jobs.py`
