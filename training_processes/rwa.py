import torch
import torch.optim as optim
import numpy as np
import os
import time
import copy
import pickle
import h5py

from decision_makers.rwa_agent import initialize, agent_learn, get_actions, save_model
from environment.data_loader import load_config_file
from environment._pathfinding import haversine
from .odt.odt_helpers.utils import format_data, save_to_h5, save_temp_checkpoint
from training_processes.writer_proccess import printer_queue


def train_rwa(queue,
              data_dir,
              ev_info,
              experiment_number,
              chargers, environment,
              routes, date,
              action_dim,
              global_weights,
              aggregation_num,
              zone_index,
              seed,
              main_seed,
              device,
              agent_by_zone,
              variant,
              args,
              fixed_attributes=None,
              verbose=False,
              display_training_times=False,
              dtype=torch.float32,
              save_offline_data=False,
              train_model=True,
              old_buffers=None):
    """
    Trains a policy using RWA (RL with Attention) for Electric Vehicle (EV) routing and charging optimization.

    Parameters:
        queue (multiprocessing.Queue): Queue for inter-process communication.
        data_dir (str): Directory for saving data.
        ev_info (list): Information about electric vehicles.
        experiment_number (int): Experiment number for tracking.
        chargers (array): Array of charger locations and their properties.
        environment (dict): Class containing information about the electric vehicles.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the RWA networks.
        aggregation_num (int): Aggregation step number for tracking.
        zone_index (int): Index of the current zone being processed.
        seed (int): Seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        device (str): Device to run the training on (e.g., 'cpu', 'cuda:0').
        agent_by_zone (bool): True if using one neural network for each zone, False if using one per car.
        variant (dict): Configuration variant containing hyperparameters.
        args (argparse.Namespace): Command-line arguments.
        fixed_attributes (list, optional): List of fixed attributes for redefining weights in the graph.
        verbose (bool, optional): Flag to enable detailed logging.
        display_training_times (bool, optional): Flag to display training times for different operations.
        dtype (torch.dtype, optional): Data type for tensors.
        save_offline_data (bool, optional): Flag to save offline data for later analysis.
        train_model (bool, optional): True if training the model, False if evaluating.
        old_buffers (list, optional): Previous experience replay buffers.

    Returns:
        tuple: A tuple containing:
            - List of trained RWA network state dictionaries.
            - List of average rewards for each episode.
            - List of average output values for each episode.
            - Experience replay buffers (or None if not using replay).
    """

    print(f'Running RWA (RL with Attention)')

    # Getting Neural Network parameters
    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    nn_c = load_config_file(config_fname)['nn_hyperparameters']
    eval_c = load_config_file(config_fname)['eval_config']
    federated_c = load_config_file(config_fname)['federated_learning_settings']
    environment_c = load_config_file(config_fname)['environment_settings']

    # TODO: Load RWA-specific hyperparameters (attention_hyperparameters)
    # rwa_c = load_config_file(config_fname).get('attention_hyperparameters', {})
    # embed_dim = rwa_c.get('embed_dim', 128)
    # num_heads = rwa_c.get('num_heads', 4)
    # num_layers = rwa_c.get('num_layers', 2)
    # attention_dropout = rwa_c.get('attention_dropout', 0.1)

    start_sequence = environment_c.get('start_sequence', 'static')
    carbon_save_interval = environment_c.get('carbon_save_interval', 1)

    discount_factor = nn_c['discount_factor'] if 'discount_factor' in nn_c else 0.99
    learning_rate = nn_c['learning_rate']
    num_episodes = nn_c['num_episodes']
    max_timesteps = environment.max_steps
    layers = nn_c['layers']
    aggregation_count = federated_c['aggregation_count']

    tracker = None

    epsilon = nn_c['epsilon']
    target_episode_epsilon_frac = nn_c['target_episode_epsilon_frac'] if 'target_episode_epsilon_frac' in nn_c else 0.3

    if eval_c['evaluate_on_diff_zone'] or args.eval:
        target_episode_epsilon_frac = 0.1

    epsilon_decay = 10 ** (-1 / ((num_episodes * aggregation_count) * target_episode_epsilon_frac))
    epsilon = epsilon * epsilon_decay ** (num_episodes * aggregation_num)

    eps_per_save = int(nn_c['eps_per_save']) if 'eps_per_save' in nn_c else 1

    avg_reward = -np.inf
    avg_rewards = []

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))

    state_dimension = (environment.num_chargers * 3 * 2) + 6
    model_indices = environment.info['model_indices']

    rwa_networks = []
    optimizers = []
    num_cars = environment.num_cars

    # Calling log and console printer standardized
    print_l, print_et = printer_queue(queue)

    # TODO: Initialize RWA network(s)
    # if agent_by_zone:
    #     num_agents = 1
    #     rwa_net = initialize(state_dimension, action_dim, layers, device, embed_dim, num_heads, attention_dropout)
    #
    #     if global_weights is not None:
    #         if eval_c['evaluate_on_diff_zone'] or args.eval:
    #             rwa_net.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)])
    #         else:
    #             rwa_net.load_state_dict(global_weights[zone_index])
    #
    #     optimizer = optim.AdamW(rwa_net.parameters(), lr=learning_rate)
    #     rwa_networks.append(rwa_net)
    #     optimizers.append(optimizer)
    # else:
    #     num_agents = num_cars
    #     for agent_ind in range(num_agents):
    #         rwa_net = initialize(state_dimension, action_dim, layers, device, embed_dim, num_heads, attention_dropout)
    #
    #         if global_weights is not None:
    #             if eval_c['evaluate_on_diff_zone'] or args.eval:
    #                 rwa_net.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)][model_indices[agent_ind]])
    #             else:
    #                 rwa_net.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])
    #
    #         optimizer = optim.AdamW(rwa_net.parameters(), lr=learning_rate)
    #         rwa_networks.append(rwa_net)
    #         optimizers.append(optimizer)

    trajectories = []
    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None
    avg_output_values = []

    # TODO: Initialize tensors for storing experiences
    # distributions = torch.zeros((num_episodes, num_cars, max_timesteps, action_dim), dtype=dtype, device=device)
    # actions = torch.zeros((num_episodes, num_cars, max_timesteps, action_dim), dtype=dtype, device=device)
    # states = torch.zeros((num_episodes, num_cars, max_timesteps+1, state_dimension), dtype=dtype, device=device)
    # rewards = torch.zeros((num_episodes, num_cars, max_timesteps), dtype=dtype, device=device)
    # dones = torch.zeros((num_episodes, num_cars, max_timesteps), dtype=dtype, device=device)

    # TODO: Implement training loop
    # - Initialize simulation
    # - Loop through episodes
    # - Sample actions using attention-based policy
    # - Execute actions in environment
    # - Compute rewards
    # - Update policy using attention-weighted gradients
    # - Track metrics

    # Placeholder return values
    weights = []  # TODO: Extract weights from trained networks
    # weights = [rwa_network.cpu().state_dict() for rwa_network in rwa_networks]

    # Return final network states, rewards, outputs, and buffers
    return weights, avg_rewards, avg_output_values, old_buffers
