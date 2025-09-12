import torch
import torch.optim as optim
import numpy as np
import os
import time
import copy
import pickle
import h5py

# Replaced import from dqn_agent with reinforce_agent
from decision_makers.reinforce_agent import initialize, agent_learn, get_actions, save_model
from environment.data_loader import load_config_file
from environment._pathfinding import haversine
from .odt.odt_helpers.utils import format_data, save_to_h5, save_temp_checkpoint
from training_processes.writer_proccess import printer_queue

def train_reinforce(queue,
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
    Trains a policy using REINFORCE for Electric Vehicle (EV) routing and charging optimization.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        environment (dict): Class containing information about the electric vehicles.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the policy networks.
        aggregation_num (int): Aggregation step number for tracking.
        zone_index (int): Index of the current zone being processed.
        seed (int): Seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        args (argparse.Namespace): Command-line arguments.
        fixed_attributes (list, optional): List of fixed attributes for redefining weights in the graph.
        devices (list, optional): list of two devices to run the environment and model, default both are cpu.
        verbose (bool, optional): Flag to enable detailed logging.
        display_training_times (bool, optional): Flag to display training times for different operations.
        agent_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car.
        train_model (bool): True if training the model, False if evaluating.

    Returns:
        tuple: A tuple containing:
            - List of trained policy state dictionaries.
            - List of average rewards for each episode.
            - List of average output values for each episode.
    """

    print(f'Running REINFORCE V2')

    # Getting Neural Network parameters
    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    nn_c = load_config_file(config_fname)['nn_hyperparameters']
    eval_c = load_config_file(config_fname)['eval_config']
    federated_c = load_config_file(config_fname)['federated_learning_settings']
    environment_c = load_config_file(config_fname)['environment_settings']

    start_sequence = environment_c.get('start_sequence', 'static')

    carbon_save_interval = environment_c.get('carbon_save_interval', 1)

    # For policy gradient, discount factor might still be used in returns
    discount_factor = nn_c['discount_factor'] if 'discount_factor' in nn_c else 0.99
    learning_rate = nn_c['learning_rate']
    num_episodes = nn_c['num_episodes']
    max_timesteps = environment.max_steps
    layers = nn_c['layers']
    aggregation_count = federated_c['aggregation_count'] if not args.eval else federated_c['aggregation_count_eval']

    if zone_index == 0:
        from carbontracker.tracker import CarbonTracker
        tracker = CarbonTracker(epochs=num_episodes, epochs_before_pred=0, monitor_epochs=-1, update_interval=1, verbose=0, ignore_errors=True)
    else:
        tracker = None

    epsilon = nn_c['epsilon']
    target_episode_epsilon_frac = nn_c['target_episode_epsilon_frac'] if 'target_episode_epsilon_frac' in nn_c else 0.3

    if eval_c['evaluate_on_diff_zone'] or args.eval:
        target_episode_epsilon_frac = 0.1

    epsilon_decay =  10 ** (-1/((num_episodes * aggregation_count) * target_episode_epsilon_frac))

    epsilon = epsilon * epsilon_decay ** (num_episodes * aggregation_num)

    # For saving metrics
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

    policy_networks = []
    optimizers = []
    num_cars = environment.num_cars

    # Calling log and console printer standardized
    print_l, print_et = printer_queue(queue)

    # Initialize policy network(s)
    if agent_by_zone:
        num_agents = 1
        policy_net = initialize(state_dimension, action_dim, layers, device)

        if global_weights is not None:
            if eval_c['evaluate_on_diff_zone'] or args.eval:
                policy_net.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)])
            else:
                policy_net.load_state_dict(global_weights[zone_index])

        optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)
        
        # Store individual networks
        policy_networks.append(policy_net)
        optimizers.append(optimizer)
    else:
        num_agents = num_cars
        for agent_ind in range(num_agents):
            policy_net = initialize(state_dimension, action_dim, layers, device)

            if global_weights is not None:
                if eval_c['evaluate_on_diff_zone'] or args.eval:
                    policy_net.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)][model_indices[agent_ind]])
                else:
                    policy_net.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])

            optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)

            # Store individual networks/optimizers
            policy_networks.append(policy_net)
            optimizers.append(optimizer)

    trajectories = []
    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None
    avg_output_values = []  # List to store the average values of output neurons for each episode

    distributions = torch.zeros((num_episodes, num_cars, max_timesteps, action_dim), dtype=dtype, device=device)
    actions = torch.zeros((num_episodes, num_cars, max_timesteps, action_dim), dtype=dtype, device=device)
    states  = torch.zeros((num_episodes, num_cars, max_timesteps+1, state_dimension), dtype=dtype, device=device)
    rewards = torch.zeros((num_episodes, num_cars, max_timesteps), dtype=dtype, device=device)
    dones   = torch.zeros((num_episodes, num_cars, max_timesteps), dtype=dtype, device=device)

    # Initialize simulation for the aggregation step
    environment.init_sim(aggregation_num)
    for i in range(num_episodes):

        random_threshold = rng.random((num_episodes, num_cars))

        if tracker is not None and i % carbon_save_interval == 0:
            tracker.epoch_start() # Start tracking carbon emissions

        if save_offline_data:
            trajectories.extend([
                {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'terminals_car': [],
                    'zone': zone_index,
                    'aggregation': aggregation_num,
                    'episode': i,      # Critical: Keep this inside the loop for correct episode tracking
                    'car_idx': car_idx
                }
                for car_idx in range(num_cars)
            ])

        # Reset environment for this episode
        environment.reset_episode(chargers, routes, unique_chargers)
        sim_done = False
        time_start_paths = time.time()

        new_rewards = []
        list_rewards= []

        while not sim_done:
            timestep = environment.init_routing()
            start_time_step = time.time()

            if start_sequence == 'random':
                starting_number = rng.integers(0, num_cars)
            else:
                starting_number = 0

            # Sample actions for each EV
            for car_idx in range(num_cars):
                car_idx = (starting_number + car_idx) % num_cars

                if save_offline_data:
                    car_traj = next((t for t in trajectories if (
                        t['car_idx'] == car_idx and t['zone'] == zone_index and
                        t['aggregation'] == aggregation_num and t['episode'] == i
                    )), None)

                state_np = environment.reset_agent(car_idx)
                state = torch.tensor(state_np, dtype=dtype, device=device)  # Convert state to tensor
                states[i, car_idx, timestep] = state  # Save state for each car on states

                if save_offline_data:
                    car_traj['observations'].append(state_np)

                # Get action distribution from policy
                action_probs = get_actions(state, policy_networks, i, car_idx, device, epsilon, random_threshold, agent_by_zone)  
                
                actions[i, car_idx, timestep] = action_probs 

                if save_offline_data:
                    #Save unmodified action
                    car_traj['actions'].append(distribution.detach().cpu().numpy().tolist()) 

                distribution = torch.sigmoid(action_probs)
                distributions[i, car_idx, timestep] = distribution 

                environment.generate_paths(distribution, fixed_attributes, car_idx)

            # If there's only 1 episode and no fixed attributes, we might load best paths from file
            if num_episodes == 1 and fixed_attributes is None:
                if os.path.isfile(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy'):
                    paths = np.load(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy',\
                                    allow_pickle=True).tolist()

            paths_copy = None
            paths_copy = copy.deepcopy(environment.paths)

            # Track output distribution stats
            episode_avg_output_values = actions[i, :,:timestep,:].mean(axis=(0, 1))
            avg_output_values.append((episode_avg_output_values.tolist(), i,\
                                      aggregation_num, zone_index, main_seed))

            if display_training_times:
                print_et('Get Paths', time_start_paths)

            ########### GET SIMULATION RESULTS ###########

            # Run simulation and get results
            sim_done, timestep_reward, arrived_at_final = environment.simulate_routes()
            
            dones[i,:,timestep] = arrived_at_final

            # Accumulate episode rewards
            if timestep == 0:
                episode_rewards = torch.unsqueeze(timestep_reward, 0)
            else:
                episode_rewards = torch.cat((episode_rewards, torch.unsqueeze(timestep_reward, 0)), dim=0)

            # Train the model only using the average of all timestep rewards
            if nn_c['average_rewards_when_training']: 
                avg_reward = timestep_reward.sum(axis=0) / len(timestep_reward)
                timestep_reward_avg = [avg_reward for _ in timestep_reward]
                rewards[i,:,timestep] = timestep_reward_avg
            # Train the model using the rewards from it's own experiences
            else:
                rewards[i,:,timestep] = timestep_reward

            if save_offline_data:
                arrived = environment.get_odt_info()
                for traj in trajectories:
                    if traj['episode'] == i:
                        traj['terminals'].append(sim_done)
                        car_idx = traj['car_idx']
                        traj['rewards'].append(timestep_reward[car_idx])
                        traj['terminals_car'].append(bool(arrived[car_idx].item()))

            if timestep >= environment.max_steps:
                raise Exception("MAX TIME-STEPS EXCEEDED!")

        if train_model:
            st = time.time()
            for agent_ind in range(num_agents):
                experiences = (states[i, agent_ind, :timestep], actions[i, agent_ind, :timestep], rewards[i, agent_ind, :timestep], dones[i, agent_ind, :timestep])

                if agent_by_zone:
                    agent_learn(experiences, discount_factor, policy_networks[0],\
                                optimizers[0], device)
                else:
                    agent_learn(experiences, discount_factor, policy_networks[agent_ind],\
                                optimizers[agent_ind], device)

            if verbose:
                print_et(f'Spent training', st)


        epsilon *= epsilon_decay  # Decay epsilon
        if train_model:
            epsilon = max(0.1, epsilon) # Minimal learning threshold

        avg_reward = episode_rewards.sum(axis=0).mean()
        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed)) 

        base_path = f'saved_networks/Experiment {experiment_number}'

        if save_offline_data and (i + 1) % eps_per_save == 0:
            metrics_base_path = f"{data_dir}_{experiment_number}"
            dataset_path = f"{metrics_base_path}/data_zone_{zone_index}.h5"
            checkpoint_dir = os.path.join(os.path.dirname(metrics_base_path),\
                                          f"temp/Exp_{experiment_number}_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
        
            # Format current trajectories
            traj_format = format_data(trajectories)
        
            # Save a temp checkpoint for ODT offline data
            temp_path = os.path.join(checkpoint_dir,\
                        f"data_zone_{zone_index}_checkpoint_{(i + 1) // eps_per_save}.tmp.h5")
            with h5py.File(temp_path, 'w') as f:
                zone_grp = f.create_group(f"zone_{zone_index}")
                for i_traj, entry in enumerate(traj_format):
                    traj_grp = zone_grp.create_group(f"traj_{i_traj}")
                    for key, value in entry.items():
                        if isinstance(value, (list, np.ndarray)):
                            traj_grp.create_dataset(key, data=np.array(value))
                        else:
                            traj_grp.attrs[key] = value
        
            # Verify data
            try:
                with h5py.File(temp_path, "r") as f:
                    _ = f[f"zone_{zone_index}"]["traj_0"]["observations"][:5]
            except Exception as e:
                print_l(f"[ERROR] Failed to verify checkpoint (zone {zone_index}, episode {i + 1}): {e}")
                os.remove(temp_path)
                trajectories.clear()
                continue  # Skip appending and move to next episode

        
            # Append to main .h5 dataset incrementally
            with h5py.File(dataset_path, 'a') as main_f, h5py.File(temp_path, 'r') as temp_f:
                main_zone_grp = main_f.require_group(f"zone_{zone_index}")
                temp_zone_grp = temp_f[f"zone_{zone_index}"]
                existing_keys = list(main_zone_grp.keys())
                offset = len(existing_keys)
        
                for i, traj_key in enumerate(temp_zone_grp):
                    traj_data = temp_zone_grp[traj_key]
                    new_grp = main_zone_grp.create_group(f"traj_{offset + i}")
                    for key in traj_data:
                        new_grp.create_dataset(key, data=traj_data[key][:])
                    for attr_key in traj_data.attrs:
                        new_grp.attrs[attr_key] = traj_data.attrs[attr_key]
        
            os.remove(temp_path)
            trajectories.clear()

        ### Saving metrics per episode ###
        station_data, agent_data = environment.get_data()
        # Saving as CSV data using the the writer proccess
        queue.put({
            'tag': 'csv',
            'station_data': station_data,
            'agent_data': agent_data
        })
        station_data = None
        agent_data = None
        
        if avg_reward > best_avg:
            best_avg = avg_reward
            best_paths = paths_copy
            if verbose:
                print_l(f'Zone: {zone_index + 1} - New Best: {best_avg}')

        # Some average intermediate result (avg_ir)
        avg_ir = 0
        # ir_count = 0
        # for distribution in distributions:
        #     for out in distribution:
        #         avg_ir += out
        #         ir_count += 1
        # avg_ir = avg_ir / ir_count if ir_count else 0
     
        if verbose:
            et = time.time() - start_time
            to_print =  f"(Agg.: {aggregation_num + 1} - Zone: {zone_index + 1}"+\
                        f" - Episode: {i + 1}/{num_episodes})\t"+\
                        f" et: {int(et // 3600):02d}h{int((et % 3600) // 60):02d}m{int(et % 60):02d}s"+\
                        f"- Avg. Reward {round(float(avg_reward.cpu().numpy()), 3):0.3f} - Time-steps: {timestep},"+\
                        f" Avg. IR: {round(avg_ir, 3):0.3f} - Epsilon: {round(epsilon, 3):0.3f}"
            print_l(to_print)

        if tracker is not None and i % carbon_save_interval == carbon_save_interval - 1:
            tracker.epoch_end() # End tracking carbon emissions

            try:
                # kWh used in each finished epoch; take the last one
                epoch_kwh = float(tracker.tracker.total_energy_per_epoch()[-1])
                # Average carbon intensity during this run (gCO2/kWh)
                avg_ci = tracker.intensity_updater.average_carbon_intensity()
                episode_co2_g = float(epoch_kwh * avg_ci.carbon_intensity)

                queue.put({
                    'tag': 'sustainability_episode',
                    'kwh': epoch_kwh,              # kWh for this episode
                    'co2': episode_co2_g,          # grams CO2e for this episode
                    'episode': i,
                    'zone_index': zone_index,
                    'aggregation_step': aggregation_num
                })
            except Exception as e:
                # If Carbontracker wasn’t able to read (e.g., permissions/NVML), push a minimal record
                queue.put({
                    'tag': 'sustainability_episode',
                    'kwh': None,
                    'co2': None,
                    'error': f'carbontracker_read_failed: {e}',
                    'episode': i,
                    'zone_index': zone_index,
                    'aggregation_step': aggregation_num
                })

    if tracker is not None:
        tracker.stop() # Stop tracking carbon emissions

    weights = [policy_network.cpu().state_dict() for policy_network in policy_networks]
    del policy_networks, optimizers
    torch.cuda.empty_cache()  # if using GPU

    # Save best paths info
    # np.save(f'outputs/best_paths/route_{zone_index}_seed_{seed}.npy', np.array(best_paths, dtype=object))

    # Return final policy states, rewards, outputs, and metrics (and old buffers since REINFORCE does not use a replay buffer)
    return [net.cpu().state_dict() for net in policy_networks], avg_rewards, avg_output_values, old_buffers