import torch
import torch.optim as optim
import numpy as np
import os
import time
import copy
import pickle
import h5py

from environment.data_loader import load_config_file, save_to_csv
from environment._pathfinding import haversine
from .odt.odt_helpers.utils import format_data, save_to_h5, save_temp_checkpoint
from training_processes.writer_proccess import printer_queue

from decision_makers.pyvrp_agent import generate_path_with_pyvrp

from pyvrp import Model, ProblemData, Client, VehicleType, Depot

from carbontracker.tracker import CarbonTracker

def train_pyvrp(queue, 
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
    Trains a Deep Q-Network (DQN) for Electric Vehicle (EV) routing and charging optimization.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        environment (dict): Class containing information about the electric vehicles.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the Q-networks.
        aggregation_num (int): Aggregation step number for tracking.
        zone_index (int): Index of the current zone being processed.
        seed (int): Seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        args (argparse.Namespace): Command-line arguments.
        fixed_attributes (list, optional): List of fixed attributes for redefining weights in the graph.
        devices (list, optional): list of two devices to run the environment and model, default both are cpu. 
                                 device[0] for environment setting, device[1] for model trainning.
        verbose (bool, optional): Flag to enable detailed logging.
        display_training_times (bool, optional): Flag to display training times for different operations.
        agent_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car
        train_model (bool): True if training the model, False if evaluating
        old_buffers (list, optional): List of old buffers to be used for experience replay.

    Returns:
        tuple: A tuple containing:
            - List of trained Q-network state dictionaries.
            - List of average rewards for each episode.
            - List of average output values for each episode.
    """

    print(f'Running PyVRP')

    # Getting Neural Network parameters
    config_fname = f'experiments/Exp_{experiment_number:04d}/config.yaml'
    nn_c = load_config_file(config_fname)['nn_hyperparameters']
    environment_c = load_config_file(config_fname)['environment_settings']

    start_sequence = environment_c.get('start_sequence', 'static')

    num_episodes = nn_c['num_episodes']
    max_timesteps = environment.max_steps

    tracker = CarbonTracker(epochs=num_episodes, epochs_before_pred=0, monitor_epochs=-1, update_interval=1, verbose=0, ignore_errors=True)

    eps_per_save = int(nn_c['eps_per_save'])

    avg_reward = -np.inf
    avg_rewards = []

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
    
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))

    state_dimension = (environment.num_chargers * 3 * 2) + 6

    num_cars = environment.num_cars

    track_times = False
    last_time = time.time()
    episode_start_time = time.time()

    # Calling log and console printer standardized
    print_l, print_et = printer_queue(queue)

    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None

    # Initialize simulation for the aggregation step
    environment.init_sim(aggregation_num)
    for i in range(num_episodes): # For each episode

        tracker.epoch_start() # Start tracking carbon emissions    

        states  = torch.zeros((num_cars, max_timesteps+1, state_dimension), dtype=dtype, device=device)
        rewards = torch.zeros((num_cars, max_timesteps), dtype=dtype, device=device)
        dones   = torch.zeros((num_cars, max_timesteps), dtype=dtype, device=device)
        
        # Episode includes every car reaching their destination
        environment.reset_episode(chargers, routes, unique_chargers)  
        sim_done = False
        time_start_paths = time.time()

        if zone_index == 0 and track_times:
            now = time.time()
            if last_time is not None:
                print(f"Environment Episode Reset: {now - last_time:.4f}s")
            last_time = now

        new_rewards = []
        list_rewards= []

        while not sim_done:  # Keep going until every EV reaches its destination
            timestep = environment.init_routing()

            print(f"Timestep: {timestep} | Zone: {zone_index} | Aggregation: {aggregation_num} | Episode: {i}")

            start_time_step = time.time()

            if start_sequence == 'random':
                starting_number = rng.integers(0, num_cars)
            else:
                starting_number = 0

            # Build path for each EV
            for car_idx in range(num_cars): # For each car
                car_idx = (starting_number + car_idx) % num_cars

                ########### Starting environment routing
                state_np = environment.reset_agent(car_idx)
                state = torch.tensor(state_np, dtype=dtype, device=device)  # Convert state to tensor
                states[car_idx, timestep] = state  # Save state for each car on states
                t1 = time.time()

                ####### Getting actions from agents

                preset_path = generate_path_with_pyvrp(environment, car_idx)

                t3 = time.time()
                environment.generate_paths(None, fixed_attributes, car_idx, preset_path)

                t4 = time.time()
                if car_idx == 0 and display_training_times:
                    print_l("Get actions", (t3 - t1))
                    print_l("Generate paths in environment", (t4 - t3))

            if num_episodes == 1 and fixed_attributes is None:
                if os.path.isfile(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy'):
                    paths = np.load(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy',\
                                    allow_pickle=True).tolist()

            paths_copy = None
            paths_copy = copy.deepcopy(environment.paths)

            if display_training_times:
                print_et('Get Paths', time_start_paths)

            ########### GET SIMULATION RESULTS ###########

            # Run simulation and get results
            sim_done, timestep_reward, arrived_at_final = environment.simulate_routes()

            dones[:,timestep] = arrived_at_final

            if timestep == 0:
                episode_rewards = torch.unsqueeze(timestep_reward, 0)
            else:
                episode_rewards = torch.cat((episode_rewards, torch.unsqueeze(timestep_reward, 0)), dim=0)
            
            # Train the model only using the average of all timestep rewards
            if nn_c['average_rewards_when_training']: 
                avg_reward = timestep_reward.sum(axis=0) / len(timestep_reward)
                timestep_reward_avg = [avg_reward for _ in timestep_reward]
                rewards[:,timestep] = timestep_reward_avg
            # Train the model using the rewards from it's own experiences
            else:
                rewards[:,timestep] = timestep_reward           

            if timestep >= environment.max_steps:
                raise Exception("MAX TIME-STEPS EXCEEDED!")

        if zone_index == 0 and track_times:
            now = time.time()
            if last_time is not None:
                print(f"Main Timestep Loop (Path Gen & Sim): {now - last_time:.4f}s")
            last_time = now

        # Saving last state for next state
        for car_idx in range(num_cars): # For each car
            states[car_idx, timestep+1] = state  # Save state for each car on states

        avg_reward = episode_rewards.sum(axis=0).mean()
        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed)) 

        if zone_index == 0 and track_times:
            now = time.time()
            if last_time is not None:
                print(f"Epsilon/Reward Update: {now - last_time:.4f}s")
            last_time = now
        
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
        
        if zone_index == 0 and track_times:
            now = time.time()
            if last_time is not None:
                print(f"Metrics Saving (CSV): {now - last_time:.4f}s")
            last_time = now
        
        if avg_reward > best_avg:
            best_avg = avg_reward
            best_paths = paths_copy
            if verbose:
                print_l(f'Zone: {zone_index + 1} - New Best: {best_avg}')

        if verbose:
            et = time.time() - start_time
            to_print =  f"(Agg.: {aggregation_num + 1} - Zone: {zone_index + 1}"+\
                        f" - Episode: {i + 1}/{num_episodes})\t"+\
                        f" et: {int(et // 3600):02d}h{int((et % 3600) // 60):02d}m{int(et % 60):02d}s"+\
                        f"- Avg. Reward {round(float(avg_reward.cpu().numpy()), 3):0.3f} - Time-steps: {timestep}"
            print_l(to_print)

        if zone_index == 0 and track_times:
            now = time.time()
            if last_time is not None:
                print(f"End of Episode Misc & Logging: {now - last_time:.4f}s")
            print(f"--- TOTAL EPISODE TIME: {now - episode_start_time:.4f}s ---")

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

    # np.save(f'outputs/best_paths/route_{zone_index}_seed_{seed}.npy', np.array(best_paths, dtype=object))

    tracker.stop() # Stop tracking carbon emissions

    torch.cuda.empty_cache()  # if using GPU

    return None, avg_rewards, None, None