import torch
import numpy as np
import os
import time
import copy
import h5py

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from environment.data_loader import load_config_file
from training_processes.writer_proccess import printer_queue
from .odt.odt_helpers.utils import format_data # Assuming this is still needed for offline data
from carbontracker.tracker import CarbonTracker
from decision_makers.bayesian_agent import create_search_space, get_best_action_from_result

def train_bayesian(queue,
                   data_dir,
                   ev_info,
                   experiment_number,
                   chargers, environment,
                   routes, date,
                   action_dim,
                   global_weights, # Note: Not used by Bayesian Optimization
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
                   train_model=True, # Note: Bayesian Opt. doesn't "train" a model, it finds parameters
                   old_buffers=None): # Note: No buffers needed

    """
    Finds optimal weights for EV routing using Bayesian Optimization.
    This function is structured to be a drop-in replacement for train_dqn.
    """

    print(f'Running Bayesian Optimization')

    # Getting parameters from config files
    config_fname = f'experiments/Exp_{experiment_number:04d}/config.yaml'
    # Use 'bopt_hyperparameters' section in your config for bayesian specific settings
    bopt_c = load_config_file(config_fname)['bopt_hyperparameters']
    nn_c = load_config_file(config_fname)['nn_hyperparameters'] # num_episodes is still needed
    environment_c = load_config_file(config_fname)['environment_settings']

    num_episodes = nn_c['num_episodes']
    max_timesteps = environment.max_steps
    carbon_save_interval = environment_c.get('carbon_save_interval', 1)

    # Bayesian Optimization specific parameters
    n_calls_per_step = bopt_c.get('n_calls_per_step', 50)
    n_initial_points = bopt_c.get('n_initial_points', 10)
    search_range_min = bopt_c.get('search_range_min', -5.0)
    search_range_max = bopt_c.get('search_range_max', 5.0)

    tracker = CarbonTracker(epochs=num_episodes, epochs_before_pred=0, monitor_epochs=-1, update_interval=1, verbose=0, ignore_errors=True)
    
    avg_rewards = []
    avg_output_values = [] # This will store the best weights found
    
    # Set seed for reproducibility
    if seed is not None:
        bo_rng = np.random.RandomState(seed)

    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))
    num_cars = environment.num_cars

    # Calling log and console printer standardized
    print_l, print_et = printer_queue(queue)

    # Define the search space for the optimizer
    search_space = create_search_space(action_dim, search_range_min, search_range_max)
    
    @use_named_args(search_space)
    def evaluate_weights(**params):
        # Create a deep copy of the environment for an isolated evaluation run
        env_copy = copy.deepcopy(environment)
        
        # This is the car we are currently finding the best weights for
        current_car_idx = evaluate_weights.car_idx
        
        # Get the weights from the optimizer for the target car
        weights_vector = np.array(list(params.values()))
        optimized_distribution = torch.sigmoid(torch.tensor(weights_vector, dtype=dtype, device=device))

        # Generate paths for ALL cars to create a valid simulation state

        # Create a default "uninformed" distribution for other cars
        dummy_distribution = torch.full_like(optimized_distribution, 0.5)

        for car_idx in range(num_cars):
            # The environment needs a state reset for each car before path generation
            _ = env_copy.reset_agent(car_idx)

            if car_idx == current_car_idx:
                # Use the optimizer's weights for the target car
                env_copy.generate_paths(optimized_distribution, fixed_attributes, car_idx)
            else:
                # Use default/dummy weights for all other cars
                env_copy.generate_paths(dummy_distribution, fixed_attributes, car_idx)
        

        sim_done, timestep_reward, arrived_at_final = env_copy.simulate_routes()
        
        # The cost is still based only on the performance of the car we are optimizing
        cost = -timestep_reward[current_car_idx].item()
        return cost

    # Initialize simulation for the aggregation step
    environment.init_sim(aggregation_num)
    for i in range(num_episodes): # For each episode
        if i % carbon_save_interval == 0:
            tracker.epoch_start()

        actions = torch.zeros((num_cars, max_timesteps, action_dim), dtype=dtype, device=device)
        rewards = torch.zeros((num_cars, max_timesteps), dtype=dtype, device=device)
        
        # Episode includes every car reaching their destination
        environment.reset_episode(chargers, routes, unique_chargers)
        sim_done = False

        while not sim_done:
            timestep = environment.init_routing()
            start_time_step = time.time()

            # Build path for each EV by running a separate optimization for each
            for car_idx in range(num_cars):
                _ = environment.reset_agent(car_idx)
                
                # Set the context for the objective function
                evaluate_weights.car_idx = car_idx

                # Run the Bayesian Optimization to find the best weights for this specific car and timestep
                result = gp_minimize(
                    func=evaluate_weights,
                    dimensions=search_space,
                    n_calls=n_calls_per_step,
                    n_initial_points=n_initial_points,
                    random_state=bo_rng
                )

                # Get the best action (weights) found by the optimizer
                action_values = get_best_action_from_result(result, device=device, dtype=dtype)
                actions[car_idx, timestep] = action_values

                # Apply the best found weights to the main environment to generate the final path
                distribution = torch.sigmoid(action_values)
                environment.generate_paths(distribution, fixed_attributes, car_idx)
                
                if verbose:
                    print_l(f"  (Zone {zone_index+1}, Ep {i+1}, TS {timestep}) Car {car_idx}: Opt. complete. Best cost: {result.fun:.3f}")

            # --- GET SIMULATION RESULTS ---
            sim_done, timestep_reward, arrived_at_final = environment.simulate_routes()

            # Store rewards for this timestep
            rewards[:, timestep] = timestep_reward
            
            if timestep >= environment.max_steps:
                raise Exception("MAX TIME-STEPS EXCEEDED!")

        # --- End of Episode ---
        avg_reward = rewards.sum(axis=1).mean()
        avg_rewards.append((avg_reward.item(), aggregation_num, zone_index, main_seed))

        # Store the mean of the best weights found across all timesteps and cars
        episode_avg_output_values = actions[:, :timestep+1, :].mean(axis=(0, 1))
        avg_output_values.append((episode_avg_output_values.tolist(), i, aggregation_num, zone_index, main_seed))

        # --- Logging and Metrics ---
        station_data, agent_data = environment.get_data()
        queue.put({'tag': 'csv', 'station_data': station_data, 'agent_data': agent_data})
        
        if verbose:
            et = time.time() - start_time_step
            to_print = (f"(Agg: {aggregation_num + 1} - Zone: {zone_index + 1} - Ep: {i + 1}/{num_episodes})\t"
                        f"ET: {int(et // 60):02d}m{int(et % 60):02d}s - Avg. Reward: {avg_reward.item():.3f} - Timesteps: {timestep+1}")
            print_l(to_print)
            
        if i % carbon_save_interval == carbon_save_interval - 1:
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

    tracker.stop()

    weights_to_return = [None] * (1 if agent_by_zone else num_cars)
    buffers_to_return = None

    return weights_to_return, avg_rewards, avg_output_values, buffers_to_return