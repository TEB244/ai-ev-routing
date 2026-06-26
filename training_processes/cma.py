# Adapted by Santiago August 9, 2024
import os
import sys
import time
import copy
import torch
import numpy as np
import torch.multiprocessing as mp

from decision_makers.cma_agent import CMAAgent
from environment.data_loader import save_to_csv, load_config_file
from environment._pathfinding import haversine
from training_processes.writer_proccess import printer_queue


def train_cma(queue,
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
    Trains decision-making agents using the Covariance Matrix Adaptation (CMA) algorithm.

    Parameters:
        chargers (array): Array containing charger configurations.
        environment (Environment): The simulation environment where agents operate.
        routes (array): Array of available routes for agents to navigate.
        date (str): Date string for logging purposes.
        action_dim (int): Dimensionality of the action space for the agents.
        global_weights (array): Pre-trained weights for initializing the agents.
        aggregation_num (int): The current aggregation number in the experiment.
        zone_index (int): Index of the zone being trained in the simulation.
        seed (int): Random seed for reproducibility in agent training.
        main_seed (int): Seed used for the overall experiment setup.
        device (torch.device): Device (CPU/GPU) where the models are trained.
        agent_by_zone (bool): Flag indicating if agents are assigned by zone or by car.
        args (argparse.Namespace): Command-line arguments.
        fixed_attributes (dict): Fixed attributes for initializing the environment.
        verbose (bool): Flag to control the verbosity of output messages.
        display_training_times (bool): Flag to display the total training time after execution.
        dtype (torch.dtype): Data type for the tensors used in training.
        save_offline_data (bool): Flag indicating whether to save data for offline analysis.

    Returns:
        weights_list (list): List containing the trained weights for each agent.
        avg_rewards (list): List of average rewards obtained per generation.
        avg_output_values (ndarray): Array of average output values across generations.
        trajectories (list): Saved trajectories for further analysis.
    """
    start_time = time.time()  # Start timing the training process
    avg_rewards = []  # List to store average rewards for each generation

    # Set seeds for reproducibility, ensuring consistent results across runs
    if seed is not None:
        torch.manual_seed(seed)
        dqn_rng = np.random.default_rng(seed)

    # Extract unique charger configurations and prepare environment-specific variables
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))),
                                         dtype=[('id', int), ('lat', float), ('lon', float)]))
    state_dimension = (environment.num_chargers * 3 * 2) + 5  # Calculate the state dimension
    model_indices = environment.info['model_indices']

    # Getting Neural Network parameters
    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    parameters = load_config_file(config_fname)
    nn_c = parameters['nn_hyperparameters']
    environment_c = parameters['environment_settings']
    population_c = parameters['cma_parameters']
    eval_c = parameters['eval_config']
    eps_per_save = int(nn_c['eps_per_save'])

    # Forward-only inference (10xxx energy batch): run the trained solution once,
    # skipping the CMA-ES population search and the tell() optimizer step.
    inference_only = bool(getattr(args, 'eval', False) and eval_c.get('inference_only', False))
    num_episodes = nn_c['num_episodes'] if (not args.eval or inference_only) else 100

    num_cars = environment_c['num_of_cars']
    num_agents = 1 if agent_by_zone else num_cars  # Determine number of agents based on assignment mode

    run_mode = 'Evaluating' if args.eval else "Training"

    # Only track carbon emissions for the first zone
    carbon_save_interval = environment_c.get('carbon_save_interval', 1)
    if zone_index == -1:
        from carbontracker.tracker import CarbonTracker
        tracker = CarbonTracker(epochs=num_episodes, epochs_before_pred=0, monitor_epochs=-1, update_interval=3, verbose=0, ignore_errors=True)

        if args.server == 'DRAC':
            print(f'Stopping intensity updater')
            if getattr(tracker, "intensity_updater", None) is not None:
                try:
                    tracker.intensity_updater.stop()   # stop background fetches
                    print(f'Intensity updater stopped')
                except Exception:
                    pass
                tracker.intensity_updater = None
    else:
        tracker = None
    # log_path = f'logs/{date}-{run_mode}_logs.txt'
    # Get logging functions
    print_l, print_et = printer_queue(queue)
    
    # Initialize CMA agents
    cma_agents_list = []

    for agent_idx in range(num_agents):
        initial_weights = None
        if global_weights is not None:
            # Assign weights based on whether agents are assigned by zone or by car
            if agent_by_zone:
                initial_weights = global_weights[zone_index]
            else:
                initial_weights = global_weights[zone_index][model_indices[agent_idx]]

        # Create a new CMA agent with the provided parameters and initial weights
        cma_agent = CMAAgent(state_dimension, action_dim, num_cars, seed, agent_idx,\
                             initial_weights, experiment_number, device, dtype)
        cma_agents_list.append(cma_agent)

    # Initialize output values storage
    avg_output_values = torch.zeros((num_episodes, action_dim), device=device)  
    best_avg = float('-inf')  # Track the best average reward encountered
    best_paths = None  # Store the best paths observed
    trained = False  # Track whether training occurred

    # Reset the environment for a new training episode
    environment.reset_episode(chargers, routes, unique_chargers)
    environment.init_sim(aggregation_num)
    
    # --- Forward-only inference (10xxx energy batch) --------------------------
    # Run the trained (loaded) CMA solution for a single rollout, write per-episode
    # metrics, then return. Skips CMA-ES population search, the tell() optimizer
    # step, and the .pkl model save (which would overwrite the pretrained source).
    if inference_only:
        environment.reset_episode(chargers, routes, unique_chargers)
        sim_done = False
        timestep_rewards = None
        while not sim_done:
            environment.init_routing()
            for car_idx in range(num_cars):
                state = environment.reset_agent(car_idx)
                agent_idx = 0 if agent_by_zone else car_idx
                agent = cma_agents_list[agent_idx]
                weights = agent.get_loaded_weights()
                car_route = agent.model(state, weights)
                environment.generate_paths(torch.tensor(car_route, device=device), None, agent_idx)
            sim_done, timestep_rewards, _ = environment.simulate_routes()
        station_data, agent_data = environment.get_data()
        queue.put({'tag': 'csv', 'station_data': station_data, 'agent_data': agent_data})
        reward_val = float(np.mean(timestep_rewards)) if timestep_rewards is not None else 0.0
        avg_rewards.append((reward_val, aggregation_num, zone_index, main_seed))
        torch.cuda.empty_cache()
        return [], avg_rewards, [], None
    # -------------------------------------------------------------------------

    # Save the current state of the environment for later restoration during evolution
    environment.population_mode_store()
    
    population_size = population_c['population_dimension']  # Size of the population for evolution
    #Setting max generations
    
    generation_weights = torch.empty((num_agents, action_dim),device=device)  # Storage for weights per generation


    # Initialize matrices for storing solutions and fitness values during evolution
    matrix_solutions = torch.zeros((population_size, num_agents, action_dim), device=device)
    fitnesses = torch.empty((population_size, num_agents), device=device)

    # Evolution process: Loop over generations to evolve the population
    for generation in range(num_episodes):

        if tracker is not None and (generation % carbon_save_interval) == 0:
            tracker.epoch_start() # Start tracking carbon emissions

        # CMA matrix able to work with 120 K dimensions but no more than that
        # Resetting the matrix if it goes beyond 120K 
        # Matrix has reached max limit? then, restart cma-es model
        max_limit = (generation%2000+1)*population_size*action_dim
        if max_limit >= 120000:
            for agent in cma_agents_list:
                agent.cma_restart()

        # Generate solutions for each agent in the population
        for agent_idx, agent in enumerate(cma_agents_list):
            matrix_solutions[:, agent_idx, :] = agent.get_solutions()

        sim_done = False
        time_start_paths = time.time()

        while not sim_done:  # Keep going until every EV reaches its destination

            timestep = environment.init_routing()
            reward_timestep = 0

            start_time_step = time.time()
    
            # Evaluate each individual in the population
            for pop_idx in range(population_size):
                environment.population_mode_copy_store()  # Restore environment to its stored state
    
                # Simulate the environment for each car
                for car_idx in range(num_cars):
                    # Reset environment for the car #MAYBE A PROBLEM HERE!
                    state = environment.reset_agent(car_idx)  
                    agent_idx = 0 if agent_by_zone else car_idx  # Determine the agent to use
                    agent = cma_agents_list[agent_idx]
                    weights = matrix_solutions[pop_idx, agent_idx, :]  # Get the agent's weights
                    car_route = agent.model(state, weights)  # Get the route from the agent's model
                    # Stack the generated paths in the environment
                    environment.generate_paths(torch.tensor(car_route, device=device), None, agent_idx)  
    
                # Once all cars have routes, simulate routes in the environment and get results
                sim_done, rewards_pop, _ = environment.simulate_routes(population_mode=True)

                if agent_by_zone:
                    fitnesses[pop_idx] = -1 * rewards_pop.sum(axis=0).mean()
                elif 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']: 
                    fitnesses[pop_idx] =  np.array(-1 * rewards_pop.sum(axis=0).mean())*num_cars
                    
                else:
                    fitnesses[pop_idx] = -1 * rewards_pop.sum(axis=0)

                
        # Update the agents based on the fitness of the solutions
        for agent_idx, agent in enumerate(cma_agents_list):
            agent.tell(matrix_solutions[:, agent_idx, :], fitnesses[:, agent_idx].flatten())

        #--- Start simulate route with best CMA agents
        # Get rewards with best solutions after evolving population
        environment.reset_episode(chargers, routes, unique_chargers)

        sim_done = False

        rewards = []
        station_data = []
        agent_data = []
        # Get the model index by using car_models[zone_index][agent_index]
        car_models = np.column_stack([info['model_type'] for info in ev_info]).T
        while not sim_done:  # Keep going until every EV reachewr its destination
            timestep = environment.init_routing()
            start_time_step = time.time()
            for car_idx in range(num_cars):
                state = environment.reset_agent(car_idx)
                agent_idx = 0 if agent_by_zone else car_idx  # Determine the agent to use
                agent = cma_agents_list[agent_idx]
                weights = agent.get_best_solutions()  # Get the best solutions
                car_route = agent.model(state, weights)  # Generate paths based on best weights
                environment.generate_paths(torch.tensor(car_route, device=device), None, agent_idx)
                generation_weights[agent_idx] = weights  # Store the best weights for this generation

            # Simulate the environment with the best solutions
            sim_done, timestep_rewards,_ = environment.simulate_routes()

            if timestep == 0:
                episode_rewards = np.expand_dims(timestep_rewards,axis=0)
            else:
                episode_rewards = np.vstack((episode_rewards,timestep_rewards))

            # Train the model only using the average of all timestep rewards
            if 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']: 
                avg_reward = timestep_rewards.sum(axis=0).mean()
                timestep_rewards_avg = [avg_reward for _ in timestep_rewards]
                rewards.extend(timestep_rewards_avg)
            # Train the model using the rewards from it's own experiences
            else:
                rewards.extend(timestep_rewards)

          
       
        # Saving data per episode
        station_data, agent_data = environment.get_data()
        # Saving as CSV data using the the writer proccess
        queue.put({
            'tag': 'csv',
            'station_data': station_data,
            'agent_data': agent_data
        })

        station_data = None
        agent_data = None
        
        avg_reward = episode_rewards.sum(axis=0).mean().item()
        # Store the average reward
        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed))  
        
        # Print information at the log and command line
        if verbose:            
            to_print = f'(Aggregation: {aggregation_num + 1} Zone: {zone_index + 1} ' +\
                        f'Generation: {generation + 1}/{num_episodes}) -'+\
                        f'avg reward {avg_rewards[-1][0]:.3f}'
            print_et(to_print, start_time)
            
        # Compare each generation's best reward and save scores and actions
        if avg_reward > best_avg:
            best_avg = avg_reward
            if verbose:
                to_print = (f' Zone: {zone_index + 1} Gen: {generation + 1}/{num_episodes}'+\
                            f' - New Best: {best_avg:.3f}')
                print_l(to_print)

        # Store the average weights for the generation
        avg_output_values[generation] = generation_weights.mean(axis=0)
        if tracker is not None and (generation % carbon_save_interval) == carbon_save_interval - 1:
            tracker.epoch_end() # End tracking carbon emissions

            try:
                # kWh used in each finished epoch; take the last one
                epoch_kwh = float(tracker.tracker.total_energy_per_epoch()[-1])

                if args.server == 'DRAC':
                    OFFLINE_CI_G_PER_KWH = 300.0 # gCO2/kWh
                    episode_co2_g = epoch_kwh * OFFLINE_CI_G_PER_KWH
                else:
                    episode_co2_g = tracker.intensity_updater.average_carbon_intensity()

                queue.put({
                    'tag': 'sustainability_episode',
                    'kwh': epoch_kwh, # kWh for this episode
                    'co2': episode_co2_g, # Cannot track on DRAC
                    'episode': generation,
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
                    'episode': generation,
                    'zone_index': zone_index,
                    'aggregation_step': aggregation_num
                })

    # Population evolution ends
    if tracker is not None:
        tracker.stop() # Stop tracking carbon emissions

    # Retrieve and print results for the best population after evolution
    final_rewards = environment.get_rewards(population_mode=True)
    # print(f'Rewards for population evolution: {final_rewards.mean():.3f}'+\
    #       f' after {num_episodes} generations')
    to_print = f'Rewards for population evolution: {final_rewards.mean():.3f}'+\
               f' after {num_episodes} generations'
    print_l(to_print)

    # Save the trained models to disk
    folder_path = 'saved_networks'
    fname = f'{folder_path}/cma_model_{main_seed}_z{zone_index}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, agent in enumerate(cma_agents_list):
        agent.save_model(f'{fname}_agent{idx}.pkl')  # Save each agent's model

   


    ########### STORE EXPERIENCES ########

    if verbose and trained:
        with open(f'logs/{date}-{run_mode}_logs.txt', 'a') as file:
            print(f'Trained for {et:.3f}s', file=file)  # Print training time with 3 decimal places

        print(f'Trained for {et:.3f}s')  # Print training time with 3 decimal places

    trajectories = []
    weights_list = [agent.get_weights() for agent in cma_agents_list]

    # Clean variables
    environment = None
    cma_agents_list = None
    # Clean up resources (e.g., GPU memory)
    torch.cuda.empty_cache()
   
    return weights_list, avg_rewards, avg_output_values, None
  

    