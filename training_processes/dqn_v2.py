import torch
import torch.optim as optim
import numpy as np
import os
import time
import copy
import pickle
import h5py

from decision_makers.dqn_agent import initialize, agent_learn, get_actions, soft_update, save_model
from environment.data_loader import load_config_file, save_to_csv
from environment._pathfinding import haversine
from misc.utils import format_data, save_to_h5, save_temp_checkpoint
from training_processes.writer_proccess import printer_queue

def train_dqn(queue, 
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

    print(f'Running DQN V2')

    # Getting Neural Network parameters
    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    nn_c = load_config_file(config_fname)['nn_hyperparameters']
    eval_c = load_config_file(config_fname)['eval_config']
    federated_c = load_config_file(config_fname)['federated_learning_settings']

    epsilon = nn_c['epsilon']

    discount_factor = nn_c['discount_factor']
    learning_rate= nn_c['learning_rate']
    num_episodes = nn_c['num_episodes']
    batch_size   = int(nn_c['batch_size'])
    buffer_limit = int(nn_c['buffer_limit'])
    max_timesteps = environment.max_steps
    layers = nn_c['layers']
    aggregation_count = federated_c['aggregation_count'] if not args.eval else federated_c['aggregation_count_eval']

    target_network_update_frequency = nn_c['target_network_update_frequency'] if 'target_network_update_frequency' in nn_c else 25

    eps_per_save = int(nn_c['eps_per_save'])
    
    target_episode_epsilon_frac = nn_c['target_episode_epsilon_frac'] if 'target_episode_epsilon_frac' in nn_c else 0.3

    if eval_c['evaluate_on_diff_zone'] or args.eval:
        target_episode_epsilon_frac = 0.1

    # Decay epsilon such that by the target_episode_epsilon_frac * num_episodes it is 0.1
    epsilon_decay =  10 ** (-1/((num_episodes * aggregation_count) * target_episode_epsilon_frac))

    avg_reward = -np.inf
    avg_rewards = []

    # Carry over epsilon from last aggregation
    epsilon = epsilon * epsilon_decay ** (num_episodes * aggregation_num)

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        dqn_rng = np.random.default_rng(seed)
    
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))

    state_dimension = (environment.num_chargers * 3 * 2) + 6

    model_indices = environment.info['model_indices']
    
    q_networks = []
    target_q_networks = []
    optimizers = []

    num_cars = environment.num_cars

    # Calling log and console printer standardized
    print_l, print_et = printer_queue(queue)
    
    if agent_by_zone:  # Use same NN for each zone
        # Initialize networks
        num_agents = 1
        q_network, target_q_network = initialize(state_dimension, action_dim, layers, device) 

        if global_weights is not None:
            if eval_c['evaluate_on_diff_zone'] or args.eval:
                q_network.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)])
                target_q_network.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)])
            else:
                q_network.load_state_dict(global_weights[zone_index])
                target_q_network.load_state_dict(global_weights[zone_index])

        optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate) # Use RMSprop optimizer

        # Store individual networks
        q_networks.append(q_network)
        target_q_networks.append(target_q_network)
        optimizers.append(optimizer)

    else: # Assign unique agent for each car
        num_agents = num_cars
        for agent_ind in range(num_agents):
            # Initialize networks
            q_network, target_q_network = initialize(state_dimension, action_dim, layers, device)  

            if global_weights is not None:
                if eval_c['evaluate_on_diff_zone'] or args.eval:
                    q_network.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)][model_indices[agent_ind]])
                    target_q_network.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)][model_indices[agent_ind]])
                else:
                    q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])
                    target_q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])

            optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate) # Use RMSprop optimizer
            
            # Store individual networks/optimizers
            q_networks.append(q_network)
            target_q_networks.append(target_q_network)
            optimizers.append(optimizer)

    random_threshold = dqn_rng.random((num_episodes, num_cars))


    buffers = [
        ExperienceBuffer(buffer_limit, state_dimension, action_dim, device=device) for _ in range(num_cars)]


    if old_buffers is not None and len(old_buffers) > 0:
        buffers = old_buffers

    trajectories = []
    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None
    avg_output_values = [] # List to store the average values of output neurons for each episode

    # Initialize simulation for the aggregation step
    environment.init_sim(aggregation_num)
    for i in range(num_episodes): # For each episode
        if save_offline_data:
            print(f'Here in line 186, saving offline data')
            trajectories.extend([
                {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'terminals_car': [],
                    'zone': zone_index,
                    'aggregation': aggregation_num,
                    'episode': i, # Critical: Keep this inside the loop for correct episode tracking
                    'car_idx': car_idx
                }
                for car_idx in range(num_cars)
            ])

        distributions = torch.zeros((num_cars, max_timesteps, action_dim), dtype=dtype, device=device)
        actions = torch.zeros((num_cars, max_timesteps, action_dim), dtype=dtype, device=device)
        states  = torch.zeros((num_cars, max_timesteps+1, state_dimension), dtype=dtype, device=device)
        rewards = torch.zeros((num_cars, max_timesteps), dtype=dtype, device=device)
        dones   = torch.zeros((num_cars, max_timesteps), dtype=dtype, device=device)
        
        # Episode includes every car reaching their destination
        environment.reset_episode(chargers, routes, unique_chargers)  
        sim_done = False
        time_start_paths = time.time()

        new_rewards = []
        list_rewards= []

        while not sim_done:  # Keep going until every EV reaches its destination
            timestep = environment.init_routing()
            start_time_step = time.time()

            # Build path for each EV
            for car_idx in range(num_cars): # For each car
                if save_offline_data:
                    # Retrieve car trajectory
                    car_traj = next((t for t in trajectories if (
                        t['car_idx'] == car_idx and t['zone'] == zone_index and
                        t['aggregation'] == aggregation_num and t['episode'] == i
                    )), None)

                ########### Starting environment routing
                state_np = environment.reset_agent(car_idx)
                state = torch.tensor(state_np, dtype=dtype, device=device)  # Convert state to tensor
                states[car_idx, timestep] = state  # Save state for each car on states
                t1 = time.time()

                ####### Getting actions from agents
                
                # Get the action values from the agent
                action_values = get_actions(state, q_networks, random_threshold, epsilon, i,\
                                            car_idx, device, agent_by_zone)  

                t2 = time.time()
                if save_offline_data:
                    #Save state for each car
                    car_traj['observations'].append(state_np)
                    #Save unmodified action
                    car_traj['actions'].append(action_values.detach().cpu().numpy().tolist()) 
                
                # Track outputs before the sigmoid application
                actions[car_idx, timestep] = action_values
                # Apply sigmoid function to the entire tensor
                distribution = torch.sigmoid(action_values)
                # Convert to list and append
                distributions[car_idx, timestep] = distribution 

                t3 = time.time()
                environment.generate_paths(distribution, fixed_attributes, car_idx)

                t4 = time.time()
                if car_idx == 0 and display_training_times:
                    print_l("Get actions", (t2 - t1))
                    print_l("Get distributions", (t3 - t2))
                    print_l("Generate paths in environment", (t4 - t3))

            if num_episodes == 1 and fixed_attributes is None:
                if os.path.isfile(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy'):
                    paths = np.load(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy',\
                                    allow_pickle=True).tolist()

            paths_copy = None
            paths_copy = copy.deepcopy(environment.paths) # Check with Lucas, is still needed?

            # Calculate the average values of the output neurons for this episode
            # episode_avg_output_values = np.mean(actions, axis=0)
            episode_avg_output_values = actions[:,:timestep,:].mean(axis=(0, 1))
            avg_output_values.append((episode_avg_output_values.tolist(), i,\
                                      aggregation_num, zone_index, main_seed)) # Double check, likely error src

            if display_training_times:
                print_et('Get Paths', time_start_paths)

            ########### GET SIMULATION RESULTS ###########

            # Run simulation and get results
            sim_done, timestep_reward, timestep_counter,\
                        arrived_at_final = environment.simulate_routes()

            dones[:,timestep] = arrived_at_final

            if timestep_counter == 0: # Needs double check with Lucas, something is wrong here
                episode_rewards = np.expand_dims(timestep_reward,axis=0)
            else:
                episode_rewards = np.vstack((episode_rewards,timestep_reward))
            
            # Train the model only using the average of all timestep rewards
            if nn_c['average_rewards_when_training']: 
                avg_reward = timestep_reward.sum(axis=0) / len(timestep_reward)
                timestep_reward_avg = [avg_reward for _ in timestep_reward]
                rewards[:,timestep] = timestep_reward_avg
            # Train the model using the rewards from it's own experiences
            else:
                rewards[:,timestep] = timestep_reward

            time_step_time = time.time() - start_time_step

            if save_offline_data:
                arrived = environment.get_odt_info()
                for traj in trajectories:
                    car_idx = traj['car_idx']
                    if traj['episode'] == i:
                        traj['terminals'].append(sim_done)
                        traj['rewards'].append(episode_rewards[-1,car_idx])
                        traj['terminals_car'].append(bool(arrived[car_idx].item()))                

            time_step_time = time.time() - start_time_step

            if timestep_counter >= environment.max_steps:
                raise Exception("MAX TIME-STEPS EXCEEDED!")

        # Saving last state for next state
        for car_idx in range(num_cars): # For each car
            states[car_idx, timestep+1] = state  # Save state for each car on states
        
        ########### STORE EXPERIENCES ###########

        # car_dones = [item for sublist in dones for item in sublist]
        

        # for d in range(len(actions)):

        #     # buffers[d % num_cars].append(Experience(states[d], 
        #     #                                         actions[d],\
        #     #                                         rewards[d],
        #     #                                         states[(d + num_cars) if d + num_cars < len(states) else d],
        #     #                                         True if car_dones[d] == 1 else False))  # Store experience
        #     print(f'line 341 d {d}')
        #     action = actions[d]
        #     next_state = states[d % num_cars]
        #     car_done = True if dones[d % num_cars] == 1 else False
        #     buffers[d % num_cars].add(state[d], action, rewards[d], next_state, car_done)

        for car_idx in range(num_cars):
            state_car  = states[car_idx,:timestep]
            action_car = actions[car_idx,:timestep]
            reward_car = rewards[car_idx,:timestep]
            next_state = states[car_idx,1:timestep+1]
            done_car   = dones[car_idx,:timestep]
            buffers[car_idx].add(state_car, action_car, reward_car, next_state, done_car, timestep)
        

        st = time.time()

        trained = False

        for agent_ind in range(num_cars):
            if len(buffers[agent_ind]) >= batch_size: # Buffer is full enough
                trained = True

                # mini_batch = None
                # mini_batch = dqn_rng.choice(np.array([Experience(exp.state.cpu().numpy(), exp.distribution, exp.reward, exp.next_state.cpu().numpy(), exp.done) if isinstance(exp.state, torch.Tensor) else exp for exp in buffers[agent_ind]], dtype=object), batch_size, replace=False)
                # experiences = map(np.stack, zip(*mini_batch))  # Format experiences

                experiences = buffers[agent_ind].sample(batch_size, dqn_rng)

                # Update networks
                if agent_by_zone:
                    agent_learn(experiences, discount_factor, q_networks[0], target_q_networks[0],\
                                optimizers[0], device)
                else:
                    agent_learn(experiences, discount_factor, q_networks[agent_ind], \
                                target_q_networks[agent_ind], optimizers[agent_ind], device)
        
        et = time.time() - st

        if verbose and trained:
            to_print = f'Trained for {et:.3f}s'
            print_l(to_print)
            

        epsilon *= epsilon_decay  # Decay epsilon
        if train_model:
            epsilon = max(0.1, epsilon) # Minimal learning threshold

        avg_reward = episode_rewards.sum(axis=0).mean()
        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed)) 

        base_path = f'saved_networks/Experiment {experiment_number}'

        if ((i + 1) % target_network_update_frequency == 0) and len(buffers[agent_ind]) >= batch_size:
            to_print = f'Updating target network at episode {i}'
            print_l(to_print)
            if agent_by_zone:                
                soft_update(target_q_networks[0], q_networks[0])

                # Add this before you save your model
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
            else:
                for agent_ind in range(num_cars):
                    soft_update(target_q_networks[agent_ind], q_networks[agent_ind])

                    # Add this before you save your model
                    if not os.path.exists(base_path):
                        os.makedirs(base_path)

        if save_offline_data and (i + 1) % eps_per_save == 0:
            metrics_base_path = f"{eval_c['save_path_metrics'][arg.server]}_{experiment_number}"
            dataset_path = f"{metrics_base_path}/data_zone_{zone_index}.h5"
            checkpoint_dir = os.path.join(os.path.dirname(metrics_base_path),\
                                          f"temp/Exp_{experiment_number}_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
        
            # Format current trajectories
            traj_format = format_data(trajectories)
        
            #Save a temp checkpoint
            temp_path = os.path.join(checkpoint_dir, \
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
        
            try:
                with h5py.File(temp_path, "r") as f:
                    _ = f[f"zone_{zone_index}"]["traj_0"]["observations"][:5]
            except Exception as e:
                print_l(f"[ERROR] Failed to verify checkpoint (zone {zone_index}, episode {i + 1}): {e}")
                os.remove(temp_path)
                trajectories.clear()
                continue

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

        avg_ir = 0
        # ir_count = 0
        # for distribution in distributions:
        #     for out in distribution:
        #         avg_ir += out
        #         ir_count += 1
        # avg_ir /= ir_count


        if verbose:
            et = time.time() - start_time
            to_print =  f"(Agg.: {aggregation_num + 1} - Zone: {zone_index + 1}"+\
                        f" - Episode: {i + 1}/{num_episodes})\t"+\
                        f" et: {int(et // 3600):02d}h{int((et % 3600) // 60):02d}m{int(et % 60):02d}s"+\
                        f"- Avg. Reward {round(avg_reward, 3):0.3f} - Time-steps: {timestep_counter},"+\
                        f" Avg. IR: {round(avg_ir, 3):0.3f} - Epsilon: {round(epsilon, 3):0.3f}"
            print_l(to_print)

    # np.save(f'outputs/best_paths/route_{zone_index}_seed_{seed}.npy', np.array(best_paths, dtype=object))

    weights = [q_network.cpu().state_dict() for q_network in q_networks]
    del q_networks, target_q_networks, optimizers
    torch.cuda.empty_cache()  # if using GPU
    return weights, avg_rewards, avg_output_values, buffers



class ExperienceBuffer:
    def __init__(self, buffer_limit, state_dim, action_dim, device='cpu', dtype=torch.float32):
        self.size = buffer_limit
        self.index = 0
        self.full = False
        self.device = device
        self.dtype = dtype

        self.states = torch.zeros((buffer_limit, state_dim), dtype=dtype, device=device)
        self.actions = torch.zeros((buffer_limit, action_dim), dtype=dtype, device=device)
        self.rewards = torch.zeros((buffer_limit,), dtype=dtype, device=device)
        self.next_states = torch.zeros((buffer_limit, state_dim), dtype=dtype, device=device)
        self.dones = torch.zeros((buffer_limit,), dtype=torch.bool, device=device)

    def add(self, states, actions, rewards, next_states, dones, timestep):
        # Handle wrap-around (circular buffer logic)
        end_index = self.index + timestep
        if end_index <= self.size:
            self.states[self.index:end_index] = states
            self.actions[self.index:end_index] = actions
            self.rewards[self.index:end_index] = rewards
            self.next_states[self.index:end_index] = next_states
            self.dones[self.index:end_index] = dones
        else:
            first_part = self.size - self.index
            second_part = end_index % self.size
    
            # Wraparound for all components
            self.states[self.index:] = states[:first_part]
            self.states[:second_part] = states[first_part:]
    
            self.actions[self.index:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]
    
            self.rewards[self.index:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]
    
            self.next_states[self.index:] = next_states[:first_part]
            self.next_states[:second_part] = next_states[first_part:]
    
            self.dones[self.index:] = dones[:first_part]
            self.dones[:second_part] = dones[first_part:]
    
        self.index = end_index % self.size
        if end_index >= self.size:
            self.full = True

    def sample(self, batch_size, rng):
        max_len = self.size if self.full else self.index
        indices = torch.tensor(rng.choice(max_len, size=batch_size, replace=False), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size if self.full else self.index

