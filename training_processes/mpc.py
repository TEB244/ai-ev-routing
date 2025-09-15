import torch
import numpy as np
import os
import time
import copy

# Import the MPC agent functions
from decision_makers.mpc_agent import initialize_solver, get_actions
from environment.data_loader import load_config_file
from training_processes.writer_proccess import printer_queue

# Function signature now matches train_dqn
def train_mpc(queue,
              data_dir,
              ev_info,
              experiment_number,
              chargers, environment,
              routes, date,
              action_dim,
              global_weights, # Included for signature consistency
              aggregation_num,
              zone_index,
              seed,
              main_seed,
              device, # Corrected typo from "evice"
              agent_by_zone,
              variant,
              args,
              fixed_attributes=None,
              verbose=False,
              display_training_times=False,
              dtype=torch.float32,
              save_offline_data=False, # Included for signature consistency
              train_model=True, # Included for signature consistency
              old_buffers=None): # Included for signature consistency
    """
    Evaluates an MPC controller for Electric Vehicle (EV) routing.
    This function's signature and structure are based on train_dqn.
    """
    print(f'Running MPC Evaluation')

    # Getting config parameters
    config_fname = f'experiments/Exp_{experiment_number:04d}/config.yaml'
    mpc_c = load_config_file(config_fname)['mpc_hyperparameters']
    
    num_cars = environment.num_cars

    print_l, print_et = printer_queue(queue)

    solver = initialize_solver(solver_name='glpk')

    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))

    start_time = time.time()
    best_avg_reward = float('-inf')
    best_paths = None
    
    # Run for one "episode" as MPC is deterministic and doesn't train over episodes
    num_episodes = 1 

    for i in range(num_episodes):
        # Using aggregation_num from the function arguments
        environment.init_sim(aggregation_num)
        
        environment.reset_episode(chargers, routes, unique_chargers)
        
        sim_done = False
        episode_rewards_tensor = torch.tensor([])

        while not sim_done:
            timestep = environment.init_routing()
            
            for car_idx in range(num_cars):
                state_np = environment.reset_agent(car_idx)
                
                current_state_info = environment.get_mpc_state_info(car_idx)
                env_info = environment.get_mpc_env_info(car_idx)

                preset_path = get_actions(current_state_info, env_info, mpc_c, solver)
                environment.generate_paths(None, fixed_attributes, car_idx, preset_path)

            paths_copy = copy.deepcopy(environment.paths)

            sim_done, timestep_reward, arrived_at_final = environment.simulate_routes()
            
            # Ensure timestep_reward is a tensor for concatenation
            if isinstance(timestep_reward, np.ndarray):
                timestep_reward = torch.from_numpy(timestep_reward)

            if episode_rewards_tensor.numel() == 0:
                episode_rewards_tensor = timestep_reward.clone().detach().unsqueeze(0)
            else:
                episode_rewards_tensor = torch.cat((episode_rewards_tensor, timestep_reward.clone().detach().unsqueeze(0)), dim=0)

            if verbose:
                print_l(f"Zone: {zone_index + 1} - Timestep: {timestep} - Avg Reward: {timestep_reward.mean():.3f}")

        avg_reward = episode_rewards_tensor.sum(axis=0).mean().item()

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
        
        station_data, agent_data = environment.get_data()
        queue.put({
            'tag': 'csv',
            'station_data': station_data,
            'agent_data': agent_data
        })
        
        if verbose:
            et = time.time() - start_time
            to_print =  f"(Agg.: {aggregation_num + 1} - Zone: {zone_index + 1} - MPC Run Complete)\t" + \
                        f"et: {int(et // 3600):02d}h{int((et % 3600) // 60):02d}m{int(et % 60):02d}s" + \
                        f"- Final Avg. Reward {float(best_avg_reward):0.3f}"
            print_l(to_print)

    # Return values now match the structure of train_dqn's return
    # We return dummy values for weights and buffers as MPC is stateless.
    avg_rewards = [(best_avg_reward, aggregation_num, zone_index, main_seed)]
    return [], avg_rewards, [], None