import copy
import time
import sys
import torch

def train_route(queue, ev_info, experiment_number, chargers, environment, routes, date, action_dim, global_weights,
                aggregate_step, ind, algorithm_dm, sub_seed, main_seed, args, fixed_attributes, local_weights_list,
                rewards, output_values, barrier, device, verbose, display_training_times, agent_by_zone, variant,
                save_offline_data, train_model, old_buffers, process_buffers, weights_to_save, num_zones):

    """
    Trains a single route for the VRP-CSP problem using reinforcement learning in a multiprocessing environment.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        environment (class): Class containing information about the environment.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the Q-networks.
        aggregate_step (int): Aggregation step number for tracking.
        ind (int): Index of the current process.
        sub_seed (int): Sub-seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        num_of_cars (int): Number of agents (EVs) in the environment.
        num_of_chargers (int): Number of charging stations.
        args (argparse.Namespace): Command-line arguments.
        fixed_attributes (list): List of fixed attributes for redefining weights in the graph.
        local_weights_list (list): List to store the local weights of each agent.
        rewards (list): List to store the average rewards for each episode.
        output_values (list): List to store the average output values for each episode.
        barrier (multiprocessing.Barrier): Barrier for synchronizing multiprocessing tasks.
        verbose (bool): Flag to enable detailed logging.
        display_training_times (bool): Flag to display training times for different operations.
        agent_by_zone (bool): True if using one agent for each zone, and false if using a agent for each car
        train_model (bool): True if training the model, False if evaluating the model

    Returns:
        None
    """

    try:
        # Create a deep copy of the environment for this thread
        chargers_copy = copy.deepcopy(chargers)

        print(f'algorithm dm {algorithm_dm}')

        if algorithm_dm == 'DQN':
            from training_processes.dqn_v2 import train_dqn as train

        elif algorithm_dm == 'ODT':
            from training_processes.odt.train_odt import train_odt as train

        elif algorithm_dm == 'REINFORCE':
            from training_processes.reinforce import train_reinforce as train
        
        elif algorithm_dm == 'CMA':
            from training_processes.cma import train_cma as train
            
        else:
            raise RuntimeError(f'model {algorithm_dm} algorithm not found.')

        local_weights_per_agent, avg_rewards, avg_output_values, new_buffers =\
            train(queue, ev_info, experiment_number, chargers_copy, environment, routes, \
                  date, action_dim, global_weights, aggregate_step, ind, sub_seed, main_seed, str(device), \
                  agent_by_zone, variant, args, fixed_attributes, verbose, display_training_times, torch.float32, \
                  save_offline_data, train_model, old_buffers)

        # Save results of training
        st = time.time()
        rewards.append(avg_rewards)
        output_values.append(avg_output_values)
        process_buffers[ind] = new_buffers
        et = time.time() - st

        if verbose:
            run_mode = 'Evaluating' if args.eval else "Training" 
            with open(f'logs/{date}-{run_mode}_logs.txt', 'a') as file:
                print(f'Spent {et:.3f} seconds saving results', file=file)  # Print saving time with 3 decimal places
            print(f'Spent {et:.3f} seconds saving results')  # Print saving time with 3 decimal places

        if train_model:
            local_weights_list[ind] = local_weights_per_agent
            weights_to_save[ind] = local_weights_per_agent

        print(f"Thread {ind} waiting")

        if train_model and num_zones > 1:
            barrier.wait()  # Wait for all threads to finish before proceeding

    except Exception as e:
        import traceback
        print(f"Error in process {ind} during aggregate step {aggregate_step}: {str(e)}")
        traceback.print_exc()
        sys.exit(1) # Exit the program with a non-zero status

