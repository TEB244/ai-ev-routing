import os
import argparse
import warnings
import time
import copy
from datetime import datetime
import numpy as np
import torch
# from codecarbon import EmissionsTracker
import shutil
import pandas as pd
import cProfile
from collections import defaultdict
import torch
warnings.filterwarnings("ignore")

# Importing proprietary modules
from environment.data_loader import *
from environment.evaluation import clear_metrics
from environment.environment_main import EnvironmentClass
from training_processes.federated_learning import get_global_weights
from training_processes.train_selector import train_route
from training_processes.writer_proccess import *

# Setting for multiprocessing using pytorch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
mp.set_start_method('spawn', force=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main_loop(args):

    """
    Trains models for vehicle routing and charging station placement (VRP-CSP).

    Parameters:
        args (argparse.Namespace): Arguments from command line
        args.number_processors (int): Number of processors used to run MERL
        args.list_gpus (list): List of GPU indices to run MERL
        args.experiments_list (int): Experiment number to run
        args.data_dir (str): Directory to save data to
        args.verbose (bool): Whether to print verbose output
        args.eval (bool): Whether to evaluate the model
        args.profile (bool): Whether to profile the code

    Returns:
        None
    """

    ############ Initialization ############
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d_%H-%M')
    print(f'Begining date and time {date}')

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        available_gpus = cuda_visible_devices.split(',')
    else:
        available_gpus = [str(i) for i in range(torch.cuda.device_count())]

    # Initializing GPUs for training
    n_gpus = len(args.list_gpus)
    if n_gpus == 0:
        gpus = ['cpu'] # no gpus assigned, then everything running on cpu
        print(f'Woring with CPUs for all zones, with following configuration for zones and devices:')
    elif n_gpus == 1:
        gpus = [f'cuda:{args.list_gpus[0]}']
        print(f'Woring with one GPU -> {gpus[0]} for all zones, with following configuration for zones and devices:')

    elif (n_gpus > 1) & (n_gpus < torch.cuda.device_count()+1):
        gpus = [f'cuda:{args.list_gpus[i]}' for i in range(n_gpus)]
        print(f'Woring with {n_gpus} GPUs, with following configuration for zones and devices:')
    else:
        raise RuntimeError('Number of GPUs requested higher than available GPUs at server.')

    # Verify that the GPUs requested by the user are available
    for gpu in gpus:
        gpu_index = gpu.replace('cuda:', '')
        if gpu_index not in available_gpus and gpu != 'cpu':
            raise RuntimeError(f"Requested GPU {gpu_index} is not available.")

    verbose = args.verbose

    # Getting experiment number to run
    experiment_number = args.experiments_list
    # Getting into Training or Evaluating mode to run experiments
    run_mode = 'Evaluating' if args.eval else "Training" 

    # Fire up initialization by config file
    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    c = load_config_file(config_fname)
    env_c = c['environment_settings']
    eval_c = c['eval_config']
    algorithm_dm = c['algorithm_settings']['algorithm']
    agent_by_zone= c['algorithm_settings']['agent_by_zone']
    federated_c = c['federated_learning_settings']

    # Data directory where metrics are saved
    data_dir = args.data_dir if args.data_dir else f"{c['eval_config']['save_path_metrics'][args.server]}"
    
    # Whether to continue the training
    load_existing_model = eval_c['continue_training']

    # Directory where logs are saved
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Directory where models are saved (the global weights)
    save_global_path = f'saved_networks/Exp_{experiment_number}/'
    if not os.path.exists(save_global_path):
        os.makedirs(save_global_path)

    metrics_base_path = f'{data_dir}_{experiment_number}'

    print(f"Metrics base path: {metrics_base_path}")

    print("Removing previous metrics if they exist")
    clear_metrics(f"{metrics_base_path}/{'eval' if args.eval else 'train'}/metrics")

    # Subdirectory where metrics are saved
    sub_dir = '/train' if run_mode == 'Training' else '/eval'
    metrics_with_sub_dir = metrics_base_path + sub_dir
    if os.path.exists(metrics_with_sub_dir):
        shutil.rmtree(metrics_with_sub_dir)
    os.makedirs(metrics_with_sub_dir)

    # Carbon emissions directory and path
    emission_output_dir = metrics_base_path + sub_dir
    if not os.path.exists(emission_output_dir):
        os.makedirs(emission_output_dir)


    if algorithm_dm in ["DQN", "PPO", "DDPG", "REINFORCE", "ODT"]: # reinforcement learning
        num_episodes = c['nn_hyperparameters']['num_episodes']
        variant = c
    elif algorithm_dm in ['CMA', 'DENSER', 'NEAT']: # population based
        num_episodes = c['cma_parameters']['max_generations']
        variant = c      
    else:
        variant = None

    # Runing experiment selected
    action_dim = env_c['action_dim'] * env_c['num_of_chargers']

    # Assign GPUs to zones
    n_zones = len(env_c['coords'])
    gpus_size = len(gpus)
    exp_devices = [gpus[i % gpus_size] for i in range(n_zones)]
    if n_gpus == 0:
        devices = ['cpu' for _ in range(n_zones)]
    else:
        for i, gpu in enumerate(exp_devices):
            print(f'Zone {i} with GPU {gpu} - {torch.cuda.get_device_name(gpu)}')
    devices = [gpus[i % gpus_size] for i in range(n_zones)]

    # Get seed for current experiment
    seed = env_c['seed']

    # Retrieve Training or Evaluation mode and Continue or from scrath model training
    if (run_mode == "Evaluating") or (load_existing_model):
        global_weights = torch.load(f'saved_networks/Exp_{experiment_number}/global_weights.pth')

    # Start writer proccess
    metrics_path = f"{metrics_base_path}/{'eval' if args.eval else 'train'}"
    log_path = f'{logs_dir}/{date}-{run_mode}_logs.txt'
    queue = mp.Queue()
    data_level = env_c['saving_data_deepness']
    writer_process = mp.Process(target=multiprocess_writer, 
                                args=(queue, log_path, metrics_path, data_level, verbose))
    writer_process.start()

    print_l, print_et = printer_queue(queue)
    print_l(f"Saving metrics to base path: {metrics_with_sub_dir}", )

    
    # If evaluating on different seed, change the seed using round robin
    if eval_c['evaluate_on_diff_seed'] or args.eval:
        seed_options = [1234, 5555, 2020, 2468, 11110, 4040, 3702, 16665, 6060]
        seed_index = seed_options.index(seed)
        old_seed = seed
        seed = seed_options[(seed_index + 1) % len(seed_options)]
        print_l(f'Running experiments with model trained on seed {old_seed} on new seed {seed}', )
    else:
        print_l(f'Running experiment {experiment_number} in {run_mode} mode with seed -> {seed}', )

    # Creating and seeding a random generator from Numpy
    rng = np.random.default_rng(seed)
    # Generating sub seeds to run on each environment
    chargers_seeds = rng.integers(low=0, high=10000, size=len(env_c['coords']))

    # Initializing enviroment
    environment_list = []
    ev_info = []
    start_time = time.time()
    for area_idx in range(n_zones):
        environment = EnvironmentClass(config_fname, seed, chargers_seeds[area_idx],\
                                       area_idx, args.server, device=devices[area_idx], dtype=torch.float32)

        environment_list.append(environment)
        ev_info.append(environment.get_ev_info())

    # Get EV Info
    print_et('Get EV Info:', start_time)
    start_time = time.time()
    all_routes = [None for route in env_c['coords']]
    for index, (city_lat, city_long) in enumerate(env_c['coords']):
        array_org_angle = rng.random(env_c['num_of_cars'])*2*np.pi # generating a list of random angles 
        all_routes[index] = get_org_dest_coords((city_lat, city_long), env_c['radius'], array_org_angle)
    
    # Get Routes
    print_et('Get Routes:', start_time)
    start_time = time.time()
    chargers = np.zeros(shape=[len(all_routes), env_c['num_of_cars'], env_c['num_of_chargers'] * 3, 3])
    
    for route_id,  route in enumerate(all_routes):
        for agent_id, (org_lat, org_long, dest_lat, dest_long) in enumerate(route):
            data = get_charger_data()
            charger_info = np.c_[data['latitude'].to_list(), data['longitude'].to_list()]
            charger_list = get_charger_list(charger_info, org_lat, org_long,\
                                            dest_lat, dest_long, env_c['num_of_chargers'])
            chargers[route_id][agent_id] = charger_list

    # Get Charging Stations
    print_et('Get Chargers:', start_time)
    print(f"Starting training at {current_datetime.strftime('%Y-%m-%d_%H-%M')}")
    
    if run_mode == "Training":
        print_l(f"Training using {algorithm_dm} - Seed {seed}", )

        print(f"CHARGERS: {len(chargers)}")

        rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
        output_values = []  # Array of [(episode_avg_output_values, episode_number,
                            #aggregation_num, route_index, seed)]
        # Hold the buffers for the previous aggregation step
        old_buffers = [None for _ in range(len(chargers))] 
        global_weights = None


        # Initialize weights_to_save as a manager list to persist between aggregations
        manager = mp.Manager()
        weights_to_save = manager.list([None for _ in range(len(chargers))])

        # Loop through aggregation steps
        for aggregate_step in range(federated_c['aggregation_count']):
            try:
                # # Start tracking emissions
                # tracker = EmissionsTracker(
                #     output_dir=emission_output_dir,
                #     save_to_file=f"emissions.csv",  # Temporary file
                #     tracking_mode='process',
                #     allow_multiple_runs=True,
                #     log_level='error'
                # )
                # tracker.start()

                # Print aggregation step
                agg_print = f"{aggregate_step + 1}/{federated_c['aggregation_count']}"
                print_l(f"\n\n############ Aggregation {agg_print} ############\n\n",)

                
                # Check if we have only one zone - if so, don't use multiprocessing
                if len(chargers) == 1:
                    print("Only one zone detected, running without multiprocessing")
                    local_weights_list = [None]
                    process_rewards = []
                    process_output_values = []
                    process_buffers = [None]
                    
                    # Run directly without multiprocessing
                    train_route(queue, ev_info, experiment_number, chargers[0],\
                                copy.deepcopy(environment_list[0]), all_routes[0], date, 
                                action_dim, global_weights, aggregate_step, 0, algorithm_dm,\
                                chargers_seeds[0], seed, args, eval_c['fixed_attributes'], \
                                local_weights_list, process_rewards,\
                                process_output_values, None, devices[0], verbose, \
                                eval_c['display_training_times'], agent_by_zone, variant,\
                                eval_c['save_offline_data'], True, old_buffers[0],\
                                process_buffers, weights_to_save, len(chargers))
                    
                # If there are multiple zones, use multiprocessing
                else:
                    # Initialize manager for multiprocessing
                    manager = mp.Manager()
                    local_weights_list = manager.list([None for _ in range(len(chargers))])
                    process_rewards = manager.list()
                    process_output_values = manager.list()
                    process_buffers = manager.list([None for _ in range(len(chargers))])

                    # Barrier for synchronization
                    barrier = mp.Barrier(len(chargers))

                    # Initialize processes
                    processes = []
                    for ind, charger_list in enumerate(chargers):
                        # Create arguments tuple for each process
                        args_tuple = (queue, ev_info, experiment_number, charger_list,\
                                  copy.deepcopy(environment_list[ind]), all_routes[ind], date,\
                                  action_dim, global_weights, aggregate_step, ind, algorithm_dm,\
                                  chargers_seeds[ind], seed, args, eval_c['fixed_attributes'],\
                                  local_weights_list, process_rewards,\
                                  process_output_values, barrier, devices[ind], verbose,\
                                  eval_c['display_training_times'], agent_by_zone, variant,\
                                  eval_c['save_offline_data'], True, old_buffers[ind], \
                                  process_buffers, weights_to_save, len(chargers))
                        
                        # Enter the training loop for each zone
                        process = mp.Process(target=train_route, args=args_tuple)
                        processes.append(process)
                        process.start()

                    print("Join Processes")

                    for process in processes:
                        process.join()

                    for p in processes:
                        if p.is_alive():
                            p.terminate()  # Just in case

                rewards = []

                print("Join Weights")

                # Aggregate the weights from all local models
                if algorithm_dm == 'ODT':
                    global_weights = get_global_weights(local_weights_list, ev_info,\
                                                        federated_c['city_multiplier'],\
                                                        federated_c['zone_multiplier'],\
                                                        federated_c['model_multiplier'],\
                                                        agent_by_zone)
                elif algorithm_dm == 'DENSER': 
                    # Cannot aggregate weights for DENSER because architecture is different between agents
                    pass
                else:
                    # Aggregate the weights from all local models
                    global_weights = get_global_weights(local_weights_list, ev_info,\
                                                        federated_c['city_multiplier'],\
                                                        federated_c['zone_multiplier'],\
                                                        federated_c['model_multiplier'],\
                                                        agent_by_zone)


                # Save the global weights
                torch.save(global_weights, f'{save_global_path}/global_weights.pth')

                # Extend the main lists with the contents of the process lists
                sorted_list = sorted([val[0] for sublist in process_rewards for val in sublist])
                
                if sorted_list:
                    print_l('Min and Max rewards for the aggregation step:'+\
                            f'{sorted_list[0], sorted_list[-1]}')
                else:
                    print_l("No rewards found for this aggregation step.")
                rewards.extend(process_rewards)
                output_values.extend(process_output_values)
                old_buffers = list(process_buffers)

            finally:
                aux = None
            #     # Stop tracking emissions
            #     # emissions = tracker.stop()
            #     # print_l(f"Total CO₂ emissions: {emissions} kg")
            #     try:
            #         # Read the temporary emissions report
            #         temp_df = pd.read_csv(f"{emission_output_dir}/emissions.csv")

            #         # Remove the temporary emissions file
            #         os.remove(f"{emission_output_dir}/emissions.csv")

            #         # Add the aggregate_step column
            #         temp_df['aggregate_step'] = aggregate_step

            #         # Determine the write mode based on the aggregate_step
            #         write_mode = 'w' if aggregate_step == 0 else 'a'

            #         # Write or append the updated DataFrame to the main CSV
            #         with open(f"{emission_output_dir}/emissions_report.csv", write_mode) as f:
            #             temp_df.to_csv(f, header=(write_mode == 'w'), index=False)
            #     except Exception as e:
            #         print_l(f"Error saving emissions report")

        # # Save the aggregated data
        # if eval_c['save_aggregate_rewards']:
        #     save_to_csv(rewards, 'outputs/rewards.csv')
        #     save_to_csv(output_values, 'outputs/output_values.csv')

    # In evaluation mode, we don't need to train, we just evaluate the already saved model
    elif run_mode == "Evaluating":
        print_l(f"Evaluating using {algorithm_dm} - Seed {seed}", )

        print_l(f"Loading saved models - Seed {seed}")
        
        try:
            global_weights = torch.load(f'saved_networks/Exp_{experiment_number}/global_weights.pth')
            assert global_weights is not None, "Global weights are None"
        except Exception as e:
            print_l(f"No saved model found for experiment {experiment_number}")
            raise Exception(f"No saved model found for experiment {experiment_number}")

        rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
        output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
        old_buffers = [None for _ in range(len(chargers))] # Hold the buffers for the previous aggregation step

        # Initialize weights_to_save as a manager list to persist between aggregations
        manager = mp.Manager()
        weights_to_save = manager.list([None for _ in range(len(chargers))])

        for aggregate_step in range(federated_c['aggregation_count_eval']):
            try:
                # Start tracking emissions
                # tracker = EmissionsTracker(
                #     output_dir=emission_output_dir,
                #     save_to_file=f"emissions.csv",  # Temporary file
                #     tracking_mode='process',
                #     allow_multiple_runs=True,
                #     log_level='error'
                # )
                # tracker.start()

                # Check if we have only one zone - if so, don't use multiprocessing
                agg_print = f"{aggregate_step + 1}/{federated_c['aggregation_count_eval']}"
                print_l(f"\n\n############ Aggregation {agg_print} ############\n\n",)

                if len(chargers) == 1:
                    print("Only one zone detected, running without multiprocessing")
                    local_weights_list = [None]
                    process_rewards = []
                    process_output_values = []
                    process_buffers = [None]
                    weights_to_save = [None]

                    # Run directly without multiprocessing
                    train_route(queue, ev_info, experiment_number, chargers[0],\
                                copy.deepcopy(environment_list[0]), all_routes[0], date,\
                                action_dim, global_weights, aggregate_step, 0, algorithm_dm,\
                                chargers_seeds[0], seed, args, eval_c['fixed_attributes'],\
                                local_weights_list, process_rewards,\
                                process_output_values, None, devices[0], verbose,\
                                eval_c['display_training_times'], agent_by_zone, variant,\
                                eval_c['save_offline_data'], True, old_buffers[0],\
                                process_buffers, weights_to_save, len(chargers))
                else:
                    manager = mp.Manager()
                    local_weights_list = manager.list([None for _ in range(len(chargers))])
                    process_rewards = manager.list()
                    process_output_values = manager.list()
                    process_buffers = manager.list([None for _ in range(len(chargers))])
                    weights_to_save = manager.list([None for _ in range(len(chargers))])

                    # Barrier for synchronization
                    barrier = mp.Barrier(len(chargers))

                    processes = []
                    for ind, charger_list in enumerate(chargers):
                        args_tuple = (queue, ev_info, experiment_number, charger_list,\
                                  copy.deepcopy(environment_list[ind]), all_routes[ind], date,\
                                  action_dim, global_weights, aggregate_step, ind, algorithm_dm,\
                                  chargers_seeds[ind], seed, args, eval_c['fixed_attributes'],\
                                  local_weights_list, process_rewards,\
                                  process_output_values, barrier, devices[ind], verbose,\
                                  eval_c['display_training_times'], agent_by_zone, variant,\
                                  eval_c['save_offline_data'], True, old_buffers[ind], \
                                  process_buffers, weights_to_save, len(chargers))

                        process = mp.Process(target=train_route, args=args_tuple)
                        processes.append(process)
                        process.start()

                    print("Join Processes")

                    for process in processes:
                        process.join()

                    for p in processes:
                        if p.is_alive():
                            p.terminate()  # Just in case

                rewards = []

                print("Join Weights")

                # Aggregate the attention layers from all local agents
                if algorithm_dm == 'ODT':
                    global_weights = get_global_weights(local_weights_list, ev_info,\
                                                        federated_c['city_multiplier'],\
                                                        federated_c['zone_multiplier'],\
                                                        federated_c['model_multiplier'],\
                                                        agent_by_zone)
                elif algorithm_dm == 'DENSER': 
                    # Cannot aggregate weights for DENSER because architecture is different between agents
                    pass
                else:
                    # Aggregate the weights from all local models
                    global_weights = get_global_weights(local_weights_list, ev_info,\
                                                        federated_c['city_multiplier'],\
                                                        federated_c['zone_multiplier'],\
                                                        federated_c['model_multiplier'],\
                                                        agent_by_zone)

                # Extend the main lists with the contents of the process lists
                sorted_list = sorted([val[0] for sublist in process_rewards for val in sublist])
                
                if sorted_list:
                    print_l(f'Min and Max rewards for the aggregation step: {sorted_list[0], sorted_list[-1]}')
                else:
                    print_l("No rewards found for this aggregation step.")
                rewards.extend(process_rewards)
                output_values.extend(process_output_values)
                old_buffers = list(process_buffers)

            finally:
                aux = None
            #     # # Stop tracking emissions
            #     # emissions = tracker.stop()
            #     # print_l(f"Total CO₂ emissions: {emissions} kg")
            #     try:
            #         # Read the temporary emissions report
            #         temp_df = pd.read_csv(f"{emission_output_dir}/emissions.csv")

            #         # Remove the temporary emissions file
            #         os.remove(f"{emission_output_dir}/emissions.csv")

            #         # Add the aggregate_step column
            #         temp_df['aggregate_step'] = aggregate_step

            #         # Determine the write mode based on the aggregate_step
            #         write_mode = 'w' if aggregate_step == 0 else 'a'

            #         # Write or append the updated DataFrame to the main CSV
            #         with open(f"{emission_output_dir}/emissions_report.csv", write_mode) as f:
            #             temp_df.to_csv(f, header=(write_mode == 'w'), index=False)
            #     except Exception as e:
            #         print_l(f"Error saving emissions report")

    # Print total run time
    print_et(f'Experiment_number: {experiment_number}, run time', start_time, )

    # Stop the logger
    queue.put("__STOP__")
    writer_process.join()


    
    



if __name__ == '__main__':

    # Parse arguments from command line

    parser = argparse.ArgumentParser(description=('MERL Project'))
    parser.add_argument('-c','--number_processors', type=int, default=1,help='number of processors used to run MERL')
    parser.add_argument('-g','--list_gpus', nargs='*', type=int, default=[], help ='Request of enumerated gpus run MERL.')
    parser.add_argument('-e','--experiments_list', type=int, default=0, help ='Get experiment to run.')
    parser.add_argument('-d','--data_dir', type=str, default='', help='Directory to save data to')
    parser.add_argument('-verb','--verbose', type=bool, default=False, help='Verbose')
    parser.add_argument('-eval', type=bool, default=False, help='Evaluate the model')
    parser.add_argument('-profile', type=bool, default=False, help='Profile the code')
    parser.add_argument('-server', type=str, default='DRAC', help='Choose between DRAC or Local server. By default is DRAC.')
    args = parser.parse_args()

    if args.profile:
        cProfile.run('main_loop(args)', 'profile.out')
    else:
        main_loop(args)
