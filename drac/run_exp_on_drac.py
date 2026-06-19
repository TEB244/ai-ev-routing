import subprocess
import argparse
import yaml
import os
import socket
from parallel_gpu_H100 import run_parallel_gpu_H100
from parallel_no_gpu import run_parallel_no_gpu

def run_exp_on_drac(experiments_list, algorithm=None, eval=False, seed=None, aggregation=None):
    """
    Run experiments using the servers from the Digital Research Alliance of Canada (DRAC)

    Parameters:
        experiments_list (list): List of experiment numbers to run
        algorithm (str): Algorithm to run
        eval (bool): Whether to evaluate the model
        seed (int): Seed to run
        aggregation (int): Aggregation count to run
    """
    start_experiment = experiments_list[0]
    end_experiment = experiments_list[-1]
    if algorithm:
        print(f"Running experiments in range {start_experiment} to {end_experiment} using ONLY algorithm {algorithm}")
    else:
        print(f"Running experiments in range {start_experiment} to {end_experiment}")

    for experiment_number in range(start_experiment, end_experiment + 1):
        try:
            # Load config.yaml file for experiment
            config_file = os.path.join(f"experiments/Exp_{experiment_number}", "config.yaml")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

                if algorithm: # Filter by algorithm
                    if config.get("algorithm_settings", {}).get("algorithm") != algorithm:
                        print(f"Experiment {experiment_number} does not use algorithm {algorithm}")
                        continue

                if seed: # Filter by seed
                    if int(config.get("environment_settings", {}).get("seed")) != int(seed):
                        print(f"Experiment {experiment_number} does not use seed {seed}")
                        continue

                if aggregation: # Filter by aggregation count
                    if int(config.get("federated_learning_settings", {}).get("aggregation_count")) != int(aggregation):
                        print(f"Experiment {experiment_number} does not use aggregation {aggregation}")
                        continue

        except Exception as e:
            print(f"Error loading config.yaml file for experiment {experiment_number}: {e}")
            continue

        cmd = f"sbatch experiments/Exp_{experiment_number}/{'eval' if eval else 'train'}_job.sh"
        print(f"Running command: {cmd}")
        try:
            subprocess.run(cmd, shell=True)
        except Exception as e:
            print(f"Error running command: {cmd}")
            print(f"Error message: {e}")

def check_same_algorithm(experiments_list):
    """
    Check if all experiments in the given range use the same algorithm.
    Returns a tuple: (is_same, algorithm_name or None)

    Args:
        experiments_list (list): List of experiment numbers to check.

    Returns:
        (bool, str or None): True and the algorithm name if all use the same, else False and None.
    """
    algorithms_found = set()
    algo_name = None

    for experiment_number in range(experiments_list[0], experiments_list[-1] + 1):
        config_file = os.path.join(f"experiments/Exp_{experiment_number}", "config.yaml")
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                algorithm = config.get("algorithm_settings", {}).get("algorithm")
                if algorithm is not None:
                    algorithms_found.add(algorithm)
        except Exception as e:
            print(f"Error loading config.yaml for experiment {experiment_number}: {e}")
            continue

    if len(algorithms_found) == 1:
        algo_name = algorithms_found.pop()
        return True, algo_name
    else:
        return False, None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=('MERL Project'))
    parser.add_argument('-e','--experiments_list', nargs='*', type=int, default=[], help ='Get the list of experiment to run.')
    parser.add_argument('-a', '--algorithm', type=str, help='Algorithm to run.')
    parser.add_argument('-s', '--seed', type=str, default=None, help='Seed to run.')
    parser.add_argument('-agg', '--aggregation', type=str, default=None, help='Aggregation count to run.')
    parser.add_argument('-eval', type=bool, default=False, help="Evaluate the model")
    parser.add_argument('-pgpu','--parallel_gpu', type=str, default=None, help="Run in parallel on the same GPU, depending on the GPU type (H100, A100, V100, etc.)")
    parser.add_argument('-parallel', '--parallel', type=bool, default=False, help="Run in parallel on the same GPU, depending on the GPU type (H100, A100, V100, etc.)")
    args = parser.parse_args()

    cluster_str = socket.gethostname()
    if "rorqual" in cluster_str:
        cluster = "rorqual"
    
    algorithm_is_same, algorithm_name = check_same_algorithm(args.experiments_list)
    print(f"Algorithm is same: {algorithm_is_same}, Algorithm name: {algorithm_name}")

    if not args.parallel:
        print('Running experiments on DRAC single experiment per job')
        run_exp_on_drac(args.experiments_list, args.algorithm, args.eval, args.seed, args.aggregation)
    elif args.parallel and algorithm_is_same:
        exp_size = args.experiments_list[-1] - args.experiments_list[0] + 1
        if exp_size < 4:
            print(f'Submitting single experiment per job as the number of experiments is less than 4.')
            run_exp_on_drac(args.experiments_list, args.algorithm, args.eval, args.seed, args.aggregation)
        elif algorithm_name == "CMA":
            print(f'Running experiments {args.experiments_list[0]}-{args.experiments_list[-1]} without GPU.')
            run_parallel_no_gpu(args.experiments_list, algorithm_name, args)
        elif algorithm_name == "ODT" and cluster == "rorqual":
            print(f'Running experiments {args.experiments_list[0]}-{args.experiments_list[-1]} in parallel with GPU H100')
            run_parallel_gpu_H100(args.experiments_list, algorithm_name, args)
        else:
            print('Especial case, either not ODT or roqual cluster is detected. Existing...')
            exit(1)
    else:
        print(f"Not all algorithms are the same in the range requested. Please confirm the range of experiments. Exiting...")
        exit(1)
