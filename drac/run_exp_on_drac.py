import subprocess
import argparse
import yaml
import os
from run_parallel_gpu_H100 import run_parallel_gpu_H100

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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=('MERL Project'))
    parser.add_argument('-e','--experiments_list', nargs='*', type=int, default=[], help ='Get the list of experiment to run.')
    parser.add_argument('-a', '--algorithm', type=str, help='Algorithm to run.')
    parser.add_argument('-s', '--seed', type=str, default=None, help='Seed to run.')
    parser.add_argument('-agg', '--aggregation', type=str, default=None, help='Aggregation count to run.')
    parser.add_argument('-eval', type=bool, default=False, help="Evaluate the model")
    parser.add_argument('-pgpu','--parallel_gpu', type=str, default=None, help="Run in parallel on the same GPU, depending on the GPU type (H100, A100, V100, etc.)")
    args = parser.parse_args()


    if args.parallel_gpu == None:
        print('Running experiments on DRAC single experiment per job')
        run_exp_on_drac(args.experiments_list, args.algorithm, args.eval, args.seed, args.aggregation)
    elif args.parallel_gpu == "H100":
        if len(args.experiments_list) > 4:
            print(f'Running experiments in parallel with GPU H100')
            run_parallel_gpu_H100(args.experiments_list, args.algorithm, args)
        else:
            print(f'Submitting single experiment per job as the number of experiments is less than 4 for H100 GPU.')
            run_exp_on_drac(args.experiments_list, args.algorithm, args.eval, args.seed, args.aggregation)
    else:
        print(f"{args.parallel_gpu} not implemented yet or not valid entry. Exiting...")
        exit(1)
