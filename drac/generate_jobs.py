# Run this script to generate the job configuration files for the experiments

import argparse
import yaml

    
def create_job(args):    
    experiment_list = []
    if args.e:
        if 'all' in args.e:
            import os
            experiment_list = [int(d.split('_')[1]) for d in os.listdir('experiments') if d.startswith('Exp_')]
        else:
            experiment_list = range(int(args.e[0]), int(args.e[-1]))
    
    print("Experiments to create job config for:", experiment_list)
    # Read config file for each experiment
    for experiment in experiment_list:
        try:
            with open(f'experiments/Exp_{experiment}/config.yaml', 'r') as file:
                config = yaml.safe_load(file)
    
            # Use 1 gpu per zone
            # num_gpus = len(config['environment_settings']['coords'])

            # Get the number of episodes and aggregations
            num_episodes = config['nn_hyperparameters']['num_episodes']
            num_aggregations = config['federated_learning_settings']['aggregation_count']
            num_generations = config['cma_parameters']['max_generations']
            
        
            # Get the algorithm
            algorithm = config['algorithm_settings']['algorithm']

            if algorithm == 'CMA' or algorithm == 'DQN':
                num_gpus = 0
                allocation = "def-mcapretz"
            elif algorithm == 'PPO':
                num_gpus = 1
                allocation = "rrg-kgroling"
            elif algorithm == 'ODT':
                num_gpus = 4
                allocation = "def-mcapretz"
        
            # Calculate the time based on the total number of episodes
            # Note: these are rough estimates based on how long takes to train 10k episodes
            algorithm_time_mapping = {
                'DQN': (45 / 6000) / 5, # 45 hours / 6k episodes / 5 zones
                'PPO': (80 / 6000) / 5, # 80 hours / 6k episodes / 5 zones
                'CMA': (25 / 10000) / 4, # 16 hours / 10k generations 4 zones
                'ODT': (14 / 2000) , # 15 hours for 2k episodes, 5 zones
            }
        
            if algorithm in algorithm_time_mapping:
                if algorithm == 'CMA':
                    total_episodes = num_generations * num_aggregations
                    mem_size = "6G"
                    num_cpus = len(config['environment_settings']['coords']) + 1
                    if args.eval:
                        calculated_time = algorithm_time_mapping[algorithm] * 100 * len(config['environment_settings']['coords']) * 1.5 
                    else:
                        calculated_time = algorithm_time_mapping[algorithm] * total_episodes * len(config['environment_settings']['coords'])
                elif algorithm == 'ODT':
                    mem_size = "35G"
                    num_cpus = len(config['environment_settings']['coords'])
                    total_episodes = num_episodes * num_aggregations
                    calculated_time = algorithm_time_mapping[algorithm] * total_episodes
                else:
                    # num_cpus = num_gpus + 1
                    # For now, use 1 cpu per zone
                    num_cpus = len(config['environment_settings']['coords'])
                    mem_size = "32G"
                    total_episodes = num_episodes * num_aggregations
                    calculated_time = algorithm_time_mapping[algorithm] * total_episodes * len(config['environment_settings']['coords'])
            else:
                print(f"Algorithm {algorithm} not supported. Need to add estimated duration for this algorithm.")
                continue
        
            data_dir = f"/home/{args.u}/scratch/metrics/Exp"
        
            # Generate the job config files
            job_script_content = f"""#!/bin/bash
#SBATCH --job-name=Exp_{experiment}_{'eval' if args.eval else 'train'}
#SBATCH --output=experiments/Exp_{experiment}/output.log
#SBATCH --error=experiments/Exp_{experiment}/error.log
#SBATCH -A {allocation}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --time={str(int(calculated_time // 1)).zfill(2)}:{str(int((calculated_time * 60) % 60)).zfill(2)}:{str(int((calculated_time * 3600) % 60)).zfill(2)}
#SBATCH --mem={mem_size}
{f"#SBATCH --gpus-per-node={num_gpus}" if num_gpus > 0 else""}

echo "Starting {'evaluation' if args.eval else 'training'} for experiment {experiment}"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py {"" if num_gpus==0 else "-g "}{" ".join(str(g) for g in range(num_gpus)) if num_gpus > 0 else ""} -e {experiment} -d "{data_dir}" {"-eval True" if args.eval else ""}
    """
            
            # Save job script to file
            with open(f'experiments/Exp_{experiment}/{"eval" if args.eval else "train"}_job.sh', 'w') as job_file:
                job_file.write(job_script_content)
        except Exception as e:
            print(f"Error creating job for experiment {experiment}: {e}")


if __name__ == "__main__":
    
    # Determine which experiments to create job config for
    parser = argparse.ArgumentParser(description="Generate job configuration files for experiments")
    parser.add_argument('-e', type=str, nargs='+', help="List of experiment numbers or 'all' to include all experiments")
    parser.add_argument('-u', type=str, default='hartman', help="user account at DRAC")
    parser.add_argument('-eval', type=bool, default=False, help="Evaluate the model")
    args = parser.parse_args()
    create_job(args)
