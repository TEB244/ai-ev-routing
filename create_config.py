# Script that loads in experiments/Exp_4000 directory, and creates new directories with all of the same files but slightly modified

import os
import shutil
import yaml

base_dir = "experiments/Exp_4000" # Base directory to copy initial config from
job_base_dir = "experiments/Exp_4000" # Base directory to copy initial job files from


# Total permutations: 4 * 12 * 3 = 144

config_permutations = {
    'algorithms': ['DQN', 'REINFORCE', 'CMA', 'ODT'],
    'num_eps_per_agg': [2, 5, 10, 50, 80, 100, 200, 500, 1000, 2000, 5000, 10000],
    'seeds': [1234, 2020, 3030],
}

num_permutations = len(config_permutations['algorithms']) * len(config_permutations['num_eps_per_agg']) * len(config_permutations['seeds'])

# Fields to modify in the config file
fields_to_modify = ['algorithm_settings.algorithm',
                    'nn_hyperparameters.eps_per_save',
                    'nn_hyperparameters.num_episodes',
                    'federated_learning_settings.aggregation_count',
                    'environment_settings.seed']

initial_exp_number = 2000 # Range from 2000 to 2132 (132 total)

for i in range(num_permutations):

    if i % 36 == 0:
        job_base_num = i

    job_base_dir = f"experiments/Exp_4{job_base_num:03d}" # Update the base of the job file directories

    new_exp_number = initial_exp_number + i

    # Create new directory Exp_<new_exp_number>
    new_dir = os.path.join(f'experiments/Exp_{new_exp_number:04d}')

    config_files_to_copy = ['config.yaml', 'description.txt']
    job_files_to_copy = ['train_job.sh']
    # Ensure the new directory exists before copying files
    os.makedirs(new_dir, exist_ok=True)
    # Copy config files from base_dir
    for file in config_files_to_copy:
        src = os.path.join(base_dir, file)
        dst = os.path.join(new_dir, file)
        shutil.copy(src, dst)
    # Copy job files from job_base_dir
    for file in job_files_to_copy:
        src = os.path.join(job_base_dir, file)
        dst = os.path.join(new_dir, file)
        shutil.copy(src, dst)

    # Get the permutation specific to this experiment based on the index
    permutation_list = {
        'environment_settings.seed': config_permutations['seeds'][i  % len(config_permutations['seeds'])],
        'nn_hyperparameters.eps_per_save': config_permutations['num_eps_per_agg'][i // len(config_permutations['seeds']) % len(config_permutations['num_eps_per_agg'])],
        'nn_hyperparameters.num_episodes': config_permutations['num_eps_per_agg'][i // len(config_permutations['seeds']) % len(config_permutations['num_eps_per_agg'])],
        'federated_learning_settings.aggregation_count': 10000 // config_permutations['num_eps_per_agg'][i // len(config_permutations['seeds']) % len(config_permutations['num_eps_per_agg'])],
        'algorithm_settings.algorithm': config_permutations['algorithms'][i // len(config_permutations['seeds']) // len(config_permutations['num_eps_per_agg']) % len(config_permutations['algorithms'])],
    }
        
    # Modify the config file
    with open(os.path.join(new_dir, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Helper to set nested field by dot notation
    def set_nested(config, dotted_key, value):
        keys = dotted_key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    for field, value in permutation_list.items():
        set_nested(config, field, value)

    with open(os.path.join(new_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Modify the description.txt file
    with open(os.path.join(new_dir, 'description.txt'), 'r') as f:
        description = f.read()

    description = description.replace('Experiment 4000', f'Experiment {new_exp_number:04d}')
    description = description.replace('Model: DQN', f'Model: {permutation_list["algorithm_settings.algorithm"]}')
    description = description.replace('Number of episodes: 200', f'Number of episodes: {permutation_list["nn_hyperparameters.num_episodes"]}')
    description = description.replace('Number of aggregations: 50', f'Number of aggregations: {permutation_list["federated_learning_settings.aggregation_count"]}')
    description = description.replace('Seed: 1234', f'Seed: {permutation_list["environment_settings.seed"]}')

    with open(os.path.join(new_dir, 'description.txt'), 'w') as f:
        f.write(description)

    print(f"Job base number: 4{job_base_num:03d} - New experiment number: {new_exp_number}")

    # Update the train_job.sh and eval_job.sh files
    with open(os.path.join(new_dir, 'train_job.sh'), 'r') as f:
        train_job = f.read()
        train_job = train_job.replace(str(f'4{job_base_num:03d}'), str(f'{new_exp_number:04d}'))

    with open(os.path.join(new_dir, 'train_job.sh'), 'w') as f:
        f.write(train_job)