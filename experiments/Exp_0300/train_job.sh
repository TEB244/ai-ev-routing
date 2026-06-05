#!/bin/bash
#SBATCH --job-name=Exp_300_train
#SBATCH --output=experiments/Exp_0300/output.log
#SBATCH --error=experiments/Exp_0300/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=4
#SBATCH --time=17:00:00
#SBATCH --mem=180G

echo "Starting training for experiment 0300"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 1 2 3 -e 0300 -d "/home/hartman/scratch/metrics/Exp" 
    