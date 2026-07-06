#!/bin/bash
#SBATCH --job-name=Exp_4105_train
#SBATCH --output=experiments/Exp_4105/output.log
#SBATCH --error=experiments/Exp_4105/error.log
#SBATCH -A def-mcapretz
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=18:30:00
#SBATCH --mem=3584M


echo "Starting training for experiment 4105"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 4105 -d "/home/hartman/scratch/metrics/Exp" 
    