#!/bin/bash
#SBATCH --job-name=Exp_4003_train
#SBATCH --output=experiments/Exp_4003/output.log
#SBATCH --error=experiments/Exp_4003/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --mem=8G


echo "Starting training for experiment 4003"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 4003 -d "/home/sgomezro/scratch/metrics/Exp" 
    