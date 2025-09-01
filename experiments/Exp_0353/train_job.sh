#!/bin/bash
#SBATCH --job-name=Exp_353_train
#SBATCH --output=experiments/Exp_0353/output.log
#SBATCH --error=experiments/Exp_0353/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=30:00:00
#SBATCH --mem=32G


echo "Starting training for experiment 0353"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 0353 -d "/home/sgomezro/scratch/metrics/Exp" 
    