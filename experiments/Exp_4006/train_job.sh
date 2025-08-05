#!/bin/bash
#SBATCH --job-name=Exp_4006_train
#SBATCH --output=experiments/Exp_4006/output.log
#SBATCH --error=experiments/Exp_4006/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
#SBATCH --mem=8G


echo "Starting training for experiment 4006"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 4006 -d "/home/sgomezro/scratch/metrics/Exp"  
    