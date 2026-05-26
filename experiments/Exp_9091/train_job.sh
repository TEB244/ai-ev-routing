#!/bin/bash
#SBATCH --job-name=Exp_9091_train
#SBATCH --output=experiments/Exp_9091/output.log
#SBATCH --error=experiments/Exp_9091/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=1792M


echo "Starting training for experiment 9091"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 9091 -d "/home/sgomezro/scratch/metrics/Exp"

