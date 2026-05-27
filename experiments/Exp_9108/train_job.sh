#!/bin/bash
#SBATCH --job-name=Exp_9108_train
#SBATCH --output=experiments/Exp_9108/output.log
#SBATCH --error=experiments/Exp_9108/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=1608M


echo "Starting training for experiment 9108"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 9108 -d "/home/sgomezro/scratch/metrics/Exp"

