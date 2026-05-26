#!/bin/bash
#SBATCH --job-name=Exp_9116_train
#SBATCH --output=experiments/Exp_9116/output.log
#SBATCH --error=experiments/Exp_9116/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=1608M


echo "Starting training for experiment 9116"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 9116 -d "/home/sgomezro/scratch/metrics/Exp"

