#!/bin/bash
#SBATCH --job-name=Exp_9117_train
#SBATCH --output=experiments/Exp_9117/output.log
#SBATCH --error=experiments/Exp_9117/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=130:00:00
#SBATCH --mem=1608M


echo "Starting training for experiment 9117"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 9117 -d "/home/sgomezro/scratch/metrics/Exp"

