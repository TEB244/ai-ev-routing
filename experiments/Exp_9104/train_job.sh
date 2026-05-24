#!/bin/bash
#SBATCH --job-name=Exp_9104_train
#SBATCH --output=experiments/Exp_9104/output.log
#SBATCH --error=experiments/Exp_9104/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --mem=4608M


echo "Starting training for experiment 9104"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 9104 -d "/home/sgomezro/scratch/metrics/Exp"

