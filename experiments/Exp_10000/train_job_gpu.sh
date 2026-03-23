#!/bin/bash
#SBATCH --job-name=Exp_10000_train_GPU
#SBATCH --output=experiments/Exp_10000/output_GPU.log
#SBATCH --error=experiments/Exp_10000/error_GPU.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --mem=12G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment 10000"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 -e 10000 -d "/home/hartman/scratch/metrics/Exp" -verb True
