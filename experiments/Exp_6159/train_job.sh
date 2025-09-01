#!/bin/bash
#SBATCH --job-name=Exp_6159_train
#SBATCH --output=experiments/Exp_6159/output.log
#SBATCH --error=experiments/Exp_6159/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=30:00:00
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 6159"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4

# Activate Nvidia MPS:
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d


python app_v2.py -g 0 -e 6159 -d "/home/epigou/scratch/metrics/Exp"
