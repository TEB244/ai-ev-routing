#!/bin/bash
#SBATCH --job-name=Exp_7163_train
#SBATCH --output=experiments/Exp_7163/output.log
#SBATCH --error=experiments/Exp_7163/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=22:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 7163"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4

# Activate Nvidia MPS:
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d


python main.py -g 0 -e 7163 -d "/home/epigou/scratch/metrics/Exp"
