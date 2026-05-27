#!/bin/bash
#SBATCH --job-name=Exp_9318_train
#SBATCH --output=experiments/Exp_9318/output.log
#SBATCH --error=experiments/Exp_9318/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=5:30:00
#SBATCH --mem=12G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 9318"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

python main.py -g 0 -e 9318 -d "/scratch/epigou/metrics/Exp"
