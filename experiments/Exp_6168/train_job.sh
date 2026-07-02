#!/bin/bash
#SBATCH --job-name=Exp_6168_train
#SBATCH --output=experiments/Exp_6168/output.log
#SBATCH --error=experiments/Exp_6168/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment 6168"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4

# Activate Nvidia MPS:
# Per-job MPS dirs so co-located ODT jobs don't collide on /tmp/nvidia-mps (CUDA err 805)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
cleanup_mps() { echo quit | nvidia-cuda-mps-control 2>/dev/null || true; rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; }
trap cleanup_mps EXIT
nvidia-cuda-mps-control -d


python main.py -g 0 -e 6168 -d "/home/hartman/links/scratch/metrics/Exp"
