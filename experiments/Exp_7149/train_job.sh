#!/bin/bash
#SBATCH --job-name=Exp_7149_train
#SBATCH --output=experiments/Exp_7149/output.log
#SBATCH --error=experiments/Exp_7149/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=22:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 7149"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4



python main.py -g 0 -e 7149 -d "/home/epigou/scratch/metrics/Exp"
