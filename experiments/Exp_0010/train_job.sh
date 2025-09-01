#!/bin/bash
#SBATCH --job-name=Exp_10_train
#SBATCH --output=experiments/Exp_0010/output.log
#SBATCH --error=experiments/Exp_0010/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=83:20:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1

echo "Starting training for experiment 0010"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.0010 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 -e 0010 -d "/home/hartman/scratch/metrics/Exp" 
    