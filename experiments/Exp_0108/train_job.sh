#!/bin/bash
#SBATCH --job-name=Exp_108_train
#SBATCH --output=experiments/Exp_0108/output.log
#SBATCH --error=experiments/Exp_0108/error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=32:00:00
#SBATCH --mem=24G

echo "Starting training for experiment 0108"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -e 0108 -d "/home/hartman/scratch/metrics/Exp" 
    