#!/bin/bash
#SBATCH --job-name=Exp_4075_train
#SBATCH --output=experiments/Exp_4075/output_v4.log
#SBATCH --error=experiments/Exp_4075/error_v4.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=15:00:00
#SBATCH --mem=6G


echo "Starting training for experiment 4075"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 4075 -d "/home/sgomezro/scratch/metrics/Exp" -verb True
    
