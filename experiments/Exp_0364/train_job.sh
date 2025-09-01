#!/bin/bash
#SBATCH --job-name=Exp_364_train
#SBATCH --output=experiments/Exp_0364/output.log
#SBATCH --error=experiments/Exp_0364/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=05:20:00
#SBATCH --mem=6G


echo "Starting training for experiment 0364"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 0364 -d "/home/sgomezro/scratch/metrics/Exp" 
    