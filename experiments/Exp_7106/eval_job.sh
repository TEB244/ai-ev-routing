#!/bin/bash
#SBATCH --job-name=Exp_7106_eval
#SBATCH --output=experiments/Exp_7106/output.log
#SBATCH --error=experiments/Exp_7106/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:56:10
#SBATCH --mem=6G


echo "Starting evaluation for experiment 7106"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 7106 -d "/home/sgomezro/scratch/metrics/Exp" -eval True
    