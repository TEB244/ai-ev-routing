#!/bin/bash
#SBATCH --job-name=Exp_5081_eval
#SBATCH --output=experiments/Exp_5081/output.log
#SBATCH --error=experiments/Exp_5081/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:22:30
#SBATCH --mem=6G


echo "Starting evaluation for experiment 5081"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 5081 -d "/home/sgomezro/scratch/metrics/Exp" -eval True
    