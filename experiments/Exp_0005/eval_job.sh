#!/bin/bash
#SBATCH --job-name=Exp_5_eval
#SBATCH --output=experiments/Exp_0005/output.log
#SBATCH --error=experiments/Exp_0005/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=0005
#SBATCH --time=41:40:00
#SBATCH --mem=24G


echo "Starting evaluation for experiment 0005"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py   -e 0005 -d "/home/hartman/scratch/metrics/Exp" -eval True
    