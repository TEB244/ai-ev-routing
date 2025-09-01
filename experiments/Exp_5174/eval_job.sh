#!/bin/bash
#SBATCH --job-name=Exp_5066_eval
#SBATCH --output=experiments/Exp_5066/output.log
#SBATCH --error=experiments/Exp_5066/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=90:00:00
#SBATCH --mem=6G


#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting evaluation for experiment 5066"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 5066 -d "/home/hartman/scratch/metrics/Exp" -eval True
    