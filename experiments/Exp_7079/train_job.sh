#!/bin/bash
#SBATCH --job-name=Exp_7079_train
#SBATCH --output=experiments/Exp_7079/output.log
#SBATCH --error=experiments/Exp_7079/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --mem=6G


#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment 7079"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 7079 -d "/home/hartman/scratch/metrics/Exp" -verb True

