#!/bin/bash
#SBATCH --job-name=Exp_4011_train
#SBATCH --output=experiments/Exp_4011/output.log
#SBATCH --error=experiments/Exp_4011/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --mem=6G


#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment 4011"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 4011 -d "/home/hartman/scratch/metrics/Exp" 
    