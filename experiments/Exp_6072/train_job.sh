#!/bin/bash
#SBATCH --job-name=Exp_6072_train
#SBATCH --output=experiments/Exp_6072/output.log
#SBATCH --error=experiments/Exp_6072/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=11:20:00
#SBATCH --mem=6G
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca


echo "Starting training for experiment 6072"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 6072 -d "/home/hartman/scratch/metrics/Exp" 
    