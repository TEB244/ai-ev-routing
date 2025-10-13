#!/bin/bash
#SBATCH --job-name=Exp_2063_eval
#SBATCH --output=experiments/Exp_2063/output.log
#SBATCH --error=experiments/Exp_2063/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem=12G
#SBATCH --gpus-per-node=2

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting evaluation for experiment 2063"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 2063 -d "/home/hartman/scratch/metrics/Exp" -g 0 1 -verb True -eval True
    
