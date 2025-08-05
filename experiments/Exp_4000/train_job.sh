#!/bin/bash
#SBATCH --job-name=Exp_4000_train
#SBATCH --output=experiments/Exp_4000/output.log
#SBATCH --error=experiments/Exp_4000/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:10:00
#SBATCH --mem=6G


echo "Starting training for experiment 4000"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 4000 -d "/home/sgomezro/scratch/metrics/Exp" -verb True
    
