#!/bin/bash
#SBATCH --job-name=Exp_69_eval
#SBATCH --output=experiments/Exp_0069/output.log
#SBATCH --error=experiments/Exp_0069/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=0
#SBATCH --time=00:00:00
#SBATCH --mem=8G

echo "Starting evaluation for experiment 0069"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py   -e 0069 -d "/home/hartman/scratch/metrics/Exp" -eval True
    