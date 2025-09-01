#!/bin/bash
#SBATCH --job-name=Exp_3001_train
#SBATCH --output=experiments/Exp_3001/output.log
#SBATCH --error=experiments/Exp_3001/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=160G

echo "Starting training for experiment 3001"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 -e 3001 -d "/home/epigou/scratch/metrics/Exp"