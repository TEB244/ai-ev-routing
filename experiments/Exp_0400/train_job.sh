#!/bin/bash
#SBATCH --job-name=Exp_400_train
#SBATCH --output=experiments/Exp_0400/output.log
#SBATCH --error=experiments/Exp_0400/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --mem=110G

echo "Starting training for experiment 0400"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 1 2 3 -e 0400 -d "/home/epigou/scratch/metrics/Exp"