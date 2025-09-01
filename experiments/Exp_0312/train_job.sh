#!/bin/bash
#SBATCH --job-name=Exp_312_train
#SBATCH --output=experiments/Exp_0312/output.log
#SBATCH --error=experiments/Exp_0312/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=16:40:00
#SBATCH --mem=64G

echo "Starting training for experiment 0312"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 -e 0312 -d "/home/epigou/scratch/metrics/Exp" 
    