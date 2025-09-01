#!/bin/bash
#SBATCH --job-name=Exp_16_eval
#SBATCH --output=experiments/Exp_0016/output.log
#SBATCH --error=experiments/Exp_0016/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem=24G

echo "Starting evaluation for experiment 0016"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 -e 0016 -d "/home/hartman/scratch/metrics/Exp" -eval True
    