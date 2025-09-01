#!/bin/bash
#SBATCH --job-name=Exp_47_eval
#SBATCH --output=experiments/Exp_0047/output.log
#SBATCH --error=experiments/Exp_0047/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --time=53:19:59
#SBATCH --mem=24G

echo "Starting evaluation for experiment 0047"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 -e 0047 -d "/home/hartman/scratch/metrics/Exp" -eval True
    