#!/bin/bash
#SBATCH --job-name=Exp_300_eval
#SBATCH --output=experiments/Exp_0300/output.log
#SBATCH --error=experiments/Exp_0300/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=05:39:59
#SBATCH --mem=64G

echo "Starting evaluation for experiment 0300"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 -e 0300 -d "/home/hartman/scratch/metrics/Exp" -eval True
    