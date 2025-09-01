#!/bin/bash
#SBATCH --job-name=Exp_146_eval
#SBATCH --output=experiments/Exp_0146/output.log
#SBATCH --error=experiments/Exp_0146/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --time=00:01:21
#SBATCH --mem=64G

echo "Starting evaluation for experiment 0146"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 1 -e 0146 -d "/home/hartman/scratch/metrics/Exp" -eval True
    