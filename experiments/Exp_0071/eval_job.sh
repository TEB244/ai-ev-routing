#!/bin/bash
#SBATCH --job-name=Exp_71_eval
#SBATCH --output=experiments/Exp_0071/output.log
#SBATCH --error=experiments/Exp_0071/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --time=53:19:59
#SBATCH --mem=24G

echo "Starting evaluation for experiment 0071"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 0071 -d "/home/hartman/scratch/metrics/Exp" -eval True
    