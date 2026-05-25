#!/bin/bash
#SBATCH --job-name=Exp_9995_train
#SBATCH --output=experiments/Exp_9995/output.log
#SBATCH --error=experiments/Exp_9995/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=35G
#SBATCH --gpus-per-node=4

echo "Starting training for experiment 9995"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=2

python main.py -e 9995 -server DRAC -g 0 1 2 3
