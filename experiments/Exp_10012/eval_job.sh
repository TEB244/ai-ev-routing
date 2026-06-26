#!/bin/bash
#SBATCH --job-name=Exp_10012_eval
#SBATCH --output=experiments/Exp_10012/output.log
#SBATCH --error=experiments/Exp_10012/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --mem=6G

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting inference (eval) for experiment 10012"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=2

python main.py -e 10012 -d "/home/hartman/scratch/metrics/Exp" -eval True
