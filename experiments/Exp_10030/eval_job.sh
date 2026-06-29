#!/bin/bash
#SBATCH --job-name=Exp_10030_eval
#SBATCH --output=experiments/Exp_10030/output.log
#SBATCH --error=experiments/Exp_10030/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:30:00
#SBATCH --mem=3584M

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting inference (eval) for experiment 10030"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=2

python main.py -e 10030 -d "/home/hartman/scratch/metrics/Exp" -eval True
