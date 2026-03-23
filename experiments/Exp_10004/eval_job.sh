#!/bin/bash
#SBATCH --job-name=Exp_10004_eval
#SBATCH --output=experiments/Exp_10004/output.log
#SBATCH --error=experiments/Exp_10004/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=2:00:00
#SBATCH --mem=4G

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting evaluation for experiment 10004"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py  -e 10004 -d "/home/hartman/scratch/metrics/Exp" -eval True
