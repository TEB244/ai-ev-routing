#!/bin/bash
#SBATCH --job-name=Exp_7155_train
#SBATCH --output=experiments/Exp_7155/output.log
#SBATCH --error=experiments/Exp_7155/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 7155"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4



python main.py -g 0 -e 7155 -d "/home/epigou/scratch/metrics/Exp"
