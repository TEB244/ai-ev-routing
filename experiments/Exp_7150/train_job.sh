#!/bin/bash
#SBATCH --job-name=Exp_7150_train
#SBATCH --output=experiments/Exp_7150/output.log
#SBATCH --error=experiments/Exp_7150/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=22:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment 7150"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4



python main.py -g 0 -e 7150 -d "/home/hartman/links/scratch/metrics/Exp"
