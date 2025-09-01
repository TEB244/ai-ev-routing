#!/bin/bash
#SBATCH --job-name=Exp_3000_eval
#SBATCH --output=experiments/Exp_3000/output.log
#SBATCH --error=experiments/Exp_3000/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=00:00:20
#SBATCH --mem=64G

echo "Starting evaluation for experiment 3000"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python main.py -g 0 -e 3000 -d "/home/hartman/scratch/metrics/Exp" -eval True
    