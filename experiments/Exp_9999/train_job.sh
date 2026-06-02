#!/bin/bash
#SBATCH --job-name=Exp_9999_train
#SBATCH --output=experiments/Exp_9999/output.log
#SBATCH --error=experiments/Exp_9999/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --gpus-per-node=t4:1
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --mail-user=epigou@uwo.ca

echo "=== Exp_9999 e2e smoke test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=1

python main.py -e 9999 -server DRAC -g 0

echo "End: $(date)"
