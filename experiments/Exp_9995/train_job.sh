#!/bin/bash
#SBATCH --job-name=Exp_9995_train
#SBATCH --output=experiments/Exp_9995/output.log
#SBATCH --error=experiments/Exp_9995/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --mail-user=epigou@uwo.ca

echo "=== Exp_9995 training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=2

# Poll GPU utilisation every 30s in the background
nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 \
    > experiments/Exp_9995/gpu.log &
GPU_MONITOR_PID=$!

python main.py -e 9995 -server DRAC -g 0 1 2 3

kill $GPU_MONITOR_PID 2>/dev/null || true

echo "End: $(date)"
echo "Run 'seff $SLURM_JOB_ID' for CPU/memory efficiency summary."
