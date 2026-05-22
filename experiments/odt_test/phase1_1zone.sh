#!/bin/bash
#SBATCH --job-name=odt_test_phase1
#SBATCH --output=experiments/odt_test/phase1_output.log
#SBATCH --error=experiments/odt_test/phase1_error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --mem=12G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --mail-user=epigou@uwo.ca

# Phase 1: single-zone functionality check and resource baseline.
# Exp_9996 is a 1-zone, 100-car ODT experiment (2 aggregations, 2 online episodes).
# After this job completes run: seff $SLURM_JOB_ID
# GPU time-series will be in experiments/odt_test/phase1_gpu.log

echo "=== Phase 1: 1-zone ODT test ==="
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
    > experiments/odt_test/phase1_gpu.log &
GPU_MONITOR_PID=$!

python main.py -e 9996 -g 0 -verb True

kill $GPU_MONITOR_PID 2>/dev/null || true

echo "End: $(date)"
echo "Run 'seff $SLURM_JOB_ID' for CPU/memory efficiency summary."
