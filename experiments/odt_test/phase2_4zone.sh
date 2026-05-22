#!/bin/bash
#SBATCH --job-name=odt_test_phase2
#SBATCH --output=experiments/odt_test/phase2_output.log
#SBATCH --error=experiments/odt_test/phase2_error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --mail-user=epigou@uwo.ca

# Phase 2: full 4-zone ODT run to measure real resource usage.
# Exp_9997 is a 4-zone, 100-car ODT experiment (2 aggregations, 2 online episodes).
# main.py spawns 4 zone processes (mp.Process), hence 10 CPUs (4 zones x 2 threads + main).
# After this job completes run: seff $SLURM_JOB_ID
# GPU time-series will be in experiments/odt_test/phase2_gpu.log
#
# What to record for parallel job sizing:
#   - Wall time from phase1 output (scale x4 for 4 zones, minus parallelism)
#   - Peak GPU memory from phase2_gpu.log (memory.used column)
#   - Peak RAM from seff output
#   - CPU efficiency from seff output

echo "=== Phase 2: 4-zone ODT test ==="
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
    > experiments/odt_test/phase2_gpu.log &
GPU_MONITOR_PID=$!

python main.py -e 9997 -g 0 -verb True

kill $GPU_MONITOR_PID 2>/dev/null || true

echo "End: $(date)"
echo "Run 'seff $SLURM_JOB_ID' for CPU/memory efficiency summary."
