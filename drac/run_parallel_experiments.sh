#!/bin/bash
#SBATCH --job-name=Exp_parallel_train_7136_7138
#SBATCH --output=experiments/parallel/output.log
#SBATCH --error=experiments/parallel/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 #16        # Scale up: ~4-6 CPUs per experiment
#SBATCH --time=00:05:00
#SBATCH --mem=20G                  # Scale up: ~12G per experiment x4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1



# ─── Configuration ────────────────────────────────────────────────────────────
# List all experiment IDs to run in parallel on the same GPU
EXPERIMENTS=(7135 7136)
DATA_DIR="/home/sgomezro/scratch/metrics/Exp"
LOGS_DIR="parallel_tests"
# ──────────────────────────────────────────────────────────────────────────────

echo "Starting parallel training for experiments: ${EXPERIMENTS[*]}"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading (shared across all processes)
export OMP_NUM_THREADS=2           # Lower per-process since sharing CPUs

# Activate Nvidia MPS — allows multiple CUDA processes to share one GPU
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
sleep 10                            # Give MPS daemon time to start

# Pre-warm the CUDA context
python -c "import torch; torch.zeros(1).cuda(); print('CUDA context ready')"
sleep 2

# Create output dirs for all experiments upfront
for EXP_ID in "${EXPERIMENTS[@]}"; do
    mkdir -p "${LOGS_DIR}/Exp_${EXP_ID}"
done

# Launch all experiments in parallel (background processes)
PIDS=()
for EXP_ID in "${EXPERIMENTS[@]}"; do
    echo "Launching experiment ${EXP_ID}..."
    sleep $((i * 30))    # 0s, 30s, 60s, 90s stagger
    python main.py -g 0 -e "${EXP_ID}" -d "${DATA_DIR}" \
        > "${LOGS_DIR}/Exp_${EXP_ID}/parallel_output.log" \
        2> "${LOGS_DIR}/Exp_${EXP_ID}/parallel_error.log" &
    PIDS+=($!)
    sleep 3                        # Small stagger to avoid race conditions at startup
done

echo "All experiments launched. PIDs: ${PIDS[*]}"

# Wait for all experiments and collect exit codes
FAILED=()
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    EXP_ID=${EXPERIMENTS[$i]}
    if wait "$PID"; then
        echo "Experiment ${EXP_ID} (PID ${PID}) completed successfully."
    else
        echo "Experiment ${EXP_ID} (PID ${PID}) FAILED with exit code $?."
        FAILED+=("${EXP_ID}")
    fi
done

# Shut down MPS daemon cleanly
echo quit | nvidia-cuda-mps-control

# Report summary
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All experiments completed successfully."
else
    echo "The following experiments FAILED: ${FAILED[*]}"
    exit 1
fi
