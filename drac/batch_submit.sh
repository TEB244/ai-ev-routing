#!/bin/bash
#SBATCH --job-name=odt_batch
#SBATCH --output=drac/logs/odt_batch_%j.log
#SBATCH --error=drac/logs/odt_batch_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

# Usage: sbatch drac/batch_submit.sh 9135 9136 9137 ... (up to ~16 experiments)
# Experiments are assigned round-robin across 4 GPUs via MPS.

EXPERIMENTS=("$@")
NUM_GPUS=4

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "No experiments provided. Usage: sbatch batch_submit.sh 9135 9136 ..."
    exit 1
fi

echo "Starting ${#EXPERIMENTS[@]} experiments across ${NUM_GPUS} GPUs: ${EXPERIMENTS[*]}"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

mkdir -p drac/logs

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    gpu=$(( i % NUM_GPUS ))
    echo "Launching Exp_${exp} on GPU ${gpu}"
    python main.py -g ${gpu} -e ${exp} -d "/scratch/epigou/metrics/Exp" \
        > experiments/Exp_${exp}/output.log \
        2> experiments/Exp_${exp}/error.log &
done

wait
echo "All experiments finished"
