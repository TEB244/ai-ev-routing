#!/bin/bash
#SBATCH --job-name=odt_4zone
#SBATCH --output=drac/logs/odt_4zone_%j.log
#SBATCH --error=drac/logs/odt_4zone_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

# Usage: sbatch drac/batch_4zone_odt.sh <exp_number>

EXP=$1

if [ -z "$EXP" ]; then
    echo "Usage: sbatch drac/batch_4zone_odt.sh <exp_number>"
    exit 1
fi

echo "Starting 4-zone ODT experiment ${EXP} on single GPU via MPS"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=1

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

mkdir -p drac/logs

python main.py -g 0 -e ${EXP} -d "/scratch/epigou/metrics/Exp"

echo "Experiment ${EXP} finished"
