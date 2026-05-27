#!/bin/bash
#SBATCH --job-name=batch_test_9135_9138
#SBATCH --output=drac/logs/batch_test_9135_9138_%j.log
#SBATCH --error=drac/logs/batch_test_9135_9138_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting parallel batch test: Exp 9135, 9136, 9137, 9138 (all on GPU 0 via MPS)"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=1

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

mkdir -p drac/logs

python main.py -g 0 -e 9135 -d "/scratch/epigou/metrics/Exp" > experiments/Exp_9135/output.log 2> experiments/Exp_9135/error.log &
python main.py -g 0 -e 9136 -d "/scratch/epigou/metrics/Exp" > experiments/Exp_9136/output.log 2> experiments/Exp_9136/error.log &
python main.py -g 0 -e 9137 -d "/scratch/epigou/metrics/Exp" > experiments/Exp_9137/output.log 2> experiments/Exp_9137/error.log &
python main.py -g 0 -e 9138 -d "/scratch/epigou/metrics/Exp" > experiments/Exp_9138/output.log 2> experiments/Exp_9138/error.log &

wait
echo "All experiments finished"
