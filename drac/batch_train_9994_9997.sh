#!/bin/bash
#SBATCH --job-name=batch_9994_9997
#SBATCH --output=drac/logs/batch_9994_9997_%j.log
#SBATCH --error=drac/logs/batch_9994_9997_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=4

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting parallel batch: Exp 9994, 9995, 9996, 9997"


module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

mkdir -p drac/logs

python main.py -g 0 1 2 3 -e 9994 -server DRAC > experiments/Exp_9994/output.log 2> experiments/Exp_9994/error.log &
python main.py -g 0 1 2 3 -e 9995 -server DRAC > experiments/Exp_9995/output.log 2> experiments/Exp_9995/error.log &
python main.py -g 0 1 2 3 -e 9996 -server DRAC > experiments/Exp_9996/output.log 2> experiments/Exp_9996/error.log &
python main.py -g 0 1 2 3 -e 9997 -server DRAC > experiments/Exp_9997/output.log 2> experiments/Exp_9997/error.log &

wait
echo "All experiments finished"
