#!/bin/bash
#SBATCH --job-name=sens_7159_7162
#SBATCH --output=drac/logs/sens_7159_7162_%j.log
#SBATCH --error=drac/logs/sens_7159_7162_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7159 7160 7161 7162"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7159 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7159/output.log 2>experiments/Exp_7159/error.log &
PID_7159=$!
python main.py -g 0 -e 7160 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7160/output.log 2>experiments/Exp_7160/error.log &
PID_7160=$!
python main.py -g 0 -e 7161 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7161/output.log 2>experiments/Exp_7161/error.log &
PID_7161=$!
python main.py -g 0 -e 7162 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7162/output.log 2>experiments/Exp_7162/error.log &
PID_7162=$!

wait $PID_7159; E_7159=$?
wait $PID_7160; E_7160=$?
wait $PID_7161; E_7161=$?
wait $PID_7162; E_7162=$?

if [ $E_7159 -ne 0 ] || [ $E_7160 -ne 0 ] || [ $E_7161 -ne 0 ] || [ $E_7162 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7159 7160 7161 7162"
