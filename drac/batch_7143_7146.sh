#!/bin/bash
#SBATCH --job-name=sens_7143_7146
#SBATCH --output=drac/logs/sens_7143_7146_%j.log
#SBATCH --error=drac/logs/sens_7143_7146_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7143 7144 7145 7146"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7143 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7143/output.log 2>experiments/Exp_7143/error.log &
PID_7143=$!
python main.py -g 0 -e 7144 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7144/output.log 2>experiments/Exp_7144/error.log &
PID_7144=$!
python main.py -g 0 -e 7145 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7145/output.log 2>experiments/Exp_7145/error.log &
PID_7145=$!
python main.py -g 0 -e 7146 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7146/output.log 2>experiments/Exp_7146/error.log &
PID_7146=$!

wait $PID_7143; E_7143=$?
wait $PID_7144; E_7144=$?
wait $PID_7145; E_7145=$?
wait $PID_7146; E_7146=$?

if [ $E_7143 -ne 0 ] || [ $E_7144 -ne 0 ] || [ $E_7145 -ne 0 ] || [ $E_7146 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7143 7144 7145 7146"
