#!/bin/bash
#SBATCH --job-name=sens_7147_7150
#SBATCH --output=drac/logs/sens_7147_7150_%j.log
#SBATCH --error=drac/logs/sens_7147_7150_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7147 7148 7149 7150"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7147 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7147/output.log 2>experiments/Exp_7147/error.log &
PID_7147=$!
python main.py -g 0 -e 7148 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7148/output.log 2>experiments/Exp_7148/error.log &
PID_7148=$!
python main.py -g 0 -e 7149 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7149/output.log 2>experiments/Exp_7149/error.log &
PID_7149=$!
python main.py -g 0 -e 7150 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7150/output.log 2>experiments/Exp_7150/error.log &
PID_7150=$!

wait $PID_7147; E_7147=$?
wait $PID_7148; E_7148=$?
wait $PID_7149; E_7149=$?
wait $PID_7150; E_7150=$?

if [ $E_7147 -ne 0 ] || [ $E_7148 -ne 0 ] || [ $E_7149 -ne 0 ] || [ $E_7150 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7147 7148 7149 7150"
