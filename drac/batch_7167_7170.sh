#!/bin/bash
#SBATCH --job-name=sens_7167_7170
#SBATCH --output=drac/logs/sens_7167_7170_%j.log
#SBATCH --error=drac/logs/sens_7167_7170_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=34:00:00
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7167 7168 7169 7170"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7167 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7167/output.log 2>experiments/Exp_7167/error.log &
PID_7167=$!
python main.py -g 0 -e 7168 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7168/output.log 2>experiments/Exp_7168/error.log &
PID_7168=$!
python main.py -g 0 -e 7169 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7169/output.log 2>experiments/Exp_7169/error.log &
PID_7169=$!
python main.py -g 0 -e 7170 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7170/output.log 2>experiments/Exp_7170/error.log &
PID_7170=$!

wait $PID_7167; E_7167=$?
wait $PID_7168; E_7168=$?
wait $PID_7169; E_7169=$?
wait $PID_7170; E_7170=$?

if [ $E_7167 -ne 0 ] || [ $E_7168 -ne 0 ] || [ $E_7169 -ne 0 ] || [ $E_7170 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7167 7168 7169 7170"
