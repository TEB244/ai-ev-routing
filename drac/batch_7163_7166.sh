#!/bin/bash
#SBATCH --job-name=sens_7163_7166
#SBATCH --output=drac/logs/sens_7163_7166_%j.log
#SBATCH --error=drac/logs/sens_7163_7166_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=34:00:00
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7163 7164 7165 7166"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7163 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7163/output.log 2>experiments/Exp_7163/error.log &
PID_7163=$!
python main.py -g 0 -e 7164 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7164/output.log 2>experiments/Exp_7164/error.log &
PID_7164=$!
python main.py -g 0 -e 7165 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7165/output.log 2>experiments/Exp_7165/error.log &
PID_7165=$!
python main.py -g 0 -e 7166 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7166/output.log 2>experiments/Exp_7166/error.log &
PID_7166=$!

wait $PID_7163; E_7163=$?
wait $PID_7164; E_7164=$?
wait $PID_7165; E_7165=$?
wait $PID_7166; E_7166=$?

if [ $E_7163 -ne 0 ] || [ $E_7164 -ne 0 ] || [ $E_7165 -ne 0 ] || [ $E_7166 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7163 7164 7165 7166"
