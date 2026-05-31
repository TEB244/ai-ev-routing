#!/bin/bash
#SBATCH --job-name=sens_7151_7154
#SBATCH --output=drac/logs/sens_7151_7154_%j.log
#SBATCH --error=drac/logs/sens_7151_7154_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=34:00:00
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7151 7152 7153 7154"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7151 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7151/output.log 2>experiments/Exp_7151/error.log &
PID_7151=$!
python main.py -g 0 -e 7152 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7152/output.log 2>experiments/Exp_7152/error.log &
PID_7152=$!
python main.py -g 0 -e 7153 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7153/output.log 2>experiments/Exp_7153/error.log &
PID_7153=$!
python main.py -g 0 -e 7154 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7154/output.log 2>experiments/Exp_7154/error.log &
PID_7154=$!

wait $PID_7151; E_7151=$?
wait $PID_7152; E_7152=$?
wait $PID_7153; E_7153=$?
wait $PID_7154; E_7154=$?

if [ $E_7151 -ne 0 ] || [ $E_7152 -ne 0 ] || [ $E_7153 -ne 0 ] || [ $E_7154 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7151 7152 7153 7154"
