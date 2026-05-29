#!/bin/bash
#SBATCH --job-name=sens_7175_7178
#SBATCH --output=drac/logs/sens_7175_7178_%j.log
#SBATCH --error=drac/logs/sens_7175_7178_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=22:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7175 7176 7177 7178"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7175 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7175/output.log 2>experiments/Exp_7175/error.log &
PID_7175=$!
python main.py -g 0 -e 7176 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7176/output.log 2>experiments/Exp_7176/error.log &
PID_7176=$!
python main.py -g 0 -e 7177 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7177/output.log 2>experiments/Exp_7177/error.log &
PID_7177=$!
python main.py -g 0 -e 7178 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7178/output.log 2>experiments/Exp_7178/error.log &
PID_7178=$!

wait $PID_7175; E_7175=$?
wait $PID_7176; E_7176=$?
wait $PID_7177; E_7177=$?
wait $PID_7178; E_7178=$?

if [ $E_7175 -ne 0 ] || [ $E_7176 -ne 0 ] || [ $E_7177 -ne 0 ] || [ $E_7178 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7175 7176 7177 7178"
