#!/bin/bash
#SBATCH --job-name=sens_7135_7138
#SBATCH --output=drac/logs/sens_7135_7138_%j.log
#SBATCH --error=drac/logs/sens_7135_7138_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=34:00:00
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7135 7136 7137 7138"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7135 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7135/output.log 2>experiments/Exp_7135/error.log &
PID_7135=$!
python main.py -g 0 -e 7136 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7136/output.log 2>experiments/Exp_7136/error.log &
PID_7136=$!
python main.py -g 0 -e 7137 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7137/output.log 2>experiments/Exp_7137/error.log &
PID_7137=$!
python main.py -g 0 -e 7138 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7138/output.log 2>experiments/Exp_7138/error.log &
PID_7138=$!

wait $PID_7135; E_7135=$?
wait $PID_7136; E_7136=$?
wait $PID_7137; E_7137=$?
wait $PID_7138; E_7138=$?

if [ $E_7135 -ne 0 ] || [ $E_7136 -ne 0 ] || [ $E_7137 -ne 0 ] || [ $E_7138 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7135 7136 7137 7138"
