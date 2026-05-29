#!/bin/bash
#SBATCH --job-name=sens_7171_7174
#SBATCH --output=drac/logs/sens_7171_7174_%j.log
#SBATCH --error=drac/logs/sens_7171_7174_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=22:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7171 7172 7173 7174"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7171 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7171/output.log 2>experiments/Exp_7171/error.log &
PID_7171=$!
python main.py -g 0 -e 7172 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7172/output.log 2>experiments/Exp_7172/error.log &
PID_7172=$!
python main.py -g 0 -e 7173 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7173/output.log 2>experiments/Exp_7173/error.log &
PID_7173=$!
python main.py -g 0 -e 7174 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7174/output.log 2>experiments/Exp_7174/error.log &
PID_7174=$!

wait $PID_7171; E_7171=$?
wait $PID_7172; E_7172=$?
wait $PID_7173; E_7173=$?
wait $PID_7174; E_7174=$?

if [ $E_7171 -ne 0 ] || [ $E_7172 -ne 0 ] || [ $E_7173 -ne 0 ] || [ $E_7174 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7171 7172 7173 7174"
