#!/bin/bash
#SBATCH --job-name=sens_7179_7179
#SBATCH --output=drac/logs/sens_7179_7179_%j.log
#SBATCH --error=drac/logs/sens_7179_7179_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=34:00:00
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7179"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7179 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7179/output.log 2>experiments/Exp_7179/error.log &
PID_7179=$!

wait $PID_7179; E_7179=$?

if [ $E_7179 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7179"
