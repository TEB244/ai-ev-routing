#!/bin/bash
#SBATCH --job-name=sens_7139_7142
#SBATCH --output=drac/logs/sens_7139_7142_%j.log
#SBATCH --error=drac/logs/sens_7139_7142_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=34:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7139 7140 7141 7142"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7139 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7139/output.log 2>experiments/Exp_7139/error.log &
PID_7139=$!
python main.py -g 0 -e 7140 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7140/output.log 2>experiments/Exp_7140/error.log &
PID_7140=$!
python main.py -g 0 -e 7141 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7141/output.log 2>experiments/Exp_7141/error.log &
PID_7141=$!
python main.py -g 0 -e 7142 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7142/output.log 2>experiments/Exp_7142/error.log &
PID_7142=$!

wait $PID_7139; E_7139=$?
wait $PID_7140; E_7140=$?
wait $PID_7141; E_7141=$?
wait $PID_7142; E_7142=$?

if [ $E_7139 -ne 0 ] || [ $E_7140 -ne 0 ] || [ $E_7141 -ne 0 ] || [ $E_7142 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7139 7140 7141 7142"
