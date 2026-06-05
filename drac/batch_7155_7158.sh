#!/bin/bash
#SBATCH --job-name=sens_7155_7158
#SBATCH --output=drac/logs/sens_7155_7158_%j.log
#SBATCH --error=drac/logs/sens_7155_7158_%j.err
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=34:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting experiments: 7155 7156 7157 7158"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4

mkdir -p drac/logs

python main.py -g 0 -e 7155 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7155/output.log 2>experiments/Exp_7155/error.log &
PID_7155=$!
python main.py -g 0 -e 7156 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7156/output.log 2>experiments/Exp_7156/error.log &
PID_7156=$!
python main.py -g 0 -e 7157 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7157/output.log 2>experiments/Exp_7157/error.log &
PID_7157=$!
python main.py -g 0 -e 7158 -d "/home/epigou/scratch/metrics/Exp" > experiments/Exp_7158/output.log 2>experiments/Exp_7158/error.log &
PID_7158=$!

wait $PID_7155; E_7155=$?
wait $PID_7156; E_7156=$?
wait $PID_7157; E_7157=$?
wait $PID_7158; E_7158=$?

if [ $E_7155 -ne 0 ] || [ $E_7156 -ne 0 ] || [ $E_7157 -ne 0 ] || [ $E_7158 -ne 0 ]; then
    echo "One or more experiments failed"
    exit 1
fi

echo "All experiments finished: 7155 7156 7157 7158"
