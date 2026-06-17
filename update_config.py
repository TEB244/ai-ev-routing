import os
import re
base_dir = "experiments/"

# exp_range = [9090, 9134]
exp_range = [4072, 4107]
for exp_num in range(exp_range[0], exp_range[1]+1):
    exp_dir = os.path.join(base_dir, f"Exp_{exp_num:04d}")

    # Load train_job.sh
    with open(os.path.join(exp_dir, "train_job.sh"), "r") as f:
        train_job = f.read()

    # train_job = train_job.replace("--time=48:00:00", "--time=48:00:00\n#SBATCH --mem=12G\n#SBATCH --gpus-per-node=2")

    # train_job = train_job.replace("--cpus-per-task=2", "--cpus-per-task=1")
    # train_job = train_job.replace("--mem=1792M", "--mem=1608M")
    # train_job = train_job.replace("--mem=6G", "--mem=3584M")
    # train_job = train_job.replace("--time=*", "--time=18:30:00")
    train_job = re.sub(r'--time=\d{2}:\d{2}:\d{2}', '--time=18:30:00', train_job)

    # Save train_job.sh
    with open(os.path.join(exp_dir, "train_job.sh"), "w") as f:
        f.write(train_job)
