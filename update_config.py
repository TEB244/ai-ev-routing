import os

base_dir = "experiments/"

exp_range = [2037, 2072]

for exp_num in range(exp_range[0], exp_range[1]):
    exp_dir = os.path.join(base_dir, f"Exp_{exp_num:04d}")

    # Load train_job.sh
    with open(os.path.join(exp_dir, "train_job.sh"), "r") as f:
        train_job = f.read()

    train_job = train_job.replace("--time=48:00:00", "--time=48:00:00\n#SBATCH --mem=12G\n#SBATCH --gpus-per-node=2")

    train_job = train_job.replace("-verb True ", "-g 0 1 -verb True ")

    # Save train_job.sh
    with open(os.path.join(exp_dir, "train_job.sh"), "w") as f:
        f.write(train_job)
