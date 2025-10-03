import os

base_dir = "experiments/"

exp_range = [2000, 2072]

for exp_num in range(exp_range[0], exp_range[1]):
    exp_dir = os.path.join(base_dir, f"Exp_{exp_num:04d}")

    # Load train_job.sh
    with open(os.path.join(exp_dir, "train_job.sh"), "r") as f:
        train_job = f.read()

    # Change --time=02:00:00 to --time=48:00:00
    train_job = train_job.replace("--time=02:00:00", "--time=48:00:00")

    # Change --mem=6G to --mem=12G
    train_job = train_job.replace("--mem=6G", "--mem=12G")

    # Change /home/sgomezro/scratch/metrics/Exp to /home/hartman/scratch/metrics/Exp
    train_job = train_job.replace("/home/sgomezro/scratch/metrics/Exp", "/home/hartman/scratch/metrics/Exp")

    # Save train_job.sh
    with open(os.path.join(exp_dir, "train_job.sh"), "w") as f:
        f.write(train_job)
