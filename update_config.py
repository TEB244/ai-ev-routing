import os

base_dir = "experiments"

# Get all experiment directories
for dir in os.listdir(base_dir):
    if dir.startswith("Exp_"):
        if os.path.isdir(os.path.join(base_dir, dir)):
            train_job_path = os.path.join(base_dir, dir, "train_job.sh")
            eval_job_path = os.path.join(base_dir, dir, "eval_job.sh")

            if os.path.exists(train_job_path):

                with open(train_job_path, "r") as f:
                    train_job = f.read()
                    train_job = train_job.replace("app_v2.py", "main.py")
                    
                with open(train_job_path, "w") as f:
                    f.write(train_job)

            if os.path.exists(eval_job_path):

                with open(eval_job_path, "r") as f:
                    eval_job = f.read()
                    eval_job = eval_job.replace("app_v2.py", "main.py")
                    
                with open(eval_job_path, "w") as f:
                    f.write(eval_job)