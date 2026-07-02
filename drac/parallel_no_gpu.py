#!/usr/bin/env python3
"""
parallel_no_gpu.py

Generates a SLURM batch script for parallel experiment training on DRAC (no GPU required)
and optionally submits it via sbatch.

Usage:
    python parallel_no_gpu.py [--dry-run] [--output PATH]

Examples:
    # Generate and submit immediately
    python parallel_no_gpu.py

    # Preview the generated script without submitting
    python parallel_no_gpu.py --dry-run

    # Save script to a custom path and submit
    python parallel_no_gpu.py --output my_job.sh
"""

import argparse
import os
import subprocess
import sys
import getpass
import socket
from pathlib import Path



# ─── User-Configurable Variables ──────────────────────────────────────────────
config_general = {
    "account": "def-mcapretz",
    "ntasks": 1,
    "omp_num_threads": 2, 
    "data_dir": f"scratch/metrics/Exp",
    "parallel_dir": "experiments/",
    "email": "lhartma8@uwo.ca",
}

config_cma = {
    "batch_size": 8,
    "time": "18:30:00",
    "cpu_per_experiment": 5,
    "mem_per_experiment": 3584 # Mbytes
}

# config_dqn = {
#     "batch_size": 8,
#     "time": "13:00:00",
#     "cpu_per_experiment": 6,
#     "mem_per_experiment": 6 # Gigabytes
# }

# config_reinforce = {
#     "batch_size": 8,
#     "time": "12:00:00",
#     "cpu_per_experiment": 6,
#     "mem_per_experiment": 32 # Gigabytes
# }

# ──────────────────────────────────────────────────────────────────────────────


def generate_script(config, experiment_list) -> str:
    """Generate the SLURM batch script as a string."""

    job_name = f"Exp_parallel_train_{experiment_list[0]}_{experiment_list[-1]}-NoGPU"
    exp_array = " ".join(str(e) for e in experiment_list)

    # Retrieve DRAC username and cluster
    drac_username = os.environ.get("DRAC_USERNAME") or getpass.getuser()
    cluster_str = socket.gethostname()
    if "rorqual" in cluster_str:
        data_dir = f"/home/{drac_username}/links/{config['data_dir']}"
    else:
        data_dir = f"/home/{drac_username}/{config['data_dir']}"
    # Build SBATCH directives
    experiment_size = len(experiment_list)
    cpus_per_task = config['cpu_per_experiment']*experiment_size
    mem = config['mem_per_experiment']*experiment_size*(1.2 if experiment_size < 4 else 1)
    mem_post = "G" if mem < 1024 else "MB"
    time = config['time']
    output_dir = config['parallel_dir'] + f"batch_train_{experiment_list[0]}-{experiment_list[-1]}"
    output_log = output_dir + "/output.log"
    error_log = output_dir + "/error.log"
    parallel_dir = config['parallel_dir']

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_log}",
        f"#SBATCH --error={error_log}",
        f"#SBATCH -A {config['account']}",
        "#SBATCH --mail-type=FAIL,TIME_LIMIT",
        f"#SBATCH --mail-user={config['email']}",
        f"#SBATCH --ntasks={config['ntasks']}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time}",
        f"#SBATCH --mem={int(mem)}{mem_post}",
        "",
        "",
        "# ─── Configuration ────────────────────────────────────────────────────────────",
        f"EXPERIMENTS=({exp_array})",
        f'DATA_DIR="{data_dir}"',
        f'LOGS_DIR="{parallel_dir}"',
        "# ──────────────────────────────────────────────────────────────────────────────",
        "",
        'echo "Starting parallel training for experiments: ${EXPERIMENTS[*]}"',
        "",
        "set -e",
        "",
        f"module load python/3.10 cuda cudnn",
        f"source ~/envs/merl_env/bin/activate",
        "",
        "# Enable multi-threading (shared across all processes)",
        f"export OMP_NUM_THREADS={config['omp_num_threads']}",
        "",
        "",
        "# Create output dirs for all experiments upfront",
        'for EXP_ID in "${EXPERIMENTS[@]}"; do',
        '    mkdir -p "${LOGS_DIR}/Exp_${EXP_ID}"',
        "done",
        "",
        "# Launch all experiments in parallel (background processes)",
        "PIDS=()",
        'for EXP_ID in "${EXPERIMENTS[@]}"; do',
        '    echo "Launching experiment ${EXP_ID}..."',
        '    python main.py -e "${EXP_ID}" -d "${DATA_DIR}" \\',
        f'        > "${{LOGS_DIR}}/Exp_${{EXP_ID}}/output.log" \\',
        f'        2> "${{LOGS_DIR}}/Exp_${{EXP_ID}}/error.log" &',
        "    PIDS+=($!)",
        "done",
        "",
        'echo "All experiments launched. PIDs: ${PIDS[*]}"',
        "",
        "# Wait for all experiments and collect exit codes",
        "FAILED=()",
        'for i in "${!PIDS[@]}"; do',
        "    PID=${PIDS[$i]}",
        "    EXP_ID=${EXPERIMENTS[$i]}",
        '    if wait "$PID"; then',
        '        echo "Experiment ${EXP_ID} (PID ${PID}) completed successfully."',
        "    else",
        '        echo "Experiment ${EXP_ID} (PID ${PID}) FAILED with exit code $?."',
        '        FAILED+=("${EXP_ID}")',
        "    fi",
        "done",
        "",
        "# Report summary",
        "if [ ${#FAILED[@]} -eq 0 ]; then",
        '    echo "All experiments completed successfully."',
        "else",
        '    echo "The following experiments FAILED: ${FAILED[*]}"',
        "    exit 1",
        "fi",
        "",
    ]

    return "\n".join(lines)


def write_script(script_content: str, output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script_content)
    path.chmod(0o755)
    print(f"[✓] Script written to: {path.resolve()}")
    return path


def submit_job(script_path: Path) -> None:
    print(f"[→] Submitting job: sbatch {script_path}")
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"[✓] {result.stdout.strip()}")
    else:
        print(f"[✗] sbatch failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(result.returncode)


def run_parallel_no_gpu(exp_bounds: list, algorithm: str, args: argparse.Namespace = None) -> None:
    
    if algorithm == 'CMA':
        config = {**config_general, **config_cma}
    # elif algorithm == 'DQN':
    #     config = {**config_general, **config_dqn}
    # elif algorithm == 'REINFORCE':
    #     config = {**config_general, **config_reinforce}
    else:
        print(f"Algorithm {algorithm} not supported yet. Need to add config information to run this algorithm. Exiting...")
        exit(1)
    exp_list = list(range(exp_bounds[0], exp_bounds[1] + 1))
    batches = [exp_list[i:i + config['batch_size']] for i in range(0, len(exp_list), config['batch_size'])]
    print(f"debug line 181: {batches}")
    for experiment_list in batches:
        script_content = generate_script(config, experiment_list)

        script_path = write_script(script_content, f"{config['parallel_dir']}batch_job/job_{experiment_list[0]}-{experiment_list[-1]}.sh")

        if hasattr(args, "dry_run") and args.dry_run == True:
            print("─── Generated SLURM script (dry run) ───────────────────────────────────────")
            print(script_content)
            print("─────────────────────────────────────────────────────────────────────────────")
            print(f" Dry run — script saved at {script_path}.")
        else:   
            print(f"Submitting job {script_path}...")
            submit_job(script_path)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit a SLURM parallel experiment job."
    )
    parser.add_argument(
        "-d","--dry-run",
        type=bool,
        default=False,
        help="Print the generated script and exit without submitting.",
    )
    parser.add_argument('-e','--experiments_list', nargs='*', type=int, default=[], help ='Get the list of experiment to run.')
    parser.add_argument('-a','--algorithm', type=str, help='Algorithm to run.')
    args = parser.parse_args()
    run_parallel_no_gpu(args.experiments_list, args.algorithm, args)