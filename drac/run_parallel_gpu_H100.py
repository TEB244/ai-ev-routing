#!/usr/bin/env python3
"""
submit_parallel_experiments.py

Generates a SLURM batch script for parallel experiment training on DRAC
and optionally submits it via sbatch.

Usage:
    python submit_parallel_experiments.py [--dry-run] [--output PATH]

Examples:
    # Generate and submit immediately
    python submit_parallel_experiments.py

    # Preview the generated script without submitting
    python submit_parallel_experiments.py --dry-run

    # Save script to a custom path and submit
    python submit_parallel_experiments.py --output my_job.sh
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
    "stagger_seconds": 30,
    "startup_stagger": 3,
    "omp_num_threads": 2, 
    "data_dir": f"scratch/metrics/Exp",
    "parallel_dir": "parallel_tests/",

}
config_odt = {
    "batch_size": 8,
    "time": "01:00:00",
    "cpu_per_experiment": 4,
    "mem_per_experiment": 10,
    "gpus": "h100:1",
}

# ──────────────────────────────────────────────────────────────────────────────


def generate_script(config, experiment_list) -> str:
    """Generate the SLURM batch script as a string."""

    job_name = f"Exp_parallel_train_{experiment_list[0]}_{experiment_list[-1]}-GPUH100"
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
        f"#SBATCH --ntasks={config['ntasks']}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time}",
        f"#SBATCH --mem={mem}G",
        f"#SBATCH --gpus={config['gpus']}",
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
        "# Activate Nvidia MPS — allows multiple CUDA processes to share one GPU",
        "export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps",
        "export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log",
        "nvidia-cuda-mps-control -d",
        "sleep 10                            # Give MPS daemon time to start",
        "",
        "# Pre-warm the CUDA context",
        "python -c \"import torch; torch.zeros(1).cuda(); print('CUDA context ready')\"",
        "sleep 2",
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
        f"    sleep {config['stagger_seconds']}",
        '    python main.py -g 0 -e "${EXP_ID}" -d "${DATA_DIR}" \\',
        f'        > "${{LOGS_DIR}}/Exp_${{EXP_ID}}/output.log" \\',
        f'        2> "${{LOGS_DIR}}/Exp_${{EXP_ID}}/error.log" &',
        "    PIDS+=($!)",
        f"    sleep {config['startup_stagger']}                        # Small stagger to avoid race conditions at startup",
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
        "# Shut down MPS daemon cleanly",
        "echo quit | nvidia-cuda-mps-control",
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


def run_parallel_gpu_H100(exp_list: list, algorithm: str, args: argparse.Namespace) -> None:
    
    if algorithm == 'ODT':
        config = {**config_general, **config_odt}
    else:
        print(f"Algorithm {algorithm} not supported yet. Need to add config information to run under H100 for this algorithm. Exiting...")
        exit(1)
    
    batches = [exp_list[i:i + config['batch_size']] for i in range(0, len(exp_list), config['batch_size'])]
    for experiment_list in batches:
        script_content = generate_script(config, experiment_list)

        script_path = write_script(script_content, f"{config['parallel_dir']}batch_job/job_{experiment_list[0]}-{experiment_list[-1]}.sh")

        if args.dry_run:
            print("─── Generated SLURM script (dry run) ───────────────────────────────────────")
            print(script_content)
            print("─────────────────────────────────────────────────────────────────────────────")
            print(f" Dry run — script saved at {script_path}.")
            return
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
    run_parallel_gpu_H100(args.experiments_list, args.algorithm, args)