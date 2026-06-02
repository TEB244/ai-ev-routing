#!/bin/bash
# Recreates merl_env on nibi from scratch.
#
# Usage:
#   bash drac/setup_env.sh [env_path]
#
# Default env_path: ~/envs/merl_env
#
# IMPORTANT: Run this from the repo root on a LOGIN NODE, not a compute node.
# IMPORTANT: GPU jobs must use h100 nodes. a100 nodes have AMD EPYC 7742 (Zen 2)
#            which lacks AVX-512 and is incompatible with the nibi software stack.

set -e

ENV_PATH="${1:-$HOME/envs/merl_env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Loading modules ==="
module load python/3.10 cuda/12.2 cudnn

echo "=== Creating virtual environment at $ENV_PATH ==="
virtualenv --no-download "$ENV_PATH"
source "$ENV_PATH/bin/activate"

echo "=== Installing Compute Canada wheels (--no-index) ==="
pip install --no-index -r "$SCRIPT_DIR/cc_wheels_requirements.txt"

echo "=== Installing PyPI packages ==="
pip install -r "$SCRIPT_DIR/pypi_requirements.txt"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  module load python/3.10 cuda/12.2 cudnn"
echo "  source $ENV_PATH/bin/activate"
echo ""
echo "To run an experiment:"
echo "  python main.py -e <experiment_number> -server DRAC -g 0"
echo ""
echo "NOTE: Submit GPU jobs with --gpus-per-node=h100:1 (not a100)."
