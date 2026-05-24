#!/bin/bash
# Update the SLURM memory request from 6G to 4608M in train_job.sh
# for every experiment from Exp_9090 to Exp_9134.
#
# Usage:
#   bash _scripts/update_mem_9090_9134.sh           # apply changes
#   bash _scripts/update_mem_9090_9134.sh --dry-run # preview without modifying

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"

OLD_LINE="#SBATCH --mem=6G"
NEW_LINE="#SBATCH --mem=4608M"

START_EXP=9090
END_EXP=9134

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "Running in dry-run mode (no files will be modified)."
fi

updated=0
skipped=0
missing=0

for ((exp = START_EXP; exp <= END_EXP; exp++)); do
    file="${EXPERIMENTS_DIR}/Exp_${exp}/train_job.sh"

    if [[ ! -f "${file}" ]]; then
        echo "MISSING: ${file}"
        missing=$((missing + 1))
        continue
    fi

    if ! grep -qE "^${OLD_LINE}$" "${file}"; then
        echo "SKIP   : ${file} (no '${OLD_LINE}' line found)"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "WOULD  : ${file}"
    else
        # Use a portable in-place edit that works on both GNU and BSD sed.
        tmp="$(mktemp)"
        sed "s|^${OLD_LINE}\$|${NEW_LINE}|" "${file}" > "${tmp}"
        mv "${tmp}" "${file}"
        echo "UPDATED: ${file}"
    fi
    updated=$((updated + 1))
done

echo
echo "Summary: updated=${updated}, skipped=${skipped}, missing=${missing}"
