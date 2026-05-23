#!/bin/bash
# Launcher: submits exp1, exp2, exp3, centralized as 4 parallel GPU jobs.
# Run from the Snellius login node: bash run_gpu_all.sh
# Do NOT submit this script itself with sbatch.

set -euo pipefail

WORKDIR=/home/asoare/vfl_mlt
cd $WORKDIR

echo "Submitting GPU experiment jobs..."
sbatch run_exp1.sh
sbatch run_exp2.sh
sbatch run_exp3.sh
sbatch run_centralized_gpu.sh
echo "All GPU jobs submitted. Monitor with: squeue -u $USER"
