#!/bin/bash
#SBATCH --job-name=eda_analysis7
#SBATCH --partition=rome
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j-analysis7.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j-analysis7.err

set -euo pipefail

source /home/asoare/vfl_mlt/.venv/bin/activate
cd /home/asoare/vfl_mlt

echo "=== Analysis 7 started: $(date) ==="
python figures/run_analysis7.py
echo "=== Analysis 7 finished: $(date) ==="
