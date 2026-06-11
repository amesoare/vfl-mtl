#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --job-name=vfl_bound
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-bound_validation.out
#SBATCH --error=slurm-%j-bound_validation.err

# Multi-task label inference bound validation (Paper 2, Figure 3).
# Pure analysis — computes g(σ, ρ) = Φ(C√(1+ρ)/σ) and compares to
# empirical label inference AUROCs from results/label_inference.csv.
# No training; runs in seconds. CPU-only (no GPU needed).
#
# PREREQUISITE: run_attacks.sh must be complete (label_inference.csv must exist).

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== Bound validation started: $(date) ==="

python experiments/validate_bound.py \
    --label_inference_csv results/label_inference.csv \
    --splits_dir $SPLITS_DIR \
    --n_rounds 100 \
    --batch_size 64 \
    --output results/bound_validation.csv

echo "=== Bound validation finished: $(date) ==="
