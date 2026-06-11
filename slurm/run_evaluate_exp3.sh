#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_exp3_test_eval
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-exp3_test_eval.out
#SBATCH --error=slurm-%j-exp3_test_eval.err

# Test-set evaluation for Exp 3 (scalability: n_sites in {2, 3}).
# Loads best checkpoint per configuration; no re-training.
# n_sites=2 evaluates IHM + Decomp only (Pheno head untrained).
# Output: results/test_exp3.csv
#
# PREREQUISITE: run_exp3.sh must be complete and checkpoints saved
# under checkpoints/best_exp3_sites{n}_seed{seed}.pt

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== Exp 3 test-set evaluation started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

python experiments/evaluate_exp3.py \
    --root $WORKDIR \
    --ckpt_dir checkpoints \
    --batch_size 64 \
    --device cpu \
    --output results/test_exp3.csv

echo "=== Exp 3 test-set evaluation finished: $(date) ==="
