#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_exp2_test_eval
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-exp2_test_eval.out
#SBATCH --error=slurm-%j-exp2_test_eval.err

# Test-set evaluation for Exp 2 (task relatedness / negative transfer).
# Loads best checkpoint per config; no re-training.
# Configs: all_tasks, ihm_only, ihm_decomp, ihm_pheno  x  3 seeds
# Output: results/test_exp2.csv
#
# PREREQUISITE: run_exp2.sh must be complete and checkpoints saved
# under checkpoints/best_exp2_{config_name}_seed{seed}.pt

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== Exp 2 test-set evaluation started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

python experiments/evaluate_exp2.py \
    --root $WORKDIR \
    --ckpt_dir checkpoints \
    --batch_size 64 \
    --device cpu \
    --output results/test_exp2.csv

echo "=== Exp 2 test-set evaluation finished: $(date) ==="
