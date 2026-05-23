#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_test_eval
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-test_eval.out
#SBATCH --error=slurm-%j-test_eval.err

# Final test-set evaluation for Paper 1 — all models in one pass.
# Loads saved checkpoints; no re-training.
# Covers: VFL-MTL, ST-IHM, ST-Decomp, ST-Pheno, local_A/B/C, centralized_oracle.
# Output: results/test_results.csv
#
# PREREQUISITE: all training jobs must be complete
# (run_exp1.sh, run_baselines.sh, run_centralized.sh).
#
# Wall time: pure inference (~33 forward passes on test set) — well under 1h.

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== Test-set evaluation started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python experiments/evaluate_test.py \
    --root $WORKDIR \
    --ckpt_dir checkpoints \
    --batch_size 64 \
    --device cpu \
    --output results/test_results_nodp.csv

echo "=== Test-set evaluation finished: $(date) ==="
