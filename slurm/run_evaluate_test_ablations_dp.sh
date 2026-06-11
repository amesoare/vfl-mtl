#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_abl_dp_test_eval
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-abl_dp_test_eval.out
#SBATCH --error=slurm-%j-abl_dp_test_eval.err

# Test-set evaluation for Paper 2 DP ablation checkpoints (Abl 2 + Abl 3).
# Loads best checkpoints saved by run_ablations_dp.sh — no re-training.
#
# Abl 2: best_abl2_{ihm_decomp,ihm_pheno}_seed{N}.pt
# Abl 3: best_abl3_embed{D}_eps{E}_seed{N}.pt
#
# Output: results/test_ablations_dp.csv
#
# PREREQUISITE: run_ablations_dp.sh must be complete and checkpoints present.

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== DP ablation test-set evaluation started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

python experiments/evaluate_test_ablations_dp.py \
    --root $WORKDIR \
    --ckpt_dir checkpoints \
    --batch_size 64 \
    --device cpu \
    --output results/test_ablations_dp.csv

echo "=== DP ablation test-set evaluation finished: $(date) ==="
