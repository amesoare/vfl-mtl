#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --job-name=vfl_test_eval_dp
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-test_eval_dp.out
#SBATCH --error=slurm-%j-test_eval_dp.err

# Final test-set evaluation for all Paper 2 DP checkpoints.
# Loads saved checkpoints — no re-training.
# Produces results/test_results_dp.csv with:
#   - Task AUC on test set (IHM, Decomp, Pheno) per (ε, mode, seed)
#   - Label inference attack (LR probe on train embeddings, tested on test)
#   - MIA (train = members, test = non-members)
#
# PREREQUISITE: all run_privacy_curves_eps_*.sh jobs must be complete
# (checkpoints must exist in checkpoints/).
#
# Wall time estimate: 7 combos × 3 seeds × ~5 min inference = ~1.75h;
# 2h is a safe margin (inference only — no training).

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== Test-set DP evaluation started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python experiments/evaluate_test_dp.py \
    --splits_dir $SPLITS_DIR \
    --ckpt_dir checkpoints \
    --output results/test_results_dp.csv \
    --num_workers 4 \
    --batch_size 64 \
    --device cpu

echo "=== Test-set DP evaluation finished: $(date) ==="
