#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --mem=56G
#SBATCH --job-name=vfl_dp_eps2
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-dp_eps2.out
#SBATCH --error=slurm-%j-dp_eps2.err

# ε=2 uniform — moderate DP budget.
# 3 seeds × 100 rounds × ~1.2 DP overhead ≈ 0.6h; 6h is 10× safety margin.
# Must finish before run_attacks.sh is submitted.

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== DP curves ε=2 started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python -u experiments/privacy_utility_curves.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 100 \
    --batch_size 64 \
    --device cuda \
    --epsilon_levels 2.0 \
    --output results/privacy_utility_eps2.csv

echo "=== DP curves ε=2 finished: $(date) ==="
