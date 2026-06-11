#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:30:00
#SBATCH --mem=56G
#SBATCH --job-name=vfl_dp_eps5
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-dp_eps5.out
#SBATCH --error=slurm-%j-dp_eps5.err

# ε=5 uniform + stratified (ε_IHM=2, ε_Decomp=2, ε_Pheno=1).
# Stratified variant runs here because it shares ε_total=5 with uniform.
# 6 runs total (3 uniform + 3 stratified) × 100 rounds × ~1.2 overhead ≈ 1.2h.
# 8h wall time is 6× safety margin.
# Must finish before run_attacks.sh and run_ablations_dp.sh are submitted.

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== DP curves ε=5 (uniform + stratified) started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python -u experiments/privacy_utility_curves.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 100 \
    --batch_size 64 \
    --device cuda \
    --epsilon_levels 5.0 \
    --run_stratified \
    --output results/privacy_utility_eps5.csv

echo "=== DP curves ε=5 finished: $(date) ==="
