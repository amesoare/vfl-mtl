#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --mem=56G
#SBATCH --job-name=vfl_eicu_dp
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-eicu_dp.out
#SBATCH --error=slurm-%j-eicu_dp.err

# Full ε sweep for eICU external validity (Paper 2 PRISM section).
# ε ∈ {∞, 10, 5, 2, 1, 0.5} uniform + stratified at ε_total=5.
# 21 runs total (6×3 uniform + 3 stratified) × 100 rounds.
# Seeds: [42, 123, 7] — matches MIMIC-III sweep for direct comparison.

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/eicu_vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== eICU DP privacy curves started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python -u experiments/privacy_utility_curves.py \
    --dataset eicu \
    --splits_dir $SPLITS_DIR \
    --n_rounds 100 \
    --batch_size 64 \
    --device cuda \
    --run_stratified \
    --num_workers 4 \
    --output results/eicu_privacy_utility_combined.csv

echo "=== eICU DP privacy curves finished: $(date) ==="
