#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --mem=56G
#SBATCH --job-name=vfl_abl_dp
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-ablations_dp.out
#SBATCH --error=slurm-%j-ablations_dp.err

# DP ablations (Paper 2):
#   Abl 1 — uniform σ vs. task-stratified σ at ε_total=5
#            Reads test_results_dp.csv (no new training).
#            PREREQUISITE: evaluate_test_dp job must be complete.
#   Abl 2 — related task pair (IHM+Decomp) vs. unrelated (IHM+Pheno) under DP.
#            Trains 2 task configs × 3 seeds at ε=5; logs gradient similarity (ρ).
#   Abl 3 — embed_dim ∈ {32, 64, 128} × ε ∈ {1, 5, ∞}.
#            Trains 3 × 3 × 3 = 27 configs; validates ε* conditional on embed_dim=64.
#
# Total training: ~33 runs × ~25 min/run × 1.2 DP overhead ≈ 10–12h.

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== DP ablations started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python experiments/ablations_dp.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 100 \
    --batch_size 64 \
    --device cuda \
    --test_dp_csv results/test_results_dp.csv \
    --output results/dp_ablations.csv \
    --split test

echo "=== DP ablations finished: $(date) ==="
