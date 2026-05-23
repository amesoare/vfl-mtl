#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_exp3_2sites
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-exp3-n_sites_2.out
#SBATCH --error=slurm-%j-exp3-n_sites_2.err

# Rerun exp3 n_sites=2 only (server.py uncertainty_weighting bug fix applied).
# n_sites=3 rows in exp3.csv are clean (all tasks active, no zero-weight task issue).
# Writes to results/exp3_n_sites_2.csv — merge into exp3.csv afterwards with:
#   python experiments/merge_exp3_rerun.py

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== exp3 rerun (n_sites=2) started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python -u experiments/run_exp3.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 100 \
    --device cuda \
    --n_sites 2 \
    --output results/exp3_n_sites_2.csv

echo "=== exp3 rerun (n_sites=2) finished: $(date) ==="
