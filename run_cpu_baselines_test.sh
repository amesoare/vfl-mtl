#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --job-name=vfl_local_test
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-local-test.out
#SBATCH --error=slurm-%j-local-test.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== CPU smoke test started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

for site in A B C; do
    echo "--- Smoke test: Local-only site $site ---"
    python baselines/local_only.py \
        --site $site \
        --root $WORKDIR \
        --n_epochs 3 \
        --use_synthetic \
        --output results/test_local_only_${site}.csv \
        --ckpt_dir $WORKDIR/checkpoints
done

echo "=== CPU smoke test finished: $(date) ==="
