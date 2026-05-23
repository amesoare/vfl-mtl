#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_gpu_test
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-gpu-test.out
#SBATCH --error=slurm-%j-gpu-test.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== GPU smoke test started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo "--- Smoke test: Exp 1 ---"
python experiments/run_exp1.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 3 \
    --device cuda \
    --use_synthetic \
    --output results/test_exp1.csv

echo "--- Smoke test: Exp 2 ---"
python experiments/run_exp2.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 3 \
    --device cuda \
    --use_synthetic \
    --output results/test_exp2.csv

echo "--- Smoke test: Exp 3 ---"
python experiments/run_exp3.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 3 \
    --device cuda \
    --use_synthetic \
    --output results/test_exp3.csv

echo "--- Smoke test: Centralized ---"
python baselines/centralized.py \
    --root $WORKDIR \
    --n_epochs 3 \
    --use_synthetic \
    --output results/test_centralized.csv \
    --ckpt_dir $WORKDIR/checkpoints

echo "=== GPU smoke test finished: $(date) ==="
