#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --job-name=vfl_ablations
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-ablations.out
#SBATCH --error=slurm-%j-ablations.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== Ablations started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python experiments/run_ablations.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 100 \
    --patience 15 \
    --device cuda \
    --output results/ablations.csv

echo "=== Ablations finished: $(date) ==="
