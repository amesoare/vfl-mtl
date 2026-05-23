#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --job-name=vfl_mia
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-mia.out
#SBATCH --error=slurm-%j-mia.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== Embedding MIA started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python attacks/embedding_mia.py \
    --splits_dir $SPLITS_DIR \
    --ckpt_dir checkpoints \
    --output results/embedding_mia.csv \
    --split test \
    --device cpu

echo "=== Embedding MIA finished: $(date) ==="
