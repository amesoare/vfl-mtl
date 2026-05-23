#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --mem=56G
#SBATCH --job-name=vfl_central
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-central.out
#SBATCH --error=slurm-%j-central.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== Centralized oracle started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python baselines/centralized.py \
    --root $WORKDIR \
    --n_epochs 100 \
    --patience 15 \
    --output results/centralized.csv \
    --ckpt_dir $WORKDIR/checkpoints

echo "=== Centralized oracle finished: $(date) ==="
