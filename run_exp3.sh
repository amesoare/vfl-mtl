#!/bin/bash
#SBATCH --job-name=vfl_exp3
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

echo "[$(date)] Starting Exp 3: Task relatedness and negative transfer"

source $VENV/bin/activate

cd $WORKDIR

python experiments/run_exp3.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 50 \
    --batch_size 64 \
    --device cuda \
    --output $WORKDIR/results/exp3.csv

echo "[$(date)] Exp 3 complete."
