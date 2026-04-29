#!/bin/bash
#SBATCH --job-name=vfl_baselines
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

echo "[$(date)] Starting baselines: local-only (A, B, C) + centralized oracle"

source $VENV/bin/activate

cd $WORKDIR

python baselines/local_only.py \
    --site A \
    --root $WORKDIR \
    --n_epochs 50 \
    --batch_size 64 \
    --seeds 42 123 7 \
    --output $WORKDIR/results/local_only_A.csv

echo "[$(date)] local_only A done."

python baselines/local_only.py \
    --site B \
    --root $WORKDIR \
    --n_epochs 50 \
    --batch_size 64 \
    --seeds 42 123 7 \
    --output $WORKDIR/results/local_only_B.csv

echo "[$(date)] local_only B done."

python baselines/local_only.py \
    --site C \
    --root $WORKDIR \
    --n_epochs 50 \
    --batch_size 64 \
    --seeds 42 123 7 \
    --output $WORKDIR/results/local_only_C.csv

echo "[$(date)] local_only C done."

python baselines/centralized.py \
    --root $WORKDIR \
    --n_epochs 50 \
    --batch_size 64 \
    --seeds 42 123 7 \
    --output $WORKDIR/results/centralized.csv

echo "[$(date)] Centralized oracle done."

# Aggregate into comparison table (VFL-MTL loaded from results/exp1.csv)
python experiments/run_baselines.py \
    --skip_rerun \
    --local_a  $WORKDIR/results/local_only_A.csv \
    --local_b  $WORKDIR/results/local_only_B.csv \
    --local_c  $WORKDIR/results/local_only_C.csv \
    --central  $WORKDIR/results/centralized.csv \
    --vfl_results $WORKDIR/results/exp1.csv \
    --output $WORKDIR/results/baselines_comparison.csv

echo "[$(date)] Baselines complete. Results → $WORKDIR/results/"
