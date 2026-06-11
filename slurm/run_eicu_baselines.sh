#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_eicu_baselines
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-eicu_baselines.out
#SBATCH --error=slurm-%j-eicu_baselines.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== eICU baselines started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

python baselines/local_only.py \
    --site A \
    --root $WORKDIR \
    --dataset eicu \
    --n_epochs 100 \
    --patience 15 \
    --batch_size 64 \
    --seeds 42 123 7 \
    --output $WORKDIR/results/eicu_local_only_A.csv \
    --ckpt_dir $WORKDIR/checkpoints/eicu

echo "[$(date)] local_only A done."

python baselines/local_only.py \
    --site B \
    --root $WORKDIR \
    --dataset eicu \
    --n_epochs 100 \
    --patience 15 \
    --batch_size 64 \
    --seeds 42 123 7 \
    --output $WORKDIR/results/eicu_local_only_B.csv \
    --ckpt_dir $WORKDIR/checkpoints/eicu

echo "[$(date)] local_only B done."

python baselines/local_only.py \
    --site C \
    --root $WORKDIR \
    --dataset eicu \
    --n_epochs 100 \
    --patience 15 \
    --batch_size 64 \
    --seeds 42 123 7 \
    --output $WORKDIR/results/eicu_local_only_C.csv \
    --ckpt_dir $WORKDIR/checkpoints/eicu

echo "[$(date)] local_only C done."

python baselines/centralized.py \
    --root $WORKDIR \
    --dataset eicu \
    --n_epochs 100 \
    --patience 15 \
    --batch_size 64 \
    --seeds 42 123 7 \
    --output $WORKDIR/results/eicu_centralized.csv \
    --ckpt_dir $WORKDIR/checkpoints/eicu

echo "[$(date)] Centralized oracle done."

python experiments/run_eicu_baselines.py \
    --skip_rerun \
    --local_a     $WORKDIR/results/eicu_local_only_A.csv \
    --local_b     $WORKDIR/results/eicu_local_only_B.csv \
    --local_c     $WORKDIR/results/eicu_local_only_C.csv \
    --central     $WORKDIR/results/eicu_centralized.csv \
    --vfl_results $WORKDIR/results/eicu_exp1.csv \
    --output      $WORKDIR/results/eicu_baselines_comparison.csv

echo "=== eICU baselines finished: $(date) ==="
