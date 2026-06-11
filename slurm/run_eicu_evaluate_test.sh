#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_eicu_test_eval
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-eicu_test_eval.out
#SBATCH --error=slurm-%j-eicu_test_eval.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== eICU test-set evaluation started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

python experiments/evaluate_eicu_test.py \
    --root     $WORKDIR \
    --ckpt_dir checkpoints/eicu \
    --batch_size 64 \
    --device   cpu \
    --output   results/eicu_test_results.csv

echo "=== eICU test-set evaluation finished: $(date) ==="
