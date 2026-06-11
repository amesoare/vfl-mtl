#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --job-name=vfl_eicu_test_dp
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-eicu_test_dp.out
#SBATCH --error=slurm-%j-eicu_test_dp.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== eICU DP test-set evaluation started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

python experiments/evaluate_eicu_test.py \
    --dp \
    --root       $WORKDIR \
    --ckpt_dir   checkpoints/eicu \
    --batch_size 64 \
    --device     cpu

echo "=== eICU DP test-set evaluation finished: $(date) ==="
