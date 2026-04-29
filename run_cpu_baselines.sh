#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --job-name=vfl_local
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%A-%a-local.out
#SBATCH --error=slurm-%A-%a-local.err

set -euo pipefail

SITES=(A B C)
SITE=${SITES[$SLURM_ARRAY_TASK_ID]}

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== Local-only site=$SITE started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

python baselines/local_only.py \
    --site $SITE \
    --root $WORKDIR \
    --n_epochs 100 \
    --patience 15 \
    --output results/local_only_${SITE}.csv \
    --ckpt_dir $WORKDIR/checkpoints

echo "=== Local-only site=$SITE finished: $(date) ==="
