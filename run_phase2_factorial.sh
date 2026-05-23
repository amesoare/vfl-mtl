#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_phase2
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-phase2-embed%a.out
#SBATCH --error=slurm-%j-phase2-embed%a.err
#SBATCH --array=0-2

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

EMBED_DIMS=(32 64 128)
EMBED_DIM=${EMBED_DIMS[$SLURM_ARRAY_TASK_ID]}

source $VENV/bin/activate
cd $WORKDIR

echo "=== Phase 2 factorial started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES  embed_dim: $EMBED_DIM"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python -u experiments/run_factorial_pcmu.py \
    --splits_dir $SPLITS_DIR \
    --n_rounds 100 \
    --device cuda \
    --embed_dim $EMBED_DIM \
    --skip_existing

echo "=== Phase 2 factorial (embed_dim=$EMBED_DIM) finished: $(date) ==="
