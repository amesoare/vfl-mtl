#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=4:00:00
#SBATCH --mem=56G
#SBATCH --job-name=vfl_eicu_exp1
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-eicu_exp1.out
#SBATCH --error=slurm-%j-eicu_exp1.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== eICU Exp 1 started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python -u experiments/run_eicu_exp1.py \
    --root    $WORKDIR \
    --n_rounds 100 \
    --patience 15 \
    --device  cuda \
    --output  results/eicu_exp1.csv

echo "=== eICU Exp 1 finished: $(date) ==="
