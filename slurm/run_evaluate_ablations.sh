#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --job-name=vfl_abl_test_eval
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-abl_test_eval.out
#SBATCH --error=slurm-%j-abl_test_eval.err

# Test-set evaluation for Week 4 ablation models.
# Loads best checkpoint per ablation variant; no re-training.
# Covers: VFL-MTL, abl_no_mmoe, abl_uniform_gating,
#         abl_experts_2, abl_experts_8, abl_embed_32, abl_embed_128
# Output: results/test_ablations.csv
#
# PREREQUISITE: run_ablations.sh must be complete and checkpoints saved
# under checkpoints/best_{model_name}_seed{seed}.pt

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt

source $VENV/bin/activate
cd $WORKDIR

echo "=== Ablation test-set evaluation started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

python experiments/evaluate_ablations.py \
    --root $WORKDIR \
    --ckpt_dir checkpoints \
    --batch_size 64 \
    --device cpu \
    --output results/test_ablations.csv

echo "=== Ablation test-set evaluation finished: $(date) ==="
