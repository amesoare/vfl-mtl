#!/bin/bash
#SBATCH --job-name=eicu_split
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j-eicu_split.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j-eicu_split.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
PROJECT=/home/asoare/vfl_mlt
EICU_ROOT=$PROJECT/data/eicu_root
EICU_CRD=$PROJECT/data/eicu-crd-2.0
SPLITS_OUT=$PROJECT/data/eicu_vertical_splits

source $VENV/bin/activate
cd $PROJECT
export PYTHONPATH=$PROJECT

echo "[$(date)] Step 1: vertical split"
python data_prep/eicu_vertical_split.py \
    --root_dir  "$EICU_ROOT" \
    --eicu_dir  "$EICU_CRD" \
    --output    "$SPLITS_OUT" \
    --seed 42

echo "[$(date)] Step 2: PSI alignment"
python data_prep/eicu_psi_alignment.py \
    --site_a "$SPLITS_OUT/site_A_eicu.csv" \
    --site_b "$SPLITS_OUT/site_B_eicu.csv" \
    --site_c "$SPLITS_OUT/site_C_eicu.csv" \
    --output "$SPLITS_OUT/aligned_patient_ids_eicu.csv"

echo "[$(date)] Done. Output: $SPLITS_OUT"
