#!/bin/bash
#SBATCH --job-name=eicu_preprocess
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j.err

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
EICU_DATA=/home/asoare/vfl_mlt/data/eicu-crd-2.0
BENCHMARK_REPO=/home/asoare/vfl_mlt/data_prep/eicu_benchmark

# Write per-patient dirs to SLURM $TMPDIR (per-job scratch-local, avoids home quota)
# $TMPDIR is set automatically by SLURM to /scratch-local/$SLURM_JOB_ID
SCRATCH=$TMPDIR/eicu_root
EICU_ROOT_HOME=/home/asoare/vfl_mlt/data/eicu_root

source $VENV/bin/activate

echo "[$(date)] Starting eICU preprocessing pipeline"

echo "[$(date)] Installing Python dependencies..."
pip install numpy pandas scikit-learn joblib --prefer-binary --quiet

mkdir -p "$SCRATCH"
mkdir -p "$EICU_ROOT_HOME"

cd "$BENCHMARK_REPO"
export PYTHONPATH="$BENCHMARK_REPO"

# --- Step 1: Extract per-patient timeseries + eicu_all_data.csv into scratch ---
echo "[$(date)] Step 1: data_extraction_root (writing to scratch)"
python data_extraction/data_extraction_root.py \
    --eicu_dir "$EICU_DATA" \
    --output_dir "$SCRATCH"

# --- Step 2: Copy only the final consolidated CSV to home ---
echo "[$(date)] Step 2: copying eicu_all_data.csv to home"
cp "$SCRATCH/eicu_all_data.csv" "$EICU_ROOT_HOME/eicu_all_data.csv"

echo "[$(date)] Preprocessing complete."
echo "[$(date)] Output: $EICU_ROOT_HOME/eicu_all_data.csv"
echo "[$(date)] Scratch intermediates remain at: $SCRATCH (safe to delete)"
