#!/bin/bash
#SBATCH --job-name=eda_notebook
#SBATCH --partition=rome
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j.err

set -euo pipefail

# --- Activate virtualenv ---
source /home/asoare/vfl_mlt/.venv/bin/activate

NOTEBOOK=/home/asoare/vfl_mlt/ExploratoryDataAnalysis.ipynb
OUTDIR=/home/asoare/vfl_mlt/results/eda
HTML=$OUTDIR/EDA_report.html

mkdir -p "$OUTDIR"

# --- Ensure kernel and nbconvert are available in the env ---
pip install --quiet --upgrade nbconvert ipykernel
python -m ipykernel install --user --name vfl_mlt_env

echo "[$(date)] Executing notebook..."
jupyter nbconvert \
    --to notebook \
    --execute "$NOTEBOOK" \
    --output "$NOTEBOOK" \
    --inplace \
    --ExecutePreprocessor.timeout=3600

echo "[$(date)] Converting to HTML..."
jupyter nbconvert \
    --to html \
    "$NOTEBOOK" \
    --output "$HTML"

echo "[$(date)] Done. Report saved to: $HTML"
