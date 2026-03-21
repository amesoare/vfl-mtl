#!/bin/bash
#SBATCH --job-name=eda_notebook
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j.err

set -euo pipefail

# --- Activate conda environment ---
module load 2023
module load Anaconda3/2023.07-2
source activate vfl_mlt_env

NOTEBOOK=/home/asoare/vfl_mlt/ExploratoryDataAnalysis.ipynb
OUTDIR=/home/asoare/vfl_mlt/results/eda
EXECUTED=$OUTDIR/EDA_executed.ipynb
HTML=$OUTDIR/EDA_report.html

mkdir -p "$OUTDIR"

# --- Ensure kernel and nbconvert are available in the env ---
pip install --quiet --upgrade nbconvert ipykernel
python -m ipykernel install --user --name vfl_mlt_env

echo "[$(date)] Executing notebook..."
jupyter nbconvert \
    --to notebook \
    --execute "$NOTEBOOK" \
    --output "$EXECUTED" \
    --ExecutePreprocessor.timeout=3600

echo "[$(date)] Converting to HTML..."
jupyter nbconvert \
    --to html \
    "$EXECUTED" \
    --output "$HTML"

echo "[$(date)] Done. Report saved to: $HTML"
