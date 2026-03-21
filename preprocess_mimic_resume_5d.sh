#!/bin/bash
#SBATCH --job-name=mimic_resume_5d
#SBATCH --partition=rome
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j.err

set -euo pipefail

BENCHMARK_REPO=/home/asoare/vfl_mlt/mimic3-benchmarks
BENCHMARK_OUT=/home/asoare/vfl_mlt/data/mimic3-benchmarks/data
OUTPUT_ROOT=$BENCHMARK_OUT/root

cd "$BENCHMARK_REPO"

# --- Step 5d: Train/test split ---
echo "[$(date)] Step 5d: split_train_and_test"
python -m mimic3benchmark.scripts.split_train_and_test \
    "$OUTPUT_ROOT"

# --- Step 5e: Create per-task datasets ---
echo "[$(date)] Step 5e: create_in_hospital_mortality"
python -m mimic3benchmark.scripts.create_in_hospital_mortality \
    "$OUTPUT_ROOT" \
    "$BENCHMARK_OUT/in-hospital-mortality/"

echo "[$(date)] Step 5e: create_decompensation"
python -m mimic3benchmark.scripts.create_decompensation \
    "$OUTPUT_ROOT" \
    "$BENCHMARK_OUT/decompensation/"

echo "[$(date)] Step 5e: create_length_of_stay"
python -m mimic3benchmark.scripts.create_length_of_stay \
    "$OUTPUT_ROOT" \
    "$BENCHMARK_OUT/length-of-stay/"

echo "[$(date)] Step 5e: create_phenotyping"
python -m mimic3benchmark.scripts.create_phenotyping \
    "$OUTPUT_ROOT" \
    "$BENCHMARK_OUT/phenotyping/"

echo "[$(date)] Preprocessing complete."
