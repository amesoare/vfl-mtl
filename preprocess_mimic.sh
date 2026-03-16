#!/bin/bash
#SBATCH --job-name=mimic_preprocess
#SBATCH --partition=rome
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/asoare/vfl_mlt/slurm-%j.out
#SBATCH --error=/home/asoare/vfl_mlt/slurm-%j.err

set -euo pipefail

MIMIC_DATA=/home/asoare/vfl_mlt/data/mimic-iii-clinical-database-1.4
BENCHMARK_REPO=/home/asoare/vfl_mlt/mimic3-benchmarks
BENCHMARK_OUT=/home/asoare/vfl_mlt/data/mimic3-benchmarks/data
OUTPUT_ROOT=$BENCHMARK_OUT/root

echo "[$(date)] Starting MIMIC-III preprocessing pipeline"

# --- Clone YerevaNN benchmark repo if not already present ---
if [ ! -d "$BENCHMARK_REPO" ]; then
    echo "[$(date)] Cloning mimic3-benchmarks..."
    git clone https://github.com/YerevaNN/mimic3-benchmarks.git "$BENCHMARK_REPO"
fi

cd "$BENCHMARK_REPO"

# --- Install dependencies ---
# Skip tensorflow/Keras (TF 1.x unavailable on Python 3.9; not needed for preprocessing)
# Skip numpy (1.16.5 has no Python 3.9 wheel; system numpy 1.23.5 is sufficient)
echo "[$(date)] Installing Python dependencies..."
grep -vE "tensorflow|Keras|numpy" requirements.txt | pip install -r /dev/stdin --prefer-binary --quiet

# --- Step 5a: Extract per-subject data ---
echo "[$(date)] Step 5a: extract_subjects"
python -m mimic3benchmark.scripts.extract_subjects \
    "$MIMIC_DATA" "$OUTPUT_ROOT"

# --- Step 5b: Validate events ---
echo "[$(date)] Step 5b: validate_events"
python -m mimic3benchmark.scripts.validate_events \
    "$OUTPUT_ROOT"

# --- Step 5c: Extract per-ICU-stay episodes ---
echo "[$(date)] Step 5c: extract_episodes_from_subjects"
python -m mimic3benchmark.scripts.extract_episodes_from_subjects \
    "$OUTPUT_ROOT"

# --- Step 5d: Train/val/test split ---
echo "[$(date)] Step 5d: split_train_val_test"
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
