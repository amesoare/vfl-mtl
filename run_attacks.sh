#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --job-name=vfl_attacks
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm-%j-attacks.out
#SBATCH --error=slurm-%j-attacks.err

# Label inference attack + membership inference attack.
# PREREQUISITE: all run_privacy_curves_eps_*.sh jobs must be complete
# (checkpoints in checkpoints/ and per-ε CSVs in results/).
#
# Step 1: merge per-ε CSVs into results/privacy_utility.csv
# Step 2: run label inference probe (LR on cut-layer embeddings)
# Step 3: run embedding MIA (binary LR on concatenated embeddings)

set -euo pipefail

VENV=/home/asoare/vfl_mlt/.venv
WORKDIR=/home/asoare/vfl_mlt
SPLITS_DIR=$WORKDIR/data/vertical_splits

source $VENV/bin/activate
cd $WORKDIR

echo "=== Attacks started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# ---- Step 1: check privacy_utility.csv (optional — scripts use defaults if absent) ----
echo "--- Checking privacy_utility.csv ---"
python - <<'PYEOF'
import pandas as pd
from pathlib import Path

f = Path("results/privacy_utility.csv")
if not f.exists():
    print(f"[WARNING] {f} not found — attack scripts will use default (ε, mode) combos.")
else:
    df = pd.read_csv(f)
    print(f"Found {f}: {len(df)} rows, "
          f"{df['epsilon_level'].nunique()} ε levels, "
          f"{df['mode'].nunique()} modes, "
          f"{df['seed'].nunique()} seeds")
PYEOF

# ---- Step 2: label inference attack ----
echo "--- Label inference attack ---"
python attacks/label_inference.py \
    --splits_dir $SPLITS_DIR \
    --ckpt_dir checkpoints \
    --privacy_csv results/privacy_utility.csv \
    --output results/label_inference.csv \
    --split test \
    --device cpu

# ---- Step 3: embedding membership inference attack ----
echo "--- Embedding MIA ---"
python attacks/embedding_mia.py \
    --splits_dir $SPLITS_DIR \
    --ckpt_dir checkpoints \
    --privacy_csv results/privacy_utility.csv \
    --output results/embedding_mia.csv \
    --split test \
    --device cpu

echo "=== Attacks finished: $(date) ==="
