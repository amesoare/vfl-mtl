#!/bin/bash
# Copy project code to Snellius and install the venv for each job.
# Usage: ./copy_to_snellius.sh
# Reads credentials from .env in the same directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# --- Parse .env (supports spaces around =) ---
get_env() { grep "^$1" "$ENV_FILE" | sed 's/.*= *//;s/ *$//'; }

SNELLIUS_PASSWORD=$(get_env SNELLIUS_PASSWORD)
USERNAME=$(get_env USERNAME)
DOMAIN=$(get_env DOMAIN)
REMOTE="$USERNAME@$DOMAIN"
REMOTE_DIR="/home/$USERNAME/vfl_mlt"

echo "[$(date)] Syncing code to $REMOTE:$REMOTE_DIR ..."

# rsync: code only — exclude data, venv, caches, results
SSHPASS="$SNELLIUS_PASSWORD" sshpass -e rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no" \
    --exclude='.venv/' \
    --exclude='data/' \
    --exclude='results/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.env' \
    --exclude='slurm-*.out' \
    --exclude='slurm-*.err' \
    --exclude='checkpoints/' \
    "$SCRIPT_DIR/" \
    "$REMOTE:$REMOTE_DIR/"

echo "[$(date)] Code synced."

# --- Install venv on Snellius for each job ---
# Jobs are identified by SLURM .sh scripts in the repo root.
JOBS=$(find "$SCRIPT_DIR" -maxdepth 1 -name "*.sh" ! -name "copy_to_snellius.sh" ! -name "sync_with_snellius.sh" -printf '%f\n' 2>/dev/null || \
      ls "$SCRIPT_DIR"/*.sh 2>/dev/null | xargs -n1 basename | grep -v "copy_to_snellius\|sync_with_snellius")

echo "[$(date)] Setting up Python environment on Snellius..."

SSHPASS="$SNELLIUS_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no "$REMOTE" bash <<REMOTE_SCRIPT
set -euo pipefail
cd "$REMOTE_DIR"

# Load Python module (Snellius uses module system)
module load 2>/dev/null || true
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || \
module load python3 2>/dev/null || true

VENV_DIR="$REMOTE_DIR/.venv"

if [ ! -d "\$VENV_DIR" ]; then
    echo "  Creating venv at \$VENV_DIR ..."
    python3 -m venv "\$VENV_DIR"
fi

echo "  Installing requirements..."
"\$VENV_DIR/bin/pip" install --upgrade pip --quiet
"\$VENV_DIR/bin/pip" install -r "$REMOTE_DIR/requirements.txt" --quiet

echo "  Venv ready: \$VENV_DIR"
REMOTE_SCRIPT

echo "[$(date)] Environment installed on Snellius."
echo ""
echo "Jobs available to submit:"
for job in $JOBS; do
    echo "  sbatch $REMOTE_DIR/$job"
done
