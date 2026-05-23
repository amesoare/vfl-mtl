#!/bin/bash
# Sync results from Snellius to local machine.
# DATA IS NEVER SYNCED — only results/, logs, and outputs.
# Usage: ./sync_with_snellius.sh
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

echo "[$(date)] Syncing results from $REMOTE:$REMOTE_DIR ..."

# Pull back: results/, executed notebook, slurm logs — never data/
SSHPASS="$SNELLIUS_PASSWORD" sshpass -e rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no" \
    --exclude='data/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --filter='+ results/***' \
    --filter='+ ExploratoryDataAnalysis.ipynb' \
    --filter='+ slurm-*.out' \
    --filter='+ slurm-*.err' \
    --filter='- *' \
    "$REMOTE:$REMOTE_DIR/" \
    "$SCRIPT_DIR/"

echo "[$(date)] Sync complete."
echo "Results saved to: $SCRIPT_DIR/results/"
echo "Executed notebook: $SCRIPT_DIR/ExploratoryDataAnalysis.ipynb"
