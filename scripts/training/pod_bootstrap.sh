#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# pod_bootstrap.sh — Zero-ephemeral-storage setup for training on K8s
#
# Creates a persistent Python venv on PVC so pip installs never touch the
# node's overlay filesystem (prevents ephemeral-storage eviction).
#
# Usage (run once after pod start, or it detects existing venv):
#   bash scripts/training/pod_bootstrap.sh
#
# Then launch training:
#   bash scripts/training/pod_bootstrap.sh --run-chain
###############################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="/home/jovyan/work/.venv-train"
PIP_CACHE="/home/jovyan/work/.cache/pip"
CONDA_PYTHON="/opt/conda/bin/python3"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [BOOTSTRAP] $*"; }

# --- Redirect all caches to PVC (zero ephemeral writes) ---
export PIP_CACHE_DIR="$PIP_CACHE"
export TMPDIR="/home/jovyan/work/.tmp"
export XDG_CACHE_HOME="/home/jovyan/work/.cache"
mkdir -p "$PIP_CACHE" "$TMPDIR" "$XDG_CACHE_HOME"

# --- Create or reuse persistent venv ---
if [[ -f "$VENV_DIR/bin/python3" ]]; then
    log "Reusing existing venv at $VENV_DIR"
else
    log "Creating persistent venv at $VENV_DIR ..."
    "$CONDA_PYTHON" -m venv --system-site-packages "$VENV_DIR"
    log "Venv created."
fi

# Activate
source "$VENV_DIR/bin/activate"
log "Python: $(which python3) ($(python3 --version))"

# --- Install/update fmri2img into venv (writes to PVC, not overlay) ---
pip install --cache-dir "$PIP_CACHE" -e "$REPO_ROOT[train]" 2>&1 | tail -3
python3 -c "import fmri2img; print('fmri2img OK:', fmri2img.__file__)"

# --- Source .env ---
set -a && source "$REPO_ROOT/.env" && set +a
log "NSD_DATA_ROOT=$NSD_DATA_ROOT"

# --- Git safe dir ---
git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true

# --- Fix permissions on output dirs ---
chmod -R a+rw "$REPO_ROOT/runtime_logs" 2>/dev/null || true
chmod -R a+rw "$REPO_ROOT/experimental_results" 2>/dev/null || true

# --- Verify ephemeral usage is minimal ---
log "Overlay usage: $(df -h / | tail -1 | awk '{print $3, "used,", $4, "avail,", $5}')"

# --- Optionally launch auto-chain ---
if [[ "${1:-}" == "--run-chain" ]]; then
    log "Launching auto-chain..."
    mkdir -p "$REPO_ROOT/runtime_logs"
    nohup bash "$REPO_ROOT/scripts/training/v55_auto_chain.sh" \
        >> "$REPO_ROOT/runtime_logs/v55_chain.log" 2>&1 &
    CHAIN_PID=$!
    log "Auto-chain started: PID=$CHAIN_PID"
    sleep 10
    ps aux | grep -E "train_unified|v55_auto_chain" | grep -v grep || log "WARNING: no training process detected yet"
fi

log "Bootstrap complete."
