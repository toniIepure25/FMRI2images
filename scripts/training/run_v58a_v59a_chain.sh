#!/usr/bin/env bash
set -euo pipefail

cd /home/jovyan/work/FMRI2images
set -a && source .env 2>/dev/null || true && set +a
export CUDA_VISIBLE_DEVICES=0
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/home/jovyan/work/FMRI2images/cache/tmp
export TORCH_HOME=/home/jovyan/work/FMRI2images/cache/torch
export XDG_CACHE_HOME=/home/jovyan/work/FMRI2images/cache/xdg
mkdir -p "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Starting V58a training"
python scripts/training/train_unified.py --config configs/experiments/V58a_improved_197k.yaml
V58A_EXIT=$?
log "V58a finished with exit code $V58A_EXIT"

if [ $V58A_EXIT -eq 0 ]; then
    log "Starting V59a training"
    python scripts/training/train_unified.py --config configs/experiments/V59a_retrieval_only_768d.yaml
    log "V59a finished with exit code $?"
else
    log "V58a failed, skipping V59a"
fi

log "Chain complete"
