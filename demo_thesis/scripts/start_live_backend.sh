#!/usr/bin/env bash
# Cortex2Canvas — Start Live Backend
# Usage: bash scripts/start_live_backend.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$DEMO_DIR")"

# --- Artifact paths (absolute) ---
export C2C_V62_CHECKPOINT_PATH="${DEMO_DIR}/local_backend_data/checkpoints/V62a_model_only.pt"
export C2C_FMRI_FEATURES_PATH="${DEMO_DIR}/local_backend_data/subj01/fmri_features.npy"
export C2C_TRIAL_INDEX_PATH="${DEMO_DIR}/local_backend_data/subj01/index.parquet"
export C2C_CLIP_GALLERY_768_PATH="${DEMO_DIR}/local_backend_data/clip/clip.parquet"
export C2C_SUBJECT="subj01"

# --- Backend mode ---
export C2C_BACKEND_MODE="${C2C_BACKEND_MODE:-v62_single}"
export C2C_DEVICE="${C2C_DEVICE:-auto}"

# --- Reconstruction (default: auto; set "live" to attempt live generation) ---
export C2C_RECON_MODE="${C2C_RECON_MODE:-auto}"
export C2C_RECON_ALLOW_DOWNLOAD="${C2C_RECON_ALLOW_DOWNLOAD:-false}"
export C2C_RECON_STEPS="${C2C_RECON_STEPS:-25}"
export C2C_RECON_GUIDANCE="${C2C_RECON_GUIDANCE:-8.0}"
export C2C_RECON_SEED="${C2C_RECON_SEED:-42}"

# --- Python path so fmri2img and backend.recon are importable ---
export PYTHONPATH="${ROOT_DIR}/src:${DEMO_DIR}/backend:${PYTHONPATH:-}"

echo "=== Cortex2Canvas Live Backend ==="
echo "  Mode:        ${C2C_BACKEND_MODE}"
echo "  Checkpoint:  ${C2C_V62_CHECKPOINT_PATH}"
echo "  Subject:     ${C2C_SUBJECT}"
echo "  Device:      ${C2C_DEVICE}"
echo "  Recon mode:  ${C2C_RECON_MODE}"
echo "  PYTHONPATH:  ${PYTHONPATH}"
echo ""

cd "${DEMO_DIR}/backend"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 "${@}"
