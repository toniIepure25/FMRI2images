#!/usr/bin/env bash
set -euo pipefail

# V56 Auto-Chain: Run V56a → V56c → PPR eval sequentially
# Crash-resilient with retry logic and proper environment setup.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/runtime_logs"
mkdir -p "$LOG_DIR"

# --- Environment: zero ephemeral writes ---
VENV_DIR="/home/jovyan/work/.venv-train"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
fi

set -a && source "$REPO_ROOT/.env" && set +a

export PIP_CACHE_DIR="/home/jovyan/work/.cache/pip"
export TMPDIR="/home/jovyan/work/.cache/tmp"
export TORCH_HOME="/home/jovyan/work/.cache/torch"
export XDG_CACHE_HOME="/home/jovyan/work/.cache"
export MPLCONFIGDIR="/home/jovyan/work/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

SUBJECT="subj01"
GPU=${GPU:-0}
MAX_RETRIES=3

run_with_retry() {
    local config="$1"
    local experiment_name="$2"
    local extra_args="${3:-}"
    local attempt=0
    local exit_code=0

    while [[ $attempt -lt $MAX_RETRIES ]]; do
        attempt=$((attempt + 1))
        echo "[$(date)] === $experiment_name attempt $attempt/$MAX_RETRIES ==="

        local resume_flag=""
        local ckpt="experimental_results/$experiment_name/$SUBJECT/checkpoint_last.pt"
        if [[ -f "$ckpt" ]]; then
            resume_flag="--resume $ckpt"
            echo "[$(date)] Resuming from $ckpt"
        fi

        set +e
        python3 scripts/training/train_unified.py \
            --config "$config" \
            --subject "$SUBJECT" \
            --gpu "$GPU" \
            $resume_flag \
            $extra_args \
            2>&1 | tee -a "$LOG_DIR/${experiment_name}.log"
        exit_code=$?
        set -e

        if [[ $exit_code -eq 0 ]]; then
            echo "[$(date)] $experiment_name completed successfully"
            return 0
        fi

        echo "[$(date)] $experiment_name attempt $attempt failed (exit=$exit_code)"
        sleep 30
    done

    echo "[$(date)] $experiment_name FAILED after $MAX_RETRIES attempts"
    return 1
}

echo "[$(date)] ========================================"
echo "[$(date)] V56 Auto-Chain: Road to 90%+ R@1"
echo "[$(date)] ========================================"

# --- Phase 2a: V56a — Fusion Distillation (vmf_triple + dual_target) ---
echo ""
echo "[$(date)] --- PHASE 2a: V56a (Fusion Distillation) ---"
run_with_retry \
    "configs/experiments/V56a_fusion_distill_fixed.yaml" \
    "V56a_fusion_distill_fixed"

# --- Phase 2b: V56c — Projection Head + R-Drop ---
echo ""
echo "[$(date)] --- PHASE 2b: V56c (Projection Head + R-Drop) ---"

# V56c initializes from V56a if available, else V55b
V56A_CKPT="experimental_results/V56a_fusion_distill_fixed/$SUBJECT/checkpoint_best.pt"
if [[ -f "$V56A_CKPT" ]]; then
    echo "[$(date)] V56c will initialize from V56a best: $V56A_CKPT"
    sed -i "s|pretrained_model_path:.*|pretrained_model_path: \"$V56A_CKPT\"|" \
        configs/experiments/V56c_projection_rdrop.yaml 2>/dev/null || true
fi

run_with_retry \
    "configs/experiments/V56c_projection_rdrop.yaml" \
    "V56c_projection_rdrop"

# --- Phase 1 evaluation on V56a/V56c ---
echo ""
echo "[$(date)] --- Running Phase 1 evaluation on V56a/V56c ---"
python3 scripts/evaluation/phase1_197k_fusion.py 2>&1 | tee "$LOG_DIR/phase2_eval.log"

echo ""
echo "[$(date)] ========================================"
echo "[$(date)] V56 Auto-Chain COMPLETE"
echo "[$(date)] ========================================"
