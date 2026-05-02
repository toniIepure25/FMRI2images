#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# V60 Chain: Multi-Subject 197K-D Pretraining -> Subj01 Finetuning
#
# Usage:
#   bash scripts/training/run_v60_chain.sh [standard|kappa|both]
#
# Modes:
#   standard  — V60a (multi-subject) -> V60b (subj01 finetune)
#   kappa     — V60c (kappa-gated)   -> V60d (subj01 finetune)
#   both      — Run standard first, then kappa (DEFAULT)
#
# Prerequisites:
#   - Pre-extracted fMRI features for all 8 subjects
#   - N1v28a checkpoint at experimental_results/N1v28a_dual_head/subj01/checkpoint_best.pt
#   - Token cache at outputs/clip_cache/tokens_ViT-L-14_projected.h5
#   - CLIP embeddings at outputs/clip_cache/clip.parquet
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-both}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/logs/v60_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR="${REPO_ROOT}/tmp"
mkdir -p "$TMPDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/chain.log"; }

check_prerequisites() {
    log "Checking prerequisites..."

    local n28a_ckpt="experimental_results/N1v28a_dual_head/subj01/checkpoint_best.pt"
    if [ ! -f "$n28a_ckpt" ]; then
        log "ERROR: N1v28a checkpoint not found: $n28a_ckpt"
        exit 1
    fi

    local token_cache="outputs/clip_cache/tokens_ViT-L-14_projected.h5"
    if [ ! -f "$token_cache" ]; then
        log "ERROR: Token cache not found: $token_cache"
        exit 1
    fi

    local clip_cache="outputs/clip_cache/clip.parquet"
    if [ ! -f "$clip_cache" ]; then
        log "ERROR: CLIP cache not found: $clip_cache"
        exit 1
    fi

    local missing_subjects=()
    for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
        local feat="cache/preextracted/subject=${subj}/fmri_features.npy"
        if [ ! -f "$feat" ]; then
            missing_subjects+=("$subj")
        fi
    done

    if [ ${#missing_subjects[@]} -gt 0 ]; then
        log "WARNING: Missing pre-extracted features for: ${missing_subjects[*]}"
        log "Attempting to pre-extract missing subjects..."
        for subj in "${missing_subjects[@]}"; do
            log "Pre-extracting $subj..."
            python scripts/preprocessing/preextract_fmri.py --subject "$subj" \
                2>&1 | tee -a "$LOG_DIR/preextract_${subj}.log" || {
                log "ERROR: Pre-extraction failed for $subj"
                exit 1
            }
        done
    fi

    log "All prerequisites satisfied."
}

run_training() {
    local config="$1"
    local name="$2"
    local extra_args="${3:-}"
    local logfile="$LOG_DIR/${name}.log"

    log "=== Starting $name ==="
    log "Config: $config"
    log "Log: $logfile"

    python scripts/training/train_unified.py \
        --config "$config" $extra_args \
        2>&1 | tee "$logfile"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        log "ERROR: $name failed with exit code $exit_code"
        return $exit_code
    fi

    log "=== $name completed successfully ==="
    return 0
}

run_standard() {
    log "=========================================="
    log "STANDARD BRANCH: V60a -> V60b"
    log "=========================================="

    local v60a_ckpt="experimental_results/V60a_cross_subject_197k/subj01/checkpoint_best.pt"
    local v60a_args="--save-checkpoints best"
    if [ -f "$v60a_ckpt" ]; then
        log "V60a checkpoint found — resuming from $v60a_ckpt"
        v60a_args="$v60a_args --resume $v60a_ckpt"
    fi

    run_training "configs/experiments/V60a_cross_subject_197k.yaml" "V60a" "$v60a_args" || return 1

    if [ ! -f "$v60a_ckpt" ]; then
        log "ERROR: V60a checkpoint not found: $v60a_ckpt"
        return 1
    fi

    run_training "configs/experiments/V60b_subj01_finetune_197k.yaml" "V60b" || return 1

    log "Standard branch (V60a -> V60b) completed."
}

run_kappa() {
    log "=========================================="
    log "KAPPA-GATED BRANCH: V60c -> V60d"
    log "=========================================="

    local v60c_ckpt="experimental_results/V60c_kappa_gated_197k/subj01/checkpoint_best.pt"
    local v60c_args="--save-checkpoints best"
    if [ -f "$v60c_ckpt" ]; then
        log "V60c checkpoint found — resuming from $v60c_ckpt"
        v60c_args="$v60c_args --resume $v60c_ckpt"
    fi

    run_training "configs/experiments/V60c_kappa_gated_197k.yaml" "V60c" "$v60c_args" || return 1

    if [ ! -f "$v60c_ckpt" ]; then
        log "ERROR: V60c checkpoint not found: $v60c_ckpt"
        return 1
    fi

    run_training "configs/experiments/V60d_subj01_finetune_kappa.yaml" "V60d" || return 1

    log "Kappa-gated branch (V60c -> V60d) completed."
}

log "V60 Chain Pipeline — mode: $MODE"
check_prerequisites

case "$MODE" in
    standard)
        run_standard
        ;;
    kappa)
        run_kappa
        ;;
    both)
        run_standard || log "WARNING: Standard branch failed, continuing to kappa..."
        run_kappa
        ;;
    *)
        log "ERROR: Unknown mode '$MODE'. Use: standard, kappa, or both"
        exit 1
        ;;
esac

log "=========================================="
log "V60 Chain Pipeline completed."
log "Logs: $LOG_DIR"
log "=========================================="
