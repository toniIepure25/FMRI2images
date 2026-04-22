#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# v55_auto_chain.sh — Crash-Resilient V55 Pipeline Auto-Chainer
#
# Runs V55a with automatic restart from checkpoint on OOM/crash, then chains:
#   V55a (multi-subject pretrain) -> V55b (subj01 finetune) -> V55c (fusion distill)
#   -> PPR evaluation -> fusion sweep
#
# Usage:
#   nohup bash scripts/training/v55_auto_chain.sh > runtime_logs/v55_chain.log 2>&1 &
#
# Prerequisites:
#   - Environment sourced: set -a && source .env && set +a
#   - Package installed: pip install -e ".[train]"
###############################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="runtime_logs"
mkdir -p "$LOG_DIR"

TRAIN_CMD="python3 scripts/training/train_unified.py"

V55A_CONFIG="configs/experiments/V55a_multi_subject_dual_head.yaml"
V55B_CONFIG="configs/experiments/V55b_subj01_finetune.yaml"
V55C_CONFIG="configs/experiments/V55c_fusion_distill.yaml"

V55A_DIR="experimental_results/V55a_multi_subject_dual_head/subj01"
V55A_CKPT_BEST="$V55A_DIR/checkpoint_best.pt"
V55A_CKPT_LAST="$V55A_DIR/checkpoint_last.pt"
V55B_CKPT="experimental_results/V55b_subj01_finetune/subj01/checkpoint_best.pt"
V55C_CKPT="experimental_results/V55c_fusion_distill/subj01/checkpoint_best.pt"

MAX_RESTARTS=10
POLL_INTERVAL=120

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [CHAIN] $*"; }

check_checkpoint() {
    local ckpt="$1"
    local name="$2"
    if [[ -f "$ckpt" ]]; then
        local size
        size=$(du -sh "$ckpt" | cut -f1)
        log "$name checkpoint exists: $ckpt ($size)"
        return 0
    else
        log "$name checkpoint NOT found at $ckpt"
        return 1
    fi
}

report_final_metrics() {
    local name="$1"
    local logfile="$2"
    log "=== $name Final Metrics ==="
    grep -E "CSLS Retrieval|PPR Retrieval|Retrieval.*high-D|Retrieval:" "$logfile" 2>/dev/null | tail -6
    grep -E "New best|best.*csls" "$logfile" 2>/dev/null | tail -1
    log "=== End $name Metrics ==="
}

run_with_retry() {
    local config="$1"
    local name="$2"
    local logfile="$3"
    local ckpt_last="$4"
    local ckpt_best="$5"
    local attempt=0

    while (( attempt < MAX_RESTARTS )); do
        attempt=$((attempt + 1))
        log "--- $name attempt $attempt/$MAX_RESTARTS ---"

        local resume_flag=""
        if [[ -f "$ckpt_last" ]]; then
            resume_flag="--resume $ckpt_last"
            log "$name: Resuming from $ckpt_last"
        elif [[ $attempt -gt 1 ]]; then
            log "$name: No checkpoint_last.pt found after crash — restarting from scratch"
        fi

        $TRAIN_CMD --config "$config" --subject subj01 $resume_flag \
            >> "$logfile" 2>&1
        local exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            log "$name completed successfully on attempt $attempt"
            report_final_metrics "$name" "$logfile"
            return 0
        fi

        log "$name died with exit code $exit_code on attempt $attempt"

        if [[ $exit_code -eq 137 ]]; then
            log "$name was OOMKilled (signal 9). Will restart from checkpoint after cooldown."
        else
            log "$name failed with non-OOM exit code $exit_code."
        fi

        if ! check_checkpoint "$ckpt_last" "$name-last" && ! check_checkpoint "$ckpt_best" "$name-best"; then
            if [[ $attempt -ge 3 ]]; then
                log "FATAL: $name has crashed $attempt times with no checkpoint saved. Aborting."
                return 1
            fi
        fi

        local cooldown=$((30 + attempt * 15))
        log "Cooling down for ${cooldown}s before restart..."
        sleep "$cooldown"
    done

    log "FATAL: $name exhausted $MAX_RESTARTS restart attempts."
    return 1
}

###############################################################################
# PHASE 1: V55a with crash recovery
###############################################################################

log "=========================================="
log "V55 AUTO-CHAIN PIPELINE STARTED"
log "=========================================="

V55A_PID=$(ps aux | grep "train_unified.*V55a" | grep -v grep | awk '{print $2}' | head -1)

if [[ -n "$V55A_PID" ]]; then
    log "Detected existing V55a process: PID=$V55A_PID — waiting for it"
    while kill -0 "$V55A_PID" 2>/dev/null; do
        local_epoch=$(grep -oP "Epoch \d+/\d+" "$LOG_DIR/v55a_training.log" 2>/dev/null | tail -1 || echo "unknown")
        log "[V55a] Still running — $local_epoch (PID=$V55A_PID)"
        sleep "$POLL_INTERVAL"
    done
    wait "$V55A_PID" 2>/dev/null || true
    log "Existing V55a process finished."
fi

if check_checkpoint "$V55A_CKPT_BEST" "V55a-best"; then
    log "V55a best checkpoint already exists — checking if training completed."
    v55a_last_epoch=$(grep -oP "Epoch \d+" "$LOG_DIR/v55a_training.log" 2>/dev/null | tail -1 | grep -oP "\d+" || echo "0")
    if (( v55a_last_epoch >= 50 )); then
        log "V55a reached epoch $v55a_last_epoch (>=50). Skipping to V55b."
    else
        log "V55a only reached epoch $v55a_last_epoch. Will resume."
        run_with_retry "$V55A_CONFIG" "V55a" "$LOG_DIR/v55a_training.log" "$V55A_CKPT_LAST" "$V55A_CKPT_BEST"
    fi
else
    run_with_retry "$V55A_CONFIG" "V55a" "$LOG_DIR/v55a_training.log" "$V55A_CKPT_LAST" "$V55A_CKPT_BEST"
fi

if ! check_checkpoint "$V55A_CKPT_BEST" "V55a" && ! check_checkpoint "$V55A_CKPT_LAST" "V55a-last"; then
    log "FATAL: V55a produced no checkpoint. Pipeline aborted."
    exit 1
fi

###############################################################################
# PHASE 2: V55b (subj01 fine-tuning)
###############################################################################

log ""
log "=========================================="
log "PHASE 2: V55b (subj01 fine-tuning)"
log "=========================================="

V55B_LAST="experimental_results/V55b_subj01_finetune/subj01/checkpoint_last.pt"
run_with_retry "$V55B_CONFIG" "V55b" "$LOG_DIR/v55b_training.log" "$V55B_LAST" "$V55B_CKPT" || true

###############################################################################
# PHASE 3: V55c (fusion distillation)
###############################################################################

log ""
log "=========================================="
log "PHASE 3: V55c (fusion distillation)"
log "=========================================="

V55C_LAST="experimental_results/V55c_fusion_distill/subj01/checkpoint_last.pt"
run_with_retry "$V55C_CONFIG" "V55c" "$LOG_DIR/v55c_training.log" "$V55C_LAST" "$V55C_CKPT" || true

###############################################################################
# PHASE 4: PPR Evaluation
###############################################################################

log ""
log "=========================================="
log "PHASE 4: PPR Evaluation"
log "=========================================="

BEST_CKPT=""
for ckpt in "$V55C_CKPT" "$V55B_CKPT" "$V55A_CKPT_BEST" "$V55A_CKPT_LAST"; do
    if [[ -f "$ckpt" ]]; then
        BEST_CKPT="$ckpt"
        break
    fi
done

if [[ -n "$BEST_CKPT" ]]; then
    log "Running PPR evaluation on: $BEST_CKPT"
    python3 scripts/evaluation/evaluate_ppr.py \
        --checkpoint "$BEST_CKPT" \
        --subject subj01 \
        > "$LOG_DIR/v55_ppr_eval.log" 2>&1 || \
        log "WARNING: PPR evaluation failed (non-critical)"
    log "PPR evaluation complete."
else
    log "No checkpoint found for PPR evaluation."
fi

###############################################################################
# PHASE 5: Fusion sweep
###############################################################################

log ""
log "=========================================="
log "PHASE 5: Fusion Sweep"
log "=========================================="

if [[ -n "$BEST_CKPT" ]]; then
    log "Running fusion sweep..."
    python3 scripts/evaluation/sweep_fusion_ppr.py \
        --compact-checkpoint "$BEST_CKPT" \
        --legacy-checkpoint "experimental_results/N1v28a_dual_head/subj01/checkpoint_best.pt" \
        --subject subj01 \
        > "$LOG_DIR/v55_fusion_sweep.log" 2>&1 || \
        log "WARNING: Fusion sweep failed (non-critical)"
    log "Fusion sweep complete."
fi

###############################################################################
# SUMMARY
###############################################################################

log ""
log "=========================================="
log "V55 AUTO-CHAIN PIPELINE COMPLETE"
log "=========================================="
log "V55a best : $(ls -lh "$V55A_CKPT_BEST" 2>/dev/null | awk '{print $5, $6, $7, $8}' || echo 'MISSING')"
log "V55a last : $(ls -lh "$V55A_CKPT_LAST" 2>/dev/null | awk '{print $5, $6, $7, $8}' || echo 'MISSING')"
log "V55b best : $(ls -lh "$V55B_CKPT" 2>/dev/null | awk '{print $5, $6, $7, $8}' || echo 'MISSING')"
log "V55c best : $(ls -lh "$V55C_CKPT" 2>/dev/null | awk '{print $5, $6, $7, $8}' || echo 'MISSING')"
log ""
log "All logs in: $LOG_DIR/v55_*.log"
log "Pipeline finished at $(timestamp)"
