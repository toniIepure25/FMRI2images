#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# v55_auto_chain.sh — Sequential V55 Pipeline Auto-Chainer
#
# Monitors a running V55a training process, then automatically chains:
#   V55a (multi-subject pretrain) -> V55b (subj01 finetune) -> V55c (fusion distill)
#   -> PPR evaluation -> fusion sweep
#
# Usage:
#   nohup bash scripts/training/v55_auto_chain.sh > runtime_logs/v55_chain.log 2>&1 &
#
# Prerequisites:
#   - V55a must already be running (PID detected automatically)
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

V55A_CKPT="experimental_results/V55a_multi_subject_dual_head/subj01/checkpoint_best.pt"
V55B_CKPT="experimental_results/V55b_subj01_finetune/subj01/checkpoint_best.pt"
V55C_CKPT="experimental_results/V55c_fusion_distill/subj01/checkpoint_best.pt"

POLL_INTERVAL=120  # seconds between status checks

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] [CHAIN] $*"; }

wait_for_process() {
    local pid="$1"
    local name="$2"
    local logfile="$3"

    log "Waiting for $name (PID=$pid) to complete..."
    log "Monitoring log: $logfile"

    local checks=0
    while kill -0 "$pid" 2>/dev/null; do
        checks=$((checks + 1))
        if (( checks % 10 == 0 )); then
            local last_epoch
            last_epoch=$(grep -oP "Epoch \d+/\d+" "$logfile" 2>/dev/null | tail -1 || echo "unknown")
            local last_r1
            last_r1=$(grep "CSLS Retrieval" "$logfile" 2>/dev/null | tail -1 | grep -oP "R@1=[\d.]+" || echo "R@1=pending")
            log "[$name] Still running — $last_epoch — $last_r1 (check #$checks)"
        fi
        sleep "$POLL_INTERVAL"
    done

    wait "$pid" 2>/dev/null
    local exit_code=$?
    log "$name process finished with exit code: $exit_code"
    return $exit_code
}

check_checkpoint() {
    local ckpt="$1"
    local name="$2"
    if [[ -f "$ckpt" ]]; then
        local size
        size=$(du -sh "$ckpt" | cut -f1)
        log "$name checkpoint exists: $ckpt ($size)"
        return 0
    else
        log "ERROR: $name checkpoint NOT found at $ckpt"
        return 1
    fi
}

report_final_metrics() {
    local name="$1"
    local logfile="$2"

    log "=== $name Final Metrics ==="
    grep -E "CSLS Retrieval|PPR Retrieval|Retrieval:" "$logfile" 2>/dev/null | tail -6
    grep -E "SHARED.?1000|shared_1000" "$logfile" 2>/dev/null | tail -3
    grep -E "New best|best.*csls" "$logfile" 2>/dev/null | tail -1
    log "=== End $name Metrics ==="
}

###############################################################################
# PHASE 1: Wait for V55a
###############################################################################

log "=========================================="
log "V55 AUTO-CHAIN PIPELINE STARTED"
log "=========================================="

V55A_PID=$(ps aux | grep "train_unified.*V55a" | grep -v grep | awk '{print $2}' | head -1)

if [[ -z "$V55A_PID" ]]; then
    log "WARNING: No running V55a process detected."
    if check_checkpoint "$V55A_CKPT" "V55a"; then
        log "V55a checkpoint already exists — skipping to V55b."
    else
        log "ERROR: V55a not running and no checkpoint found. Cannot proceed."
        log "Start V55a first: nohup $TRAIN_CMD --config $V55A_CONFIG --subject subj01 > $LOG_DIR/v55a_training.log 2>&1 &"
        exit 1
    fi
else
    log "Detected V55a training process: PID=$V55A_PID"
    wait_for_process "$V55A_PID" "V55a" "$LOG_DIR/v55a_training.log" || true
    report_final_metrics "V55a" "$LOG_DIR/v55a_training.log"

    if ! check_checkpoint "$V55A_CKPT" "V55a"; then
        log "FATAL: V55a finished but no checkpoint produced. Pipeline aborted."
        exit 1
    fi
fi

###############################################################################
# PHASE 2: Run V55b (subj01 fine-tuning from V55a)
###############################################################################

log ""
log "=========================================="
log "PHASE 2: Starting V55b (subj01 fine-tuning)"
log "=========================================="

$TRAIN_CMD --config "$V55B_CONFIG" --subject subj01 \
    > "$LOG_DIR/v55b_training.log" 2>&1 &
V55B_PID=$!
log "V55b launched: PID=$V55B_PID"

wait_for_process "$V55B_PID" "V55b" "$LOG_DIR/v55b_training.log" || true
report_final_metrics "V55b" "$LOG_DIR/v55b_training.log"

if ! check_checkpoint "$V55B_CKPT" "V55b"; then
    log "WARNING: V55b checkpoint not found. Attempting V55c anyway (may fail)."
fi

###############################################################################
# PHASE 3: Run V55c (fusion topology distillation)
###############################################################################

log ""
log "=========================================="
log "PHASE 3: Starting V55c (fusion distillation)"
log "=========================================="

$TRAIN_CMD --config "$V55C_CONFIG" --subject subj01 \
    > "$LOG_DIR/v55c_training.log" 2>&1 &
V55C_PID=$!
log "V55c launched: PID=$V55C_PID"

wait_for_process "$V55C_PID" "V55c" "$LOG_DIR/v55c_training.log" || true
report_final_metrics "V55c" "$LOG_DIR/v55c_training.log"

###############################################################################
# PHASE 4: PPR Evaluation on best available checkpoint
###############################################################################

log ""
log "=========================================="
log "PHASE 4: PPR Evaluation"
log "=========================================="

BEST_CKPT=""
for ckpt in "$V55C_CKPT" "$V55B_CKPT" "$V55A_CKPT"; do
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
    log "PPR evaluation complete. Results in $LOG_DIR/v55_ppr_eval.log"
else
    log "No checkpoint found for PPR evaluation."
fi

###############################################################################
# PHASE 5: Fusion sweep (if applicable)
###############################################################################

log ""
log "=========================================="
log "PHASE 5: Fusion Sweep"
log "=========================================="

if [[ -f "$BEST_CKPT" ]]; then
    log "Running fusion sweep..."
    python3 scripts/evaluation/sweep_fusion_ppr.py \
        --compact-checkpoint "$BEST_CKPT" \
        --legacy-checkpoint "experimental_results/N1v28a_dual_head/subj01/checkpoint_best.pt" \
        --subject subj01 \
        > "$LOG_DIR/v55_fusion_sweep.log" 2>&1 || \
        log "WARNING: Fusion sweep failed (non-critical)"
    log "Fusion sweep complete. Results in $LOG_DIR/v55_fusion_sweep.log"
fi

###############################################################################
# SUMMARY
###############################################################################

log ""
log "=========================================="
log "V55 AUTO-CHAIN PIPELINE COMPLETE"
log "=========================================="
log "V55a checkpoint: $(ls -lh "$V55A_CKPT" 2>/dev/null | awk '{print $5, $6, $7, $8}' || echo 'MISSING')"
log "V55b checkpoint: $(ls -lh "$V55B_CKPT" 2>/dev/null | awk '{print $5, $6, $7, $8}' || echo 'MISSING')"
log "V55c checkpoint: $(ls -lh "$V55C_CKPT" 2>/dev/null | awk '{print $5, $6, $7, $8}' || echo 'MISSING')"
log ""
log "All logs in: $LOG_DIR/v55_*.log"
log "Pipeline finished at $(timestamp)"
