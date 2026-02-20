#!/bin/bash
# Run all experiments (EXP0-EXP6) sequentially with comprehensive logging
# 
# Usage:
#   bash scripts/run_all_experiments.sh [GPU_ID]
#
# Example:
#   bash scripts/run_all_experiments.sh 0

set -e

GPU=${1:-0}
TRAIN_SCRIPT="scripts/train_unified.py"

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="experimental_results/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_all_experiments_${TIMESTAMP}.log"

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Function to log separator
log_separator() {
    echo "=================================================================================" | tee -a "${LOG_FILE}"
}

# Redirect all output to both console and log file
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

log_separator
log "ABLATION SUITE - Full Experiment Run"
log_separator
log "Start time: $(date)"
log "GPU: ${GPU}"
log "Log file: ${LOG_FILE}"
log ""

# Log system information
log "System Information:"
log "  Hostname: $(hostname)"
log "  CPUs: $(nproc)"
log "  Memory: $(free -h | grep Mem | awk '{print $2}')"
if command -v nvidia-smi &> /dev/null; then
    log "  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        log "    ${line}"
    done
fi
log ""

# Check if preprocessors exist
if [ ! -f "cache/embedding_preproc/subj01_center_pcr_k8.pkl" ]; then
    log "⚠️  Preprocessor artifacts not found!"
    log "   Running build_all_preprocessors.sh first..."
    bash scripts/build_all_preprocessors.sh
    log ""
fi

# Array of experiments (EXP0-EXP6: original ablation, EXP7-EXP8: novel contributions)
experiments=(
    "exp0_baseline"
    "exp1_preproc"
    "exp2_queue"
    "exp3_gaussian_nll"
    "exp4_gaussian_nce"
    "exp5_kl_anneal"
    "exp6_whiten"
    "exp7_vmf_nce"
    "exp8_roi_transformer"
)

for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    config="configs/experiments/${exp}.yaml"
    exp_num=$((i + 1))
    
    log_separator
    log "[${exp_num}/${#experiments[@]}] Starting: ${exp}"
    log_separator
    log "Config: ${config}"
    log "Start time: $(date)"
    log ""
    
    if [ ! -f "${config}" ]; then
        log "❌ Config not found: ${config}"
        exit 1
    fi
    
    # Record start time
    exp_start=$(date +%s)
    
    # Train model
    log "Training ${exp}..."
    if python3 "${TRAIN_SCRIPT}" \
        --config "${config}" \
        --gpu "${GPU}"; then
        
        # Record end time
        exp_end=$(date +%s)
        exp_duration=$((exp_end - exp_start))
        
        log ""
        log "✅ ${exp} training complete"
        log "   Duration: $((exp_duration / 3600))h $((exp_duration % 3600 / 60))m $((exp_duration % 60))s"
        log ""
    else
        log ""
        log "❌ ${exp} training FAILED"
        log "   Check logs above for errors"
        log ""
        continue  # Skip evaluation and move to next experiment
    fi
    
    # Log experiment completion
    log ""
    log "Completed ${exp_num}/${#experiments[@]} experiments"
    log "---"
    log ""
    
    # GPU status check
    if command -v nvidia-smi &> /dev/null; then
        log "GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
            awk -F', ' '{printf "  GPU Util: %s, Memory: %s / %s, Temp: %s\n", $1, $2, $3, $4}' | tee -a "${LOG_FILE}"
        log ""
    fi
done

log_separator
log "All Experiments Complete!"
log_separator
log "End time: $(date)"
log ""

# Calculate total duration
if [ -n "${SECONDS}" ]; then
    total_duration=${SECONDS}
    log "Total duration: $((total_duration / 3600))h $((total_duration % 3600 / 60))m $((total_duration % 60))s"
    log ""
fi

log "Results saved to: experimental_results/"
log "Log file: ${LOG_FILE}"
log ""

# Summary table
log "Experiment Summary:"
log "-------------------"
for exp in "${experiments[@]}"; do
    eval_dir="experimental_results/${exp}"
    checkpoint="${eval_dir}/checkpoint.pth"
    if [ -f "${checkpoint}" ]; then
        status="✅ COMPLETE"
    else
        status="❌ FAILED"
    fi
    log "${status} ${exp}"
done
log ""

log "Next steps:"
log "  1. Review log: cat ${LOG_FILE}"
log "  2. Compare results: python3 scripts/compare_experiments.py"
log "  3. Generate figures for paper"
log "  4. Write paper using docs/paper_outline.md"
log ""
log_separator
