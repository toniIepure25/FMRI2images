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

# Array of experiments
experiments=(
    "exp0_baseline"
    "exp1_preproc"
    "exp2_queue"
    "exp3_gaussian_nll"
    "exp4_gaussian_nce"
    "exp5_kl_anneal"
    "exp6_whiten"
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
    if python "${TRAIN_SCRIPT}" \
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
    
    # Evaluate model
    log "Evaluating ${exp}..."
    checkpoint="experimental_results/${exp}/checkpoints/best.ckpt"
    eval_output="experimental_results/${exp}/evaluation"
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
    eval_dir="experimental_results/${exp}/evaluation"
    if [ -f "${eval_dir}/metrics.json" ]; then
        if command -v jq &> /dev/null; then
            R1=$(jq -r '.retrieval."R@1" // "N/A"' "${eval_dir}/metrics.json")
            oracle=$(jq -r '.oracle.passed // "N/A"' "${eval_dir}/metrics.json")
            status="✅"
            [ "${oracle}" == "false" ] && status="⚠️ "
        else
            R1="N/A"
            status="✅"
        fi
    else
        R1="FAILED"
        status="❌"
    fi
    log "${status} ${exp}: R@1=${R1}"
done
log ""

log "Next steps:"
log "  1. Review log: cat ${LOG_FILE}"
log "  2. Compare results: python scripts/compare_experiments.py"
log "  3. Generate figures for paper"
log "  4. Write paper using docs/paper_outline.md"
log ""
log_separator     --checkpoint "${checkpoint}" \
            --output "${eval_output}" \
            --split test \
            --preprocessor "${preprocessor_path}"; then
            
            log "✅ Evaluation complete"
            
            # Log key metrics if available
            if [ -f "${eval_output}/metrics.json" ]; then
                log ""
                log "Key Metrics for ${exp}:"
                if command -v jq &> /dev/null; then
                    R1=$(jq -r '.retrieval."R@1" // "N/A"' "${eval_output}/metrics.json")
                    MeanR=$(jq -r '.retrieval.MeanR // "N/A"' "${eval_output}/metrics.json")
                    oracle=$(jq -r '.oracle.passed // "N/A"' "${eval_output}/metrics.json")
                    log "  R@1: ${R1}"
                    log "  MeanR: ${MeanR}"
                    log "  Oracle passed: ${oracle}"
                fi
            fi
        else
            log "❌ Evaluation FAILED"
        fi
    else
        log "⚠️  Checkpoint not found, skipping evaluation"
        log "   Expected: ${checkpoint}"
    fi
    
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

echo "="
echo "All experiments complete!"
echo "="
echo ""
echo "Results saved to: experimental_results/"
echo ""
echo "Next steps:"
echo "  1. Compare results: python scripts/compare_experiments.py"
echo "  2. Generate figures for paper"
echo "  3. Write paper using docs/paper_outline.md"
echo ""
