#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh - Run multiple experiments in sequence
# =============================================================================
# Runs a series of experiments from config files, logging results and
# supporting resume capability for partial sweeps.
#
# Usage:
#   bash scripts/run_sweep.sh configs/experiments/*.yaml
#   bash scripts/run_sweep.sh --sweep-name my_sweep configs/experiments/exp_*.yaml
#
# Options:
#   --sweep-name NAME    Name for the sweep (default: sweep_<timestamp>)
#   --skip-preflight     Skip preflight checks for each experiment
#   --continue-on-error  Continue running even if one experiment fails
#   --help               Show this help message
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_header() {
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}$*${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
CONFIGS=()
SWEEP_NAME="sweep_$(date +%Y%m%d_%H%M%S)"
SKIP_PREFLIGHT=false
CONTINUE_ON_ERROR=false

show_help() {
    grep '^#' "$0" | grep -v '#!/usr/bin/env' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep-name)
            SWEEP_NAME="$2"
            shift 2
            ;;
        --skip-preflight)
            SKIP_PREFLIGHT=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            CONFIGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    log_error "No config files specified"
    echo "Usage: bash scripts/run_sweep.sh [options] <config1.yaml> <config2.yaml> ..."
    exit 1
fi

log_header "🔄 EXPERIMENT SWEEP: ${SWEEP_NAME}"
log_info "Number of experiments: ${#CONFIGS[@]}"
log_info "Continue on error: ${CONTINUE_ON_ERROR}"

# Create sweep directory
SWEEP_DIR="${PROJECT_ROOT}/runs/${SWEEP_NAME}"
mkdir -p "${SWEEP_DIR}"
log_info "Sweep directory: ${SWEEP_DIR}"

# Create sweep manifest
MANIFEST_FILE="${SWEEP_DIR}/sweep_manifest.json"
cat > "${MANIFEST_FILE}" << EOF
{
  "sweep_name": "${SWEEP_NAME}",
  "start_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "num_experiments": ${#CONFIGS[@]},
  "configs": [
$(for i in "${!CONFIGS[@]}"; do
    echo -n "    \"${CONFIGS[$i]}\""
    [[ $i -lt $((${#CONFIGS[@]} - 1)) ]] && echo "," || echo ""
done)
  ],
  "status": "running"
}
EOF

log_success "Created sweep manifest: ${MANIFEST_FILE}"

# Run experiments
SUCCESSES=0
FAILURES=0
SKIPPED=0

RESULTS_FILE="${SWEEP_DIR}/sweep_results.txt"
echo "Sweep Results: ${SWEEP_NAME}" > "${RESULTS_FILE}"
echo "Started: $(date)" >> "${RESULTS_FILE}"
echo "======================================" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    EXP_NUM=$((i + 1))
    
    log_header "Experiment ${EXP_NUM}/${#CONFIGS[@]}: $(basename ${CONFIG})"
    
    if [[ ! -f "${CONFIG}" ]]; then
        log_error "Config not found: ${CONFIG}"
        echo "❌ Experiment ${EXP_NUM}: FAILED (config not found) - ${CONFIG}" >> "${RESULTS_FILE}"
        ((FAILURES++))
        
        if [[ "${CONTINUE_ON_ERROR}" == "false" ]]; then
            break
        fi
        continue
    fi
    
    # Build run command
    RUN_CMD="${SCRIPT_DIR}/run_experiment_simple.sh ${CONFIG}"
    if [[ "${SKIP_PREFLIGHT}" == "true" ]]; then
        RUN_CMD="${RUN_CMD} --skip-preflight"
    fi
    
    # Run experiment
    START_TIME=$(date +%s)
    
    if eval "${RUN_CMD}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        log_success "Experiment ${EXP_NUM} completed successfully! (${DURATION}s)"
        echo "✓ Experiment ${EXP_NUM}: SUCCESS (${DURATION}s) - ${CONFIG}" >> "${RESULTS_FILE}"
        ((SUCCESSES++))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        log_error "Experiment ${EXP_NUM} failed! (${DURATION}s)"
        echo "❌ Experiment ${EXP_NUM}: FAILED (${DURATION}s) - ${CONFIG}" >> "${RESULTS_FILE}"
        ((FAILURES++))
        
        if [[ "${CONTINUE_ON_ERROR}" == "false" ]]; then
            log_error "Stopping sweep due to failure (use --continue-on-error to continue)"
            break
        fi
    fi
    
    echo "" >> "${RESULTS_FILE}"
done

# Update sweep manifest
python3 << PYEOF
import json
from pathlib import Path
from datetime import datetime

manifest_file = Path("${MANIFEST_FILE}")
with open(manifest_file) as f:
    manifest = json.load(f)

manifest.update({
    "end_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "completed",
    "results": {
        "total": ${#CONFIGS[@]},
        "successes": ${SUCCESSES},
        "failures": ${FAILURES},
        "skipped": ${SKIPPED}
    }
})

with open(manifest_file, 'w') as f:
    json.dump(manifest, f, indent=2)
PYEOF

# Final summary
echo "" >> "${RESULTS_FILE}"
echo "======================================" >> "${RESULTS_FILE}"
echo "Completed: $(date)" >> "${RESULTS_FILE}"
echo "Total: ${#CONFIGS[@]}" >> "${RESULTS_FILE}"
echo "Successes: ${SUCCESSES}" >> "${RESULTS_FILE}"
echo "Failures: ${FAILURES}" >> "${RESULTS_FILE}"

log_header "📊 SWEEP SUMMARY"
echo ""
log_info "Total experiments: ${#CONFIGS[@]}"
log_success "Successes: ${SUCCESSES}"
log_error "Failures: ${FAILURES}"
echo ""
log_info "Results saved to: ${RESULTS_FILE}"
log_info "Sweep directory: ${SWEEP_DIR}"
echo ""

if [[ ${FAILURES} -eq 0 ]]; then
    log_success "✓ All experiments completed successfully!"
    exit 0
else
    log_error "✗ Some experiments failed"
    exit 1
fi
