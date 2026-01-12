#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh - Run a single experiment with preflight checks
# =============================================================================
# Orchestrates the complete experiment workflow:
# 1. Load environment
# 2. Run preflight checks
# 3. Execute training
# 4. Log results
#
# Usage:
#   bash scripts/run_experiment.sh <config.yaml> [options]
#   bash scripts/run_experiment.sh configs/experiments/example.yaml
#   bash scripts/run_experiment.sh configs/experiments/example.yaml --skip-preflight
#   bash scripts/run_experiment.sh configs/experiments/example.yaml --name custom_name
#
# Arguments:
#   config.yaml          Path to experiment configuration file
#
# Options:
#   --skip-preflight     Skip preflight checks
#   --skip-gpu           Skip GPU checks in preflight
#   --name NAME          Override experiment name
#   --seed SEED          Override random seed
#   --resume             Resume from latest checkpoint
#   --resume-from PATH   Resume from specific checkpoint
#   --help               Show this help message
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

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

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
CONFIG_FILE=""
SKIP_PREFLIGHT=false
SKIP_GPU=false
EXP_NAME=""
SEED=""
RESUME=false
RESUME_FROM=""

show_help() {
    grep '^#' "$0" | grep -v '#!/usr/bin/env' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-preflight)
            SKIP_PREFLIGHT=true
            shift
            ;;
        --skip-gpu)
            SKIP_GPU=true
            shift
            ;;
        --name)
            EXP_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            if [[ -z "${CONFIG_FILE}" ]]; then
                CONFIG_FILE="$1"
                shift
            else
                log_error "Unknown argument: $1"
                echo "Use --help for usage information"
                exit 1
            fi
            ;;
    esac
done

# Validate arguments
if [[ -z "${CONFIG_FILE}" ]]; then
    log_error "Missing required argument: config file"
    echo ""
    echo "Usage: bash scripts/run_experiment.sh <config.yaml> [options]"
    echo "Use --help for more information"
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    log_error "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

log_header "🚀 EXPERIMENT RUNNER"

log_info "Project root: ${PROJECT_ROOT}"
log_info "Config file: ${CONFIG_FILE}"
log_info "Timestamp: $(date)"
log_info "User: $(whoami)@$(hostname)"

# Load environment
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    log_info "Loading environment from .env"
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
else
    log_info "No .env file found, using defaults"
fi

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_error "Virtual environment not active!"
    log_info "Please activate it first:"
    echo ""
    echo "  source venv/bin/activate"
    echo "  # or"
    echo "  source activate_env.sh"
    echo ""
    exit 1
fi

log_success "Virtual environment active: ${VIRTUAL_ENV}"

# Run preflight checks
if [[ "${SKIP_PREFLIGHT}" == "false" ]]; then
    log_header "Pre-flight Checks"
    
    PREFLIGHT_SCRIPT="${SCRIPT_DIR}/preflight.sh"
    if [[ ! -f "${PREFLIGHT_SCRIPT}" ]]; then
        log_error "Preflight script not found: ${PREFLIGHT_SCRIPT}"
        exit 1
    fi
    
    PREFLIGHT_ARGS=""
    if [[ "${SKIP_GPU}" == "true" ]]; then
        PREFLIGHT_ARGS="--skip-gpu"
    fi
    
    if bash "${PREFLIGHT_SCRIPT}" ${PREFLIGHT_ARGS} --quiet; then
        log_success "All preflight checks passed!"
    else
        log_error "Preflight checks failed!"
        exit 1
    fi
else
    log_info "Skipping preflight checks (--skip-preflight)"
fi

# Build training command
log_header "Starting Training"

TRAIN_CMD="python -m src.train --config ${CONFIG_FILE}"

if [[ -n "${EXP_NAME}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --name ${EXP_NAME}"
    log_info "Experiment name: ${EXP_NAME}"
fi

if [[ -n "${SEED}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --seed ${SEED}"
    log_info "Random seed: ${SEED}"
fi

if [[ "${RESUME}" == "true" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --resume"
    log_info "Resuming from latest checkpoint"
fi

if [[ -n "${RESUME_FROM}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --resume-from ${RESUME_FROM}"
    log_info "Resuming from: ${RESUME_FROM}"
fi

log_info "Command: ${TRAIN_CMD}"
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Run training
START_TIME=$(date +%s)

if eval "${TRAIN_CMD}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log_header "✓ Experiment Completed"
    log_success "Training finished successfully!"
    log_info "Duration: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
    
    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log_header "✗ Experiment Failed"
    log_error "Training failed after ${DURATION} seconds!"
    
    exit 1
fi
