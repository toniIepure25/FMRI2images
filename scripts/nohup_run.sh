#!/usr/bin/env bash
# =============================================================================
# nohup_run.sh - Run experiment with nohup (continues after logout)
# =============================================================================
# Runs an experiment in the background using nohup, which allows the process
# to continue running even after you log out from the shell.
#
# Usage:
#   bash scripts/nohup_run.sh <config.yaml>
#   bash scripts/nohup_run.sh configs/experiments/example.yaml
#
# The output will be saved to a log file in the project logs directory.
# The process ID will be saved for later monitoring/termination.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
if [[ $# -lt 1 ]]; then
    log_error "Missing required argument: config file"
    echo ""
    echo "Usage: bash scripts/nohup_run.sh <config.yaml>"
    exit 1
fi

CONFIG_FILE="$1"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    log_error "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Create logs directory
LOGS_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOGS_DIR}"

# Generate unique log filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_NAME=$(basename "${CONFIG_FILE}" .yaml)
LOG_FILE="${LOGS_DIR}/nohup_${CONFIG_NAME}_${TIMESTAMP}.log"
PID_FILE="${LOGS_DIR}/nohup_${CONFIG_NAME}_${TIMESTAMP}.pid"

log_info "Config: ${CONFIG_FILE}"
log_info "Log file: ${LOG_FILE}"
log_info "PID file: ${PID_FILE}"

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_warn "Virtual environment not active"
    VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/venv}"
    
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        log_info "Will activate venv: ${VENV_PATH}"
        # shellcheck disable=SC1091
        source "${VENV_PATH}/bin/activate"
    else
        log_error "Virtual environment not found at: ${VENV_PATH}"
        log_error "Please activate it first or run setup_env.sh"
        exit 1
    fi
fi

# Load .env if exists
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    log_info "Loading environment from .env"
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Change to project root
cd "${PROJECT_ROOT}"

# Start experiment with nohup
log_info "Starting experiment in background..."

nohup bash "${SCRIPT_DIR}/run_experiment_simple.sh" "${CONFIG_FILE}" > "${LOG_FILE}" 2>&1 &
PID=$!

# Save PID
echo "${PID}" > "${PID_FILE}"

# Wait a moment to check if process started successfully
sleep 2

if kill -0 "${PID}" 2>/dev/null; then
    log_success "Experiment started successfully!"
    echo ""
    log_info "Process ID: ${PID}"
    log_info "Log file: ${LOG_FILE}"
    log_info "PID file: ${PID_FILE}"
    echo ""
    log_info "To monitor the log:"
    echo "  tail -f ${LOG_FILE}"
    echo ""
    log_info "To check if process is running:"
    echo "  ps -p ${PID}"
    echo ""
    log_info "To stop the experiment:"
    echo "  kill ${PID}"
    echo "  # or"
    echo "  kill \$(cat ${PID_FILE})"
    echo ""
else
    log_error "Failed to start experiment"
    log_error "Check the log file: ${LOG_FILE}"
    exit 1
fi
