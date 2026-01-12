#!/usr/bin/env bash
# =============================================================================
# tmux_run.sh - Run experiment in a detached tmux session
# =============================================================================
# Starts an experiment in a tmux session that persists after disconnection.
# Ideal for long-running experiments on remote servers.
#
# Usage:
#   bash scripts/tmux_run.sh <config.yaml> [session_name]
#   bash scripts/tmux_run.sh configs/experiments/example.yaml
#   bash scripts/tmux_run.sh configs/experiments/example.yaml my_experiment
#
# To attach to the session:
#   tmux attach -t <session_name>
#
# To list sessions:
#   tmux ls
#
# To kill a session:
#   tmux kill-session -t <session_name>
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    log_error "tmux is not installed"
    echo ""
    echo "Install tmux first:"
    echo "  # On Ubuntu/Debian:"
    echo "  sudo apt-get install tmux"
    echo "  # On CentOS/RHEL:"
    echo "  sudo yum install tmux"
    echo ""
    exit 1
fi

# Parse arguments
if [[ $# -lt 1 ]]; then
    log_error "Missing required argument: config file"
    echo ""
    echo "Usage: bash scripts/tmux_run.sh <config.yaml> [session_name]"
    exit 1
fi

CONFIG_FILE="$1"
SESSION_NAME="${2:-exp_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    log_error "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

log_info "Config: ${CONFIG_FILE}"
log_info "Session name: ${SESSION_NAME}"

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    log_error "tmux session '${SESSION_NAME}' already exists"
    echo ""
    echo "Attach to it with: tmux attach -t ${SESSION_NAME}"
    echo "Or kill it with: tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

# Create tmux session and run experiment
log_info "Creating tmux session..."

# Load .env if exists
ENV_SETUP=""
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    ENV_SETUP="set -a; source ${PROJECT_ROOT}/.env; set +a;"
fi

# Activate venv
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/venv}"
VENV_SETUP=""
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    VENV_SETUP="source ${VENV_PATH}/bin/activate;"
fi

# Build command to run in tmux
TMUX_CMD="cd ${PROJECT_ROOT}; ${ENV_SETUP} ${VENV_SETUP} bash scripts/run_experiment_simple.sh ${CONFIG_FILE}"

# Create detached tmux session
tmux new-session -d -s "${SESSION_NAME}" "${TMUX_CMD}"

log_success "tmux session '${SESSION_NAME}' created and experiment started!"
echo ""
log_info "To attach to the session:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
log_info "To detach from the session (while inside):"
echo "  Press: Ctrl+B, then D"
echo ""
log_info "To list all sessions:"
echo "  tmux ls"
echo ""
log_info "To kill the session:"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo ""
