#!/usr/bin/env bash
# =============================================================================
# preflight.sh - Pre-flight checks for experiment runs
# =============================================================================
# Validates that the environment is ready for training:
# - Virtual environment is active
# - Python and pip are accessible
# - GPU is available and working
# - Required disk space is available
# - Can write to output directories
# - PyTorch CUDA works
#
# Usage:
#   bash scripts/preflight.sh [--skip-gpu] [--min-free-gb N]
#
# Options:
#   --skip-gpu       Skip GPU checks (for CPU-only mode)
#   --min-free-gb N  Minimum free disk space in GB (default: 50)
#   --quiet          Less verbose output
#   --help           Show this help message
#
# Exit codes:
#   0  All checks passed
#   1  One or more checks failed
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓ PASS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[⚠ WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗ FAIL]${NC} $*"
}

log_check() {
    echo -e "${BLUE}[CHECK]${NC} $*"
}

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
SKIP_GPU=false
MIN_FREE_GB=50
QUIET=false

show_help() {
    grep '^#' "$0" | grep -v '#!/usr/bin/env' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-gpu)
            SKIP_GPU=true
            shift
            ;;
        --min-free-gb)
            MIN_FREE_GB="$2"
            shift 2
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Track failures
FAILED_CHECKS=0
WARNINGS=0

print_header() {
    if [[ "${QUIET}" == "false" ]]; then
        echo ""
        echo "=========================================="
        echo "$*"
        echo "=========================================="
        echo ""
    fi
}

# Load environment variables if .env exists
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    if [[ "${QUIET}" == "false" ]]; then
        log_info "Loading environment from .env file"
    fi
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

print_header "🚀 PRE-FLIGHT CHECKS"

# =============================================================================
# 1. Environment and paths
# =============================================================================
print_header "1. Environment and Paths"

log_check "Checking project root..."
if [[ -d "${PROJECT_ROOT}" ]]; then
    log_success "Project root: ${PROJECT_ROOT}"
else
    log_error "Project root not found: ${PROJECT_ROOT}"
    ((FAILED_CHECKS++))
fi

log_check "Checking user and hostname..."
log_info "User: $(whoami)"
log_info "Hostname: $(hostname)"
log_info "PWD: $(pwd)"

# =============================================================================
# 2. Virtual environment
# =============================================================================
print_header "2. Virtual Environment"

log_check "Checking if virtual environment is active..."
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    log_success "Virtual environment active: ${VIRTUAL_ENV}"
else
    log_warn "Virtual environment not active"
    log_info "To activate: source venv/bin/activate"
    log_info "Or: source activate_env.sh"
    ((WARNINGS++))
fi

log_check "Checking Python..."
if command -v python &> /dev/null; then
    PYTHON_PATH=$(which python)
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    log_success "Python found: ${PYTHON_PATH}"
    log_info "Python version: ${PYTHON_VERSION}"
else
    log_error "Python not found in PATH"
    ((FAILED_CHECKS++))
fi

log_check "Checking pip..."
if command -v pip &> /dev/null; then
    PIP_PATH=$(which pip)
    PIP_VERSION=$(pip --version | awk '{print $2}')
    log_success "pip found: ${PIP_PATH}"
    log_info "pip version: ${PIP_VERSION}"
else
    log_error "pip not found in PATH"
    ((FAILED_CHECKS++))
fi

# =============================================================================
# 3. Python packages
# =============================================================================
print_header "3. Python Packages"

check_package() {
    local package=$1
    local import_name=${2:-$1}
    
    log_check "Checking ${package}..."
    if python -c "import ${import_name}" 2>/dev/null; then
        local version
        version=$(python -c "import ${import_name}; print(getattr(${import_name}, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        log_success "${package}: ${version}"
    else
        log_error "${package} not found or cannot be imported"
        ((FAILED_CHECKS++))
    fi
}

check_package "PyTorch" "torch"
check_package "NumPy" "numpy"
check_package "Pandas" "pandas"
check_package "PyYAML" "yaml"
check_package "tqdm" "tqdm"
check_package "scikit-learn" "sklearn"

# =============================================================================
# 4. GPU and CUDA
# =============================================================================
if [[ "${SKIP_GPU}" == "false" ]]; then
    print_header "4. GPU and CUDA"
    
    log_check "Checking nvidia-smi..."
    if command -v nvidia-smi &> /dev/null; then
        log_success "nvidia-smi found"
        
        if [[ "${QUIET}" == "false" ]]; then
            log_info "GPU information:"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read -r line; do
                log_info "  ${line}"
            done
        fi
        
        # Check GPU count
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        log_info "GPU count: ${GPU_COUNT}"
    else
        log_error "nvidia-smi not found - GPU not available"
        ((FAILED_CHECKS++))
    fi
    
    log_check "Checking PyTorch CUDA availability..."
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        GPU_COUNT_TORCH=$(python -c "import torch; print(torch.cuda.device_count())")
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        
        log_success "PyTorch CUDA available"
        log_info "CUDA version: ${CUDA_VERSION}"
        log_info "GPU count (PyTorch): ${GPU_COUNT_TORCH}"
        log_info "GPU 0: ${GPU_NAME}"
    else
        log_error "PyTorch CUDA not available"
        log_error "GPU acceleration will not work"
        ((FAILED_CHECKS++))
    fi
    
    log_check "Testing CUDA tensor operations..."
    if python -c "
import torch
try:
    device = torch.device('cuda')
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = torch.matmul(x, y)
    assert z.device.type == 'cuda'
    print('CUDA tensor test passed')
    exit(0)
except Exception as e:
    print(f'CUDA tensor test failed: {e}')
    exit(1)
" 2>&1; then
        log_success "CUDA tensor operations working"
    else
        log_error "CUDA tensor operations failed"
        ((FAILED_CHECKS++))
    fi
else
    log_warn "GPU checks skipped (--skip-gpu)"
fi

# =============================================================================
# 5. System resources
# =============================================================================
print_header "5. System Resources"

log_check "Checking CPU..."
CPU_COUNT=$(python -c "import os; print(os.cpu_count())")
log_info "CPU cores: ${CPU_COUNT}"

log_check "Checking memory..."
if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
    AVAIL_MEM=$(free -h | awk '/^Mem:/ {print $7}')
    log_info "Total memory: ${TOTAL_MEM}"
    log_info "Available memory: ${AVAIL_MEM}"
else
    log_warn "Cannot check memory (free command not available)"
    ((WARNINGS++))
fi

log_check "Checking disk space..."
check_disk() {
    local path=$1
    local label=$2
    
    if [[ -d "${path}" ]] || [[ -d "$(dirname "${path}")" ]]; then
        local check_path="${path}"
        [[ ! -d "${check_path}" ]] && check_path="$(dirname "${path}")"
        
        local free_gb
        free_gb=$(df -BG "${check_path}" | awk 'NR==2 {print $4}' | sed 's/G//')
        
        log_info "${label}: ${free_gb} GB free"
        
        if [[ ${free_gb} -lt ${MIN_FREE_GB} ]]; then
            log_error "${label} has less than ${MIN_FREE_GB} GB free"
            ((FAILED_CHECKS++))
        else
            log_success "${label} has sufficient space"
        fi
    else
        log_warn "${label} path does not exist: ${path}"
        ((WARNINGS++))
    fi
}

check_disk "${PROJECT_ROOT}" "Project root"
check_disk "${RUNS_DIR:-${PROJECT_ROOT}/runs}" "Runs directory"
check_disk "${DATA_DIR:-${PROJECT_ROOT}/data}" "Data directory"
check_disk "${CACHE_DIR:-${PROJECT_ROOT}/cache}" "Cache directory"

# =============================================================================
# 6. Directory permissions
# =============================================================================
print_header "6. Directory Permissions"

check_write_permission() {
    local dir=$1
    local label=$2
    
    log_check "Checking write permission for ${label}..."
    
    # Create directory if it doesn't exist
    if [[ ! -d "${dir}" ]]; then
        if mkdir -p "${dir}" 2>/dev/null; then
            log_success "Created ${label}: ${dir}"
        else
            log_error "Cannot create ${label}: ${dir}"
            ((FAILED_CHECKS++))
            return
        fi
    fi
    
    # Test write permission
    local test_file="${dir}/.preflight_test_$$"
    if touch "${test_file}" 2>/dev/null; then
        rm -f "${test_file}"
        log_success "Write permission OK for ${label}"
    else
        log_error "No write permission for ${label}: ${dir}"
        ((FAILED_CHECKS++))
    fi
}

check_write_permission "${RUNS_DIR:-${PROJECT_ROOT}/runs}" "Runs directory"
check_write_permission "${PROJECT_ROOT}/logs" "Logs directory"
check_write_permission "${PROJECT_ROOT}/checkpoints" "Checkpoints directory"

# =============================================================================
# 7. Git repository (optional)
# =============================================================================
print_header "7. Git Repository (optional)"

log_check "Checking git..."
if command -v git &> /dev/null; then
    log_success "git found: $(which git)"
    
    if git -C "${PROJECT_ROOT}" rev-parse --git-dir > /dev/null 2>&1; then
        log_success "Git repository detected"
        COMMIT_HASH=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
        BRANCH=$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        log_info "Branch: ${BRANCH}"
        log_info "Commit: ${COMMIT_HASH}"
        
        # Check for uncommitted changes
        if git -C "${PROJECT_ROOT}" diff-index --quiet HEAD -- 2>/dev/null; then
            log_success "No uncommitted changes"
        else
            log_warn "You have uncommitted changes"
            log_info "Consider committing before running experiments for reproducibility"
            ((WARNINGS++))
        fi
    else
        log_warn "Not a git repository"
        ((WARNINGS++))
    fi
else
    log_warn "git not found (not required but recommended)"
    ((WARNINGS++))
fi

# =============================================================================
# Summary
# =============================================================================
print_header "📋 SUMMARY"

echo ""
if [[ ${FAILED_CHECKS} -eq 0 ]]; then
    log_success "All critical checks passed! ✓"
    
    if [[ ${WARNINGS} -gt 0 ]]; then
        log_warn "${WARNINGS} warning(s) - see above"
    fi
    
    echo ""
    log_info "System is ready for experiments!"
    log_info "Next step: bash scripts/run_experiment.sh <config.yaml>"
    echo ""
    exit 0
else
    log_error "${FAILED_CHECKS} check(s) failed! ✗"
    
    if [[ ${WARNINGS} -gt 0 ]]; then
        log_warn "${WARNINGS} warning(s)"
    fi
    
    echo ""
    log_error "System is NOT ready for experiments"
    log_info "Please fix the errors above before proceeding"
    echo ""
    exit 1
fi
