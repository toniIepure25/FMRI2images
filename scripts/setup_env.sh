#!/usr/bin/env bash
# =============================================================================
# setup_env.sh - Setup Python virtual environment and dependencies
# =============================================================================
# This script sets up a Python virtual environment and installs all
# dependencies needed for the project. Designed for JupyterHub/HPC
# environments where sudo is not available.
#
# Usage:
#   bash scripts/setup_env.sh [--force] [--cuda-version VERSION]
#
# Options:
#   --force            Remove existing venv and recreate from scratch
#   --cuda-version VER Specify CUDA version (default: auto-detect or 12.2)
#   --help             Show this help message
#
# Environment variables (from .env):
#   VENV_PATH          Path to virtual environment (default: ./venv)
#   CUDA_VERSION       CUDA version for PyTorch (default: 12.2)
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
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
FORCE_REBUILD=false
CUDA_VERSION_ARG=""

show_help() {
    grep '^#' "$0" | grep -v '#!/usr/bin/env' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_REBUILD=true
            shift
            ;;
        --cuda-version)
            CUDA_VERSION_ARG="$2"
            shift 2
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

log_info "Starting environment setup for project: ${PROJECT_ROOT}"
log_info "User: $(whoami), Hostname: $(hostname)"

# Load environment variables if .env exists
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    log_info "Loading environment from .env file"
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
else
    log_warn ".env file not found, using defaults"
    log_info "Consider copying .env.jupyterhub or .env.example to .env"
fi

# Set defaults
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/venv}"
CUDA_VERSION="${CUDA_VERSION_ARG:-${CUDA_VERSION:-12.2}}"

log_info "Virtual environment path: ${VENV_PATH}"
log_info "CUDA version: ${CUDA_VERSION}"

# Check Python version
log_info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
log_info "Found Python ${PYTHON_VERSION}"

# Verify Python version >= 3.10
PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)
if [[ "${PYTHON_MAJOR}" -lt 3 ]] || [[ "${PYTHON_MAJOR}" -eq 3 && "${PYTHON_MINOR}" -lt 10 ]]; then
    log_error "Python 3.10+ required, found ${PYTHON_VERSION}"
    exit 1
fi

# Handle force rebuild
if [[ "${FORCE_REBUILD}" == "true" ]] && [[ -d "${VENV_PATH}" ]]; then
    log_warn "Force rebuild requested, removing existing venv..."
    rm -rf "${VENV_PATH}"
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "${VENV_PATH}" ]]; then
    log_info "Creating virtual environment at ${VENV_PATH}..."
    python3 -m venv "${VENV_PATH}"
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists at ${VENV_PATH}"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

# Verify activation
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_error "Failed to activate virtual environment"
    exit 1
fi

log_success "Virtual environment activated: ${VIRTUAL_ENV}"

# Upgrade pip, setuptools, wheel
log_info "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
log_info "Installing dependencies..."

# Determine which requirements file to use
if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
    REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
    log_info "Using requirements.txt"
elif [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    REQUIREMENTS_FILE="${PROJECT_ROOT}/pyproject.toml"
    log_info "Using pyproject.toml"
else
    log_error "No requirements.txt or pyproject.toml found"
    exit 1
fi

# Check if torch is already installed
TORCH_INSTALLED=false
if python -c "import torch" 2>/dev/null; then
    TORCH_INSTALLED=true
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    log_info "PyTorch already installed: ${TORCH_VERSION}"
    
    # Check CUDA availability
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.version.cuda)")
        log_success "CUDA is available in PyTorch: ${CUDA_AVAILABLE}"
    else
        log_warn "PyTorch is installed but CUDA is not available"
        log_warn "You may need to reinstall PyTorch with CUDA support"
        TORCH_INSTALLED=false
    fi
fi

# Install PyTorch with CUDA if not already installed
if [[ "${TORCH_INSTALLED}" == "false" ]]; then
    log_info "Installing PyTorch with CUDA ${CUDA_VERSION} support..."
    
    # Determine PyTorch index URL based on CUDA version
    if [[ "${CUDA_VERSION}" == "12.1" ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    elif [[ "${CUDA_VERSION}" == "12.2" ]] || [[ "${CUDA_VERSION}" == "12"* ]]; then
        # Use cu121 for CUDA 12.x (most compatible)
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    elif [[ "${CUDA_VERSION}" == "11.8" ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    else
        log_warn "Unknown CUDA version ${CUDA_VERSION}, using default PyTorch"
        TORCH_INDEX=""
    fi
    
    if [[ -n "${TORCH_INDEX}" ]]; then
        log_info "Using PyTorch index: ${TORCH_INDEX}"
        pip install torch torchvision --index-url "${TORCH_INDEX}"
    else
        log_info "Installing PyTorch from default index"
        pip install torch torchvision
    fi
    
    log_success "PyTorch installed"
fi

# Install other dependencies
log_info "Installing other dependencies from ${REQUIREMENTS_FILE}..."
if [[ "${REQUIREMENTS_FILE}" == *"requirements.txt" ]]; then
    # Install from requirements.txt, skipping torch lines if already installed
    if [[ "${TORCH_INSTALLED}" == "true" ]]; then
        grep -v '^torch' "${REQUIREMENTS_FILE}" | grep -v '^#' | grep -v '^$' | xargs -r pip install
    else
        pip install -r "${REQUIREMENTS_FILE}"
    fi
elif [[ "${REQUIREMENTS_FILE}" == *"pyproject.toml" ]]; then
    pip install -e "${PROJECT_ROOT}"
fi

# Install additional ML/science packages
log_info "Installing additional scientific packages..."
pip install scikit-learn scipy joblib rich loguru

# Install deep learning optional dependencies
log_info "Installing deep learning packages..."
pip install open_clip_torch pillow requests diffusers transformers accelerate huggingface_hub

log_success "All dependencies installed"

# Verify installation
log_info "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import yaml; print('PyYAML: OK')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# Check CUDA
log_info "Checking CUDA availability..."
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_VERSION_TORCH=$(python -c "import torch; print(torch.version.cuda)")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    log_success "CUDA available: ${CUDA_VERSION_TORCH}"
    log_success "GPU detected: ${GPU_NAME}"
else
    log_warn "CUDA not available in PyTorch"
    log_warn "GPU acceleration will not be available"
fi

# Create necessary directories
log_info "Creating necessary directories..."
mkdir -p "${PROJECT_ROOT}/runs"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/cache"
mkdir -p "${PROJECT_ROOT}/checkpoints"
mkdir -p "${PROJECT_ROOT}/outputs"

log_success "Directory structure created"

# Create activation helper script
ACTIVATE_SCRIPT="${PROJECT_ROOT}/activate_env.sh"
cat > "${ACTIVATE_SCRIPT}" << 'EOF'
#!/usr/bin/env bash
# Quick activation helper
# Source this file to activate the environment:
#   source activate_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if exists
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# Activate venv
VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
    echo "Environment activated: ${VIRTUAL_ENV}"
    echo "Python: $(which python)"
    echo "To deactivate, run: deactivate"
else
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "Run: bash scripts/setup_env.sh"
fi
EOF
chmod +x "${ACTIVATE_SCRIPT}"
log_success "Created activation helper: ${ACTIVATE_SCRIPT}"

# Summary
echo ""
log_success "=========================================="
log_success "Environment setup complete!"
log_success "=========================================="
echo ""
log_info "To activate the environment in a new shell:"
echo "  source ${ACTIVATE_SCRIPT}"
echo "  # or"
echo "  source ${VENV_PATH}/bin/activate"
echo ""
log_info "Next steps:"
echo "  1. Run preflight checks: bash scripts/preflight.sh"
echo "  2. Run an experiment: bash scripts/run_experiment.sh configs/experiments/example.yaml"
echo ""
log_info "For JupyterHub, you may need to source initialSetup.sh:"
echo "  run=true source initialSetup.sh"
echo ""
