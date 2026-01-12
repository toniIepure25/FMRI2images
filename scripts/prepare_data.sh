#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh - Download/verify datasets and models
# =============================================================================
# Prepares all data needed for running experiments:
# - Verifies NSD dataset availability
# - Downloads pre-trained models (Stable Diffusion, CLIP)
# - Downloads or generates necessary cache files
#
# Usage:
#   bash scripts/prepare_data.sh
#   bash scripts/prepare_data.sh --skip-models
#   bash scripts/prepare_data.sh --download-nsd
#
# Options:
#   --skip-models      Skip model downloads (use if offline/cached)
#   --download-nsd     Download NSD dataset (WARNING: ~300GB)
#   --verify-only      Only verify, don't download anything
#   --help             Show this help message
#
# Environment variables (from .env):
#   NSD_DATA_ROOT      Where NSD dataset lives
#   CACHE_ROOT         Where to cache models/embeddings
#   HF_TOKEN           HuggingFace token (for gated models)
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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
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
SKIP_MODELS=false
DOWNLOAD_NSD=false
VERIFY_ONLY=false

show_help() {
    grep '^#' "$0" | grep -v '#!/usr/bin/env' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --download-nsd)
            DOWNLOAD_NSD=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
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

log_header "📦 DATA PREPARATION"

# Load environment
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    log_info "Loading environment from .env"
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
else
    log_warn ".env file not found"
    log_info "Using default paths"
fi

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_warn "Virtual environment not active"
    log_info "Activating environment..."
    
    VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/venv}"
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${VENV_PATH}/bin/activate"
        log_success "Environment activated"
    else
        log_error "Virtual environment not found"
        log_error "Run: bash scripts/setup_env.sh"
        exit 1
    fi
fi

cd "${PROJECT_ROOT}"

# =============================================================================
# 1. NSD Dataset Verification
# =============================================================================
log_header "1. NSD Dataset Verification"

NSD_DATA_ROOT="${NSD_DATA_ROOT:-./data/nsd}"
log_info "NSD_DATA_ROOT: ${NSD_DATA_ROOT}"

if [[ "${DOWNLOAD_NSD}" == "true" ]]; then
    log_warn "NSD dataset download requested"
    log_warn "The NSD dataset is ~300GB and requires registration"
    log_warn "Please download manually from: https://naturalscenesdataset.org/"
    echo ""
    log_info "After downloading, extract to: ${NSD_DATA_ROOT}"
    log_info "Expected structure:"
    echo "  ${NSD_DATA_ROOT}/"
    echo "    ├── nsddata/"
    echo "    ├── nsddata_betas/"
    echo "    └── nsddata_stimuli/"
    echo ""
    
    if [[ "${VERIFY_ONLY}" == "false" ]]; then
        log_error "Automatic NSD download not implemented"
        log_error "Please download manually and retry"
        exit 1
    fi
fi

# Run dataset verification
log_info "Verifying NSD dataset..."
if python scripts/verify_dataset.py; then
    log_success "NSD dataset verified!"
else
    log_error "NSD dataset verification failed"
    echo ""
    log_info "Options:"
    echo "  1. Download NSD from: https://naturalscenesdataset.org/"
    echo "  2. Extract to: ${NSD_DATA_ROOT}"
    echo "  3. Or update NSD_DATA_ROOT in .env to point to existing data"
    echo "  4. Or use S3 backend (set ALLOW_S3_ONLY=1)"
    echo ""
    
    if [[ "${VERIFY_ONLY}" == "false" ]]; then
        exit 1
    fi
fi

# =============================================================================
# 2. Pre-trained Models
# =============================================================================
if [[ "${SKIP_MODELS}" == "false" ]]; then
    log_header "2. Pre-trained Models"
    
    CACHE_ROOT="${CACHE_ROOT:-./cache}"
    log_info "CACHE_ROOT: ${CACHE_ROOT}"
    
    # Create cache directories
    mkdir -p "${CACHE_ROOT}/hf/hub"
    mkdir -p "${CACHE_ROOT}/hf/transformers"
    mkdir -p "${CACHE_ROOT}/hf/diffusers"
    
    if [[ "${VERIFY_ONLY}" == "false" ]]; then
        log_info "Downloading/verifying models..."
        
        # Set HuggingFace cache locations
        export HF_HOME="${CACHE_ROOT}/hf"
        export HF_HUB_CACHE="${CACHE_ROOT}/hf/hub"
        export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
        export DIFFUSERS_CACHE="${CACHE_ROOT}/hf/diffusers"
        
        # Run model fetcher
        if python scripts/fetch_models.py; then
            log_success "Models downloaded/verified!"
        else
            log_error "Model download failed"
            echo ""
            log_info "Common issues:"
            echo "  1. Network connection problems"
            echo "  2. Missing HF_TOKEN for gated models"
            echo "  3. Insufficient disk space"
            echo ""
            log_info "To skip model downloads: --skip-models"
            exit 1
        fi
    else
        log_info "Verify-only mode: skipping model downloads"
    fi
else
    log_info "Skipping model downloads (--skip-models)"
fi

# =============================================================================
# 3. CLIP Embeddings Cache
# =============================================================================
log_header "3. CLIP Embeddings Cache"

CLIP_CACHE="${CACHE_ROOT}/clip_embeddings"
log_info "CLIP cache: ${CLIP_CACHE}"

if [[ -d "${CLIP_CACHE}" ]] && [[ -n "$(ls -A ${CLIP_CACHE} 2>/dev/null)" ]]; then
    log_success "CLIP cache exists"
    log_info "Files: $(ls ${CLIP_CACHE} | wc -l)"
else
    log_warn "CLIP cache empty or missing"
    log_info "CLIP embeddings will be generated on first run"
    mkdir -p "${CLIP_CACHE}"
fi

# =============================================================================
# 4. Summary
# =============================================================================
log_header "📋 SUMMARY"

echo ""
log_info "Data preparation status:"
echo ""

# Check NSD
if [[ -d "${NSD_DATA_ROOT}" ]]; then
    echo "  ✓ NSD dataset: ${NSD_DATA_ROOT}"
else
    echo "  ✗ NSD dataset: NOT FOUND"
fi

# Check models
if [[ -d "${CACHE_ROOT}/hf/hub" ]] && [[ -n "$(ls -A ${CACHE_ROOT}/hf/hub 2>/dev/null)" ]]; then
    echo "  ✓ Models: ${CACHE_ROOT}/hf/"
else
    echo "  ⚠ Models: EMPTY (will download on first use)"
fi

# Check CLIP cache
if [[ -d "${CLIP_CACHE}" ]]; then
    echo "  ✓ CLIP cache: ${CLIP_CACHE}"
else
    echo "  ⚠ CLIP cache: EMPTY (will generate on first use)"
fi

echo ""

# Disk space check
log_info "Disk space:"
df -h "${PROJECT_ROOT}" | tail -1 | awk '{print "  Available: " $4 " / " $2}'

echo ""

if [[ "${VERIFY_ONLY}" == "true" ]]; then
    log_info "Verification complete (--verify-only)"
    echo ""
    log_info "To download models: bash scripts/prepare_data.sh"
    exit 0
fi

log_success "Data preparation complete!"
echo ""
log_info "Next steps:"
echo "  1. Review paths above"
echo "  2. Run experiments: bash scripts/run_experiment_simple.sh <config>"
echo ""
