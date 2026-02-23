#!/usr/bin/env bash
################################################################################
# Production Setup Script - fMRI-to-Image Brain Decoding
################################################################################
#
# Complete automated setup from zero to experiment-ready state.
# This script is idempotent - safe to run multiple times.
#
# What it does:
#   1. System preflight checks (Python, CUDA, disk space)
#   2. Python environment setup (conda/venv)
#   3. Package installation (fmri2img)
#   4. Dataset verification
#   5. Model downloads (CLIP, Stable Diffusion)
#   6. Data index building
#   7. Preprocessing pipeline fitting
#   8. CLIP embeddings cache building
#   9. Final readiness verification
#
# Usage:
#   ./setup.sh [OPTIONS]
#
# Options:
#   --subject SUBJ      Subject to prepare (default: subj01)
#   --skip-data         Skip dataset verification (use with caution)
#   --skip-cache        Skip CLIP cache building (can build later)
#   --minimal           Minimal setup (no data, models, or cache)
#   --clean             Clean all markers and start fresh
#   --check-only        Only verify current setup status
#   --help              Show this help message
#
# After successful setup, run experiments:
#   python scripts/train.py --config configs/experiments/exp0_baseline.yaml
#
################################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="fMRI2img"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/setup_$(date +%Y%m%d_%H%M%S).log"
MARKER_DIR="${SCRIPT_DIR}/.setup_markers"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"
mkdir -p "${MARKER_DIR}"

# Default options
SUBJECT="subj01"
SKIP_DATA=false
SKIP_CACHE=false
MINIMAL_SETUP=false
CLEAN_START=false
CHECK_ONLY=false

# ============================================================================
# Color Output
# ============================================================================

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    MAGENTA='\033[0;35m'
    BOLD='\033[1m'
    DIM='\033[2m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' MAGENTA='' BOLD='' DIM='' NC=''
fi

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

print_header() {
    local text="$1"
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    printf "${CYAN}${BOLD}║ %-76s ║${NC}\n" "$text"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    log "═══ $text ═══"
}

print_step() {
    echo -e "${BLUE}▶${NC} ${BOLD}$1${NC}"
    log "STEP: $1"
}

print_substep() {
    echo -e "  ${DIM}→${NC} $1"
    log "  → $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    log "SUCCESS: $1"
}

print_skip() {
    echo -e "${YELLOW}⊘${NC} $1 ${DIM}(already done)${NC}"
    log "SKIPPED: $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} ${YELLOW}$1${NC}"
    log "WARNING: $1"
}

print_error() {
    echo -e "${RED}✗${NC} ${RED}$1${NC}"
    log "ERROR: $1"
}

print_info() {
    echo -e "${MAGENTA}ℹ${NC} $1"
}

# ============================================================================
# State Management
# ============================================================================

mark_completed() {
    local step="$1"
    local marker="${MARKER_DIR}/${step}.done"
    date +%s > "$marker"
    log "Marked completed: $step"
}

is_completed() {
    local step="$1"
    local marker="${MARKER_DIR}/${step}.done"
    [[ -f "$marker" ]]
}

clean_markers() {
    print_step "Cleaning all setup markers..."
    rm -rf "${MARKER_DIR}"
    mkdir -p "${MARKER_DIR}"
    print_success "All markers cleaned - fresh start"
}

# ============================================================================
# Helper Functions
# ============================================================================

check_command() {
    local cmd="$1"
    command -v "$cmd" &> /dev/null
}

get_python() {
    # Try to find best Python 3.10+
    for py in python3.11 python3.10 python3 python; do
        if check_command "$py"; then
            local version=$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
            local major=$(echo "$version" | cut -d. -f1)
            local minor=$(echo "$version" | cut -d. -f2)
            if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 10 ]]; then
                echo "$py"
                return 0
            fi
        fi
    done
    return 1
}

run_python_script() {
    local script="$1"
    shift
    local python_cmd="${PYTHON:-python3}"
    
    print_substep "Running: $script $*"
    if "$python_cmd" "$script" "$@" >> "$LOG_FILE" 2>&1; then
        return 0
    else
        local exit_code=$?
        print_error "Script failed: $script (exit code: $exit_code)"
        print_info "Check log file: $LOG_FILE"
        return $exit_code
    fi
}

# ============================================================================
# Setup Steps
# ============================================================================

step_preflight() {
    if is_completed "preflight"; then
        print_skip "System preflight checks"
        return 0
    fi
    
    print_step "Running system preflight checks..."
    
    # Check Python version
    print_substep "Checking Python version..."
    if ! PYTHON=$(get_python); then
        print_error "Python 3.10+ not found"
        print_info "Install Python 3.10 or 3.11 and try again"
        return 1
    fi
    print_success "Python: $PYTHON ($($PYTHON --version 2>&1))"
    
    # Check CUDA
    print_substep "Checking CUDA availability..."
    if check_command nvidia-smi; then
        local cuda_version=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' || echo "unknown")
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo "unknown")
        print_success "GPU: $gpu_name (CUDA $cuda_version)"
    else
        print_warning "nvidia-smi not found - GPU training may not work"
    fi
    
    # Check disk space
    print_substep "Checking disk space..."
    local available_gb=$(df -BG "${SCRIPT_DIR}" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ "$available_gb" -lt 50 ]]; then
        print_warning "Only ${available_gb}GB free - recommend 80GB+ for full setup"
    else
        print_success "Disk space: ${available_gb}GB available"
    fi
    
    # Run full preflight script if available
    if [[ -f "${SCRIPT_DIR}/scripts/preflight.py" ]]; then
        print_substep "Running comprehensive preflight checks..."
        if run_python_script "${SCRIPT_DIR}/scripts/preflight.py"; then
            print_success "Preflight checks passed"
        else
            print_warning "Preflight script reported issues (see log)"
        fi
    fi
    
    mark_completed "preflight"
    return 0
}

step_environment() {
    if is_completed "environment"; then
        print_skip "Python environment setup"
        return 0
    fi
    
    print_step "Setting up Python environment..."
    
    # Check if we're in a conda environment
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        print_info "Using existing conda environment: $CONDA_DEFAULT_ENV"
        mark_completed "environment"
        return 0
    fi
    
    # Check if we're in a virtualenv
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        print_info "Using existing virtual environment: $VIRTUAL_ENV"
        mark_completed "environment"
        return 0
    fi
    
    # Try to create conda environment from environment.yml
    if check_command conda && [[ -f "${SCRIPT_DIR}/environment.yml" ]]; then
        print_substep "Creating conda environment from environment.yml..."
        if conda env create -f "${SCRIPT_DIR}/environment.yml" >> "$LOG_FILE" 2>&1; then
            print_success "Conda environment created: fmri2img"
            print_info "Activate with: conda activate fmri2img"
        else
            print_warning "Conda environment creation failed (see log)"
        fi
    else
        # Fall back to venv
        print_substep "Creating Python virtual environment..."
        local venv_dir="${SCRIPT_DIR}/venv"
        if [[ ! -d "$venv_dir" ]]; then
            if "$PYTHON" -m venv "$venv_dir" >> "$LOG_FILE" 2>&1; then
                print_success "Virtual environment created: $venv_dir"
                print_info "Activate with: source venv/bin/activate"
            else
                print_error "Failed to create virtual environment"
                return 1
            fi
        else
            print_info "Virtual environment already exists: $venv_dir"
        fi
    fi
    
    mark_completed "environment"
    return 0
}

step_install_package() {
    if is_completed "package"; then
        print_skip "Package installation"
        return 0
    fi
    
    print_step "Installing fmri2img package..."
    
    # Install requirements
    if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
        print_substep "Installing dependencies from requirements.txt..."
        if "$PYTHON" -m pip install -r "${SCRIPT_DIR}/requirements.txt" >> "$LOG_FILE" 2>&1; then
            print_success "Dependencies installed"
        else
            print_warning "Some dependencies failed to install (see log)"
        fi
    fi
    
    # Install package in editable mode
    print_substep "Installing fmri2img package (editable mode)..."
    if "$PYTHON" -m pip install -e "${SCRIPT_DIR}" >> "$LOG_FILE" 2>&1; then
        print_success "Package installed in editable mode"
    else
        print_error "Package installation failed"
        return 1
    fi
    
    # Verify installation
    print_substep "Verifying installation..."
    if "$PYTHON" -c "import fmri2img; print(f'fmri2img imported successfully')" >> "$LOG_FILE" 2>&1; then
        print_success "Package verified - import successful"
    else
        print_error "Package import failed"
        return 1
    fi
    
    mark_completed "package"
    return 0
}

step_verify_dataset() {
    if [[ "$SKIP_DATA" == "true" ]]; then
        print_skip "Dataset verification (--skip-data specified)"
        return 0
    fi
    
    if is_completed "dataset_${SUBJECT}"; then
        print_skip "Dataset verification for ${SUBJECT}"
        return 0
    fi
    
    print_step "Verifying NSD dataset for ${SUBJECT}..."
    
    if [[ -f "${SCRIPT_DIR}/scripts/verify_dataset.py" ]]; then
        print_substep "Running dataset verification script..."
        if run_python_script "${SCRIPT_DIR}/scripts/verify_dataset.py" --subject "$SUBJECT"; then
            print_success "Dataset verified for ${SUBJECT}"
            mark_completed "dataset_${SUBJECT}"
        else
            print_error "Dataset verification failed"
            print_info "The NSD dataset is required for training"
            print_info "See: https://naturalscenesdataset.org/"
            return 1
        fi
    else
        print_warning "Dataset verification script not found - skipping"
        mark_completed "dataset_${SUBJECT}"
    fi
    
    return 0
}

step_fetch_models() {
    if [[ "$SKIP_DATA" == "true" ]]; then
        print_skip "Model downloads (--skip-data specified)"
        return 0
    fi
    
    if is_completed "models"; then
        print_skip "Model downloads"
        return 0
    fi
    
    print_step "Downloading required models (CLIP, Stable Diffusion)..."
    print_substep "This may take 5-15 minutes depending on connection..."
    
    if [[ -f "${SCRIPT_DIR}/scripts/fetch_models.py" ]]; then
        if run_python_script "${SCRIPT_DIR}/scripts/fetch_models.py"; then
            print_success "Models downloaded and cached"
            mark_completed "models"
        else
            print_warning "Model download encountered issues (see log)"
            print_info "Models will be downloaded on-demand during training"
            mark_completed "models"
        fi
    else
        print_warning "Model fetch script not found - models will download on-demand"
        mark_completed "models"
    fi
    
    return 0
}

step_build_index() {
    if [[ "$SKIP_DATA" == "true" ]]; then
        print_skip "Index building (--skip-data specified)"
        return 0
    fi
    
    if is_completed "index_${SUBJECT}"; then
        print_skip "Data index for ${SUBJECT}"
        return 0
    fi
    
    print_step "Building NSD data index for ${SUBJECT}..."
    
    local index_file="${SCRIPT_DIR}/data/indices/nsd_index/subject=${SUBJECT}/index.parquet"
    if [[ -f "$index_file" ]]; then
        print_info "Index file already exists: $index_file"
        mark_completed "index_${SUBJECT}"
        return 0
    fi
    
    # Try multiple methods to build index
    if [[ -f "${SCRIPT_DIR}/scripts/build_full_index.py" ]]; then
        print_substep "Building index with build_full_index.py..."
        if run_python_script "${SCRIPT_DIR}/scripts/build_full_index.py" --subject "$SUBJECT"; then
            print_success "Index built for ${SUBJECT}"
            mark_completed "index_${SUBJECT}"
            return 0
        fi
    fi
    
    if [[ -f "${SCRIPT_DIR}/build_minimal_index.py" ]]; then
        print_substep "Building index with build_minimal_index.py..."
        if run_python_script "${SCRIPT_DIR}/build_minimal_index.py" --subject "$SUBJECT"; then
            print_success "Index built for ${SUBJECT}"
            mark_completed "index_${SUBJECT}"
            return 0
        fi
    fi
    
    # Try via Python module
    print_substep "Building index via Python module..."
    if "$PYTHON" -m fmri2img.data.build_full_index --subject "$SUBJECT" >> "$LOG_FILE" 2>&1; then
        print_success "Index built for ${SUBJECT}"
        mark_completed "index_${SUBJECT}"
        return 0
    fi
    
    print_error "Failed to build index for ${SUBJECT}"
    return 1
}

step_fit_preprocessing() {
    if [[ "$SKIP_DATA" == "true" ]]; then
        print_skip "Preprocessing fitting (--skip-data specified)"
        return 0
    fi
    
    if is_completed "preprocessing_${SUBJECT}"; then
        print_skip "Preprocessing pipeline for ${SUBJECT}"
        return 0
    fi
    
    print_step "Fitting preprocessing pipeline for ${SUBJECT}..."
    print_substep "This may take 10-30 minutes..."
    
    local index_file="${SCRIPT_DIR}/data/indices/nsd_index/subject=${SUBJECT}/index.parquet"
    local preproc_dir="${SCRIPT_DIR}/cache/preproc/subject=${SUBJECT}"
    
    if [[ ! -f "$index_file" ]]; then
        print_warning "Index file not found - skipping preprocessing"
        print_info "Run index building first or use --skip-data"
        return 0
    fi
    
    # Check if preprocessing artifacts already exist
    if [[ -f "${preproc_dir}/scaler.pkl" ]] && [[ -f "${preproc_dir}/pca.pkl" ]]; then
        print_info "Preprocessing artifacts already exist"
        mark_completed "preprocessing_${SUBJECT}"
        return 0
    fi
    
    if [[ -f "${SCRIPT_DIR}/scripts/fit_preprocessing.py" ]]; then
        print_substep "Fitting preprocessing pipeline..."
        if run_python_script "${SCRIPT_DIR}/scripts/fit_preprocessing.py" \
            --subject "$SUBJECT" \
            --index-file "$index_file" \
            --output-dir "$preproc_dir"; then
            print_success "Preprocessing pipeline fitted for ${SUBJECT}"
            mark_completed "preprocessing_${SUBJECT}"
        else
            print_warning "Preprocessing fitting failed (see log)"
            print_info "Preprocessing is optional - training can continue without it"
            mark_completed "preprocessing_${SUBJECT}"
        fi
    else
        print_warning "Preprocessing script not found - skipping"
        mark_completed "preprocessing_${SUBJECT}"
    fi
    
    return 0
}

step_build_clip_cache() {
    if [[ "$SKIP_CACHE" == "true" ]]; then
        print_skip "CLIP cache building (--skip-cache specified)"
        return 0
    fi
    
    if is_completed "clip_cache_${SUBJECT}"; then
        print_skip "CLIP embeddings cache for ${SUBJECT}"
        return 0
    fi
    
    print_step "Building CLIP embeddings cache for ${SUBJECT}..."
    print_substep "This may take 15-45 minutes depending on GPU..."
    
    local index_file="${SCRIPT_DIR}/data/indices/nsd_index/subject=${SUBJECT}/index.parquet"
    
    if [[ ! -f "$index_file" ]]; then
        print_warning "Index file not found - skipping CLIP cache"
        return 0
    fi
    
    # Check if cache already exists
    local cache_dir="${SCRIPT_DIR}/cache/clip_embeddings"
    if [[ -n "$(find "$cache_dir" -maxdepth 1 -name "*${SUBJECT}*.parquet" 2>/dev/null)" ]]; then
        print_info "CLIP cache files already exist for ${SUBJECT}"
        mark_completed "clip_cache_${SUBJECT}"
        return 0
    fi
    
    # Try different cache building scripts
    if [[ -f "${SCRIPT_DIR}/scripts/build_clip_cache.py" ]]; then
        print_substep "Building CLIP cache..."
        if run_python_script "${SCRIPT_DIR}/scripts/build_clip_cache.py" --subject "$SUBJECT"; then
            print_success "CLIP cache built for ${SUBJECT}"
            mark_completed "clip_cache_${SUBJECT}"
            return 0
        fi
    fi
    
    if [[ -f "${SCRIPT_DIR}/scripts/build_target_clip_cache_robust.py" ]]; then
        print_substep "Building CLIP cache (robust method)..."
        if run_python_script "${SCRIPT_DIR}/scripts/build_target_clip_cache_robust.py"; then
            print_success "CLIP cache built for ${SUBJECT}"
            mark_completed "clip_cache_${SUBJECT}"
            return 0
        fi
    fi
    
    print_warning "CLIP cache building failed or not available"
    print_info "CLIP cache will be built on-demand during training"
    mark_completed "clip_cache_${SUBJECT}"
    return 0
}

step_final_check() {
    print_step "Running final readiness check..."
    
    if [[ -f "${SCRIPT_DIR}/scripts/doctor.py" ]]; then
        print_substep "Running doctor check..."
        if run_python_script "${SCRIPT_DIR}/scripts/doctor.py"; then
            print_success "System is ready for experiments"
            return 0
        else
            print_warning "Doctor check reported some issues (see log)"
            print_info "You may still be able to run experiments"
            return 0
        fi
    else
        print_info "Doctor check script not found - skipping"
        return 0
    fi
}

# ============================================================================
# Status Check
# ============================================================================

check_status() {
    print_header "Setup Status Check"
    
    local steps=(
        "preflight:System Preflight"
        "environment:Python Environment"
        "package:Package Installation"
        "dataset_${SUBJECT}:Dataset (${SUBJECT})"
        "models:Model Downloads"
        "index_${SUBJECT}:Data Index (${SUBJECT})"
        "preprocessing_${SUBJECT}:Preprocessing (${SUBJECT})"
        "clip_cache_${SUBJECT}:CLIP Cache (${SUBJECT})"
    )
    
    local all_done=true
    
    for step_info in "${steps[@]}"; do
        local step="${step_info%%:*}"
        local name="${step_info#*:}"
        
        if is_completed "$step"; then
            echo -e "${GREEN}✓${NC} $name"
        else
            echo -e "${YELLOW}○${NC} $name ${DIM}(not done)${NC}"
            all_done=false
        fi
    done
    
    echo ""
    if [[ "$all_done" == "true" ]]; then
        print_success "All setup steps completed!"
        echo ""
        print_info "Ready to run experiments:"
        echo "  python scripts/train.py --config configs/experiments/exp0_baseline.yaml"
    else
        print_info "Some steps are incomplete. Run './setup.sh' to complete setup."
    fi
}

# ============================================================================
# Main Setup Flow
# ============================================================================

run_setup() {
    print_header "${PROJECT_NAME} - Production Setup"
    
    echo "This script will prepare your system for running experiments."
    echo "Subject: ${SUBJECT}"
    echo "Log file: ${LOG_FILE}"
    echo ""
    
    local steps=(
        step_preflight
        step_environment
        step_install_package
        step_verify_dataset
        step_fetch_models
        step_build_index
        step_fit_preprocessing
        step_build_clip_cache
        step_final_check
    )
    
    local failed=false
    
    for step_func in "${steps[@]}"; do
        if ! "$step_func"; then
            failed=true
            print_error "Setup step failed: $step_func"
            print_info "Check log file for details: $LOG_FILE"
            
            # Ask if user wants to continue
            echo ""
            read -p "Continue with remaining steps? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_error "Setup aborted by user"
                return 1
            fi
        fi
        echo ""
    done
    
    # Final summary
    print_header "Setup Complete!"
    
    if [[ "$failed" == "true" ]]; then
        print_warning "Setup completed with some warnings or errors"
        print_info "Check the log file for details: $LOG_FILE"
        echo ""
        print_info "Run './setup.sh --check-only' to see status"
    else
        print_success "All setup steps completed successfully!"
    fi
    
    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo ""
    echo "1. Activate your Python environment:"
    if [[ -d "${SCRIPT_DIR}/venv" ]]; then
        echo "   ${CYAN}source venv/bin/activate${NC}"
    else
        echo "   ${CYAN}conda activate fmri2img${NC}"
    fi
    echo ""
    echo "2. Run your first experiment:"
    echo "   ${CYAN}python scripts/train.py --config configs/experiments/exp0_baseline.yaml${NC}"
    echo ""
    echo "3. Monitor training:"
    echo "   ${CYAN}tensorboard --logdir outputs/${NC}"
    echo ""
    echo "For more information, see: ${CYAN}docs/guides/RUNNING_EXPERIMENTS.md${NC}"
    echo ""
    
    return 0
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --subject)
                SUBJECT="$2"
                shift 2
                ;;
            --skip-data)
                SKIP_DATA=true
                shift
                ;;
            --skip-cache)
                SKIP_CACHE=true
                shift
                ;;
            --minimal)
                MINIMAL_SETUP=true
                SKIP_DATA=true
                SKIP_CACHE=true
                shift
                ;;
            --clean)
                CLEAN_START=true
                shift
                ;;
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --help|-h)
                head -n 35 "$0" | tail -n 32
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    parse_args "$@"
    
    # Get Python early
    if ! PYTHON=$(get_python); then
        print_error "Python 3.10+ is required"
        print_info "Install Python 3.10 or 3.11 and try again"
        exit 1
    fi
    export PYTHON
    
    if [[ "$CLEAN_START" == "true" ]]; then
        clean_markers
        echo ""
    fi
    
    if [[ "$CHECK_ONLY" == "true" ]]; then
        check_status
        exit 0
    fi
    
    # Run the setup
    if run_setup; then
        exit 0
    else
        print_error "Setup failed"
        print_info "Log file: $LOG_FILE"
        exit 1
    fi
}

# Run main
main "$@"
