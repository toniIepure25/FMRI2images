#!/bin/bash
################################################################################
# ENHANCED Automated Setup Script for fMRI-to-Image Brain Decoding Project
# Bachelor Thesis Production Version
################################################################################
#
# This script provides a robust, production-ready setup with:
# - Automatic error recovery and resume capability
# - Smart GPU memory detection and batch size recommendation
# - Comprehensive health checks and diagnostics
# - Detailed logging for troubleshooting
#
# Usage:
#   ./setup_enhanced.sh [OPTIONS]
#
# Options:
#   --skip-data         Skip NSD data download (if already present)
#   --skip-models       Skip model checkpoint download
#   --skip-clip-cache   Skip CLIP cache building
#   --minimal           Minimal setup (environment + dependencies only)
#   --auto-yes          Skip all confirmations (non-interactive)
#   --check-only        Only run health checks, don't install
#   --help              Show this help message
#
# Requirements:
#   - Python 3.11+
#   - CUDA 11.8+ (for GPU support)
#   - 60GB+ free disk space
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Catch errors in pipes

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="fMRI2img Bachelor Thesis"
PYTHON_MIN_VERSION="3.11"
REQUIRED_DISK_GB=60
LOG_FILE="$SCRIPT_DIR/setup_$(date +%Y%m%d_%H%M%S).log"
STATE_FILE="$SCRIPT_DIR/.setup_state"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Parse arguments
SKIP_DATA=false
SKIP_MODELS=false
SKIP_CLIP_CACHE=false
MINIMAL_SETUP=false
AUTO_YES=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data) SKIP_DATA=true; shift ;;
        --skip-models) SKIP_MODELS=true; shift ;;
        --skip-clip-cache) SKIP_CLIP_CACHE=true; shift ;;
        --minimal)
            MINIMAL_SETUP=true
            SKIP_DATA=true
            SKIP_MODELS=true
            SKIP_CLIP_CACHE=true
            shift
            ;;
        --auto-yes) AUTO_YES=true; shift ;;
        --check-only) CHECK_ONLY=true; shift ;;
        --help)
            head -n 33 "$0" | tail -n 30
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

################################################################################
# Logging and Output Functions
################################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

print_header() {
    local text="$1"
    local width=80
    echo "" | tee -a "$LOG_FILE"
    echo -e "${CYAN}$(printf '═%.0s' $(seq 1 $width))${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}${BOLD}  $text${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}$(printf '═%.0s' $(seq 1 $width))${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    log "SECTION: $text"
}

print_step() {
    echo -e "${BLUE}▶ $1${NC}" | tee -a "$LOG_FILE"
    log "STEP: $1"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$LOG_FILE"
    log "SUCCESS: $1"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$LOG_FILE"
    log "WARNING: $1"
}

print_error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$LOG_FILE"
    log "ERROR: $1"
}

print_info() {
    echo -e "${MAGENTA}ℹ $1${NC}" | tee -a "$LOG_FILE"
}

################################################################################
# State Management (for resume capability)
################################################################################

mark_completed() {
    local step="$1"
    echo "$step=$(date +%s)" >> "$STATE_FILE"
    log "Marked step completed: $step"
}

is_completed() {
    local step="$1"
    [[ -f "$STATE_FILE" ]] && grep -q "^${step}=" "$STATE_FILE"
}

reset_state() {
    rm -f "$STATE_FILE"
    log "Reset setup state"
}

################################################################################
# System Checks with Diagnostics
################################################################################

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

check_python_version() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        print_info "Install Python 3.11+: sudo apt-get install python3.11"
        return 1
    fi
    
    local py_version=$(python3 --version | awk '{print $2}')
    local py_major=$(echo "$py_version" | cut -d. -f1)
    local py_minor=$(echo "$py_version" | cut -d. -f2)
    
    if [[ "$py_major" -ge 3 ]] && [[ "$py_minor" -ge 11 ]]; then
        print_success "Python $py_version detected (✓ >= 3.11)"
        return 0
    else
        print_error "Python $py_version detected (✗ requires >= 3.11)"
        print_info "Current: $py_version | Required: >= 3.11"
        return 1
    fi
}

check_gpu_detailed() {
    print_step "Checking GPU configuration..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found - No GPU support"
        return 1
    fi
    
    # Get GPU info
    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    local gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    local gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1)
    
    print_success "GPU: $gpu_name"
    print_info "    Total Memory: ${gpu_memory} MB"
    print_info "    Used Memory:  ${gpu_used} MB (${gpu_util}% utilization)"
    print_info "    Free Memory:  ${gpu_free} MB"
    
    # Check if GPU is heavily used
    if [[ "$gpu_used" -gt 15000 ]]; then
        print_warning "GPU is heavily used (${gpu_used}MB / ${gpu_memory}MB)"
        print_info "    Consider waiting for GPU to free up or using --skip-training"
    elif [[ "$gpu_free" -lt 3000 ]]; then
        print_warning "Low GPU memory available (${gpu_free}MB free)"
        print_info "    Training may require smaller batch sizes"
    else
        print_success "GPU has sufficient free memory (${gpu_free}MB available)"
    fi
    
    # Recommend batch size based on available memory
    local batch_size=32
    if [[ "$gpu_free" -lt 3000 ]]; then
        batch_size=2
        print_info "    Recommended batch_size: 2 (very constrained)"
    elif [[ "$gpu_free" -lt 5000 ]]; then
        batch_size=4
        print_info "    Recommended batch_size: 4 (constrained)"
    elif [[ "$gpu_free" -lt 8000 ]]; then
        batch_size=8
        print_info "    Recommended batch_size: 8 (moderate)"
    else
        batch_size=16
        print_info "    Recommended batch_size: 16-32 (ample memory)"
    fi
    
    # Export for later use
    export RECOMMENDED_BATCH_SIZE="$batch_size"
    
    return 0
}

check_disk_space_detailed() {
    print_step "Checking disk space..."
    
    local path="$SCRIPT_DIR"
    local available_gb=$(df -BG "$path" | awk 'NR==2 {print $4}' | sed 's/G//')
    local total_gb=$(df -BG "$path" | awk 'NR==2 {print $2}' | sed 's/G//')
    local used_gb=$(df -BG "$path" | awk 'NR==2 {print $3}' | sed 's/G//')
    
    print_info "    Disk: ${used_gb}GB used / ${total_gb}GB total"
    print_info "    Available: ${available_gb}GB"
    
    if [[ "$available_gb" -lt "$REQUIRED_DISK_GB" ]]; then
        print_error "Insufficient disk space: ${available_gb}GB < ${REQUIRED_DISK_GB}GB required"
        print_info "    Required breakdown:"
        print_info "      - NSD data: ~17GB"
        print_info "      - Model checkpoints: ~5GB"
        print_info "      - CLIP cache: ~2GB"
        print_info "      - Dependencies: ~10GB"
        print_info "      - Workspace: ~25GB"
        return 1
    elif [[ "$available_gb" -lt 100 ]]; then
        print_warning "Limited disk space: ${available_gb}GB available"
        print_info "    This should work but may fill up during training"
    else
        print_success "Sufficient disk space: ${available_gb}GB available"
    fi
    
    return 0
}

check_internet_connection() {
    print_step "Checking internet connection..."
    
    if ping -c 1 8.8.8.8 &> /dev/null; then
        print_success "Internet connection available"
        
        # Test Hugging Face connection
        if curl -s --head https://huggingface.co | head -n 1 | grep "200" > /dev/null; then
            print_success "Hugging Face is reachable"
        else
            print_warning "Hugging Face may be unreachable"
        fi
        
        return 0
    else
        print_error "No internet connection"
        print_info "Internet is required for downloading models and data"
        return 1
    fi
}

################################################################################
# Main Setup Functions
################################################################################

setup_system_checks() {
    print_header "SYSTEM REQUIREMENTS CHECK"
    
    if is_completed "system_checks"; then
        print_warning "System checks already completed. Skipping..."
        return 0
    fi
    
    local all_ok=true
    
    # Python
    print_step "Checking Python installation..."
    check_python_version || all_ok=false
    
    # Git
    print_step "Checking Git installation..."
    check_command git || all_ok=false
    
    # GPU
    check_gpu_detailed || print_warning "GPU not available - training will be slow"
    
    # Disk space
    check_disk_space_detailed || all_ok=false
    
    # Internet
    check_internet_connection || all_ok=false
    
    if [[ "$all_ok" = false ]]; then
        print_error "System requirements not met"
        exit 1
    fi
    
    print_success "All system checks passed!"
    mark_completed "system_checks"
}

setup_environment_variables() {
    print_header "CONFIGURING ENVIRONMENT"
    
    if is_completed "environment"; then
        print_warning "Environment already configured. Skipping..."
        return 0
    fi
    
    # Detect environment type
    local cache_base="$SCRIPT_DIR"
    if [[ -d "/bigdata" ]]; then
        print_step "Detected shared server (/bigdata exists)"
        cache_base="/bigdata/userhome/$(whoami)"
        print_info "    Using /bigdata for caches to avoid filling system disk"
    else
        print_step "Using local environment"
    fi
    
    # Create .env file
    local env_file="$SCRIPT_DIR/.env"
    print_step "Creating environment file: $env_file"
    
    cat > "$env_file" << EOF
# fMRI2img Environment Configuration
# Generated: $(date)
# Machine: $(hostname)
# User: $(whoami)

# Cache directories
export TORCH_HOME="${cache_base}/cache/torch"
export HF_HOME="${cache_base}/cache/huggingface"
export TRANSFORMERS_CACHE="${cache_base}/cache/transformers"
export TMPDIR="${cache_base}/tmp"

# Python
export PYTHONPATH="${SCRIPT_DIR}/src:\${PYTHONPATH:-}"

# CUDA
export CUDA_VISIBLE_DEVICES=0

# Warnings
export PYTHONWARNINGS="ignore"

# Memory optimization (for CUDA OOM issues)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
EOF
    
    print_success ".env file created"
    
    # Update shell RC
    local shell_rc="$HOME/.bashrc"
    [[ -f "$HOME/.zshrc" ]] && shell_rc="$HOME/.zshrc"
    
    if ! grep -q "source $env_file" "$shell_rc" 2>/dev/null; then
        print_step "Adding environment to $shell_rc"
        {
            echo ""
            echo "# fMRI2img environment (added $(date))"
            echo "[ -f \"$env_file\" ] && source \"$env_file\""
        } >> "$shell_rc"
        print_success "Shell configuration updated"
    else
        print_info "Environment already in $shell_rc"
    fi
    
    # Source for current session
    source "$env_file"
    
    # Create directories
    print_step "Creating directory structure..."
    local dirs=(
        "$TORCH_HOME"
        "$HF_HOME"
        "$TRANSFORMERS_CACHE"
        "$TMPDIR"
        "$SCRIPT_DIR/cache/clip_embeddings"
        "$SCRIPT_DIR/cache/stimuli"
        "$SCRIPT_DIR/data/indices"
        "$SCRIPT_DIR/checkpoints"
        "$SCRIPT_DIR/outputs"
        "$SCRIPT_DIR/logs"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "Directory structure created"
    print_success "Environment configured successfully!"
    mark_completed "environment"
}

setup_python_environment() {
    print_header "PYTHON ENVIRONMENT SETUP"
    
    if is_completed "python_env"; then
        print_warning "Python environment already set up. Skipping..."
        return 0
    fi
    
    cd "$SCRIPT_DIR"
    
    # Create venv
    if [[ ! -d "venv" ]]; then
        print_step "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate
    print_step "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_step "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel -q
    
    # Install PyTorch
    print_step "Installing PyTorch..."
    if command -v nvidia-smi &> /dev/null; then
        print_info "    Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        print_info "    Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio
    fi
    
    # Install dependencies
    if [[ -f "requirements.txt" ]]; then
        print_step "Installing dependencies from requirements.txt..."
        print_warning "This may take 5-10 minutes..."
        pip install -r requirements.txt
    fi
    
    # Install package
    print_step "Installing fmri2img package in development mode..."
    pip install -e .
    
    # Verify installation
    print_step "Verifying Python packages..."
    python3 -c "
import sys
import torch
print(f'✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
" | tee -a "$LOG_FILE"
    
    print_success "Python environment configured successfully!"
    mark_completed "python_env"
}

download_nsd_data() {
    print_header "NSD DATA DOWNLOAD"
    
    if [[ "$SKIP_DATA" = true ]]; then
        print_warning "Skipping NSD data download (--skip-data)"
        return 0
    fi
    
    if is_completed "nsd_data"; then
        print_warning "NSD data already downloaded. Skipping..."
        return 0
    fi
    
    print_step "Downloading NSD stimulus info CSV..."
    python scripts/fetch_models.py --component nsd-stim-info
    
    print_step "Downloading beta files (subject 01, session 1)..."
    print_warning "Size: ~17GB | Time: 15-30 minutes"
    python scripts/fetch_models.py --component nsd-betas --subject subj01 --session 1
    
    print_step "Building data index..."
    python scripts/build_full_index.py --subject subj01 --session 1
    
    print_success "NSD data downloaded and indexed!"
    mark_completed "nsd_data"
}

download_models() {
    print_header "MODEL CHECKPOINTS"
    
    if [[ "$SKIP_MODELS" = true ]]; then
        print_warning "Skipping model download (--skip-models)"
        return 0
    fi
    
    if is_completed "models"; then
        print_warning "Models already downloaded. Skipping..."
        return 0
    fi
    
    print_step "Downloading Stable Diffusion v1.5..."
    python scripts/download_sd_model.py --model-id runwayml/stable-diffusion-v1-5
    
    print_success "Model checkpoints downloaded!"
    mark_completed "models"
}

build_clip_cache() {
    print_header "CLIP EMBEDDINGS CACHE"
    
    if [[ "$SKIP_CLIP_CACHE" = true ]]; then
        print_warning "Skipping CLIP cache build (--skip-clip-cache)"
        return 0
    fi
    
    if is_completed "clip_cache"; then
        print_warning "CLIP cache already built. Skipping..."
        return 0
    fi
    
    print_step "Building CLIP embeddings cache..."
    print_warning "Time: 5-10 minutes on GPU, longer on CPU"
    
    python scripts/build_target_clip_cache_robust.py \
        --subject subj01 \
        --index-root data/indices/nsd_index \
        --model-id runwayml/stable-diffusion-v1-5 \
        --batch-size 100 \
        --inference-batch-size 64 \
        --output cache/clip_embeddings/nsd_clipvitl14.parquet
    
    print_success "CLIP cache built successfully!"
    mark_completed "clip_cache"
}

run_health_check() {
    print_header "POST-INSTALLATION HEALTH CHECK"
    
    local all_ok=true
    
    # Python imports
    print_step "Testing Python imports..."
    if python3 -c "
import sys
import torch
import transformers
import diffusers
import nibabel
import pandas
import numpy
print('✓ All core packages importable')
" 2>&1 | tee -a "$LOG_FILE"; then
        print_success "Python imports OK"
    else
        print_error "Python import test failed"
        all_ok=false
    fi
    
    # Data files
    print_step "Checking data files..."
    local missing_files=()
    
    [[ -f "cache/nsd_stim_info_merged.csv" ]] || missing_files+=("nsd_stim_info_merged.csv")
    [[ -d "data/indices/nsd_index/subject=subj01" ]] || missing_files+=("data index")
    [[ -f "cache/clip_embeddings/nsd_clipvitl14.parquet" ]] || missing_files+=("CLIP cache")
    
    if [[ ${#missing_files[@]} -eq 0 ]]; then
        print_success "All data files present"
    else
        print_warning "Missing files: ${missing_files[*]}"
        all_ok=false
    fi
    
    # Disk usage
    print_step "Disk usage summary:"
    du -sh "$SCRIPT_DIR" 2>/dev/null | tee -a "$LOG_FILE"
    
    if [[ "$all_ok" = true ]]; then
        print_success "Health check passed!"
    else
        print_warning "Health check completed with warnings"
    fi
}

print_next_steps() {
    print_header "SETUP COMPLETE! 🎉"
    
    cat << EOF | tee -a "$LOG_FILE"

${GREEN}${BOLD}✓ Installation successful!${NC}

${CYAN}${BOLD}Quick Start:${NC}

${YELLOW}1. Activate environment:${NC}
   ${BLUE}source venv/bin/activate${NC}
   ${BLUE}source .env${NC}

${YELLOW}2. Run smoke test (2-3 minutes):${NC}
   ${BLUE}python scripts/smoke.py${NC}

${YELLOW}3. Start training:${NC}
EOF

    # Provide GPU-specific training command
    if [[ -n "${RECOMMENDED_BATCH_SIZE:-}" ]]; then
        cat << EOF | tee -a "$LOG_FILE"
   ${MAGENTA}# Recommended batch size for your GPU: ${RECOMMENDED_BATCH_SIZE}${NC}
   ${MAGENTA}# Edit experiments/ultimate_novel_subj01.yaml and set:${NC}
   ${MAGENTA}#   training.batch_size: ${RECOMMENDED_BATCH_SIZE}${NC}
   ${MAGENTA}#   advanced.gradient_accumulation_steps: $((32 / RECOMMENDED_BATCH_SIZE))${NC}
   
EOF
    fi
    
    cat << EOF | tee -a "$LOG_FILE"
   ${BLUE}python scripts/train_ultimate_novel.py --config experiments/ultimate_novel_subj01.yaml${NC}

${YELLOW}4. Monitor training:${NC}
   ${BLUE}tensorboard --logdir outputs/runs${NC}

${CYAN}${BOLD}Documentation:${NC}
  📖 Getting Started: ${BLUE}START_HERE.md${NC}
  📖 Setup Guide: ${BLUE}QUICKSTART.md${NC}
  📖 Troubleshooting: ${BLUE}SETUP_TROUBLESHOOTING.md${NC}

${CYAN}${BOLD}Diagnostics:${NC}
  🔧 Check setup: ${BLUE}bash scripts/check_setup.sh${NC}
  🔧 Full diagnostic: ${BLUE}python scripts/doctor.py${NC}

${CYAN}${BOLD}Logs:${NC}
  Setup log saved to: ${BLUE}$LOG_FILE${NC}

${GREEN}${BOLD}Happy brain decoding! 🧠→🖼️${NC}
EOF
}

################################################################################
# Main Execution
################################################################################

main() {
    # Start logging
    log "Setup script started"
    log "Arguments: $*"
    
    clear
    
    print_header "$PROJECT_NAME - AUTOMATED SETUP"
    
    cat << EOF
${CYAN}Project:${NC}   fMRI-to-Image Brain Decoding
${CYAN}Location:${NC}  $SCRIPT_DIR
${CYAN}User:${NC}      $(whoami)
${CYAN}Machine:${NC}   $(hostname)
${CYAN}Date:${NC}      $(date)
${CYAN}Log file:${NC}  $LOG_FILE
EOF
    
    echo ""
    
    if [[ "$MINIMAL_SETUP" = true ]]; then
        print_warning "MINIMAL SETUP MODE (environment + dependencies only)"
    fi
    
    if [[ "$CHECK_ONLY" = true ]]; then
        print_warning "CHECK-ONLY MODE (no installation)"
        setup_system_checks
        exit 0
    fi
    
    # Confirmation
    if [[ "$AUTO_YES" = false ]]; then
        echo -e "${YELLOW}Continue with setup? (Y/n): ${NC}"
        read -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo "Setup cancelled."
            exit 0
        fi
    fi
    
    # Run setup
    setup_system_checks
    setup_environment_variables
    setup_python_environment
    
    if [[ "$MINIMAL_SETUP" = false ]]; then
        download_nsd_data
        download_models
        build_clip_cache
    fi
    
    run_health_check
    print_next_steps
    
    log "Setup completed successfully"
}

# Trap errors
trap 'print_error "Setup failed at line $LINENO. Check log: $LOG_FILE"' ERR

# Run main
main "$@"
