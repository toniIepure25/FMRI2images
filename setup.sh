#!/bin/bash
################################################################################
# Automated Setup Script for fMRI-to-Image Brain Decoding Project
################################################################################
#
# This script automates the complete setup process for the fMRI2img project,
# including environment configuration, dependency installation, data download,
# and cache building.
#
# Usage:
#   ./setup.sh [OPTIONS]
#
# Options:
#   --skip-data         Skip NSD data download (if already present)
#   --skip-models       Skip model checkpoint download
#   --skip-clip-cache   Skip CLIP cache building
#   --minimal           Minimal setup (environment + dependencies only)
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

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="fMRI2img"
PYTHON_VERSION="3.11"
VENV_NAME="venv"

# Parse command-line arguments
SKIP_DATA=false
SKIP_MODELS=false
SKIP_CLIP_CACHE=false
MINIMAL_SETUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-clip-cache)
            SKIP_CLIP_CACHE=true
            shift
            ;;
        --minimal)
            MINIMAL_SETUP=true
            SKIP_DATA=true
            SKIP_MODELS=true
            SKIP_CLIP_CACHE=true
            shift
            ;;
        --help)
            head -n 25 "$0" | tail -n 23
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

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
    if command -v python3 &> /dev/null; then
        PY_VERSION=$(python3 --version | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        
        if [[ "$PY_MAJOR" -ge 3 ]] && [[ "$PY_MINOR" -ge 11 ]]; then
            print_success "Python $PY_VERSION detected"
            return 0
        else
            print_error "Python $PY_VERSION detected (requires 3.11+)"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        print_success "GPU detected: $GPU_INFO"
        return 0
    else
        print_warning "No GPU detected (nvidia-smi not available)"
        return 1
    fi
}

get_disk_space() {
    local path="$1"
    df -BG "$path" | awk 'NR==2 {print $4}' | sed 's/G//'
}

################################################################################
# Main Setup Functions
################################################################################

setup_system_checks() {
    print_header "SYSTEM REQUIREMENTS CHECK"
    
    local all_ok=true
    
    # Check Python
    print_step "Checking Python installation..."
    if ! check_python_version; then
        all_ok=false
    fi
    
    # Check Git
    print_step "Checking Git installation..."
    if ! check_command git; then
        all_ok=false
    fi
    
    # Check GPU (optional but recommended)
    print_step "Checking GPU availability..."
    check_gpu || true
    
    # Check disk space
    print_step "Checking disk space..."
    AVAILABLE_SPACE=$(get_disk_space "$SCRIPT_DIR")
    if [[ "$AVAILABLE_SPACE" -lt 60 ]]; then
        print_warning "Only ${AVAILABLE_SPACE}GB available. Recommended: 60GB+"
    else
        print_success "${AVAILABLE_SPACE}GB available"
    fi
    
    if [[ "$all_ok" = false ]]; then
        print_error "System requirements not met. Please install missing dependencies."
        exit 1
    fi
    
    print_success "All system checks passed!"
}

setup_environment_variables() {
    print_header "CONFIGURING ENVIRONMENT VARIABLES"
    
    # Detect if we're on a shared server with /bigdata
    if [[ -d "/bigdata" ]]; then
        print_step "Detected shared server environment (/bigdata exists)"
        CACHE_BASE="/bigdata/userhome/$(whoami)"
    else
        print_step "Using local environment"
        CACHE_BASE="$SCRIPT_DIR"
    fi
    
    # Create environment configuration file
    ENV_FILE="$SCRIPT_DIR/.env"
    print_step "Creating environment configuration: $ENV_FILE"
    
    cat > "$ENV_FILE" << EOF
# fMRI2img Environment Configuration
# Generated on $(date)

# Cache directories (use /bigdata on shared servers to avoid filling system disk)
export TORCH_HOME="${CACHE_BASE}/cache/torch"
export HF_HOME="${CACHE_BASE}/cache/huggingface"
export TRANSFORMERS_CACHE="${CACHE_BASE}/cache/transformers"
export TMPDIR="${CACHE_BASE}/tmp"

# Python paths
export PYTHONPATH="${SCRIPT_DIR}/src:\${PYTHONPATH}"

# CUDA settings (if available)
export CUDA_VISIBLE_DEVICES=0

# Disable warnings
export PYTHONWARNINGS="ignore"
EOF
    
    # Create shell rc additions
    SHELL_RC="$HOME/.bashrc"
    if [[ -f "$HOME/.zshrc" ]]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    
    # Check if already sourced
    if ! grep -q "source $ENV_FILE" "$SHELL_RC" 2>/dev/null; then
        print_step "Adding environment sourcing to $SHELL_RC"
        echo "" >> "$SHELL_RC"
        echo "# fMRI2img environment (added by setup.sh)" >> "$SHELL_RC"
        echo "[ -f \"$ENV_FILE\" ] && source \"$ENV_FILE\"" >> "$SHELL_RC"
        print_success "Shell configuration updated"
    else
        print_warning "Environment already configured in $SHELL_RC"
    fi
    
    # Source for current session
    source "$ENV_FILE"
    
    # Create cache directories
    print_step "Creating cache directories..."
    mkdir -p "$TORCH_HOME"
    mkdir -p "$HF_HOME"
    mkdir -p "$TRANSFORMERS_CACHE"
    mkdir -p "$TMPDIR"
    mkdir -p "$SCRIPT_DIR/cache/clip_embeddings"
    mkdir -p "$SCRIPT_DIR/cache/stimuli"
    mkdir -p "$SCRIPT_DIR/data/indices"
    mkdir -p "$SCRIPT_DIR/checkpoints"
    mkdir -p "$SCRIPT_DIR/outputs"
    mkdir -p "$SCRIPT_DIR/logs"
    
    print_success "Environment configured successfully!"
}

setup_python_environment() {
    print_header "SETTING UP PYTHON ENVIRONMENT"
    
    cd "$SCRIPT_DIR"
    
    # Create virtual environment
    if [[ ! -d "$VENV_NAME" ]]; then
        print_step "Creating virtual environment..."
        python3 -m venv "$VENV_NAME"
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_step "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    print_step "Upgrading pip..."
    pip install --upgrade pip setuptools wheel -q
    
    # Install dependencies
    print_step "Installing Python dependencies..."
    print_warning "This may take 5-10 minutes..."
    
    # Install PyTorch first (CUDA 12.1)
    if check_gpu &> /dev/null; then
        print_step "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    else
        print_step "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio -q
    fi
    
    # Install other dependencies
    if [[ -f "requirements.txt" ]]; then
        print_step "Installing from requirements.txt..."
        pip install -r requirements.txt -q
    fi
    
    # Install package in development mode
    print_step "Installing fmri2img package..."
    pip install -e . -q
    
    print_success "Python environment configured successfully!"
    
    # Show installed packages
    echo ""
    print_step "Key packages installed:"
    pip list | grep -E "(torch|transformers|diffusers|nibabel|pandas|numpy)" || true
}

download_nsd_data() {
    print_header "DOWNLOADING NSD DATA"
    
    if [[ "$SKIP_DATA" = true ]]; then
        print_warning "Skipping NSD data download (--skip-data flag)"
        return 0
    fi
    
    # Check if data already exists
    if [[ -f "cache/nsd_stim_info_merged.csv" ]] && [[ -d "data/indices/nsd_index" ]]; then
        print_warning "NSD data appears to be already downloaded"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    print_step "Downloading NSD stimulus info CSV..."
    python scripts/fetch_models.py --component nsd-stim-info
    
    print_step "Downloading beta files for subject 01, session 1..."
    print_warning "This downloads ~17GB of fMRI data. It may take 15-30 minutes."
    python scripts/fetch_models.py --component nsd-betas --subject subj01 --session 1
    
    print_step "Building data index..."
    python scripts/build_full_index.py --subject subj01 --session 1
    
    print_success "NSD data downloaded and indexed!"
}

download_models() {
    print_header "DOWNLOADING MODEL CHECKPOINTS"
    
    if [[ "$SKIP_MODELS" = true ]]; then
        print_warning "Skipping model download (--skip-models flag)"
        return 0
    fi
    
    print_step "Downloading Stable Diffusion model..."
    python scripts/download_sd_model.py --model-id runwayml/stable-diffusion-v1-5
    
    print_success "Model checkpoints downloaded!"
}

build_clip_cache() {
    print_header "BUILDING CLIP EMBEDDINGS CACHE"
    
    if [[ "$SKIP_CLIP_CACHE" = true ]]; then
        print_warning "Skipping CLIP cache build (--skip-clip-cache flag)"
        return 0
    fi
    
    # Check if cache already exists
    if [[ -f "cache/clip_embeddings/nsd_clipvitl14.parquet" ]]; then
        print_warning "CLIP cache already exists"
        read -p "Rebuild? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    print_step "Building CLIP embeddings cache..."
    print_warning "This may take 5-10 minutes on GPU, longer on CPU"
    
    python scripts/build_target_clip_cache_robust.py \
        --subject subj01 \
        --index-root data/indices/nsd_index \
        --model-id runwayml/stable-diffusion-v1-5 \
        --batch-size 100 \
        --inference-batch-size 64 \
        --output cache/clip_embeddings/nsd_clipvitl14.parquet
    
    print_success "CLIP cache built successfully!"
}

verify_installation() {
    print_header "VERIFYING INSTALLATION"
    
    print_step "Running verification checks..."
    
    # Check Python imports
    python3 -c "
import sys
import torch
import transformers
import diffusers
import nibabel
import pandas
print(f'✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ Transformers {transformers.__version__}')
print(f'✓ Diffusers {diffusers.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
    
    # Check data files
    echo ""
    print_step "Checking data files..."
    if [[ -f "cache/nsd_stim_info_merged.csv" ]]; then
        print_success "Stimulus info CSV present"
    else
        print_warning "Stimulus info CSV missing"
    fi
    
    if [[ -d "data/indices/nsd_index/subject=subj01" ]]; then
        print_success "Data index present"
    else
        print_warning "Data index missing"
    fi
    
    if [[ -f "cache/clip_embeddings/nsd_clipvitl14.parquet" ]]; then
        print_success "CLIP cache present"
    else
        print_warning "CLIP cache missing"
    fi
    
    # Check disk usage
    echo ""
    print_step "Disk usage:"
    du -sh "$SCRIPT_DIR" 2>/dev/null || echo "Unable to calculate"
    
    print_success "Verification complete!"
}

print_next_steps() {
    print_header "SETUP COMPLETE!"
    
    cat << EOF
${GREEN}✓ Installation successful!${NC}

${CYAN}Next Steps:${NC}

1. ${YELLOW}Activate the environment:${NC}
   ${BLUE}source venv/bin/activate${NC}
   ${BLUE}source .env${NC}

2. ${YELLOW}Run a quick smoke test:${NC}
   ${BLUE}python scripts/smoke.py${NC}

3. ${YELLOW}Train the ultimate model (8-12 hours on A100):${NC}
   ${BLUE}python scripts/train_ultimate_novel.py --config experiments/ultimate_novel_subj01.yaml${NC}

4. ${YELLOW}View training logs:${NC}
   ${BLUE}tensorboard --logdir outputs/runs${NC}

${CYAN}Documentation:${NC}
  - Getting Started: ${BLUE}START_HERE.md${NC}
  - Architecture: ${BLUE}ARCHITECTURE.md${NC}
  - Usage Examples: ${BLUE}USAGE_EXAMPLES.md${NC}
  - Ultimate Training Guide: ${BLUE}ULTIMATE_TRAINING_GUIDE.md${NC}

${CYAN}Troubleshooting:${NC}
  - Check setup: ${BLUE}python scripts/check_setup.sh${NC}
  - Diagnose issues: ${BLUE}python scripts/doctor.py${NC}

${GREEN}Happy brain decoding! 🧠→🖼️${NC}
EOF
}

################################################################################
# Main Execution
################################################################################

main() {
    clear
    
    print_header "fMRI-TO-IMAGE BRAIN DECODING - AUTOMATED SETUP"
    
    echo -e "${CYAN}Project:${NC} $PROJECT_NAME"
    echo -e "${CYAN}Location:${NC} $SCRIPT_DIR"
    echo -e "${CYAN}User:${NC} $(whoami)"
    echo -e "${CYAN}Date:${NC} $(date)"
    echo ""
    
    if [[ "$MINIMAL_SETUP" = true ]]; then
        print_warning "Running MINIMAL setup (environment + dependencies only)"
    fi
    
    # Confirmation
    read -p "$(echo -e ${YELLOW}Continue with setup? \(Y/n\): ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    # Run setup steps
    setup_system_checks
    setup_environment_variables
    setup_python_environment
    
    if [[ "$MINIMAL_SETUP" = false ]]; then
        download_nsd_data
        download_models
        build_clip_cache
    fi
    
    verify_installation
    print_next_steps
}

# Run main function
main "$@"
