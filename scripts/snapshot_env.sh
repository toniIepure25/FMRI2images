#!/usr/bin/env bash
# =============================================================================
# snapshot_env.sh - Capture environment state for reproducibility
# =============================================================================
# Creates a comprehensive snapshot of the current environment including:
# - Python packages and versions
# - System information
# - Git commit hash
# - Environment variables
# - CUDA/GPU information
#
# Usage:
#   bash scripts/snapshot_env.sh <output_dir>
#   bash scripts/snapshot_env.sh <output_dir> --format json
#
# Arguments:
#   output_dir       Directory to write snapshot files
#
# Options:
#   --format FORMAT  Output format: txt (default) or json
#   --help           Show this help message
#
# Output files:
#   - environment.txt (or .json)  Complete environment snapshot
#   - requirements_frozen.txt      Frozen pip requirements
#   - git_info.txt (or .json)      Git repository information
#   - system_info.txt (or .json)   System and hardware information
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
OUTPUT_DIR=""
FORMAT="txt"

show_help() {
    grep '^#' "$0" | grep -v '#!/usr/bin/env' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            if [[ -z "${OUTPUT_DIR}" ]]; then
                OUTPUT_DIR="$1"
                shift
            else
                log_error "Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

# Validate arguments
if [[ -z "${OUTPUT_DIR}" ]]; then
    log_error "Missing required argument: output_dir"
    echo "Usage: bash scripts/snapshot_env.sh <output_dir> [--format txt|json]"
    exit 1
fi

if [[ "${FORMAT}" != "txt" && "${FORMAT}" != "json" ]]; then
    log_error "Invalid format: ${FORMAT}. Must be txt or json"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
log_info "Creating environment snapshot in: ${OUTPUT_DIR}"

# =============================================================================
# 1. Python packages
# =============================================================================
log_info "Capturing Python packages..."
REQUIREMENTS_FILE="${OUTPUT_DIR}/requirements_frozen.txt"
if command -v pip &> /dev/null; then
    pip freeze > "${REQUIREMENTS_FILE}"
    log_success "Saved frozen requirements to: ${REQUIREMENTS_FILE}"
else
    log_error "pip not found"
fi

# =============================================================================
# 2. Git information
# =============================================================================
log_info "Capturing git information..."
GIT_FILE="${OUTPUT_DIR}/git_info.${FORMAT}"

if command -v git &> /dev/null && git -C "${PROJECT_ROOT}" rev-parse --git-dir > /dev/null 2>&1; then
    if [[ "${FORMAT}" == "json" ]]; then
        cat > "${GIT_FILE}" << EOF
{
  "commit_hash": "$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "commit_hash_short": "$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
  "branch": "$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "remote_url": "$(git -C "${PROJECT_ROOT}" config --get remote.origin.url 2>/dev/null || echo 'unknown')",
  "has_uncommitted_changes": $(git -C "${PROJECT_ROOT}" diff-index --quiet HEAD -- 2>/dev/null && echo 'false' || echo 'true'),
  "last_commit_message": "$(git -C "${PROJECT_ROOT}" log -1 --pretty=%B 2>/dev/null | head -1 | sed 's/"/\\"/g' || echo 'unknown')",
  "last_commit_author": "$(git -C "${PROJECT_ROOT}" log -1 --pretty=%an 2>/dev/null || echo 'unknown')",
  "last_commit_date": "$(git -C "${PROJECT_ROOT}" log -1 --pretty=%ai 2>/dev/null || echo 'unknown')"
}
EOF
    else
        cat > "${GIT_FILE}" << EOF
Git Repository Information
==========================
Repository: ${PROJECT_ROOT}
Commit Hash: $(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo 'unknown')
Short Hash: $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')
Branch: $(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')
Remote URL: $(git -C "${PROJECT_ROOT}" config --get remote.origin.url 2>/dev/null || echo 'unknown')
Uncommitted Changes: $(git -C "${PROJECT_ROOT}" diff-index --quiet HEAD -- 2>/dev/null && echo 'No' || echo 'Yes')

Last Commit:
$(git -C "${PROJECT_ROOT}" log -1 --pretty=format:"  Author: %an <%ae>%n  Date: %ai%n  Message: %s" 2>/dev/null || echo '  unknown')
EOF
    fi
    log_success "Saved git info to: ${GIT_FILE}"
else
    if [[ "${FORMAT}" == "json" ]]; then
        echo '{"error": "Not a git repository or git not available"}' > "${GIT_FILE}"
    else
        echo "Not a git repository or git not available" > "${GIT_FILE}"
    fi
    log_error "Not a git repository or git not available"
fi

# =============================================================================
# 3. System information
# =============================================================================
log_info "Capturing system information..."
SYSTEM_FILE="${OUTPUT_DIR}/system_info.${FORMAT}"

if [[ "${FORMAT}" == "json" ]]; then
    cat > "${SYSTEM_FILE}" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "$(hostname)",
  "user": "$(whoami)",
  "operating_system": "$(uname -s)",
  "kernel_version": "$(uname -r)",
  "architecture": "$(uname -m)",
  "python_version": "$(python --version 2>&1 | awk '{print $2}')",
  "python_executable": "$(which python)",
  "pip_version": "$(pip --version 2>&1 | awk '{print $2}' || echo 'unknown')",
  "cpu_count": $(python -c "import os; print(os.cpu_count())" 2>/dev/null || echo 'null'),
  "virtual_env": "${VIRTUAL_ENV:-null}",
  "pwd": "$(pwd)"
}
EOF
else
    cat > "${SYSTEM_FILE}" << EOF
System Information
==================
Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Hostname: $(hostname)
User: $(whoami)
Operating System: $(uname -s)
Kernel: $(uname -r)
Architecture: $(uname -m)

Python Environment
==================
Python Version: $(python --version 2>&1)
Python Executable: $(which python)
pip Version: $(pip --version 2>&1 | awk '{print $2}' || echo 'unknown')
Virtual Environment: ${VIRTUAL_ENV:-Not active}
CPU Count: $(python -c "import os; print(os.cpu_count())" 2>/dev/null || echo 'unknown')

Working Directory: $(pwd)
EOF
fi

# Add memory info if available
if command -v free &> /dev/null; then
    if [[ "${FORMAT}" == "json" ]]; then
        # Add memory to JSON (requires editing the file)
        :
    else
        echo "" >> "${SYSTEM_FILE}"
        echo "Memory Information" >> "${SYSTEM_FILE}"
        echo "==================" >> "${SYSTEM_FILE}"
        free -h >> "${SYSTEM_FILE}"
    fi
fi

log_success "Saved system info to: ${SYSTEM_FILE}"

# =============================================================================
# 4. GPU/CUDA information
# =============================================================================
log_info "Capturing GPU/CUDA information..."
if command -v nvidia-smi &> /dev/null && python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    if [[ "${FORMAT}" == "json" ]]; then
        python << 'PYEOF' > "${OUTPUT_DIR}/gpu_info.json"
import json
import subprocess
import torch

info = {
    "pytorch_cuda_available": torch.cuda.is_available(),
    "pytorch_cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "pytorch_version": torch.__version__,
    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "gpus": []
}

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        gpu_info = {
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
            "total_memory_mb": torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
        }
        info["gpus"].append(gpu_info)
    
    # Add nvidia-smi info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        info["nvidia_driver_version"] = result.stdout.strip().split('\n')[0]
    except:
        pass

print(json.dumps(info, indent=2))
PYEOF
        log_success "Saved GPU info to: ${OUTPUT_DIR}/gpu_info.json"
    else
        GPU_FILE="${OUTPUT_DIR}/gpu_info.txt"
        cat > "${GPU_FILE}" << EOF
GPU/CUDA Information
====================
PyTorch CUDA Available: $(python -c "import torch; print(torch.cuda.is_available())")
PyTorch Version: $(python -c "import torch; print(torch.__version__)")
PyTorch CUDA Version: $(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')")
GPU Count: $(python -c "import torch; print(torch.cuda.device_count())")

EOF
        # Add GPU details
        python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Total Memory: {props.total_memory / 1024 / 1024:.0f} MB')
        print(f'  Compute Capability: {props.major}.{props.minor}')
        print()
" >> "${GPU_FILE}"

        echo "" >> "${GPU_FILE}"
        echo "nvidia-smi Output" >> "${GPU_FILE}"
        echo "=================" >> "${GPU_FILE}"
        nvidia-smi >> "${GPU_FILE}" 2>&1 || echo "nvidia-smi failed" >> "${GPU_FILE}"
        
        log_success "Saved GPU info to: ${GPU_FILE}"
    fi
else
    if [[ "${FORMAT}" == "json" ]]; then
        echo '{"pytorch_cuda_available": false, "error": "CUDA not available or nvidia-smi not found"}' > "${OUTPUT_DIR}/gpu_info.json"
    else
        echo "CUDA not available or nvidia-smi not found" > "${OUTPUT_DIR}/gpu_info.txt"
    fi
    log_error "CUDA not available or nvidia-smi not found"
fi

# =============================================================================
# 5. Environment variables
# =============================================================================
log_info "Capturing environment variables..."
ENV_FILE="${OUTPUT_DIR}/env_vars.${FORMAT}"

if [[ "${FORMAT}" == "json" ]]; then
    python << 'PYEOF' > "${ENV_FILE}"
import json
import os

# Filter relevant environment variables
relevant_prefixes = ['PYTHON', 'CUDA', 'HF_', 'TRANSFORMERS_', 'TORCH_', 'DATA_', 'CACHE_', 'OUTPUT_', 'RUNS_', 'CHECKPOINT_', 'WANDB_', 'PROJECT_']
env_vars = {}

for key, value in os.environ.items():
    if any(key.startswith(prefix) for prefix in relevant_prefixes):
        env_vars[key] = value

print(json.dumps(env_vars, indent=2, sort_keys=True))
PYEOF
else
    cat > "${ENV_FILE}" << EOF
Environment Variables
=====================
(Filtered to relevant variables)

EOF
    env | grep -E '^(PYTHON|CUDA|HF_|TRANSFORMERS_|TORCH_|DATA_|CACHE_|OUTPUT_|RUNS_|CHECKPOINT_|WANDB_|PROJECT_)' | sort >> "${ENV_FILE}" || true
fi

log_success "Saved environment variables to: ${ENV_FILE}"

# =============================================================================
# 6. Comprehensive summary
# =============================================================================
log_info "Creating comprehensive summary..."
SUMMARY_FILE="${OUTPUT_DIR}/environment.${FORMAT}"

if [[ "${FORMAT}" == "json" ]]; then
    # Combine all JSON files
    python << PYEOF > "${SUMMARY_FILE}"
import json
import os
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
summary = {
    "snapshot_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "snapshot_format_version": "1.0"
}

# Load individual JSON files
for filename in ["system_info.json", "git_info.json", "gpu_info.json", "env_vars.json"]:
    filepath = output_dir / filename
    if filepath.exists():
        with open(filepath) as f:
            try:
                key = filename.replace(".json", "")
                summary[key] = json.load(f)
            except:
                pass

print(json.dumps(summary, indent=2))
PYEOF
else
    cat > "${SUMMARY_FILE}" << EOF
=============================================================================
ENVIRONMENT SNAPSHOT
=============================================================================
Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Project: ${PROJECT_ROOT}

This snapshot contains comprehensive information about the environment
used to run experiments. It can be used to reproduce results.

Files in this snapshot:
- environment.txt           This file (summary)
- requirements_frozen.txt   Exact Python package versions
- git_info.txt              Git repository state
- system_info.txt           System and hardware information
- gpu_info.txt              GPU and CUDA information
- env_vars.txt              Relevant environment variables

=============================================================================

EOF
    
    # Append content from other files
    for file in system_info.txt git_info.txt gpu_info.txt; do
        if [[ -f "${OUTPUT_DIR}/${file}" ]]; then
            echo "" >> "${SUMMARY_FILE}"
            echo "==============================================================================" >> "${SUMMARY_FILE}"
            cat "${OUTPUT_DIR}/${file}" >> "${SUMMARY_FILE}"
            echo "" >> "${SUMMARY_FILE}"
        fi
    done
fi

log_success "Saved comprehensive summary to: ${SUMMARY_FILE}"

# =============================================================================
# Summary
# =============================================================================
echo ""
log_success "Environment snapshot complete!"
log_info "Output directory: ${OUTPUT_DIR}"
log_info "Format: ${FORMAT}"
echo ""
log_info "Files created:"
ls -lh "${OUTPUT_DIR}" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
echo ""
