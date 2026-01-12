#!/usr/bin/env bash
# =============================================================================
# verify_setup.sh - Verify JupyterHub workflow installation
# =============================================================================
# Quick verification that all components are in place.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_check() {
    echo -en "${BLUE}[CHECK]${NC} $* ... "
}

log_ok() {
    echo -e "${GREEN}OK${NC}"
}

log_missing() {
    echo -e "${RED}MISSING${NC}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "========================================"
echo "JupyterHub Workflow Verification"
echo "========================================"
echo ""

MISSING=0

# Check configuration files
log_check "Configuration: .env.jupyterhub"
[[ -f .env.jupyterhub ]] && log_ok || { log_missing; ((MISSING++)); }

log_check "Configuration: .env.example"
[[ -f .env.example ]] && log_ok || { log_missing; ((MISSING++)); }

# Check scripts
echo ""
echo "Scripts:"
for script in setup_env.sh preflight.sh snapshot_env.sh run_experiment_simple.sh run_sweep.sh tmux_run.sh nohup_run.sh; do
    log_check "  $script"
    [[ -f "scripts/$script" ]] && log_ok || { log_missing; ((MISSING++)); }
done

# Check executability
echo ""
echo "Script Permissions:"
for script in scripts/*.sh; do
    if [[ -f "$script" ]]; then
        log_check "  $(basename $script) executable"
        [[ -x "$script" ]] && log_ok || { log_missing; ((MISSING++)); }
    fi
done

# Check training script
echo ""
log_check "Training: src/train.py"
[[ -f src/train.py ]] && log_ok || { log_missing; ((MISSING++)); }

# Check configs
echo ""
echo "Example Configs:"
for config in jupyterhub_quickstart.yaml full_training_example.yaml smoke_test.yaml; do
    log_check "  $config"
    [[ -f "configs/experiments/$config" ]] && log_ok || { log_missing; ((MISSING++)); }
done

# Check documentation
echo ""
echo "Documentation:"
for doc in README_JUPYTERHUB.md QUICK_REFERENCE.md IMPLEMENTATION_SUMMARY.md START_JUPYTERHUB.md; do
    log_check "  $doc"
    [[ -f "$doc" ]] && log_ok || { log_missing; ((MISSING++)); }
done

# Check requirements
echo ""
log_check "Dependencies: requirements.txt"
[[ -f requirements.txt ]] && log_ok || { log_missing; ((MISSING++)); }

# Summary
echo ""
echo "========================================"
if [[ $MISSING -eq 0 ]]; then
    echo -e "${GREEN}✓ All components present!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. cp .env.jupyterhub .env"
    echo "  2. Edit .env (set USER)"
    echo "  3. bash scripts/setup_env.sh"
    echo "  4. source activate_env.sh"
    echo "  5. bash scripts/preflight.sh"
    echo ""
    echo "See START_JUPYTERHUB.md for details."
    exit 0
else
    echo -e "${RED}✗ Missing $MISSING component(s)!${NC}"
    echo ""
    echo "Some files are missing. Please check the implementation."
    exit 1
fi
