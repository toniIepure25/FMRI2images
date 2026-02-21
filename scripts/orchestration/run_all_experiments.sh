#!/bin/bash
# =============================================================================
# Run All Experiments (B0-N4 Ablation Ladder)
# =============================================================================
#
# Thin wrapper around the ablation ladder runner.
#
# Usage:
#   bash scripts/orchestration/run_all_experiments.sh [GPU_ID]
#
# For full control over subjects and starting point, use the ladder directly:
#   bash scripts/training/run_ablation_ladder.sh --subjects "subj01 subj02" --gpu 0 --start N1
#
# =============================================================================

set -euo pipefail

GPU=${1:-0}
SUBJECTS="${SUBJECTS:-subj01 subj02 subj05 subj07}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== Running Full Ablation Ladder (B0 -> N4) ==="
echo "GPU:      ${GPU}"
echo "Subjects: ${SUBJECTS}"
echo ""

# Ensure preprocessors are built
if [[ -f "${SCRIPT_DIR}/scripts/build/build_all_preprocessors.sh" ]]; then
    if [[ ! -f "cache/embedding_preproc/center_pcr_k8.pkl" ]]; then
        echo "Building preprocessors first..."
        bash "${SCRIPT_DIR}/scripts/build/build_all_preprocessors.sh"
        echo ""
    fi
fi

# Delegate to the canonical ablation runner
bash "${SCRIPT_DIR}/scripts/training/run_ablation_ladder.sh" \
    --subjects "${SUBJECTS}" \
    --gpu "${GPU}"
