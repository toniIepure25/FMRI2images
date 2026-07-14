#!/bin/bash
# PCD Ablation Experiments
#
# Runs all 6 ablations: full, no_errors, reversed, random, flat_transformer, mlp_baseline
# Run on subj01 (primary evaluation subject).
#
# Usage:
#   bash scripts/orchestration/run_pcd_ablations.sh [GPU_ID]
#   bash scripts/orchestration/run_pcd_ablations.sh 0 --only no_errors

set -euo pipefail

GPU=${1:-0}
ONLY=${2:-}

echo "============================================"
echo " PCD Ablation Experiments"
echo "============================================"

ARGS="--base-config configs/experiments/PCD_v1_8subject.yaml --subject subj01 --gpu $GPU"

if [ -n "$ONLY" ] && [ "$ONLY" != "--only" ]; then
    ARGS="$ARGS --only $ONLY"
elif [ "$2" = "--only" ] && [ -n "${3:-}" ]; then
    ARGS="$ARGS --only $3"
fi

python3 scripts/analysis/run_pcd_ablations.py $ARGS

echo "============================================"
echo " Ablations complete. Results in experimental_results/PCD_ablations/"
echo "============================================"
