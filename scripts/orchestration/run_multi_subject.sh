#!/usr/bin/env bash
# ============================================================================
# Multi-Subject Experiment Runner
# ============================================================================
# Trains the best-performing configuration on multiple NSD subjects and
# aggregates results for the paper (mean +/- std across subjects).
#
# Usage:
#   bash scripts/orchestration/run_multi_subject.sh [CONFIG] [GPU]
#
# Example:
#   bash scripts/orchestration/run_multi_subject.sh configs/experiments/exp_vmf.yaml 0
# ============================================================================

set -euo pipefail

CONFIG="${1:-configs/experiments/exp4_gaussian_nce.yaml}"
GPU="${2:-0}"

SUBJECTS=("subj01" "subj02" "subj05" "subj07")

echo "==========================================="
echo "Multi-Subject Experiment"
echo "Config:   $CONFIG"
echo "GPU:      $GPU"
echo "Subjects: ${SUBJECTS[*]}"
echo "==========================================="

for SUBJ in "${SUBJECTS[@]}"; do
    echo ""
    echo "--- Training $SUBJ ---"
    python scripts/training/train_unified.py \
        --config "$CONFIG" \
        --gpu "$GPU" \
        --subject "$SUBJ"
    echo "--- Finished $SUBJ ---"
done

echo ""
echo "All subjects complete.  Aggregate results with:"
echo "  python scripts/analysis/aggregate_subjects.py --subjects ${SUBJECTS[*]}"
