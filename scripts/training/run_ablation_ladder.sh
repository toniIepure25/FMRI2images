#!/bin/bash
# =============================================================================
# Full Ablation Ladder Runner: EXP0 → EXP14
# =============================================================================
#
# Runs the complete ablation study across all experiments and subjects.
# Each experiment tests ONE hypothesis; the full system combines all.
#
# Usage:
#   bash scripts/training/run_ablation_ladder.sh [--subjects "subj01 subj02"] [--gpu 0] [--start-exp 0]
#
# Results are saved to experimental_results/expN_<name>/
# =============================================================================

set -euo pipefail

SUBJECTS="${SUBJECTS:-subj01 subj02 subj05 subj07}"
GPU="${GPU:-0}"
START_EXP="${START_EXP:-0}"
CONFIG_DIR="configs/experiments"
SCRIPT="scripts/training/train_unified.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --start-exp) START_EXP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

# Experiment ladder with configs and hypotheses
declare -A EXPERIMENTS=(
    [0]="exp0_baseline.yaml|Baseline (MSE+cos, no queue)"
    [1]="exp1_preproc.yaml|H1: center_pcr improves geometry"
    [2]="exp2_queue.yaml|H2: InfoNCE+queue improves retrieval"
    [3]="exp3_gaussian_nll.yaml|H3: Gaussian NLL captures uncertainty"
    [4]="exp4_gaussian_nce.yaml|H4: Gaussian-NCE improves calibration"
    [5]="exp5_kl_anneal.yaml|H5: KL annealing stabilizes training"
    [6]="exp6_whiten.yaml|H6: Whitening vs PCR ablation"
    [7]="exp7_vmf_nce.yaml|H7: vMF > Gaussian on S^{d-1}"
    [8]="exp8_roi_transformer.yaml|H8: ROI Transformer > MLP"
    [9]="exp9_roi_dcf.yaml|H9: ROI-DCF consensus improves both"
    [10]="exp10_vmf_mixture.yaml|H10: Mixture sampling > consensus"
    [11]="exp11_dual_ua_cfg.yaml|H11: Decomposed UA-CFG > heuristic"
    [12]="exp12_ceiling_temperature.yaml|H12: Ceiling-temp improves calibration"
    [13]="exp13_kappa_spcl.yaml|H13: kappa-SPCL curriculum helps"
    [14]="exp14_full_system.yaml|H14: Full system is best overall"
)

echo "=============================================="
echo "ABLATION LADDER: EXP${START_EXP} → EXP14"
echo "Subjects: ${SUBJECTS}"
echo "GPU: ${GPU}"
echo "=============================================="

TOTAL_RUNS=0
PASSED_RUNS=0
FAILED_RUNS=0

for exp_num in $(seq "$START_EXP" 14); do
    IFS='|' read -r config_file hypothesis <<< "${EXPERIMENTS[$exp_num]}"
    config_path="${CONFIG_DIR}/${config_file}"

    if [[ ! -f "$config_path" ]]; then
        echo "[SKIP] EXP${exp_num}: Config not found: ${config_path}"
        continue
    fi

    echo ""
    echo "====== EXP${exp_num}: ${hypothesis} ======"
    echo "Config: ${config_path}"

    for subject in $SUBJECTS; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        run_id="exp${exp_num}_${subject}"
        log_file="experimental_results/${run_id}/train.log"
        mkdir -p "experimental_results/${run_id}"

        echo "  → ${subject} ... "

        if python3 "$SCRIPT" \
            --config "$config_path" \
            --subject "$subject" \
            --gpu 0 \
            > "$log_file" 2>&1; then
            echo "    ✓ PASSED"
            PASSED_RUNS=$((PASSED_RUNS + 1))
        else
            echo "    ✗ FAILED (see ${log_file})"
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
    done
done

echo ""
echo "=============================================="
echo "ABLATION LADDER COMPLETE"
echo "  Total runs: ${TOTAL_RUNS}"
echo "  Passed: ${PASSED_RUNS}"
echo "  Failed: ${FAILED_RUNS}"
echo "=============================================="
