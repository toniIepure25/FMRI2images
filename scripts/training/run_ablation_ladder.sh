#!/bin/bash
# =============================================================================
# Ablation Ladder Runner: B0 -> B1 -> N1 -> N2 -> N3 -> N4
# =============================================================================
#
# Runs the streamlined 6-experiment ablation study:
#   B0  Deterministic MLP baseline (MSE + InfoNCE + queue + PCR)
#   B1  Gaussian probabilistic baseline (Gaussian-NCE + KL)
#   N1  vMF-NCE  (novel distribution — vMF on S^{d-1})
#   N2  ROI Transformer  (novel architecture — brain-topology-aware)
#   N3  ROI-DCF  (novel fusion — per-ROI directional consensus)
#   N4  Full System  (all innovations combined)
#
# Usage:
#   bash scripts/training/run_ablation_ladder.sh
#   bash scripts/training/run_ablation_ladder.sh --subjects "subj01 subj02" --gpu 0
#   bash scripts/training/run_ablation_ladder.sh --start B1
#   bash scripts/training/run_ablation_ladder.sh --only N    # all N-series (v5+v6)
#   bash scripts/training/run_ablation_ladder.sh --only N6   # only N v6
#   bash scripts/training/run_ablation_ladder.sh --only N5   # only N v5
#   make ablation SUBJECTS="subj01" GPU=0 ONLY=N6            # same via Make
#
# Results are saved to experimental_results/<experiment_name>/
# =============================================================================

set -euo pipefail

SUBJECTS="${SUBJECTS:-subj01 subj02 subj05 subj07}"
GPU="${GPU:-0}"
START="${START:-B0v4}"
ONLY="${ONLY:-all}"
CONFIG_DIR="configs/experiments"
SCRIPT="scripts/training/train_unified.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

EXPERIMENT_ORDER=(B0v4 B1v4 N1v5 N2v5 N3v5 N4v5 N1v6 N2v6 N3v6 N4v6)

declare -A CONFIGS=(
    [B0]="B0_deterministic.yaml|Strong deterministic baseline (MLP + PCR + queue + MSE + InfoNCE)"
    [B1]="B1_gaussian.yaml|Probabilistic Gaussian baseline (Gaussian-NCE + KL annealing)"
    [N1]="N1_vmf_nce.yaml|Novel: vMF > Gaussian on S^{d-1}"
    [N2]="N2_roi_transformer.yaml|Novel: ROI Transformer > flat MLP"
    [N3]="N3_roi_dcf.yaml|Novel: ROI-DCF directional consensus fusion"
    [N4]="N4_full_system.yaml|Novel: Full system (all innovations)"
    [B0v2]="B0v2_deterministic.yaml|v2: Improved deterministic (dropout 0.3, lr 5e-5, R@1 stopping)"
    [B1v2]="B1v2_gaussian.yaml|v2: Improved Gaussian (GNCE tau 0.5, lr 3e-5)"
    [N1v2]="N1v2_vmf_nce.yaml|v2: Improved vMF-NCE (kappa_reg 0.1, dropout 0.3)"
    [N2v2]="N2v2_roi_transformer.yaml|v2: Improved ROI Transformer (kappa_reg added, kappa_max ~20)"
    [N3v2]="N3v2_roi_dcf.yaml|v2: Improved ROI-DCF (dropout 0.3, wd 0.1, lr 1e-4)"
    [N4v2]="N4v2_full_system.yaml|v2: Improved Full System (aggressive reg + faster SPCL)"
    [B0v3]="B0v3_deterministic.yaml|v3: batch 64, residual MLP, image split, 300 epochs"
    [B1v3]="B1v3_gaussian.yaml|v3: batch 64, residual MLP, GNCE tau 1.0, image split"
    [N1v3]="N1v3_vmf_nce.yaml|v3: batch 64, residual MLP, kappa_reg 0.05, image split"
    [N2v3]="N2v3_roi_transformer.yaml|v3: batch 64, no NLL, wider kappa, image split"
    [N3v3]="N3v3_roi_dcf.yaml|v3: batch 64, lighter reg, image split, 300 epochs"
    [N4v3]="N4v3_full_system.yaml|v3: batch 64, lighter reg, faster SPCL, image split"
    [B0v4]="B0v4_deterministic.yaml|v4: z-scored fMRI, no PCR, rep-avg, MixCo, wider MLP"
    [B1v4]="B1v4_gaussian.yaml|v4: z-scored fMRI, no PCR, rep-avg, MixCo, wider MLP"
    [N1v4]="N1v4_vmf_nce.yaml|v4: z-scored fMRI, no PCR, rep-avg, MixCo, wider MLP"
    [N2v4]="N2v4_roi_transformer.yaml|v4: z-scored fMRI, no PCR, rep-avg, MixCo"
    [N3v4]="N3v4_roi_dcf.yaml|v4: z-scored fMRI, no PCR, rep-avg, MixCo"
    [N4v4]="N4v4_full_system.yaml|v4: z-scored fMRI, no PCR, rep-avg, MixCo"
    [N1v5]="N1v5_vmf_nce.yaml|v5: tau=1.0, no kappa_reg, noise aug, EMA, SoftCLIP from start"
    [N2v5]="N2v5_roi_transformer.yaml|v5: tau=1.0, d_model=768, 6 layers, noise aug, EMA"
    [N3v5]="N3v5_roi_dcf.yaml|v5: tau=1.0, no kappa_reg, noise aug, EMA, SoftCLIP from start"
    [N4v5]="N4v5_full_system.yaml|v5: tau=1.0, no kappa_reg, noise aug, EMA, flagship"
    [N1v6]="N1v6_vmf_nce.yaml|v6: vMF-SoftCLIP, Slerp MixCo"
    [N2v6]="N2v6_roi_transformer.yaml|v6: vMF-SoftCLIP, Slerp MixCo"
    [N3v6]="N3v6_roi_dcf.yaml|v6: vMF-SoftCLIP, Slerp MixCo"
    [N4v6]="N4v6_full_system.yaml|v6: Delta-SPCL, vMF-SoftCLIP, Slerp MixCo, flagship"
)

CACHE_ROOT="${CACHE_ROOT:-cache}"

# Filter experiment order by series prefix and optional version number
#   ONLY=N   -> all N-series (v5+v6)
#   ONLY=N6  -> only N v6
#   ONLY=N5  -> only N v5
#   ONLY=B   -> all B-series
if [[ "$ONLY" != "all" ]]; then
    SERIES="${ONLY:0:1}"
    VERSION="${ONLY:1}"
    FILTERED=()
    for exp_id in "${EXPERIMENT_ORDER[@]}"; do
        if [[ "$exp_id" == ${SERIES}* ]]; then
            if [[ -z "$VERSION" ]] || [[ "$exp_id" == *v${VERSION}* ]]; then
                FILTERED+=("$exp_id")
            fi
        fi
    done
    EXPERIMENT_ORDER=("${FILTERED[@]}")
    START="${EXPERIMENT_ORDER[0]}"
fi

echo "=============================================="
echo "ABLATION LADDER"
echo "Experiments: ${EXPERIMENT_ORDER[*]}"
echo "Subjects: ${SUBJECTS}"
echo "GPU: ${GPU}"
echo "=============================================="

# --- Pre-extract fMRI features (one-time, ~5-10 min per subject) ---
echo ""
echo "====== Pre-extraction check ======"
for subject in $SUBJECTS; do
    feat_file="${CACHE_ROOT}/preextracted/subject=${subject}/fmri_features.npy"
    if [[ -f "$feat_file" ]]; then
        echo "  ${subject}: pre-extracted features found"
    else
        echo "  ${subject}: extracting features ..."
        make preextract SUBJECT="${subject}"
    fi
done
echo "====== Pre-extraction complete ======"

started=false
TOTAL_RUNS=0
PASSED_RUNS=0
FAILED_RUNS=0

for exp_id in "${EXPERIMENT_ORDER[@]}"; do
    if [[ "$exp_id" == "$START" ]]; then
        started=true
    fi
    if [[ "$started" != true ]]; then
        continue
    fi

    IFS='|' read -r config_file description <<< "${CONFIGS[$exp_id]}"
    config_path="${CONFIG_DIR}/${config_file}"

    if [[ ! -f "$config_path" ]]; then
        echo "[SKIP] ${exp_id}: Config not found: ${config_path}"
        continue
    fi

    echo ""
    echo "====== ${exp_id}: ${description} ======"
    echo "Config: ${config_path}"

    for subject in $SUBJECTS; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        log_dir="experimental_results/${config_file%.yaml}/${subject}/logs"
        mkdir -p "$log_dir"
        log_file="${log_dir}/train.log"

        echo "  -> ${subject} ... "

        result_dir="experimental_results/${config_file%.yaml}/${subject}"
        if [[ -f "${result_dir}/checkpoint_best.pt" ]]; then
            echo "    [SKIP] Already completed"
            PASSED_RUNS=$((PASSED_RUNS + 1))
            TOTAL_RUNS=$((TOTAL_RUNS - 1))
            continue
        fi

        if python3 "$SCRIPT" \
            --config "$config_path" \
            --subject "$subject" \
            --gpu 0 \
            > "$log_file" 2>&1; then
            echo "    PASSED"
            PASSED_RUNS=$((PASSED_RUNS + 1))
        else
            echo "    FAILED (see ${log_file})"
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
    done
done

echo ""
echo "=============================================="
echo "ABLATION LADDER COMPLETE"
echo "  Total runs: ${TOTAL_RUNS}"
echo "  Passed:     ${PASSED_RUNS}"
echo "  Failed:     ${FAILED_RUNS}"
echo "=============================================="
