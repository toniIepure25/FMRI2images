#!/bin/bash
# =============================================================================
# Full Ablation Pipeline: Data Prep -> Training -> Reconstruction -> Eval
# =============================================================================
#
# Runs the complete experiment pipeline end-to-end:
#   1. Build NSD index for each subject
#   2. Build CLIP embedding cache
#   3. Fit preprocessing pipelines
#   4. Run training for B0-N4 x all subjects
#   5. Reconstruct images from checkpoints
#   6. Evaluate reconstructions
#   7. Aggregate results into paper-ready tables
#
# Usage:
#   bash scripts/orchestration/run_full_ablation.sh
#   bash scripts/orchestration/run_full_ablation.sh --subjects "subj01 subj02" --gpu 0
#   bash scripts/orchestration/run_full_ablation.sh --skip-data --skip-recon
#
# =============================================================================

set -euo pipefail

SUBJECTS="${SUBJECTS:-subj01 subj02 subj05 subj07}"
GPU="${GPU:-0}"
SKIP_DATA="${SKIP_DATA:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_RECON="${SKIP_RECON:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"
RECON_LIMIT="${RECON_LIMIT:-16}"
SD_MODEL="${SD_MODEL:-sd2-community/stable-diffusion-2-1}"
CONFIG_DIR="configs/experiments"
RESULTS_DIR="experimental_results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --subjects)     SUBJECTS="$2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --skip-data)    SKIP_DATA=true; shift ;;
        --skip-train)   SKIP_TRAIN=true; shift ;;
        --skip-recon)   SKIP_RECON=true; shift ;;
        --skip-eval)    SKIP_EVAL=true; shift ;;
        --recon-limit)  RECON_LIMIT="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

EXPERIMENT_ORDER=(B0_deterministic B1_gaussian N1_vmf_nce N2_roi_transformer N3_roi_dcf N4_full_system)

echo "=============================================="
echo "FULL ABLATION PIPELINE"
echo "  Subjects: ${SUBJECTS}"
echo "  GPU: ${GPU}"
echo "  Skip data: ${SKIP_DATA}"
echo "  Skip train: ${SKIP_TRAIN}"
echo "  Skip recon: ${SKIP_RECON}"
echo "=============================================="

# -------------------------------------------------------
# Phase 1: Data Preparation
# -------------------------------------------------------
if [[ "$SKIP_DATA" != "true" ]]; then
    echo ""
    echo "====== PHASE 1: Data Preparation ======"
    for subject in $SUBJECTS; do
        echo "  -> Index for $subject"
        make index SUBJECT="$subject" 2>&1 | tail -1

        echo "  -> Preprocessing for $subject"
        make preprocess SUBJECT="$subject" 2>&1 | tail -1

        echo "  -> Pre-extracting fMRI features for $subject"
        make preextract SUBJECT="$subject" 2>&1 | tail -1
    done

    echo "  -> CLIP embedding cache"
    make clip-cache 2>&1 | tail -1

    echo "  -> Stable Diffusion model"
    make download-sd MODEL="$SD_MODEL" 2>&1 | tail -1

    echo "  Phase 1 complete."
else
    echo "Skipping data preparation."
fi

# -------------------------------------------------------
# Phase 2: Training (B0 -> N4 for all subjects)
# -------------------------------------------------------
if [[ "$SKIP_TRAIN" != "true" ]]; then
    echo ""
    echo "====== PHASE 2: Training ======"
    TOTAL=0; PASSED=0; FAILED=0

    for exp_id in "${EXPERIMENT_ORDER[@]}"; do
        config_path="${CONFIG_DIR}/${exp_id}.yaml"
        if [[ ! -f "$config_path" ]]; then
            echo "  [SKIP] ${exp_id}: config not found"
            continue
        fi

        echo ""
        echo "  --- ${exp_id} ---"
        for subject in $SUBJECTS; do
            TOTAL=$((TOTAL + 1))
            log_dir="${RESULTS_DIR}/${exp_id}/${subject}/logs"
            mkdir -p "$log_dir"
            log_file="${log_dir}/train.log"

            echo -n "    ${subject} ... "
            if python3 scripts/training/train_unified.py \
                --config "$config_path" \
                --subject "$subject" \
                --gpu 0 \
                > "$log_file" 2>&1; then
                echo "PASSED"
                PASSED=$((PASSED + 1))
            else
                echo "FAILED (see ${log_file})"
                FAILED=$((FAILED + 1))
            fi
        done
    done

    echo ""
    echo "  Training summary: ${PASSED}/${TOTAL} passed, ${FAILED} failed"
else
    echo "Skipping training."
fi

# -------------------------------------------------------
# Phase 3: Reconstruction (diffusion generation)
# -------------------------------------------------------
if [[ "$SKIP_RECON" != "true" ]]; then
    echo ""
    echo "====== PHASE 3: Reconstruction ======"

    for exp_id in "${EXPERIMENT_ORDER[@]}"; do
        for subject in $SUBJECTS; do
            ckpt="${RESULTS_DIR}/${exp_id}/${subject}/checkpoint_best.pt"
            if [[ ! -f "$ckpt" ]]; then
                echo "  [SKIP] ${exp_id}/${subject}: no checkpoint"
                continue
            fi

            recon_dir="${RESULTS_DIR}/${exp_id}/${subject}/reconstructions"
            if [[ -d "$recon_dir" ]] && [[ "$(ls -A "$recon_dir" 2>/dev/null)" ]]; then
                echo "  [SKIP] ${exp_id}/${subject}: reconstructions exist"
                continue
            fi

            echo "  -> ${exp_id}/${subject}"
            python3 scripts/reconstruction/decode_diffusion.py \
                --subject "$subject" \
                --encoder unified \
                --ckpt "$ckpt" \
                --clip-cache outputs/clip_cache/clip.parquet \
                --model-id "$SD_MODEL" \
                --output-dir "$recon_dir" \
                --limit "$RECON_LIMIT" \
                --guidance 7.5 \
                --steps 50 \
                2>&1 | tail -3
        done
    done
    echo "  Phase 3 complete."
else
    echo "Skipping reconstruction."
fi

# -------------------------------------------------------
# Phase 4: Evaluation
# -------------------------------------------------------
if [[ "$SKIP_EVAL" != "true" ]]; then
    echo ""
    echo "====== PHASE 4: Evaluation ======"

    for exp_id in "${EXPERIMENT_ORDER[@]}"; do
        for subject in $SUBJECTS; do
            recon_dir="${RESULTS_DIR}/${exp_id}/${subject}/reconstructions"
            if [[ ! -d "$recon_dir" ]]; then
                continue
            fi
            echo "  -> Eval ${exp_id}/${subject}"
            python3 scripts/evaluation/eval_reconstruction.py \
                --index-root data/indices/nsd_index \
                --subject "$subject" \
                --recon-dir "$recon_dir" \
                --clip-cache outputs/clip_cache/clip.parquet \
                --out-csv "${RESULTS_DIR}/${exp_id}/${subject}/metrics/recon_eval.csv" \
                --out-json "${RESULTS_DIR}/${exp_id}/${subject}/metrics/recon_eval.json" \
                --out-fig "${RESULTS_DIR}/${exp_id}/${subject}/metrics/recon_grid.png" \
                2>&1 | tail -3 || true
        done
    done
    echo "  Phase 4 complete."
fi

# -------------------------------------------------------
# Phase 5: Aggregation
# -------------------------------------------------------
echo ""
echo "====== PHASE 5: Aggregation ======"
python3 scripts/evaluation/aggregate_ablation.py \
    --results-dir "$RESULTS_DIR" \
    --subjects $SUBJECTS \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=============================================="
echo "FULL ABLATION PIPELINE COMPLETE"
echo "  Results: ${RESULTS_DIR}/"
echo "  Summary: ${RESULTS_DIR}/ablation_summary.csv"
echo "  LaTeX:   ${RESULTS_DIR}/ablation_summary.tex"
echo "  Chart:   ${RESULTS_DIR}/ablation_comparison.png"
echo "=============================================="
