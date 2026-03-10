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
#   bash scripts/training/run_ablation_ladder.sh --only N    # all N-series (v5+v6+v7+v8+v9)
#   bash scripts/training/run_ablation_ladder.sh --only N9   # only N v9
#   bash scripts/training/run_ablation_ladder.sh --only N8   # only N v8
#   bash scripts/training/run_ablation_ladder.sh --only N7   # only N v7
#   bash scripts/training/run_ablation_ladder.sh --no-checkpoints          # skip all checkpoints
#   bash scripts/training/run_ablation_ladder.sh --save-checkpoints best  # save only best
#   make ablation SUBJECTS="subj01" GPU=0 ONLY=N7            # same via Make
#   make ablation SUBJECTS="subj01" GPU=0 ONLY=N7 SAVE_CKPT=no    # no checkpoints
#   make ablation SUBJECTS="subj01" GPU=0 ONLY=N7 SAVE_CKPT=best  # best checkpoint only
#
# Results are saved to experimental_results/<experiment_name>/
# =============================================================================

set -euo pipefail

SUBJECTS="${SUBJECTS:-subj01 subj02 subj05 subj07}"
GPU="${GPU:-0}"
START="${START:-B0v4}"
ONLY="${ONLY:-all}"
SAVE_CKPT="${SAVE_CKPT:-all}"
case "${SAVE_CKPT,,}" in
    no|false|off|0|none) SAVE_CKPT=none ;;
    best)                SAVE_CKPT=best ;;
    *)                   SAVE_CKPT=all ;;
esac
CONFIG_DIR="configs/experiments"
SCRIPT="scripts/training/train_unified.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --no-checkpoints) SAVE_CKPT=none; shift ;;
        --save-checkpoints) SAVE_CKPT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

EXPERIMENT_ORDER=(B0v4 B1v4 N1v5 N2v5 N3v5 N4v5 N1v6 N2v6 N3v6 N4v6 N1v7 N2v7 N3v7 N4v7 N3v8 N4v8 N1v9 N2v9 N3v9 N4v9 N1v10 N2v10 N3v10 N4v10 N1v11 N2v11 N3v11 N4v11 N1v12 N2v12 N3v12 N4v12 N1v13 N2v13 N3v13 N4v13 N1v14 N2v14 N3v14 N4v14 N1v15 N2v15 N3v15 N4v15 N1v16 N2v16 N3v16 N4v16 N1v17 N2v17 N3v17 N4v17 N1v18 N2v18 N3v18 N4v18 N1v19 N2v19 N3v19 N4v19 N1v20 N2v20 N3v20 N4v20 N1v21 N2v21 N3v21 N4v21 N1v22 N1v22b N2v22 N3v22 N4v22 N1v23d N1v23a N1v23b N1v23c N1v24 N1v24b N1v24c N1v24d N1v25_rerun N1v25a N1v25b N2v25c N1v26a N1v26b)

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
    [N1v7]="N1v7_vmf_nce.yaml|v7: softplus kappa, fused CLIP targets"
    [N2v7]="N2v7_roi_transformer.yaml|v7: multi-subject, softplus kappa, fused targets"
    [N3v7]="N3v7_roi_dcf.yaml|v7: multi-subject ROI-DCF, softplus kappa, fused targets"
    [N4v7]="N4v7_full_system.yaml|v7: multi-subject flagship, softplus kappa, fused targets, SPCL"
    [N3v8]="N3v8_roi_dcf.yaml|v8: hierarchical CLIP + CKA + arctanh + kappa-margin"
    [N4v8]="N4v8_full_system.yaml|v8: flagship + hierarchical CLIP + per-subject CKA"
    [N1v9]="N1v9_vmf_nce.yaml|v9: MLP vMF-NCE + projection head + R-Drop + CSLS + TTA"
    [N2v9]="N2v9_roi_transformer.yaml|v9: ROI Transformer + DropPath + projection head + CSLS + TTA"
    [N3v9]="N3v9_roi_dcf.yaml|v9: anti-overfit + projection head + CSLS + TTA"
    [N4v9]="N4v9_full_system.yaml|v9: flagship anti-overfit + projection head + CSLS + TTA + SPCL"
    [N1v10]="N1v10_vmf_nce.yaml|v10: V8 recipe + V9 eval + wider MLP + hard neg + model soup"
    [N2v10]="N2v10_roi_transformer.yaml|v10: V8 recipe + V9 eval + wider Transformer + hard neg + model soup"
    [N3v10]="N3v10_roi_dcf.yaml|v10: V8 multi-loss + V9 eval + wider Transformer + hard neg + model soup"
    [N4v10]="N4v10_full_system.yaml|v10: flagship V8 multi-loss + SPCL + V9 eval + wider Transformer + hard neg + soup"
    [N1v11]="N1v11_vmf_nce.yaml|v11: CSLS training + ISF + rep avg + direct align + uniformity"
    [N2v11]="N2v11_roi_transformer.yaml|v11: ROI Transformer + CSLS training + ISF + rep avg"
    [N3v11]="N3v11_roi_dcf.yaml|v11: ROI-DCF multi-loss + CSLS training + ISF + rep avg"
    [N4v11]="N4v11_full_system.yaml|v11: flagship CSLS training + SPCL + rep avg + direct align + uniformity"
    [N1v12]="N1v12_vmf_nce.yaml|v12: two-stage training + kappa cap 50 + scaled MLP + eff batch 512"
    [N2v12]="N2v12_roi_transformer.yaml|v12: two-stage training + kappa cap 50 + wider FFN + eff batch 512"
    [N3v12]="N3v12_roi_dcf.yaml|v12: two-stage + kappa cap + simplified losses + auto-weighting"
    [N4v12]="N4v12_full_system.yaml|v12: flagship two-stage + kappa cap + simplified losses + auto-weighting + SPCL"
    [N1v13]="N1v13_vmf_nce.yaml|v13: MSE regression + contrastive + kappa cap + scaled MLP + two-stage"
    [N2v13]="N2v13_roi_transformer.yaml|v13: MSE regression + contrastive + kappa cap + wider FFN + two-stage"
    [N3v13]="N3v13_roi_dcf.yaml|v13: MSE regression + contrastive + kappa cap + wider FFN + multitask"
    [N4v13]="N4v13_full_system.yaml|v13: flagship MSE regression + SPCL + kappa cap + wider FFN + DUA-CFG"
    [N1v14]="N1v14_vmf_nce.yaml|v14: anti-hubness (CSLS training + ISF + direct alignment) + CSLS checkpoint"
    [N2v14]="N2v14_roi_transformer.yaml|v14: anti-hubness (CSLS training + ISF + direct alignment) + CSLS checkpoint"
    [N3v14]="N3v14_roi_dcf.yaml|v14: anti-hubness + hierarchical CLIP + CSLS checkpoint"
    [N4v14]="N4v14_full_system.yaml|v14: flagship anti-hubness + SPCL + hierarchical CLIP + DUA-CFG + CSLS checkpoint"
    [N1v15]="N1v15_vmf_nce.yaml|v15: PCR + large batch (256) + simplified loss"
    [N2v15]="N2v15_roi_transformer.yaml|v15: ROI Transformer + PCR + large batch + z-scoring fix"
    [N3v15]="N3v15_roi_dcf.yaml|v15: ROI-DCF + PCR + large batch + z-scoring fix"
    [N4v15]="N4v15_full_system.yaml|v15: flagship PCR + large batch + z-scoring fix + SPCL + DUA-CFG"
    [N1v16]="N1v16_vmf_nce.yaml|v16: rep-avg + low-reg + focused MSE+NCE"
    [N2v16]="N2v16_roi_transformer.yaml|v16: ROI Transformer + rep-avg + low-reg + focused MSE+NCE"
    [N3v16]="N3v16_roi_dcf.yaml|v16: ROI-DCF + rep-avg + low-reg + focused MSE+NCE"
    [N4v16]="N4v16_full_system.yaml|v16: flagship rep-avg + low-reg + focused MSE+SPCL"
    [N1v17]="N1v17_vmf_nce.yaml|v17: restore V7 baseline + CSLS + shared1000 + diagnostics"
    [N2v17]="N2v17_roi_transformer.yaml|v17: restore V7 baseline + CSLS + shared1000 + diagnostics"
    [N3v17]="N3v17_roi_dcf.yaml|v17: restore V8 baseline + CSLS + shared1000 + diagnostics"
    [N4v17]="N4v17_full_system.yaml|v17: restore V8 flagship + CSLS + shared1000 + diagnostics"
    [N1v18]="N1v18_vmf_nce.yaml|v18: V17 + MSE regression + PCR anti-hubness"
    [N2v18]="N2v18_roi_transformer.yaml|v18: V17 + MSE regression + PCR anti-hubness"
    [N3v18]="N3v18_roi_dcf.yaml|v18: V17 + MSE + PCR + kappa cap 50"
    [N4v18]="N4v18_full_system.yaml|v18: V17 flagship + MSE + PCR + kappa cap 50"
    [N1v19]="N1v19_vmf_nce.yaml|v19: V17 base + CosFace margin + DirectAlign"
    [N2v19]="N2v19_roi_transformer.yaml|v19: V17 base + margin + DirectAlign + ROI-Token Dropout"
    [N3v19]="N3v19_roi_dcf.yaml|v19: V17 base + margin + DirectAlign + simplified losses + ROI-Token Dropout"
    [N4v19]="N4v19_full_system.yaml|v19: V17 base + DirectAlign + simplified losses + ROI-Token Dropout"
    [N1v20]="N1v20_vmf_nce.yaml|v20: V17 base + sequential MixCo->SoftCLIP + batch 128"
    [N2v20]="N2v20_roi_transformer.yaml|v20: V17 base + sequential MixCo->SoftCLIP + batch 128"
    [N3v20]="N3v20_roi_dcf.yaml|v20: V17 base + sequential MixCo->SoftCLIP + batch 128"
    [N4v20]="N4v20_full_system.yaml|v20: V17 base + sequential MixCo->SoftCLIP + batch 128"
    [N1v21]="N1v21_vmf_nce.yaml|v21: ResidualMLP + kappa calibration (vmf_nll) + ISF anti-hubness + batch 1024"
    [N2v21]="N2v21_roi_transformer.yaml|v21: ROI Transformer + kappa calibration + ISF + batch 1024"
    [N3v21]="N3v21_roi_dcf.yaml|v21: ROI-DCF + kappa calibration + ISF + batch 1024"
    [N4v21]="N4v21_full_system.yaml|v21: flagship + kappa calibration + ISF + batch 1024"
    [N1v22]="N1v22_vmf_nce.yaml|v22: V17 + MSE(sum) regression — properly-scaled"
    [N1v22b]="N1v22b_vmf_nce.yaml|v22b: V17 + MSE(sum) + sequential MixCo->SoftCLIP + batch 128"
    [N2v22]="N2v22_roi_transformer.yaml|v22: V17 + MSE(sum) regression"
    [N3v22]="N3v22_roi_dcf.yaml|v22: V17 + MSE(sum) + reduced mt_aux"
    [N4v22]="N4v22_full_system.yaml|v22: V17 flagship + MSE(sum) + reduced mt_aux"
    [N1v23d]="N1v23d_v17_control.yaml|v23d: exact V17 N1 reproduction — control baseline"
    [N1v23a]="N1v23a_wider_encoder.yaml|v23a: V17 + wider [8192,8192,4096,2048] residual MLP, dropout 0.15, lr 5e-5"
    [N1v23b]="N1v23b_csls_training.yaml|v23b: V17 + differentiable CSLS training (use_csls_training=True, k=10)"
    [N1v23c]="N1v23c_mse_tuned.yaml|v23c: V17 + MSE(sum, w=0.015) calibrated 2% geometric regularizer"
    [N1v24]="N1v24_combined.yaml|v24: V23a+V23b combined — wider enc + CSLS train + hard neg w=0.3 k=16"
    [N1v24b]="N1v24b_label_smooth.yaml|v24b: V24 + label_smoothing=0.05"
    [N1v24c]="N1v24c_larger_batch.yaml|v24c: V24 + batch 128 (grad_accum 2)"
    [N1v24d]="N1v24d_sequential.yaml|v24d: V24 + sequential MixCo->SoftCLIP (softclip_from_start=false)"
    [N1v25_rerun]="N1v25_v23a_rerun.yaml|v25 rerun: reproduce V23a wider MLP for V25b checkpoint"
    [N1v25a]="N1v25a_sequential.yaml|v25a: V23a + sequential MixCo->SoftCLIP (clean, no hard neg)"
    [N1v25b]="N1v25b_cross_subject.yaml|v25b: V23a backbone + cross-subject linear adapters (4 subj, ~100K trials)"
    [N2v25c]="N2v25c_patched_roi.yaml|v25c: ROI Transformer + sub-ROI patching (250 vox/token, 8 layers)"
    [N1v26a]="N1v26a_token_targets.yaml|v26a: MindEye-style 257×768 ViT-L/14 token-level CLIP targets"
    [N1v26b]="N1v26b_cross_subject_tokens.yaml|v26b: V26a token targets + cross-subject adapters (4 subjects, ~100K trials)"
)

CACHE_ROOT="${CACHE_ROOT:-cache}"

# Filter experiment order by series prefix and optional version number
#   ONLY=N1v22  -> exact experiment ID match
#   ONLY=N      -> all N-series (v5+v6+...)
#   ONLY=N22    -> only N v22 (all N*v22* experiments)
#   ONLY=B      -> all B-series
if [[ "$ONLY" != "all" ]]; then
    FILTERED=()
    # First try exact match against experiment IDs
    for exp_id in "${EXPERIMENT_ORDER[@]}"; do
        if [[ "$exp_id" == "$ONLY" ]]; then
            FILTERED+=("$exp_id")
        fi
    done
    # If no exact match, fall back to series+version prefix filter
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        SERIES="${ONLY:0:1}"
        VERSION="${ONLY:1}"
        for exp_id in "${EXPERIMENT_ORDER[@]}"; do
            if [[ "$exp_id" == ${SERIES}* ]]; then
                if [[ -z "$VERSION" ]] || [[ "$exp_id" == *v${VERSION}* ]]; then
                    FILTERED+=("$exp_id")
                fi
            fi
        done
    fi
    EXPERIMENT_ORDER=("${FILTERED[@]}")
    if [[ ${#EXPERIMENT_ORDER[@]} -eq 0 ]]; then
        echo "ERROR: --only '$ONLY' matched zero experiments. Available: ${!CONFIGS[*]}"
        exit 1
    fi
    START="${EXPERIMENT_ORDER[0]}"
fi

echo "=============================================="
echo "ABLATION LADDER"
echo "Experiments: ${EXPERIMENT_ORDER[*]}"
echo "Subjects: ${SUBJECTS}"
echo "GPU: ${GPU}"
echo "Save checkpoints: ${SAVE_CKPT}"
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
        if [[ -f "${result_dir}/checkpoints/checkpoint_best.pt" ]] || \
           [[ -f "${result_dir}/metrics/summary.json" ]]; then
            echo "    [SKIP] Already completed"
            PASSED_RUNS=$((PASSED_RUNS + 1))
            TOTAL_RUNS=$((TOTAL_RUNS - 1))
            continue
        fi

        EXTRA_ARGS=""
        if [[ "$SAVE_CKPT" != "all" ]]; then
            EXTRA_ARGS="--save-checkpoints $SAVE_CKPT"
        fi

        if python3 "$SCRIPT" \
            --config "$config_path" \
            --subject "$subject" \
            --gpu 0 \
            $EXTRA_ARGS \
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
