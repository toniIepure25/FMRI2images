# Implementation Status

**Last Updated:** March 2026
**Current Phase:** Phase 2+ (ViT-L/14, 768-D, vMF-NCE, ROI-DCF, V9 anti-overfit)
**Hardware:** NVIDIA H100 80GB HBM3 (CUDA 12.8, bf16 mixed precision)

---

## Phase 2+: Novel Contributions

### Core Innovations (4/4 Implemented)

1. **vMF-NCE Loss (Bessel-Free)**
   - Geometry-correct contrastive loss on S^{d-1}
   - Softplus kappa parameterization (v7+), label smoothing (v9)
   - Implementation: `src/fmri2img/losses/vmf_nce.py`
   - Tests: `tests/test_vmf.py`

2. **ROI-Tokenized Transformer Encoder**
   - Brain-region-aware architecture with attention-based interpretability
   - Stochastic Depth via DropPath (v9)
   - Implementation: `src/fmri2img/models/roi_transformer.py`
   - Multi-subject variant: `src/fmri2img/models/multi_subject_encoder.py`

3. **ROI Directional Consensus Fusion (ROI-DCF)**
   - Per-ROI vMF experts with spherical weighted mean consensus
   - Produces dual uncertainty: kappa (concentration) + delta (disagreement)
   - Implementation: `src/fmri2img/models/roi_dcf.py`

4. **Decomposed Uncertainty-Aware CFG (DUA-CFG)**
   - kappa -> guidance scale, delta -> ensemble size and steps
   - Implementation: `src/fmri2img/inference/decomposed_ua_cfg.py`
   - Tests: `tests/test_decomposed_ua_cfg.py`

### V9 Innovations (New)

| Module | Location | Tests |
|--------|----------|-------|
| Stochastic Depth (DropPath) | `src/fmri2img/models/roi_transformer.py` | `tests/test_v9_features.py` |
| Contrastive Projection Head | `src/fmri2img/models/projection_head.py` | `tests/test_v9_features.py` |
| R-Drop (vMF symmetric KL) | `src/fmri2img/losses/vmf_nce.py` | `tests/test_v9_features.py` |
| CSLS Retrieval Correction | `src/fmri2img/eval/embedding_eval.py` | `tests/test_v9_features.py` |
| MC-Dropout TTA | `scripts/training/train_unified.py` | `tests/test_v9_features.py` |
| Kappa-Weighted Averaging | `scripts/training/train_unified.py` | `tests/test_v9_features.py` |
| bf16 Mixed Precision | `scripts/training/train_unified.py` | -- |

### Additional Modules

| Module | Location | Tests |
|--------|----------|-------|
| vMF Mixture Sampling | `src/fmri2img/inference/vmf_mixture.py` | `tests/test_vmf_mixture.py` |
| Risk-Coverage Curves | `src/fmri2img/eval/vmf_risk_coverage.py` | `tests/test_vmf_risk_coverage.py` |
| Noise-Ceiling Normalization | `src/fmri2img/eval/ceiling_normalized_eval.py` | -- |
| SoftCLIP / vMF-SoftCLIP Loss | `src/fmri2img/losses/softclip.py` | `tests/test_softclip.py` |
| Hierarchical CLIP Loss | `src/fmri2img/losses/hierarchical_clip_loss.py` | -- |
| CKA Loss | `src/fmri2img/losses/cka_loss.py` | -- |
| Multi-Subject Dataset | `src/fmri2img/data/multi_subject_dataset.py` | `tests/test_multi_subject.py` |
| Multi-Layer CLIP Cache | `scripts/build/build_multilayer_clip_cache.py` | -- |

### Experiment Configurations

The ablation ladder runs **B v4 + N v5-v9** (`run_ablation_ladder.sh`):

| Config | Type | Status | R@1 (subj01) |
|--------|------|--------|--------------|
| `B0v4_deterministic.yaml` | Strong deterministic baseline | Complete | ~22% |
| `B1v4_gaussian.yaml` | Probabilistic Gaussian baseline | Complete | ~22% |
| `N1v5_vmf_nce.yaml` | vMF-NCE (tau=1.0) | Complete | 39.9% |
| `N1v7_vmf_nce.yaml` | vMF-NCE (softplus kappa, fused targets) | Complete | **49%** |
| `N3v8_roi_dcf.yaml` | ROI-DCF + hierarchical CLIP + CKA | Complete | **50.3%** |
| `N4v8_full_system.yaml` | Full system + SPCL | Complete | **50.8%** |
| `N1v9_vmf_nce.yaml` | V9: proj head, R-Drop, CSLS, TTA, bf16 | **Ready** | Pending |
| `N2v9_roi_transformer.yaml` | V9: DropPath, proj head, CSLS, TTA, bf16 | **Ready** | Pending |
| `N3v9_roi_dcf.yaml` | V9: anti-overfit, proj head, CSLS, TTA, bf16 | **Ready** | Pending |
| `N4v9_full_system.yaml` | V9: flagship, SPCL, CSLS, TTA, bf16 | **Ready** | Pending |

---

## Phase 1: Foundation (Completed)

Results archived in `experimental_results/exp001_baseline_ultimate/` and `RaportPaper3/`.

| Component | Location |
|-----------|----------|
| Soft Reliability Weighting | `src/fmri2img/preprocessing/soft_reliability.py` |
| InfoNCE Contrastive Loss | `src/fmri2img/losses/contrastive.py` |
| MC Dropout Uncertainty | `src/fmri2img/eval/uncertainty.py` |
| Gaussian NLL / Gaussian-NCE | `src/fmri2img/losses/gaussian.py`, `gaussian_nce.py` |

---

## Training Infrastructure

| Feature | Status |
|---------|--------|
| Mixed precision (AMP fp16 + bf16) | Implemented (`mixed_precision_dtype: "bf16"` for H100) |
| Gradient accumulation | Implemented (v9: 8 steps, eff. batch 1024) |
| Checkpoint resume | Implemented |
| Early stopping | Implemented (on R@1, patience 30) |
| LR scheduling (cosine + warmup) | Implemented |
| Gradient clipping | Implemented (max_norm=1.0) |
| Per-session z-scoring | Implemented |
| MixCo + SoftCLIP | Implemented (simultaneous from start) |
| EMA (Exponential Moving Average) | Implemented (decay=0.999) |
| fMRI noise augmentation | Implemented (std=0.1) |
| Voxel dropout | Implemented (0.1) |
| R-Drop regularization | Implemented (v9, weight=0.5) |
| MC-Dropout TTA | Implemented (v9, 8 samples) |
| CSLS retrieval evaluation | Implemented (v9, k=10) |

---

## What to Run

### V9 Ablation (H100)

```bash
# Full V9 N-series (4 experiments)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N9 > ablation_v9.log 2>&1 &
tail -f ablation_v9.log

# Without checkpoints (save disk space)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N9 SAVE_CKPT=0 > ablation_v9.log 2>&1 &
```

### After Training

```bash
# Aggregate results
python3 scripts/evaluation/aggregate_ablation.py \
    --results-dir experimental_results \
    --subjects subj01
```
