# Implementation Status

**Last Updated:** March 2026
**Current Phase:** V15 Fix the Fundamentals (PCR + large batch + z-scoring fix)
**Hardware:** NVIDIA H100 80GB HBM3 (CUDA 12.8, bf16 mixed precision)

---

## Phase 2+: Novel Contributions

### Core Innovations (4/4 Implemented)

1. **vMF-NCE Loss (Bessel-Free)**
   - Geometry-correct contrastive loss on S^{d-1}
   - Softplus kappa parameterization (v7+), label smoothing (v9)
   - CSLS training loss + ISF anti-hubness (v14)
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

### V15 Innovations (Current)

| Module | Location | Description |
|--------|----------|-------------|
| Multi-subject z-scoring fix | `scripts/training/train_unified.py` | Per-subject per-session z-scoring for `MultiSubjectPreextractedDataset` |
| CLIP embedding PCR re-enabled | `preprocessing.enabled: true` | center_pcr k=4 removes dominant PCs causing hubness |
| Large batch contrastive | `training.batch_size: 256` | 4x more in-batch negatives (was 64) |
| Validation prediction saving | `scripts/training/train_unified.py` | Saves .npy files for `diagnose_embeddings.py` |

### V14 Innovations

| Module | Location | Description |
|--------|----------|-------------|
| Differentiable CSLS Training | `src/fmri2img/losses/vmf_nce.py` | CSLS correction on logit matrix before CE (anti-hubness) |
| Inverted Softmax (ISF) | `src/fmri2img/losses/vmf_nce.py` | Column-normalized CE penalizing hub targets |
| Direct Cosine Alignment | `src/fmri2img/losses/direct_alignment.py` | Per-sample absolute alignment signal |
| Configurable Checkpoint Metric | `scripts/training/train_unified.py` | `checkpoint_metric: csls_r@1` / `r@1` / `median_rank` |
| Embedding Diagnostics | `scripts/evaluation/diagnose_embeddings.py` | Hubness, similarity, kappa, and failure analysis |

### V13 Innovations

| Module | Location | Description |
|--------|----------|-------------|
| MSE Regression (MindEye-style) | `scripts/training/train_unified.py` | Per-sample regression from epoch 1 |
| Hierarchical CLIP Alignment | `src/fmri2img/losses/hierarchical_clip_loss.py` | BrainMCLIP-style per-tier ROI supervision |
| Two-Stage Training | `scripts/training/train_unified.py` | Contrastive (Stage 1) -> MSE-heavy (Stage 2) |

### Additional Modules

| Module | Location | Tests |
|--------|----------|-------|
| Stochastic Depth (DropPath) | `src/fmri2img/models/roi_transformer.py` | `tests/test_v9_features.py` |
| Contrastive Projection Head | `src/fmri2img/models/projection_head.py` | `tests/test_v9_features.py` |
| R-Drop (vMF symmetric KL) | `src/fmri2img/losses/vmf_nce.py` | `tests/test_v9_features.py` |
| CSLS Retrieval Correction | `src/fmri2img/eval/embedding_eval.py` | `tests/test_v9_features.py` |
| MC-Dropout TTA | `scripts/training/train_unified.py` | `tests/test_v9_features.py` |
| Kappa-Weighted Averaging | `scripts/training/train_unified.py` | `tests/test_v9_features.py` |
| bf16 Mixed Precision | `scripts/training/train_unified.py` | -- |
| vMF Mixture Sampling | `src/fmri2img/inference/vmf_mixture.py` | `tests/test_vmf_mixture.py` |
| Risk-Coverage Curves | `src/fmri2img/eval/vmf_risk_coverage.py` | `tests/test_vmf_risk_coverage.py` |
| Noise-Ceiling Normalization | `src/fmri2img/eval/ceiling_normalized_eval.py` | -- |
| SoftCLIP / vMF-SoftCLIP Loss | `src/fmri2img/losses/softclip.py` | `tests/test_softclip.py` |
| Hierarchical CLIP Loss | `src/fmri2img/losses/hierarchical_clip_loss.py` | -- |
| CKA Loss | `src/fmri2img/losses/cka_loss.py` | -- |
| Multi-Subject Dataset | `src/fmri2img/data/multi_subject_dataset.py` | `tests/test_multi_subject.py` |
| Multi-Layer CLIP Cache | `scripts/build/build_multilayer_clip_cache.py` | -- |

### Experiment Results and Configurations

| Config | Type | Status | Raw R@1 | CSLS R@1 |
|--------|------|--------|---------|----------|
| `B0v4_deterministic.yaml` | Strong deterministic baseline | Complete | ~22% | -- |
| `B1v4_gaussian.yaml` | Probabilistic Gaussian baseline | Complete | ~22% | -- |
| `N1v5_vmf_nce.yaml` | vMF-NCE (tau=1.0) | Complete | 39.9% | -- |
| `N1v7_vmf_nce.yaml` | vMF-NCE (softplus kappa, fused targets) | Complete | 49% | -- |
| `N3v8_roi_dcf.yaml` | ROI-DCF + hierarchical CLIP + CKA | Complete | 50.3% | -- |
| `N4v8_full_system.yaml` | Full system + SPCL | Complete | 50.8% | -- |
| `N1v10_vmf_nce.yaml` | V10: wider MLP + hard neg + model soup | Complete | ~48% | ~54% |
| `N4v10_full_system.yaml` | V10: wider ROI-DCF + hard neg + model soup | Complete | ~50.8% | -- |
| `N1v13_vmf_nce.yaml` | V13: MSE regression + contrastive | Complete | **~45-46%** | **~54%** |
| `N2v13_roi_transformer.yaml` | V13: ROI Transformer + MSE | Complete | ~44% | ~51-52% |
| `N3v13_roi_dcf.yaml` | V13: ROI-DCF + MSE + hierarchical CLIP | Complete | ~45% | **~54-55%** |
| `N4v13_full_system.yaml` | V13: flagship + MSE + hierarchical CLIP | Complete | ~45-46% | ~53-54% |
| `N1v14_vmf_nce.yaml` | V14: anti-hubness + CSLS checkpoint | Complete | ~43% | ~52-54% |
| `N2v14_roi_transformer.yaml` | V14: anti-hubness + CSLS checkpoint | Complete | ~38-40% | ~52-55% |
| `N3v14_roi_dcf.yaml` | V14: anti-hubness + hierarchical + CSLS ckpt | Complete | ~40%+ | ~52-55% |
| `N4v14_full_system.yaml` | V14: flagship anti-hubness + CSLS checkpoint | Complete | **~44-45%** | **~55-58%** |
| `N1v15_vmf_nce.yaml` | V15: PCR + large batch + simplified loss | **Ready** | Pending | Pending |
| `N2v15_roi_transformer.yaml` | V15: PCR + large batch + z-scoring fix | **Ready** | Pending | Pending |
| `N3v15_roi_dcf.yaml` | V15: PCR + large batch + z-scoring fix + hier | **Ready** | Pending | Pending |
| `N4v15_full_system.yaml` | V15: flagship PCR + large batch + z-scoring fix | **Ready** | Pending | Pending |

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
| Gradient accumulation | Implemented (v15: 2 steps with batch 256, eff. batch 512) |
| Checkpoint resume | Implemented |
| Early stopping | Implemented (configurable metric: r@1, csls_r@1, median_rank) |
| LR scheduling (cosine + warmup) | Implemented |
| Gradient clipping | Implemented (max_norm=1.0) |
| Per-session z-scoring | Implemented (v15: fixed for multi-subject datasets) |
| MixCo + SoftCLIP | Implemented (simultaneous from start) |
| EMA (Exponential Moving Average) | Implemented (decay=0.999) |
| fMRI noise augmentation | Implemented (std=0.1) |
| Voxel dropout | Implemented (0.1) |
| Two-stage training | Implemented (v12+: contrastive -> MSE-heavy at epoch 80) |
| MSE regression loss | Implemented (v13+: from epoch 1, weight 1.0) |
| CSLS training loss | Implemented (v14: differentiable CSLS on logit matrix) |
| Inverted softmax (ISF) | Implemented (v14: column-normalized CE, weight 0.3) |
| Direct alignment loss | Implemented (v11+: per-sample cosine alignment) |
| Configurable checkpoint metric | Implemented (v14: csls_r@1 / r@1 / median_rank) |
| MC-Dropout TTA | Implemented (v9, 8 samples) |
| CSLS retrieval evaluation | Implemented (v9, k=10) |
| Model soup | Implemented (v10, top-5 checkpoints) |

---

## Diagnostic Tools

| Tool | Location | Description |
|------|----------|-------------|
| Embedding diagnostics | `scripts/evaluation/diagnose_embeddings.py` | Hubness, similarity, kappa, failure analysis |
| Aggregation | `scripts/evaluation/aggregate_ablation.py` | Cross-experiment comparison tables |
| Val prediction saving | `scripts/training/train_unified.py` | Saves .npy after training for diagnostics (v15) |

```bash
# Run diagnostics on a trained model (requires V15+ for auto-saved .npy files)
python scripts/evaluation/diagnose_embeddings.py \
    --results-dir experimental_results/N1v15_vmf_nce/subj01
```

---

## What to Run

### V15 Ablation (H100)

```bash
# Full V15 N-series (4 experiments)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N15 > ablation_v15.log 2>&1 &
tail -f ablation_v15.log

# Without checkpoints (save disk space)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N15 SAVE_CKPT=0 > ablation_v15.log 2>&1 &
```

### After Training

```bash
# Aggregate results
python3 scripts/evaluation/aggregate_ablation.py \
    --results-dir experimental_results \
    --subjects subj01

# Run diagnostics on each experiment (V15 auto-saves .npy files)
for exp in N1v15_vmf_nce N2v15_roi_transformer N3v15_roi_dcf N4v15_full_system; do
    python3 scripts/evaluation/diagnose_embeddings.py \
        --results-dir experimental_results/${exp}/subj01
done
```
