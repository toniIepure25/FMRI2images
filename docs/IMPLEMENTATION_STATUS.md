# Implementation Status

**Last Updated:** March 2026
**Current Phase:** Phase 2 (ViT-L/14, 768-D, vMF-NCE, ROI-DCF)

---

## Phase 2: Novel Contributions

### Core Innovations (4/4 Implemented)

1. **vMF-NCE Loss (Bessel-Free)**
   - Geometry-correct contrastive loss on S^{d-1}
   - Implementation: `src/fmri2img/losses/vmf_nce.py`
   - Tests: `tests/test_vmf.py`

2. **ROI-Tokenized Transformer Encoder**
   - Brain-region-aware architecture with attention-based interpretability
   - Implementation: `src/fmri2img/models/roi_transformer.py`
   - Tests: integrated in model tests

3. **ROI Directional Consensus Fusion (ROI-DCF)**
   - Per-ROI vMF experts with spherical weighted mean consensus
   - Produces dual uncertainty: kappa (concentration) + delta (disagreement)
   - Implementation: `src/fmri2img/models/roi_dcf.py`

4. **Decomposed Uncertainty-Aware CFG (DUA-CFG)**
   - kappa -> guidance scale, delta -> ensemble size and steps
   - Implementation: `src/fmri2img/inference/decomposed_ua_cfg.py`
   - Tests: `tests/test_decomposed_ua_cfg.py`

### Additional Modules

| Module | Location | Tests |
|--------|----------|-------|
| vMF Mixture Sampling | `src/fmri2img/inference/vmf_mixture.py` | `tests/test_vmf_mixture.py` |
| Risk-Coverage Curves | `src/fmri2img/eval/vmf_risk_coverage.py` | `tests/test_vmf_risk_coverage.py` |
| Noise-Ceiling Normalization | `src/fmri2img/eval/ceiling_normalized_eval.py` | — |
| Frequency-ROI Module | `src/fmri2img/models/freq_roi.py` | `tests/test_freq_roi.py` |
| Cross-Subject Alignment | `src/fmri2img/models/cross_subject.py` | `tests/test_cross_subject.py` |
| Kappa Calibration | `src/fmri2img/eval/kappa_calibration.py` | in `tests/test_vmf.py` |
| SoftCLIP Loss | `src/fmri2img/losses/softclip.py` | `tests/test_softclip.py` |

### Experiment Configurations

The ablation ladder runs **B v4 + N v5** (`run_ablation_ladder.sh`):

| Config | Type | Status |
|--------|------|--------|
| `B0v4_deterministic.yaml` | Strong deterministic baseline | Ready |
| `B1v4_gaussian.yaml` | Probabilistic Gaussian baseline | Ready |
| `N1v5_vmf_nce.yaml` | vMF-NCE (novel distribution) | **v5: tau=1.0, kappa collapse fix** |
| `N2v5_roi_transformer.yaml` | ROI Transformer (novel architecture) | **v5: tau=1.0, d_model=768, 6 layers** |
| `N3v5_roi_dcf.yaml` | ROI-DCF consensus (novel fusion) | **v5: tau=1.0, kappa collapse fix** |
| `N4v5_full_system.yaml` | Full system (flagship) | **v5: tau=1.0, kappa collapse fix** |

Previous configs (v1-v4) remain in `configs/experiments/` for reproducibility. N-series v4 configs have a known kappa collapse bug (`tau=0.07` caps kappa gradient at ~5.6).

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

## System Components

### Models and Architecture

| Component | Location | Phase |
|-----------|----------|-------|
| UnifiedModel (MLP + decoders) | `src/fmri2img/models/unified_model.py` | 1+2 |
| VonMisesFisherDecoder | `src/fmri2img/models/vmf_decoder.py` | 2 |
| ROITransformerEncoder | `src/fmri2img/models/roi_transformer.py` | 2 |
| ROI-DCF Module | `src/fmri2img/models/roi_dcf.py` | 2 |
| Memory Queue | `src/fmri2img/contrastive/memory_queue.py` | 1+2 |

### Loss Functions

| Loss | Location | Used In |
|------|----------|---------|
| MSE + Cosine | `src/fmri2img/losses/` | B0 |
| InfoNCE | `src/fmri2img/losses/contrastive.py` | B0 |
| Gaussian NLL | `src/fmri2img/losses/gaussian.py` | B1 |
| Gaussian-NCE | `src/fmri2img/losses/gaussian_nce.py` | B1 |
| vMF-NCE | `src/fmri2img/losses/vmf_nce.py` | N1-N4 (tau=1.0 in v5) |
| Kappa Regularizer | `src/fmri2img/losses/vmf_nce.py` | N1-N4 v4 only (disabled in v5) |
| SoftCLIP KD | `src/fmri2img/losses/softclip.py` | All (B0-N4) |
| MixCo Augmentation | `src/fmri2img/losses/mixco.py` | All (B0-N4) |

### Preprocessing

| Component | Location |
|-----------|----------|
| Center + PCR (k=8) | `src/fmri2img/embedding_preproc.py` |
| Center + Whitening | `src/fmri2img/embedding_preproc.py` |

### Evaluation Suite

| Metric Category | Location |
|----------------|----------|
| Embedding (R@K, MRR, 2AFC, hubness) | `src/fmri2img/eval/embedding_eval.py` |
| Probabilistic (NLL, Energy Score, ECE) | `src/fmri2img/eval/probabilistic_eval.py` |
| Risk-Coverage (AURC, selective prediction) | `src/fmri2img/eval/vmf_risk_coverage.py` |
| Noise-Ceiling Normalization | `src/fmri2img/eval/ceiling_normalized_eval.py` |
| Kappa Calibration | `src/fmri2img/eval/kappa_calibration.py` |

### Training Infrastructure

| Feature | Status |
|---------|--------|
| Mixed precision (AMP) | Implemented |
| Gradient accumulation | Implemented |
| Checkpoint resume | Implemented |
| Early stopping | Implemented |
| LR scheduling (cosine + warmup) | Implemented |
| Gradient clipping | Implemented |
| Per-session z-scoring | Implemented (`zscore_mode: per_session`) |
| MixCo -> SoftCLIP phase schedule | Implemented (1/3 MixCo, 2/3 SoftCLIP; or simultaneous in v5) |
| EMA (Exponential Moving Average) | Implemented (`ema.enabled: true, decay: 0.999`) |
| fMRI noise augmentation | Implemented (`fmri_noise_std: 0.1`) |
| Voxel dropout | Implemented (`voxel_dropout: 0.1`) |
| TensorBoard logging | Implemented |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/training/train_unified.py` | Main training entry point |
| `scripts/training/run_ablation_ladder.sh` | Full ablation runner |
| `scripts/reconstruction/decode_diffusion.py` | Image reconstruction |
| `scripts/evaluation/evaluate_experiment.py` | Post-training evaluation |
| `scripts/evaluation/compare_experiments.py` | Cross-experiment comparison |

---

## What to Run

### Immediate (Phase 2 Experiments)

```bash
# Full ablation ladder (B0v4, B1v4, N1v5, N2v5, N3v5, N4v5)
nohup make ablation SUBJECTS="subj01" GPU=0 > ablation_v5.log 2>&1 &
tail -f ablation_v5.log

# Or individual experiments
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1v5_vmf_nce.yaml --gpu 0
```

### After Training

```bash
# Evaluate
python3 scripts/evaluation/evaluate_experiment.py \
    --config configs/experiments/N1v5_vmf_nce.yaml \
    --checkpoint experimental_results/N1v5_vmf_nce/subj01/checkpoint_best.pt

# Aggregate ablation results
python3 scripts/evaluation/aggregate_ablation.py \
    --results-dir experimental_results \
    --subjects subj01

# Generate reconstructions
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint experimental_results/N4v5_full_system/subj01/checkpoint_best.pt
```
