# PHASE 0: Current Pipeline Architecture Mapping

**Date**: November 25, 2025  
**Status**: Complete ✅  
**Purpose**: Comprehensive map of existing codebase before implementing advanced features

---

## Executive Summary

This document maps the **complete current pipeline** for fMRI → CLIP → Diffusion image reconstruction on NSD dataset. The system is already quite sophisticated, with a two-stage encoder, multi-loss training, best-of-N sampling, and BOI-lite refinement.

### Key Findings

✅ **Already Implemented:**
- Two-stage residual encoder (Stage1: fMRI→latent, Stage2: latent→CLIP)
- Multi-objective loss (MSE + Cosine + InfoNCE)
- Self-supervised pretraining (masked/denoising autoencoder)
- Best-of-N diffusion sampling
- BOI-lite iterative refinement
- Multi-target decoder (CLIP + IP-Adapter tokens + SD latent)
- Comprehensive evaluation suite
- Full 30K trial support in configs

⚠️ **Potential Issues Identified:**
1. **CLIP cache** may not be built for all 30K trials yet (needs verification)
2. **Multi-target decoder** exists but may not be fully integrated into training/inference
3. **PCA consistency** across preprocessing, training, and decoding needs verification
4. Old MLP encoder path still present (good for backward compatibility)

---

## 1. Data & Indexing

### 1.1 Index Builder & Loader

**Location**: `src/fmri2img/data/nsd_index_builder.py`

- **Class**: `NSDIndexBuilder`
- **Purpose**: Builds canonical Parquet index mapping `(subject, session, trial)` → `(nsdId, beta_path, beta_index)`
- **Uses**: Session design files for true trial order (not naive zipping)
- **Metadata**: COCO IDs, shared1000 flags, repeat indicators

**Location**: `src/fmri2img/data/nsd_index_reader.py`

- **Function**: `read_subject_index(index_root, subject)` → DataFrame
- **Supports**: Partitioned Parquet directories

### 1.2 Dataset Loader

**Location**: `src/fmri2img/data/torch_dataset.py`

- **Class**: `NSDIterableDataset`
- **Features**:
  - Streams from Parquet index
  - Optional preprocessing (T0/T1/T2)
  - Optional CLIP cache integration
  - Worker-aware sharding for DataLoader
- **Yields**: `{"fmri": (k,), "nsdId": int, "clip": (512,), ...}`

### 1.3 Preprocessing Pipeline

**Location**: `src/fmri2img/data/preprocess.py`

**Class**: `NSDPreprocessor`

**Three-Stage Transformation**:

1. **T0 (Per-Volume Z-Score)**: Online normalization
   - `(vol - vol.mean()) / (vol.std() + 1e-8)`
   - Applied at load time, always

2. **T1 (Subject-Level Standardization + Reliability Masking)**:
   - Voxelwise mean/std from training data only
   - Split-half reliability masking (Pearson r ≥ threshold)
   - Falls back to variance threshold if insufficient repeats
   - **Critical**: Only train_df used for fitting (no leakage)

3. **T2 (PCA Dimensionality Reduction)**:
   - IncrementalPCA for compression
   - Auto-capping: `k_eff = min(k, n_train_samples, n_features_kept)`
   - Saves: `pca_components.npy`, `pca_mean.npy`, `meta.json`

**Artifacts**: `outputs/preproc/{subject}/`
- `scaler_mean.npy`, `scaler_std.npy`
- `reliability_mask.npy`, `voxel_indices.npy`
- `pca_components.npy`, `pca_mean.npy`, `meta.json`

**PCA Configuration**:
- `sota_two_stage.yaml`: `pca_k: 512`
- `production_improved.yaml`: `pca_k: 100`

### 1.4 CLIP Cache

**Location**: `src/fmri2img/data/clip_cache.py`

- **Class**: `CLIPCache`
- **Storage**: Parquet file with `(nsdId, clip_embedding)` mapping
- **Builder**: `scripts/build_clip_cache.py`

**Build Script Features**:
- Loads images from HDF5 (nsd_stimuli.hdf5) via nsdId
- COCO HTTP fallback if HDF5 fails
- Batching, GPU support, resume capability
- Uses `configs/clip.yaml` for model config

**⚠️ ISSUE TO VERIFY**: Does the CLIP cache cover all ~30K trials for subj01?

---

## 2. Encoders & Models

### 2.1 Two-Stage Encoder Architecture

**Location**: `src/fmri2img/models/encoders.py`

**Classes**:

1. **`ResidualBlock`**: Pre-normalized residual block
   - `LayerNorm → Linear → GELU → Dropout → Linear → Dropout → (+x)`
   - Pre-normalization improves training stability

2. **`ResidualMLPEncoder`** (Stage 1):
   - Input: PCA fMRI vector (e.g., 512-D)
   - Output: Latent brain representation (e.g., 768-D)
   - Architecture:
     - Input projection: `Linear(input_dim, latent_dim) → GELU → Dropout`
     - N residual blocks (default: 4)
     - Final LayerNorm

3. **`CLIPMappingHead`** (Stage 2):
   - Input: Latent representation (768-D)
   - Output: L2-normalized CLIP embedding (512-D)
   - Variants: "linear" or "mlp" (with hidden layer)

4. **`TwoStageEncoder`** (Complete Model):
   - Combines Stage 1 + Stage 2
   - Methods: `freeze_stage1()`, `unfreeze_stage1()`
   - Supports staged training

5. **`SelfSupervisedPretrainer`**:
   - Pretrains Stage 1 with masked or denoising autoencoder
   - Reconstructs original fMRI from corrupted input

**Save/Load Functions**:
- `save_two_stage_encoder(model, path, config, metrics)`
- `load_two_stage_encoder(path, device)`

### 2.2 Multi-Target Decoder

**Location**: `src/fmri2img/models/multi_target_decoder.py`

**Classes**:

1. **`IPAdapterTokenHead`**:
   - Predicts N token embeddings (e.g., 16 × 1024-D)
   - For IP-Adapter-style conditioning in diffusion

2. **`SDLatentHead`**:
   - Predicts SD VAE latent (4 × 64 × 64)
   - Coarse structural prior

3. **`MultiTargetDecoder`**:
   - Combines all heads:
     - Global CLIP (512-D)
     - IP-Adapter tokens (N × 1024-D)
     - SD latent (4 × 64 × 64)

4. **`MultiTaskLoss`**:
   - Combines losses for all targets
   - Configurable weights

**⚠️ INTEGRATION STATUS**: This module exists but may not be fully integrated into `train_two_stage.py` and `decode_two_stage.py` yet.

### 2.3 Old MLP Encoder

**Location**: `src/fmri2img/models/mlp.py`, `mlp_improved.py`

- Simple MLP baseline (kept for backward compatibility)
- Training: `scripts/train_mlp.py`
- **Status**: Separate path from SOTA two-stage encoder

### 2.4 Ridge Baseline

**Location**: `src/fmri2img/models/ridge.py`

- Linear ridge regression baseline
- Training: `scripts/train_ridge.py`

### 2.5 Encoding Model (CLIP/Image → fMRI)

**Location**: `src/fmri2img/models/encoding_model.py`

**Classes**:

1. **`ImageEncoder`**: Pretrained vision backbones (CLIP/DINO/ResNet)
2. **`EncodingModel`**: Image → fMRI PCA prediction
   - Used for BOI-lite refinement
   - Not yet used for cycle-consistency loss (Phase 2 task)

---

## 3. Training

### 3.1 Two-Stage Training Script

**Location**: `scripts/train_two_stage.py`

**Features**:
- Loads data via `NSDIterableDataset` + `extract_features_and_targets()`
- Supports self-supervised pretraining
- Staged training (freeze Stage 1, train Stage 2)
- Multi-objective loss (MSE + Cosine + InfoNCE)
- Early stopping, gradient clipping
- Saves checkpoints with architecture config

**Usage**:
```bash
python scripts/train_two_stage.py \
    --subject subj01 \
    --use-preproc --pca-k 512 \
    --latent-dim 768 --n-blocks 4 \
    --head-type mlp --head-hidden 512 \
    --batch-size 128 --epochs 50
```

**Config**: `configs/sota_two_stage.yaml`

### 3.2 Loss Functions

**Location**: `src/fmri2img/training/losses.py`

**Functions**:

1. **`mse_loss(pred, target)`**: L2 distance in CLIP space
2. **`cosine_loss(pred, target)`**: 1 - cosine_similarity
3. **`info_nce_loss(pred, target, temperature)`**: Contrastive loss
   - Positive pairs: (pred[i], target[i])
   - Negative pairs: (pred[i], target[j]) for j ≠ i

**Class**: `MultiLoss`
- Combines all three losses with configurable weights
- `forward(pred, target)` → total_loss, loss_dict

**Config Example**:
```yaml
loss:
  mse_weight: 0.3
  cosine_weight: 0.3
  info_nce_weight: 0.4
  temperature: 0.05
```

---

## 4. Diffusion & Generation

### 4.1 Basic Decoding

**Location**: `scripts/decode_two_stage.py`

- Loads TwoStageEncoder from checkpoint
- Predicts CLIP embeddings from fMRI
- Generates images via Stable Diffusion
- Simple projection: Linear(512 → 1024) for SD 2.1

**Usage**:
```bash
python scripts/decode_two_stage.py \
    --ckpt checkpoints/two_stage/subj01/two_stage_best.pt \
    --subject subj01 \
    --output-dir outputs/reconstructions
```

### 4.2 Advanced Diffusion Strategies

**Location**: `src/fmri2img/generation/advanced_diffusion.py`

**Functions**:

1. **`generate_best_of_n(pipe, clip_embedding, n=8, ...)`**:
   - Generates N candidates with different seeds
   - Encodes each with CLIP
   - Selects best based on cosine similarity
   - Inspired by MindEye2

2. **`refine_with_boi_lite(pipe, clip_embedding, encoding_model, ...)`**:
   - Iterative refinement with encoding model feedback
   - Generates candidates, predicts fMRI, scores brain alignment
   - Inspired by Brain-Diffuser BOI

3. **`generate_with_all_strategies(pipe, clip_embedding, ...)`**:
   - Unified interface for all strategies
   - Returns: single, best_of_n, boi_lite images

**Config Example**:
```yaml
diffusion:
  best_of_n: 8  # Set >1 to enable
  boi_lite:
    enabled: true
    steps: 3
    candidates_per_step: 4
```

---

## 5. Evaluation & Ablation

### 5.1 Retrieval Evaluation

**Location**: `scripts/eval_retrieval.py`

**Features**:
- Retrieval@K (K=1, 5, 10, 20, 50)
- Mean/median rank, MRR
- Multiple gallery options (train/val/test/full/shared1000)
- Supports Ridge, MLP, TwoStage encoders

**Usage**:
```bash
python scripts/eval_retrieval.py \
    --subject subj01 \
    --encoder-type two_stage \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --split test --gallery full
```

### 5.2 Comprehensive Evaluation

**Location**: `scripts/eval_comprehensive.py`

**Features**:
- NSD Shared 1000 standard benchmark
- Multi-strategy generation (single/best-of-N/BOI-lite)
- Perceptual metrics: CLIPScore, SSIM, LPIPS
- Brain alignment: encoding model correlation
- Statistical testing across strategies

**Usage**:
```bash
python scripts/eval_comprehensive.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --encoder-type two_stage \
    --strategies single best_of_8 boi_lite
```

### 5.3 Ablation Driver

**Location**: `scripts/ablation_driver.py`

- Automates ablation studies
- Varies loss weights, architecture params, etc.
- Generates comparison reports

### 5.4 Reporting Tools

**Locations**:
- `scripts/generate_report.py`: LaTeX/Markdown tables
- `scripts/generate_comparison_gallery.py`: Visual comparisons
- `scripts/compare_evals.py`: Statistical comparisons

---

## 6. Configuration System

### 6.1 Main Configs

**`configs/sota_two_stage.yaml`**:
```yaml
dataset:
  max_trials: 30000  # All trials
  train_ratio: 0.80  # 24K train
  val_ratio: 0.10    # 3K val
  test_ratio: 0.10   # 3K test

preprocessing:
  pca_k: 512  # High dimensionality

encoder:
  type: "two_stage"
  latent_dim: 768
  n_blocks: 4
  head_type: "mlp"

loss:
  mse_weight: 0.3
  cosine_weight: 0.3
  info_nce_weight: 0.4
  temperature: 0.05
```

**`configs/production_improved.yaml`**:
- Similar but with `pca_k: 100` (lower)

**`configs/clip.yaml`**:
- CLIP model configuration
- Single source of truth for CLIP cache builder

**`configs/data.yaml`**:
- NSD S3 paths
- Dataset structure

---

## 7. Production vs SOTA Paths

### 7.1 SOTA Path (Recommended)

**Config**: `configs/sota_two_stage.yaml`

**Pipeline**:
1. Preprocessing: `scripts/nsd_fit_preproc.py` (or fast variants)
2. CLIP cache: `scripts/build_clip_cache.py`
3. Training: `scripts/train_two_stage.py`
4. Decoding: `scripts/decode_two_stage.py`
5. Evaluation: `scripts/eval_retrieval.py`, `eval_comprehensive.py`

**Documentation**:
- `SOTA_QUICK_START.md`: Step-by-step guide
- `SOTA_IMPLEMENTATION_SUMMARY.md`: Technical details
- `README_SOTA.md`: Current file (overview)

### 7.2 Production Path (Older, Simpler)

**Config**: `configs/production_improved.yaml`

**Pipeline**:
- Uses MLP encoder (`scripts/train_mlp.py`)
- Lower PCA dimensionality (k=100)
- Simpler single-stage architecture

**Script**: `scripts/run_production.sh`

---

## 8. Identified Issues & Inconsistencies

### 8.1 CLIP Cache Coverage

**Issue**: Need to verify that CLIP cache covers all ~30K trials.

**Check**:
```bash
# Count rows in CLIP cache
python -c "import pandas as pd; df = pd.read_parquet('cache/clip_embeddings/subj01_clip_cache.parquet'); print(f'CLIP cache: {len(df)} samples')"

# Count trials in index
python -c "import pandas as pd; df = pd.read_parquet('data/indices/nsd_index/subject=subj01/index.parquet'); print(f'Index: {len(df)} trials')"
```

**Action**: If mismatch, rebuild CLIP cache with `scripts/build_clip_cache.py` (no limit).

### 8.2 PCA Consistency

**Issue**: PCA `k` must be consistent across:
1. Preprocessing fit (`nsd_fit_preproc.py --pca-k 512`)
2. Training (`train_two_stage.py --pca-k 512`)
3. Decoding (`decode_two_stage.py` loads from checkpoint)

**Current State**:
- `sota_two_stage.yaml`: `pca_k: 512` ✅
- `production_improved.yaml`: `pca_k: 100` ✅
- Training script reads from config ✅
- Decoding script loads from checkpoint metadata ✅

**Verdict**: Likely consistent if using same config throughout.

### 8.3 Multi-Target Decoder Integration

**Issue**: `multi_target_decoder.py` exists but may not be integrated.

**Check**:
- `train_two_stage.py`: Does it use `MultiTargetDecoder`? **NO** (uses `TwoStageEncoder`)
- `decode_two_stage.py`: Does it decode tokens/latent? **NO** (only CLIP)

**Status**: Multi-target decoder is implemented but **not yet integrated** into main training/inference pipeline.

**Action**: Phase 4 will integrate this properly.

### 8.4 Encoding Model for Cycle Loss

**Issue**: Encoding model exists (`encoding_model.py`) but is only used for BOI-lite, not cycle-consistency loss.

**Status**: Phase 2 will add brain-consistency loss using this model.

### 8.5 Dataset Filtering by CLIP Cache

**Check**: Does `NSDIterableDataset` filter by CLIP cache availability?

**Code**: `torch_dataset.py`:
```python
if self.clip_cache is not None:
    # Adds 'clip' key to output dict
    # Does NOT filter rows without CLIP embeddings
```

**Status**: No explicit filtering. If a trial lacks CLIP embedding, it may cause errors during training.

**Action**: Phase 1 will ensure all training samples have CLIP embeddings.

---

## 9. File Structure Summary

```
src/fmri2img/
├── data/
│   ├── clip_cache.py               # CLIP embedding cache
│   ├── nsd_index_builder.py        # Index construction
│   ├── nsd_index_reader.py         # Index loading
│   ├── preprocess.py               # T0/T1/T2 preprocessing
│   ├── reliability.py              # Split-half reliability
│   ├── torch_dataset.py            # PyTorch dataset
│   └── ...
├── models/
│   ├── encoders.py                 # Two-stage encoder (SOTA)
│   ├── multi_target_decoder.py     # Multi-target decoder (not integrated)
│   ├── encoding_model.py           # Image→fMRI encoder (for BOI-lite)
│   ├── mlp.py, mlp_improved.py     # Old MLP encoder
│   ├── ridge.py                    # Ridge baseline
│   └── ...
├── training/
│   ├── losses.py                   # Multi-objective loss
│   └── ...
├── generation/
│   ├── advanced_diffusion.py       # Best-of-N, BOI-lite
│   └── ...
├── eval/
│   ├── retrieval.py                # Retrieval metrics
│   └── ...
└── ...

scripts/
├── train_two_stage.py              # SOTA training (main)
├── train_mlp.py                    # Old MLP training
├── train_ridge.py                  # Ridge baseline
├── decode_two_stage.py             # Decoding (main)
├── build_clip_cache.py             # Build CLIP cache
├── nsd_fit_preproc.py              # Fit preprocessing
├── eval_retrieval.py               # Retrieval evaluation
├── eval_comprehensive.py           # Full evaluation suite
├── ablation_driver.py              # Ablation studies
├── generate_report.py              # Reporting
└── ...

configs/
├── sota_two_stage.yaml             # SOTA config (main)
├── production_improved.yaml        # Production config
├── clip.yaml                       # CLIP model config
├── data.yaml                       # NSD S3 paths
└── ...
```

---

## 10. Next Steps (Phase 1+)

### Phase 1: Sanity & Full-Data Usage
1. Verify CLIP cache covers all 30K trials
2. Verify PCA consistency across pipeline
3. Ensure two-stage encoder is default SOTA path
4. Remove/guard any temporary limits (750/9000 samples)

### Phase 2: Brain-Consistency Loss
1. Train CLIP→fMRI encoder
2. Add cycle-consistency loss to decoder training
3. Make it configurable via `brain_consistency_weight`

### Phase 3: Multi-Layer CLIP Supervision
1. Extract multi-layer CLIP features
2. Add decoder heads for each layer
3. Extend loss to supervise all layers

### Phase 4: Multi-Task Decoder (Image+Text CLIP)
1. Generate captions for NSD images
2. Encode captions with CLIP text tower
3. Add text CLIP head to decoder
4. Multi-task loss (image + text)

### Phase 5: Probabilistic Decoder
1. Output (mu, logvar) instead of point estimate
2. Reparameterization trick for sampling
3. KL divergence regularization
4. Uncertainty-aware best-of-N

### Phase 6: Evaluation Polish
1. Update eval scripts for all new features
2. Ablation presets
3. Statistical tests

### Phase 7: Documentation
1. Update all docs with new features
2. Type hints, docstrings
3. Config comments

---

## Summary

The current pipeline is **already quite advanced** and implements many SOTA techniques:
- ✅ Two-stage residual encoder with self-supervised pretraining
- ✅ Multi-objective loss (MSE + Cosine + InfoNCE)
- ✅ Best-of-N sampling and BOI-lite refinement
- ✅ Comprehensive evaluation suite
- ✅ Full 30K trial support in configs

**Main gaps** to address in subsequent phases:
1. Brain-consistency (cycle) loss
2. Multi-layer CLIP supervision
3. Multi-task decoding (image + text CLIP)
4. Probabilistic decoder with uncertainty
5. Full integration of multi-target decoder

The foundation is solid. Phases 1-7 will add innovative, research-grade enhancements.

---

**Phase 0 Status**: ✅ **COMPLETE**

Ready to proceed to **Phase 1: Sanity, Correctness, and Full-Data Usage**.
