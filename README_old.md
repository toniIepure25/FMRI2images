# fMRI-to-Image Reconstruction

This project implements fMRI-to-image reconstruction using the Natural Scenes Dataset (NSD), mapping brain activity to visual stimuli via CLIP embeddings.

## Key Components

### Phase 1: Canonical Index

- `src/fmri2img/data/nsd_index_builder.py` - Builds canonical Parquet index
- `src/fmri2img/data/nsd_index.py` - Index query interface
- `scripts/nsd_build_index_s3.py` - CLI for index building

### Phase 2: IO Layer

- `src/fmri2img/io/nsd_layout.py` - Centralized path management
- `src/fmri2img/io/s3.py` - Robust S3 data loaders (NIfTI, HDF5, CSV)

### Phase 3: Preprocessing Pipeline

- `src/fmri2img/data/preprocess.py` - Production-grade preprocessing with T0/T1/T2 transforms
- `scripts/nsd_fit_preproc.py` - CLI to fit preprocessing on training data
- **T0**: Per-volume z-score normalization (online)
- **T1**: Subject-level scaler + reliability/variance mask (fit on train, persist)
- **T2**: PCA to k components or ROI pooling

### Phase 4: ROI Pooling & CLIP Cache

- `src/fmri2img/data/roi.py` - ROI pooling for anatomical region analysis
- `src/fmri2img/data/clip_cache.py` - CLIP vision embeddings cache with Parquet storage
- `scripts/nsd_build_clip_cache.py` - CLI to build CLIP embeddings cache
- `scripts/test_roi.py` - Test script for ROI functionality

### Phase 5: Ridge Baseline ✨ **NEW**

- `src/fmri2img/models/ridge.py` - Ridge regression encoder (fMRI → CLIP)
- `src/fmri2img/eval/retrieval.py` - Retrieval evaluation metrics
- `scripts/train_ridge.py` - Full training pipeline with alpha selection
- `docs/RIDGE_BASELINE.md` - Comprehensive documentation

**Features**:

- L2-regularized linear regression with hyperparameter selection
- Validation-based alpha tuning (no test leakage)
- L2-normalized predictions for cosine similarity
- Retrieval@K evaluation (K=1,5,10) + ranking metrics
- Complete train/val/test splits
- Model persistence with save/load

Note: GLMdenoise betas are already denoised; this layer standardizes & reduces dimensionality.

## Important Notes

**Trial Order**: The `nsd_stim_info_merged.csv` file is a **stimulus catalog** indexed by `nsdId`, providing COCO metadata (cocoId, cocoSplit, shared1000, filename). It is **NOT** trial order information.

**True Trial Order**: Actual trial presentation order comes from per-subject session design files located at:

```
s3://natural-scenes-dataset/nsddata/ppdata/subjXX/behav/sessionYY/
```

**Beta dtype**: NIfTI betas may be stored as `int16` (or other). After slicing a single trial, cast to `float32` if your model expects it:
`vol = img.slicer[..., beta_index].get_fdata().astype('float32')`.

**Index layout**: Primary format is partitioned by subject at `nsd_index/subject=subjXX/index.parquet`. The single-file path in `configs/data.yaml: paths.index_file` is only a local fallback.

**S3 writes**: The public NSD bucket is read-only; examples that write Parquet to S3 require your own bucket + AWS credentials. The demo falls back to local Parquet automatically.

The canonical index properly maps `(subject, session, trial_in_session) → nsdId → beta_path` using these design files, not naive pairing.

## Usage

```bash
# Build index for subjects - output is partitioned by subject as:
# .../nsd_index/subject=subjXX/index.parquet
SUBJECTS="subj01 subj02" OUT_ROOT="data/indices/nsd_index" make index

# Fit preprocessing pipeline on training data
python scripts/nsd_fit_preproc.py --subject subj01 --k 4096 --reliability-thr 0.1

# Stream a few 3D volumes from S3 via the canonical index and build small batches (no model yet)
make train-smoke

# Test with preprocessing pipeline
python scripts/train_smoke.py --use-preproc --pca-k 4096

# Run tests
make test
```

### Preprocessing Pipeline

The preprocessing pipeline implements three transformation levels:

- **T0**: Per-volume z-score normalization (applied online during data loading)
- **T1**: Subject-level scaler with reliability masking (fitted on training data)
  - Computes voxel-wise mean/std from training trials using Welford's online algorithm
  - For stimuli with repeat presentations, computes test-retest correlation per voxel
  - Keeps only voxels above reliability threshold (default r ≥ 0.1)
  - Falls back to variance threshold when repeats unavailable
- **T2**: PCA dimensionality reduction to k components OR ROI pooling

Example preprocessing workflow:

```bash
# Fit standard preprocessing with PCA
python scripts/nsd_fit_preproc.py --subject subj01 --k 4096 --reliability-thr 0.1

# Fit preprocessing with ROI pooling instead of PCA
python scripts/nsd_fit_preproc.py --subject subj01 --roi-mode pool

# Test with preprocessing in data loading
python scripts/train_smoke.py --subject subj01 --use-preproc --pca-k 4096
python scripts/train_smoke.py --subject subj01 --roi-mode pool
```

### ROI Pooling

ROI pooling extracts anatomical region means from fMRI volumes:

```python
from fmri2img.data.roi import ROIPooler

# Initialize and fit ROI pooler
pooler = ROIPooler(subject="subj01", min_voxels=50)
pooler.fit(sample_beta_path)  # Auto-discovers ROI masks via NSDLayout

# Pool volume to ROI means
vol = load_volume()  # (H, W, D)
roi_means = pooler.pool(vol)  # (n_roi,) - mean per anatomical region
```

### CLIP Embeddings Cache

CLIP cache stores precomputed ViT-B/32 embeddings (512-dim) for NSD stimuli in a Parquet file with enforced schema:

- **nsdId**: int32 (NSD stimulus identifier)
- **clip512**: fixed-length list[float32, 512] (CLIP vision embedding)

**Image Loading**: Primary path is `nsd_stimuli.hdf5` via nsdId (fast, S3-backed). Falls back to COCO HTTP if HDF5 access fails and cocoId is available.

**Build Cache**:

```bash
# From partitioned index (recommended)
python scripts/build_clip_cache.py \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --cache outputs/clip_cache/clip.parquet \
    --batch 64 --device cuda --limit 256

# From index root with subject filter
python scripts/build_clip_cache.py \
    --index-root data/indices/nsd_index \
    --subject subj01 \
    --cache outputs/clip_cache/clip.parquet \
    --batch 128 --device cuda

# Resume is automatic - skips already-cached nsdIds
# Re-run same command to continue after interruption
```

**Use in Dataset**:

```python
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.data.torch_dataset import NSDIterableDataset

# Preferred: Fluent API (load() returns self)
clip_cache = CLIPCache("outputs/clip_cache/clip.parquet").load()
dataset = NSDIterableDataset(
    index_path_or_root="data/indices/nsd_index",
    subject="subj01",
    clip_cache=clip_cache  # Add CLIP embeddings to batch output
)

# Also supported: Pass path string directly
dataset = NSDIterableDataset(
    index_path_or_root="data/indices/nsd_index",
    subject="subj01",
    clip_cache="outputs/clip_cache/clip.parquet"  # String path
)

# Each batch now includes "clip" key with (512,) float32 L2-normalized array
for batch in dataset:
    fmri = batch["fmri"]      # (H,W,D) or (k,) after PCA
    clip = batch["clip"]      # (512,) CLIP embedding (L2 normalized)
```

### Ridge Baseline Training

Train a reproducible Ridge regression baseline to map fMRI → CLIP embeddings:

```bash
# Quick test (works with current k=4 PCA, uses 256 samples)
python scripts/train_ridge.py \
    --subject subj01 \
    --use-preproc \
    --clip-cache outputs/clip_cache/clip.parquet \
    --limit 256 \
    --alpha-grid "1,10"

# Full training via Makefile
make ridge

# Full training with custom config
python scripts/train_ridge.py \
    --index-root data/indices/nsd_index \
    --subject subj01 \
    --use-preproc \
    --clip-cache outputs/clip_cache/clip.parquet \
    --alpha-grid "0.1,1,3,10,30,100" \
    --limit 2048  # Remove for all data
```

**Output**:

- **Model**: `checkpoints/ridge/subj01/ridge.pkl` (loadable via `RidgeEncoder.load()`)
- **Report**: `outputs/reports/subj01/ridge_eval.json` (cosine, MSE, R@K metrics)

**Evaluation Metrics**:

- Cosine similarity (with ground truth)
- MSE loss
- Retrieval@1/5/10 (% queries with true image in top-K)
- Mean/median rank, MRR

See `docs/RIDGE_BASELINE.md` for complete documentation.
nsd_id = batch["nsdId"] # int

````

**Common Mistake**:
```python
# ❌ Don't do this (old API returned boolean):
# cache = CLIPCache(...).load()  # Returns self now, not bool!

# ✓ Correct (fluent API):
cache = CLIPCache("path/to/cache.parquet").load()
dataset = NSDIterableDataset(..., clip_cache=cache)

# ✓ Or use string path:
dataset = NSDIterableDataset(..., clip_cache="path/to/cache.parquet")
````

**API Reference**:

```python
from fmri2img.data.clip_cache import CLIPCache

# Fluent API - load() returns self
cache = CLIPCache(cache_path="outputs/clip_cache/clip.parquet").load()

# Check if loaded
assert cache.is_loaded  # Property

# Check if nsdId is cached
if cache.contains(nsd_id=12345):
    print("Already cached!")

# Get embeddings for multiple nsdIds (L2 normalized)
embeddings = cache.get([1, 2, 3])  # Returns: {1: array(512,), 2: array(512,), ...}

# Save new embeddings
import pandas as pd
rows = pd.DataFrame({
    "nsdId": [4, 5, 6],
    "clip512": [emb1.tolist(), emb2.tolist(), emb3.tolist()]
})
cache.save_rows(rows)  # Deduplicates on nsdId, enforces schema

# Get stats
stats = cache.stats()  # {"cache_size": N, "path": "..."}
```

**Implementation Details**:

- Uses PyArrow schema enforcement for type safety
- Deduplicates automatically on nsdId (keeps latest)
- Resume support: builder skips already-cached IDs
- Batch processing with GPU autocast for efficiency
- Snappy compression for compact storage

### Test Scripts

```bash
# Test ROI functionality
python scripts/test_roi.py
```

Primary format: partitioned Parquet per subject: nsd_index/subject=subjXX/index.parquet. The single-file path (paths.index_file) is only a local fallback.

```bash
# Demo IO layer
make sanity
```

## Important Notes

- **PyTorch IterableDataset** reads one 3D trial at a time via `img.slicer[..., beta_index]` to avoid loading full 4D NIfTI.

## Architecture

1. **Stimulus Catalog**: 73K COCO images with NSD metadata
2. **Session Designs**: Per-subject trial order and timing
3. **Beta Files**: 4D NIfTI files with GLMdenoise preprocessed fMRI
4. **Canonical Index**: Parquet mapping trials → stimuli → files
   - Extra columns:
     - `stimulus_repeat_count` – count of repeats for that nsdId up to current trial
     - `has_beta_data` – boolean availability flag for mapped beta file/index
     - `data_quality_flag` – optional QC status if exposed by design
5. **S3 Streaming**: Memory-efficient data access via fsspec
