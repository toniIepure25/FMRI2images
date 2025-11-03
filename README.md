# fMRI→CLIP→Diffusion Image Reconstruction Pipeline# fMRI-to-Image Reconstruction



[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)This project implements fMRI-to-image reconstruction using the Natural Scenes Dataset (NSD), mapping brain activity to visual stimuli via CLIP embeddings.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)## Key Components



> **Publication-ready pipeline for reconstructing natural images from fMRI brain activity using CLIP embeddings and Stable Diffusion.**### Phase 1: Canonical Index



This project implements a complete neural decoding pipeline that transforms fMRI voxel patterns into semantic CLIP embeddings, then uses these embeddings to guide Stable Diffusion image generation. The pipeline includes preprocessing, model training, reconstruction, and comprehensive evaluation with multiple retrieval gallery configurations.- `src/fmri2img/data/nsd_index_builder.py` - Builds canonical Parquet index

- `src/fmri2img/data/nsd_index.py` - Index query interface

---- `scripts/nsd_build_index_s3.py` - CLI for index building



## 🎯 Overview### Phase 2: IO Layer



### Pipeline Architecture- `src/fmri2img/io/nsd_layout.py` - Centralized path management

- `src/fmri2img/io/s3.py` - Robust S3 data loaders (NIfTI, HDF5, CSV)

```

fMRI Voxels (Cortex) → Ridge Regression → CLIP Embeddings (512-D/1024-D) → Stable Diffusion → Reconstructed Images### Phase 3: Preprocessing Pipeline

                              ↓

                     CLIP Adapter (optional)- `src/fmri2img/data/preprocess.py` - Production-grade preprocessing with T0/T1/T2 transforms

                     512-D → 768-D/1024-D- `scripts/nsd_fit_preproc.py` - CLI to fit preprocessing on training data

```- **T0**: Per-volume z-score normalization (online)

- **T1**: Subject-level scaler + reliability/variance mask (fit on train, persist)

**Key Features:**- **T2**: PCA to k components or ROI pooling

- ✅ **Preprocessing**: PCA, z-scoring, split-half reliability masking

- ✅ **Encoding**: Ridge regression baseline (fMRI → CLIP)### Phase 4: ROI Pooling & CLIP Cache

- ✅ **Adaptation**: CLIP space alignment (ViT-B/32 → ViT-L/14 or ViT-H/14)

- ✅ **Decoding**: Stable Diffusion with CLIP guidance- `src/fmri2img/data/roi.py` - ROI pooling for anatomical region analysis

- ✅ **Evaluation**: CLIPScore, retrieval@K, ranking metrics with ablations- `src/fmri2img/data/clip_cache.py` - CLIP vision embeddings cache with Parquet storage

- ✅ **Automation**: Complete `Makefile` for reproducible experiments- `scripts/nsd_build_clip_cache.py` - CLI to build CLIP embeddings cache

- `scripts/test_roi.py` - Test script for ROI functionality

---

### Phase 5: Ridge Baseline ✨ **NEW**

## 📂 Project Structure

- `src/fmri2img/models/ridge.py` - Ridge regression encoder (fMRI → CLIP)

```- `src/fmri2img/eval/retrieval.py` - Retrieval evaluation metrics

Bachelor V2/- `scripts/train_ridge.py` - Full training pipeline with alpha selection

├── scripts/              # Main execution scripts- `docs/RIDGE_BASELINE.md` - Comprehensive documentation

│   ├── nsd_fit_preproc.py        # Fit preprocessing pipeline

│   ├── train_ridge.py            # Train Ridge encoder (fMRI → CLIP)**Features**:

│   ├── train_clip_adapter.py     # Train CLIP adapter (512-D → 768/1024-D)

│   ├── decode_diffusion.py       # Generate images via Stable Diffusion- L2-regularized linear regression with hyperparameter selection

│   ├── eval_reconstruction.py    # Evaluate reconstructions (enhanced)- Validation-based alpha tuning (no test leakage)

│   ├── summarize_reports.py      # Aggregate evaluation reports- L2-normalized predictions for cosine similarity

│   └── plot_metrics.py           # Generate publication figures- Retrieval@K evaluation (K=1,5,10) + ranking metrics

│- Complete train/val/test splits

├── src/fmri2img/         # Core package- Model persistence with save/load

│   ├── data/             # Dataset loaders, CLIP cache, preprocessing

│   ├── models/           # Ridge, MLP, CLIP adapter modelsNote: GLMdenoise betas are already denoised; this layer standardizes & reduces dimensionality.

│   ├── eval/             # Evaluation metrics (CLIPScore, retrieval)

│   └── utils/            # Logging, visualization, I/O utilities## Important Notes

│

├── outputs/              # Generated outputs**Trial Order**: The `nsd_stim_info_merged.csv` file is a **stimulus catalog** indexed by `nsdId`, providing COCO metadata (cocoId, cocoSplit, shared1000, filename). It is **NOT** trial order information.

│   ├── clip_cache/       # CLIP embedding caches (512-D, 1024-D)

│   ├── checkpoints/      # Trained model weights**True Trial Order**: Actual trial presentation order comes from per-subject session design files located at:

│   ├── preproc/          # Preprocessing artifacts (PCA, scalers)

│   ├── recon/            # Reconstructed images by subject```

│   └── reports/          # Evaluation reports and figuress3://natural-scenes-dataset/nsddata/ppdata/subjXX/behav/sessionYY/

│```

├── data/indices/         # NSD dataset indices (Parquet)

├── configs/              # YAML configuration files**Beta dtype**: NIfTI betas may be stored as `int16` (or other). After slicing a single trial, cast to `float32` if your model expects it:

├── Makefile              # Automation targets`vol = img.slicer[..., beta_index].get_fdata().astype('float32')`.

├── requirements.txt      # Python dependencies

└── environment.yml       # Conda environment (pinned)**Index layout**: Primary format is partitioned by subject at `nsd_index/subject=subjXX/index.parquet`. The single-file path in `configs/data.yaml: paths.index_file` is only a local fallback.

```

**S3 writes**: The public NSD bucket is read-only; examples that write Parquet to S3 require your own bucket + AWS credentials. The demo falls back to local Parquet automatically.

---

The canonical index properly maps `(subject, session, trial_in_session) → nsdId → beta_path` using these design files, not naive pairing.

## 🚀 Setup

## Usage

### 1. Environment

```bash

```bash# Build index for subjects - output is partitioned by subject as:

# Clone repository# .../nsd_index/subject=subjXX/index.parquet

git clone <repo-url>SUBJECTS="subj01 subj02" OUT_ROOT="data/indices/nsd_index" make index

cd "Bachelor V2"

# Fit preprocessing pipeline on training data

# Create conda environmentpython scripts/nsd_fit_preproc.py --subject subj01 --k 4096 --reliability-thr 0.1

conda env create -f environment.yml

conda activate fmri2img# Stream a few 3D volumes from S3 via the canonical index and build small batches (no model yet)

make train-smoke

# Or use pip

pip install -r requirements.txt# Test with preprocessing pipeline

python scripts/train_smoke.py --use-preproc --pca-k 4096

# Install package in development mode

pip install -e .# Run tests

```make test

```

### 2. Dataset Preparation

### Preprocessing Pipeline

**Natural Scenes Dataset (NSD)**

The preprocessing pipeline implements three transformation levels:

This project requires the NSD dataset. You need:

- Preprocessed fMRI beta maps (β₁, β₂, β₃ for split-half reliability)- **T0**: Per-volume z-score normalization (applied online during data loading)

- NSD stimulus images (73,000 natural scenes from COCO)- **T1**: Subject-level scaler with reliability masking (fitted on training data)

  - Computes voxel-wise mean/std from training trials using Welford's online algorithm

Download and prepare:  - For stimuli with repeat presentations, computes test-retest correlation per voxel

```bash  - Keeps only voxels above reliability threshold (default r ≥ 0.1)

# Build canonical NSD index (Parquet format)  - Falls back to variance threshold when repeats unavailable

make index- **T2**: PCA dimensionality reduction to k components OR ROI pooling



# Cache NSD stimulus imagesExample preprocessing workflow:

# Option 1: HDF5 (recommended, ~20GB)

python scripts/cache_nsd_stimuli_hdf5.py --output cache/nsd_hdf5/nsd_stimuli.hdf5```bash

# Fit standard preprocessing with PCA

# Option 2: PNG files (larger, ~30GB)python scripts/nsd_fit_preproc.py --subject subj01 --k 4096 --reliability-thr 0.1

python scripts/cache_nsd_stimuli_png.py --output-dir cache/nsd_png/

```# Fit preprocessing with ROI pooling instead of PCA

python scripts/nsd_fit_preproc.py --subject subj01 --roi-mode pool

---

# Test with preprocessing in data loading

## 📖 Usagepython scripts/train_smoke.py --subject subj01 --use-preproc --pca-k 4096

python scripts/train_smoke.py --subject subj01 --roi-mode pool

### Quick Start: Complete Pipeline```



```bash### ROI Pooling

# Run full pipeline (cache → reconstruct → evaluate → summarize)

make pipelineROI pooling extracts anatomical region means from fMRI volumes:

```

```python

This will:from fmri2img.data.roi import ROIPooler

1. Build 1024-D CLIP cache for Stable Diffusion 2.1

2. Reconstruct test sets for subj01-03# Initialize and fit ROI pooler

3. Evaluate with 3 gallery types (matched, test, all)pooler = ROIPooler(subject="subj01", min_voxels=50)

4. Generate summary CSV, Markdown report, and figurespooler.fit(sample_beta_path)  # Auto-discovers ROI masks via NSDLayout



### Step-by-Step Workflow# Pool volume to ROI means

vol = load_volume()  # (H, W, D)

#### 1. Build CLIP Cachesroi_means = pooler.pool(vol)  # (n_roi,) - mean per anatomical region

```

```bash

# Build 512-D cache (ViT-B/32 baseline)### CLIP Embeddings Cache

make build_cache

CLIP cache stores precomputed ViT-B/32 embeddings (512-dim) for NSD stimuli in a Parquet file with enforced schema:

# Build 1024-D cache (SD 2.1 target space)

make build_target_cache- **nsdId**: int32 (NSD stimulus identifier)

```- **clip512**: fixed-length list[float32, 512] (CLIP vision embedding)



#### 2. Fit Preprocessing**Image Loading**: Primary path is `nsd_stimuli.hdf5` via nsdId (fast, S3-backed). Falls back to COCO HTTP if HDF5 access fails and cocoId is available.



```bash**Build Cache**:

# Fit PCA, scaler, reliability mask for subj01

python scripts/nsd_fit_preproc.py \```bash

    --subject subj01 \# From partitioned index (recommended)

    --k 4096 \python scripts/build_clip_cache.py \

    --reliability-thr 0.1 \    --index-file data/indices/nsd_index/subject=subj01/index.parquet \

    --seed 42    --cache outputs/clip_cache/clip.parquet \

```    --batch 64 --device cuda --limit 256



#### 3. Train Ridge Encoder# From index root with subject filter

python scripts/build_clip_cache.py \

```bash    --index-root data/indices/nsd_index \

# Train Ridge regression (fMRI → CLIP 512-D)    --subject subj01 \

python scripts/train_ridge.py \    --cache outputs/clip_cache/clip.parquet \

    --subject subj01 \    --batch 128 --device cuda

    --clip-cache outputs/clip_cache/clip.parquet \

    --output checkpoints/ridge/subj01/ridge.pt# Resume is automatic - skips already-cached nsdIds

```# Re-run same command to continue after interruption

```

#### 4. Train CLIP Adapter (Optional)

**Use in Dataset**:

```bash

# Train adapter (512-D → 1024-D for SD 2.1)```python

python scripts/train_clip_adapter.py \from fmri2img.data.clip_cache import CLIPCache

    --subject subj01 \from fmri2img.data.torch_dataset import NSDIterableDataset

    --model-id stabilityai/stable-diffusion-2-1 \

    --source-cache outputs/clip_cache/clip.parquet \# Preferred: Fluent API (load() returns self)

    --target-cache outputs/clip_cache/target_clip_stabilityai_stable-diffusion-2-1.parquet \clip_cache = CLIPCache("outputs/clip_cache/clip.parquet").load()

    --output checkpoints/clip_adapter/subj01/adapter.ptdataset = NSDIterableDataset(

```    index_path_or_root="data/indices/nsd_index",

    subject="subj01",

#### 5. Reconstruct Images    clip_cache=clip_cache  # Add CLIP embeddings to batch output

)

```bash

# Generate images for test set# Also supported: Pass path string directly

python scripts/decode_diffusion.py \dataset = NSDIterableDataset(

    --subject subj01 \    index_path_or_root="data/indices/nsd_index",

    --model-id stabilityai/stable-diffusion-2-1 \    subject="subj01",

    --output-dir outputs/recon/subj01/ridge_diffusion \    clip_cache="outputs/clip_cache/clip.parquet"  # String path

    --split test \)

    --num-inference-steps 50 \

    --guidance-scale 7.5 \# Each batch now includes "clip" key with (512,) float32 L2-normalized array

    --device cudafor batch in dataset:

```    fmri = batch["fmri"]      # (H,W,D) or (k,) after PCA

    clip = batch["clip"]      # (512,) CLIP embedding (L2 normalized)

#### 6. Evaluate Reconstructions```



```bash### Ridge Baseline Training

# Evaluate with matched gallery (primary metric)

python scripts/eval_reconstruction.py \Train a reproducible Ridge regression baseline to map fMRI → CLIP embeddings:

    --subject subj01 \

    --recon-dir outputs/recon/subj01/ridge_diffusion/images \```bash

    --clip-cache outputs/clip_cache/target_clip_stabilityai_stable-diffusion-2-1.parquet \# Quick test (works with current k=4 PCA, uses 256 samples)

    --use-adapter \python scripts/train_ridge.py \

    --model-id stabilityai/stable-diffusion-2-1 \    --subject subj01 \

    --gallery matched \    --use-preproc \

    --image-source hdf5 \    --clip-cache outputs/clip_cache/clip.parquet \

    --out-csv outputs/reports/subj01/eval_matched.csv \    --limit 256 \

    --out-json outputs/reports/subj01/eval_matched.json \    --alpha-grid "1,10"

    --out-fig outputs/reports/subj01/eval_matched_grid.png

```# Full training via Makefile

make ridge

**Gallery Types:**

- `matched`: Only GT of reconstructed images (easiest, standard)# Full training with custom config

- `test`: All test split GT embeddings (medium difficulty)python scripts/train_ridge.py \

- `all`: All GT embeddings train+val+test (hardest, most realistic)    --index-root data/indices/nsd_index \

    --subject subj01 \

#### 7. Aggregate Results    --use-preproc \

    --clip-cache outputs/clip_cache/clip.parquet \

```bash    --alpha-grid "0.1,1,3,10,30,100" \

# Summarize all evaluation reports    --limit 2048  # Remove for all data

python scripts/summarize_reports.py \```

    --reports-dir outputs/reports \

    --output-csv outputs/reports/summary_by_subject.csv \**Output**:

    --output-md outputs/reports/SUMMARY.md

```- **Model**: `checkpoints/ridge/subj01/ridge.pkl` (loadable via `RidgeEncoder.load()`)

- **Report**: `outputs/reports/subj01/ridge_eval.json` (cosine, MSE, R@K metrics)

#### 8. Generate Figures

**Evaluation Metrics**:

```bash

# Create publication-quality plots- Cosine similarity (with ground truth)

python scripts/plot_metrics.py \- MSE loss

    --reports-dir outputs/reports \- Retrieval@1/5/10 (% queries with true image in top-K)

    --output-dir outputs/reports/figures \- Mean/median rank, MRR

    --subjects subj01 subj02 subj03

```See `docs/RIDGE_BASELINE.md` for complete documentation.

nsd_id = batch["nsdId"] # int

---

````

## 📊 Evaluation Metrics

**Common Mistake**:

### CLIPScore```python

Cosine similarity between generated and ground truth CLIP embeddings (Hessel et al., 2021).# ❌ Don't do this (old API returned boolean):

# cache = CLIPCache(...).load()  # Returns self now, not bool!

**Formula:** `score(gen, gt) = cos(CLIP(gen), CLIP(gt))`

# ✓ Correct (fluent API):

**Range:** [-1, 1], higher is bettercache = CLIPCache("path/to/cache.parquet").load()

dataset = NSDIterableDataset(..., clip_cache=cache)

### Retrieval@K (R@K)

Proportion of samples where GT is in top-K retrieval from gallery.# ✓ Or use string path:

dataset = NSDIterableDataset(..., clip_cache="path/to/cache.parquet")

**Metrics:** R@1, R@5, R@10````



### Ranking Metrics**API Reference**:

- **Mean Rank**: Average position of GT in retrieval ranking

- **Median Rank**: Median position (robust to outliers)```python

- **MRR**: Mean Reciprocal Rank = mean(1/rank)from fmri2img.data.clip_cache import CLIPCache



### Adapter Ablation# Fluent API - load() returns self

Automatic comparison of performance with/without CLIP adapter to quantify alignment gains.cache = CLIPCache(cache_path="outputs/clip_cache/clip.parquet").load()



---# Check if loaded

assert cache.is_loaded  # Property

## 📈 Example Outputs

# Check if nsdId is cached

### Per-Sample CSVif cache.contains(nsd_id=12345):

```csv    print("Already cached!")

nsdId,clipscore,rank,r@1,r@5,r@10,in_gallery,nn_nsdId,nn_sim,gt_sim

73000,0.652,1,1,1,1,1,73000,0.652,0.652# Get embeddings for multiple nsdIds (L2 normalized)

73001,0.548,3,0,1,1,1,73005,0.601,0.548embeddings = cache.get([1, 2, 3])  # Returns: {1: array(512,), 2: array(512,), ...}

...

```# Save new embeddings

import pandas as pd

### Aggregate JSONrows = pd.DataFrame({

```json    "nsdId": [4, 5, 6],

{    "clip512": [emb1.tolist(), emb2.tolist(), emb3.tolist()]

  "subject": "subj01",})

  "gallery_type": "matched",cache.save_rows(rows)  # Deduplicates on nsdId, enforces schema

  "n_samples": 515,

  "gallery_size": 515,# Get stats

  "clipscore": {stats = cache.stats()  # {"cache_size": N, "path": "..."}

    "mean": 0.524,```

    "std": 0.108

  },**Implementation Details**:

  "retrieval": {

    "R@1": 0.452,- Uses PyArrow schema enforcement for type safety

    "R@5": 0.712,- Deduplicates automatically on nsdId (keeps latest)

    "R@10": 0.823- Resume support: builder skips already-cached IDs

  },- Batch processing with GPU autocast for efficiency

  "ranking": {- Snappy compression for compact storage

    "mean_rank": 8.3,

    "median_rank": 2.0,### Test Scripts

    "mrr": 0.561

  },```bash

  "rank_hist": {# Test ROI functionality

    "1": 233,python scripts/test_roi.py

    "2-5": 134,```

    "6-10": 89,

    "11+": 59Primary format: partitioned Parquet per subject: nsd_index/subject=subjXX/index.parquet. The single-file path (paths.index_file) is only a local fallback.

  },

  "ablations": {```bash

    "with_adapter": {...},# Demo IO layer

    "without_adapter": {...}make sanity

  }```

}

```## Important Notes



### Visualization Grid- **PyTorch IterableDataset** reads one 3D trial at a time via `img.slicer[..., beta_index]` to avoid loading full 4D NIfTI.

Each row shows: **Ground Truth | Nearest Neighbor | Generated**

## Architecture

![Example Grid](outputs/reports/subj01/eval_matched_grid.png)

1. **Stimulus Catalog**: 73K COCO images with NSD metadata

### Distribution Plots2. **Session Designs**: Per-subject trial order and timing

- CLIPScore histogram by subject3. **Beta Files**: 4D NIfTI files with GLMdenoise preprocessed fMRI

- Rank distribution (log scale)4. **Canonical Index**: Parquet mapping trials → stimuli → files

- R@K comparison bars   - Extra columns:

- Adapter ablation comparison     - `stimulus_repeat_count` – count of repeats for that nsdId up to current trial

     - `has_beta_data` – boolean availability flag for mapped beta file/index

---     - `data_quality_flag` – optional QC status if exposed by design

5. **S3 Streaming**: Memory-efficient data access via fsspec

## 🔧 Makefile Targets

| Target | Description |
|--------|-------------|
| `make pipeline` | **Complete pipeline** (recommended) |
| `make build_target_cache` | Build 1024-D CLIP cache |
| `make reconstruct_all` | Reconstruct all subjects |
| `make eval_all_subjects` | Evaluate with all gallery types |
| `make summarize_reports` | Aggregate reports into CSV |
| `make generate_figures` | Create publication figures |
| `make eval_quick` | Quick test (subj01, 50 samples) |
| `make help` | Show all targets |

---

## 🧪 Advanced Features

### FAISS Acceleration
For large galleries (>10k embeddings), use FAISS for fast retrieval:

```bash
pip install faiss-cpu  # or faiss-gpu

python scripts/eval_reconstruction.py \
    --subject subj01 \
    --gallery all \
    --faiss \
    ...
```

### Custom Diffusion Models
Override the default SD 2.1 model:

```bash
export MODEL=runwayml/stable-diffusion-v1-5

python scripts/decode_diffusion.py \
    --model-id $MODEL \
    ...
```

### Preprocessing Ablations
Compare different preprocessing configurations:

```bash
# No PCA
python scripts/nsd_fit_preproc.py --subject subj01 --no-pca

# Lower reliability threshold
python scripts/nsd_fit_preproc.py --subject subj01 --reliability-thr 0.05

# ROI-based (visual cortex only)
python scripts/nsd_fit_preproc.py --subject subj01 --roi-mode early
```

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@misc{fmri2img2025,
  author = {Your Name},
  title = {fMRI-to-Image Reconstruction via CLIP and Stable Diffusion},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/fmri2img}
}
```

**Dataset Citation:**

```bibtex
@article{allen2021massive,
  title={A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence},
  author={Allen, Emily J and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L and Prince, Jacob S and Dowdle, Logan T and Nau, Matthias and Caron, Brad and Pestilli, Franco and Charest, Ian and others},
  journal={Nature Neuroscience},
  year={2021}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Natural Scenes Dataset (NSD)** is subject to its own terms. Please review the [NSD Data Sharing Agreement](http://naturalscenesdataset.org/) before use.

---

## 🙏 Acknowledgments

- **Natural Scenes Dataset (NSD)** by Allen et al., 2021
- **CLIP** by OpenAI (Radford et al., 2021)
- **Stable Diffusion** by Stability AI & Runway ML
- **Transformers** and **Diffusers** libraries by HuggingFace

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your.email@example.com](mailto:your.email@example.com).

---

**Happy Decoding! 🧠→🖼️**
