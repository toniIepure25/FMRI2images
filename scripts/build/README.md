# Build Scripts

Data preparation, preprocessing, and caching utilities.

**13 build scripts**

## Key Scripts

- **`build_target_clip_cache_robust.py`** ⭐ - Robust CLIP embedding cache builder
- **`fit_preprocessing.py`** ⭐ - Fit preprocessing pipeline (PCA/PCR)
- `build_full_index.py` - Build full NSD data index
- `build_full_subj01_index.py` - Build subject-01 specific index
- `build_clip_cache.py` - Standard CLIP cache builder
- `build_multilayer_clip_cache.py` - Multi-layer CLIP embeddings
- `build_text_clip_cache.py` - Text CLIP embeddings
- `build_embedding_preproc.py` - Build embedding preprocessing
- `nsd_index_builder.py` - NSD index builder utility
- `download_stimuli_hdf5.py` - Download NSD stimuli to HDF5
- `fast_preproc.py` - Fast preprocessing implementation
- `build_all_preprocessors.sh` - Batch build all preprocessors

## Usage

```bash
# Build data index
python scripts/build/build_full_index.py --subject subj01

# Fit preprocessing
python scripts/build/fit_preprocessing.py \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --output-dir cache/preproc/subject=subj01

# Build CLIP cache
python scripts/build/build_target_clip_cache_robust.py

# Build all preprocessors at once
bash scripts/build/build_all_preprocessors.sh
```
