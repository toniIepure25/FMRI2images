# Pipeline Fixes Applied

## Issues Fixed

### 1. ✅ Makefile `--max-trials` Argument Error

**Problem**:

```bash
nsd_index_builder.py: error: argument --max-trials: expected one argument
```

**Root Cause**: Makefile was passing `--max-trials` with an empty value when `MAX_TRIALS` environment variable was not set.

**Fix**: Changed Makefile line from:

```makefile
--max-trials $${MAX_TRIALS:-}
```

To:

```makefile
$${MAX_TRIALS:+--max-trials $$MAX_TRIALS}
```

This only includes the flag if `MAX_TRIALS` is actually set.

---

### 2. ✅ Missing `configs/data.yaml`

**Problem**:

```bash
FileNotFoundError: [Errno 2] No such file or directory: 'configs/data.yaml'
```

**Root Cause**: `NSDIndexBuilder` was hardcoding `config_path="configs/data.yaml"` but the file didn't exist.

**Fix**: Modified `nsd_index_builder.py` to:

1. Make config_path optional (default=None)
2. Use default paths if config doesn't exist
3. Fall back to `NSDLayout(None)` which uses built-in defaults

---

### 3. ✅ S3 Write Attempt Without Credentials

**Problem**: Index builder was trying to write to S3 without credentials

**Root Cause**: Makefile had `--use-s3` flag

**Fix**: Removed `--use-s3` flag from Makefile to write locally instead

---

### 4. ✅ Wrong Output Directory

**Problem**: Index builder wrote to `data/indices/nsd_canonical_index/` but pipeline expected `data/indices/nsd_index/`

**Fix**: Changed default output path in `nsd_index_builder.py` from:

```python
output_root = Path("data/indices/nsd_canonical_index")
```

To:

```python
output_root = Path("data/indices/nsd_index")
```

---

### 5. ✅ Missing `configs/clip.yaml`

**Problem**:

```bash
FileNotFoundError: CLIP config not found at configs/clip.yaml
```

**Fix**: Created `configs/clip.yaml` with sensible defaults:

```yaml
model_name: "ViT-L/14"
embedding_dim: 768
device: "cuda"
batch_size: 256
```

---

### 6. ✅ No Preprocessing CLI Interface

**Problem**: Pipeline was trying to call `python -m fmri2img.data.preprocess` but no CLI existed

**Fix**: Created new script `scripts/fit_preprocessing.py` with full argument parsing:

- `--subject`
- `--index-file`
- `--output-dir`
- `--reliability-mode` (hard_threshold/soft_weight/none)
- `--reliability-curve` (sigmoid/linear)
- `--reliability-temperature`
- `--n-components`
- `--pca-whiten`

---

## Files Created/Modified

### Created:

1. **`configs/clip.yaml`** - CLIP model configuration
2. **`scripts/fit_preprocessing.py`** - Preprocessing CLI wrapper

### Modified:

1. **`Makefile`** - Fixed `--max-trials` argument handling, removed `--use-s3`
2. **`src/fmri2img/data/nsd_index_builder.py`** - Made config optional, fixed output path
3. **`scripts/run_full_pipeline.py`** - Updated to use new preprocessing script

---

## Verification

### Test Index Building:

```bash
make index SUBJECTS=subj01
# Should create: data/indices/nsd_index/subject=subj01/index.parquet
```

### Test Preprocessing:

```bash
python scripts/fit_preprocessing.py \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --output-dir outputs/preproc/test \
    --reliability-mode soft_weight \
    --n-components 3072
# Should create: outputs/preproc/test/subj01/*.npy and meta.json
```

### Test Full Pipeline:

```bash
# Dry run to preview
python scripts/run_full_pipeline.py --subject subj01 --mode novel --dry-run

# Actual run
python scripts/run_full_pipeline.py --subject subj01 --mode novel
```

---

## Current Status

✅ **Environment Validation** - Working  
✅ **Index Building** - Working  
🔄 **CLIP Cache Building** - Should work (needs testing)  
✅ **Preprocessing** - Script created, needs testing  
⏸️ **Training** - Pending  
⏸️ **Evaluation** - Pending

---

## Next Steps

1. **Test CLIP Cache Building**:

   ```bash
   make build-clip-cache CACHE=outputs/clip_cache/clip.parquet BATCH=256
   ```

   - Will take 2-3 hours for full 73K images
   - Or test with small subset first

2. **Test Preprocessing**:

   ```bash
   python scripts/fit_preprocessing.py \
       --subject subj01 \
       --index-file data/indices/nsd_index/subject=subj01/index.parquet \
       --output-dir outputs/preproc/test \
       --reliability-mode soft_weight \
       --reliability-curve sigmoid \
       --reliability-temperature 0.1 \
       --n-components 3072
   ```

3. **Run Full Pipeline**:
   ```bash
   python scripts/run_full_pipeline.py --subject subj01 --mode novel
   ```

---

## Issue 7: Missing 'split' Column in Index

**Problem**: Preprocessing script expected 'split' column to filter training data, but index builder doesn't create this column.

**Error**: `KeyError: 'split'`

**Fix**: Added automatic 80/10/10 train/val/test split in `scripts/fit_preprocessing.py` when 'split' column is missing.

**Status**: ✅ Fixed

## Issue 8: Wrong NIfTILoader API

**Problem**: Called non-existent `load_beta_volume()` method. NIfTILoader only has `load()` method.

**Error**: `AttributeError: 'NIfTILoader' object has no attribute 'load_beta_volume'`

**Fix**: Updated `get_volume()` to use `loader.load()` and extract beta volume with `data[..., row['beta_index']]`.

**Status**: ✅ Fixed

---

## Summary

Total issues fixed: 8

Pipeline status:

- ✅ Environment validation working
- ✅ Index building working (73K trials)
- ✅ CLIP cache validation working
- ✅ Preprocessing working (tested with 100 samples, successfully loads from S3)
- ⏸️ Training integration pending
- ⏸️ Evaluation pending

Note: Preprocessing is slow (~6s/sample) due to S3 downloads but functional.
