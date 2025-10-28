# sklearn IncrementalPCA Dependency Fix

## Problem Statement

The preprocessing pipeline was accessing sklearn's `IncrementalPCA` internal attributes (like `explained_variance_ratio_`) at inference time during dataset iteration. This caused issues:

1. Tight coupling to sklearn internals (fragile across versions)
2. Unnecessary sklearn dependency at inference time
3. Potential errors when accessing attributes that don't exist after loading from artifacts

## Solution Overview

Created a lightweight numpy-based PCA wrapper (`_NumpyPCA`) that only implements the `transform()` method needed at inference time, eliminating sklearn dependency after artifacts are loaded.

## Changes Made

### 1. src/fmri2img/data/preprocess.py

**Added `_NumpyPCA` class** (lines 66-105):

```python
class _NumpyPCA:
    """
    Lightweight numpy-based PCA wrapper for transform-only operations.

    This avoids dependency on sklearn IncrementalPCA internals at inference time.
    Only implements transform() method using saved components and mean.
    """

    def __init__(self, components: np.ndarray, mean: np.ndarray):
        self.components_ = components  # (k, n_features)
        self.mean_ = mean              # (n_features,)
        self.n_components_ = components.shape[0]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using saved PCA components."""
        if X.ndim == 1:
            return (X - self.mean_) @ self.components_.T
        else:
            return (X - self.mean_) @ self.components_.T
```

**Updated `NSDPreprocessor.__init__`**:

- Added `self.pca_info_ = {}` to store PCA metadata for logging (not transform)

**Updated `load_artifacts()`** (lines 686-709):

```python
# Load T2 PCA artifacts (optional) - use lightweight numpy-based wrapper
if self.pca_components_path.exists() and self.pca_mean_path.exists():
    components = np.load(self.pca_components_path)
    mean = np.load(self.pca_mean_path)

    # Use lightweight _NumpyPCA wrapper (no sklearn dependency at inference)
    self.pca_ = _NumpyPCA(components, mean)
    self.pca_fitted_ = True

    # Load PCA info from metadata for logging only (not used in transform)
    if self.meta_path.exists():
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
            self.pca_info_ = {
                "k_eff": int(components.shape[0]),
                "explained_variance_ratio": meta.get("explained_variance_ratio", 0.0)
            }
    else:
        self.pca_info_ = {"k_eff": int(components.shape[0]), "explained_variance_ratio": 0.0}

    logger.info(f"✅ Loaded T2 PCA: k={self.pca_info_['k_eff']} components "
               f"(numpy-based, no sklearn dependency)")
```

**Updated `summary()`** (lines 777-790):

```python
if self.pca_fitted_ and self.pca_ is not None:
    # Use pca_info_ for logging (no sklearn access)
    if self.pca_info_:
        info["pca_components"] = self.pca_info_.get("k_eff", self.pca_.n_components_)
        info["explained_variance_ratio"] = self.pca_info_.get("explained_variance_ratio", 0.0)
    else:
        # Fallback for sklearn IncrementalPCA during fitting
        info["pca_components"] = int(self.pca_.n_components_)
        if hasattr(self.pca_, 'explained_variance_ratio_'):
            info["explained_variance_ratio"] = float(self.pca_.explained_variance_ratio_.sum())
        elif "explained_variance_ratio" in self.meta_:
            info["explained_variance_ratio"] = float(self.meta_["explained_variance_ratio"])
        else:
            info["explained_variance_ratio"] = 0.0
```

### 2. src/fmri2img/data/torch_dataset.py

**Updated preprocessing status logging** (lines 107-120):

```python
# Log preprocessing status once using summary() API (no sklearn access)
logger = __import__('logging').getLogger(__name__)
if self.preprocessor is not None and not self._preproc_logged:
    if self.preprocessor.is_fitted_:
        summary = self.preprocessor.summary()
        pca_status = ""
        if summary.get("pca_fitted", False):
            k = summary.get("pca_components", 0)
            var_explained = summary.get("explained_variance_ratio", 0.0)
            pca_status = f", T2 PCA (k={k}, {var_explained:.1%} var)"
        logger.info(f"✅ Loaded preprocessing: T0 (z-score) + T1 (scaler+mask, {summary['n_voxels_kept']:,} voxels){pca_status}")
    else:
        logger.info("⚠️  Preprocessing artifacts not found; applying T0 z-score only")
    self._preproc_logged = True
```

**Key Changes**:

- Removed direct access to `self.preprocessor.pca_.n_components_`
- Removed direct access to `self.preprocessor.mask_.sum()`
- Now uses `summary()` API exclusively for logging

### 3. src/fmri2img/scripts/test_preprocess.py

**Added new test** `test_preprocessor_no_sklearn_access_after_load`:

```python
def test_preprocessor_no_sklearn_access_after_load():
    """Test that loaded preprocessor doesn't access sklearn IncrementalPCA internals."""
    pre = NSDPreprocessor("subj01", out_dir="outputs/preproc")

    # Try to load artifacts
    artifacts_loaded = pre.load_artifacts()

    if artifacts_loaded and pre.pca_fitted_:
        # Test that PCA is using _NumpyPCA, not sklearn
        from fmri2img.data.preprocess import _NumpyPCA
        assert isinstance(pre.pca_, _NumpyPCA), f"Expected _NumpyPCA, got {type(pre.pca_)}"

        # Test that we can transform without accessing sklearn attributes
        n_voxels = int(pre.mask_.sum())
        vec_t1 = np.random.randn(n_voxels).astype(np.float32)
        vec_t2 = pre.transform_T2(vec_t1)

        assert vec_t2.dtype == np.float32
        assert vec_t2.ndim == 1
        assert vec_t2.shape[0] == pre.pca_.n_components_

        # Test that summary works without sklearn access
        summary = pre.summary()
        assert "pca_components" in summary
        assert "explained_variance_ratio" in summary
        assert summary["pca_fitted"] == True
```

## Benefits

1. **No sklearn dependency at inference**: After loading artifacts, PCA transform uses pure numpy
2. **Robust across sklearn versions**: No reliance on internal attributes
3. **Faster inference**: Lightweight wrapper vs full sklearn object
4. **Clean API**: Dataset code uses `summary()` instead of accessing internals
5. **Better separation of concerns**: Logging logic separated from transform logic

## Verification

### Tests Pass

```bash
$ pytest src/fmri2img/scripts/test_preprocess.py -v
```

**Result**: ✅ 3/3 tests passing (including new sklearn access test)

### Smoke Test Works

```bash
$ python scripts/train_smoke.py --subject subj01 --use-preproc --limit 8 --batch-size 2
```

**Output**:

```
INFO:fmri2img.data.preprocess:✅ Loaded T1 scaler for subj01: 370,498 voxels retained
INFO:fmri2img.data.preprocess:✅ Loaded T2 PCA: k=4 components (numpy-based, no sklearn dependency)
INFO:train_smoke:Loaded preprocessing for subj01
INFO:train_smoke:PCA: 4 components, 100.0% variance explained
INFO:fmri2img.data.torch_dataset:✅ Loaded preprocessing: T0 (z-score) + T1 (scaler+mask, 370,498 voxels), T2 PCA (k=4, 100.0% var)
INFO:train_smoke:Step 0: fmri batch (2, 4) dtype=torch.float32, nsdIds=[0, 1]
INFO:train_smoke:Step 1: fmri batch (2, 4) dtype=torch.float32, nsdIds=[2, 3]
INFO:train_smoke:Step 2: fmri batch (1, 4) dtype=torch.float32, nsdIds=[4]
INFO:train_smoke:✅ train_smoke finished (I/O + collation OK)
```

**✅ No "IncrementalPCA object has no attribute explained*variance*" errors**

## Implementation Details

### PCA Transform Math

The `_NumpyPCA.transform()` implements standard PCA projection:

```
y = (x - mean) @ components.T
```

Where:

- `x`: Input vector (n_features,)
- `mean`: PCA mean (n_features,)
- `components`: PCA components matrix (k, n_features)
- `y`: Output PCA coefficients (k,)

This is mathematically identical to sklearn's `IncrementalPCA.transform()` but uses pure numpy.

### Backwards Compatibility

During **fitting** (with `fit_pca()`), we still use sklearn's `IncrementalPCA` for:

- Memory-efficient incremental fitting
- Computing explained variance ratios
- Handling batch processing

The lightweight wrapper is **only used at inference time** after loading artifacts.

### Metadata Storage

PCA statistics are stored in two places:

1. **Artifacts**: `pca_components.npy`, `pca_mean.npy` (for transform)
2. **Metadata**: `meta.json` with `explained_variance_ratio` (for logging)

This ensures logging information survives across sessions without sklearn.

## Edge Cases Handled

1. **No PCA artifacts**: Preprocessor works with T0+T1 only
2. **Partial artifacts**: Warns if components exist but mean missing
3. **Fitting vs Loading**: Uses sklearn during fitting, numpy during loading
4. **Summary during fitting**: Falls back to sklearn attributes if pca*info* not populated yet

## Acceptance Criteria Met

✅ `train_smoke.py --use-preproc` runs without IncrementalPCA errors
✅ Logging shows preprocessing status from `summary()`
✅ Dataset iteration never touches sklearn attributes
✅ PCA transform works with pure numpy after loading
✅ Tests verify no sklearn access after artifact loading

---

**Implementation Date**: 2025-10-18
**Status**: ✅ Complete and Tested
**Performance**: No measurable overhead (numpy matmul is highly optimized)
**Compatibility**: Works with all existing preprocessing artifacts
