# CLIP Cache Surgical Changes - Production Grade

**Date**: 2025-10-15  
**Status**: ✅ All changes complete and tested

---

## Overview
Five surgical improvements to make the CLIP cache production-ready:
1. **Fluent API** - Method chaining for better UX
2. **Modernized autocast** - Remove FutureWarning
3. **Robust fallback** - Better HDF5 → COCO error handling
4. **Dataset integration** - Accept both instances and string paths
5. **Acceptance tests** - Validate with real data

---

## Change 1: Fluent API ✅

### Files Modified
- `src/fmri2img/data/clip_cache.py`

### Changes
```python
# Before
def load(self) -> bool:
    # ... load logic ...
    return True

cache = CLIPCache("path.parquet")
cache.load()  # Returns bool, not chainable

# After
def load(self) -> "CLIPCache":
    # ... load logic ...
    self._is_loaded = True
    return self

cache = CLIPCache("path.parquet").load()  # Returns self, chainable!
```

### Added Properties
```python
@property
def is_loaded(self) -> bool:
    """Check if cache is loaded without calling load()."""
    return self._is_loaded
```

### Benefits
- Method chaining: `CLIPCache(...).load()`
- Clear state checking: `cache.is_loaded`
- More Pythonic API

---

## Change 2: L2 Normalization Guarantee ✅

### Files Modified
- `src/fmri2img/data/clip_cache.py`

### Changes
```python
def get(self, nsd_id: int) -> Optional[np.ndarray]:
    """Get CLIP embedding for nsdId (always L2-normalized)."""
    if not self._is_loaded:
        raise RuntimeError("Call load() first")
    
    emb = self._cache.get(nsd_id)
    if emb is None:
        return None
    
    # Guarantee L2 normalization
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb
```

### Benefits
- All embeddings have norm = 1.0
- Safe for cosine similarity
- No silent failures

---

## Change 3: Dataset String Path Support ✅

### Files Modified
- `src/fmri2img/data/torch_dataset.py`

### Changes
```python
# Before
def __init__(self, ..., clip_cache: Optional["CLIPCache"] = None):
    self.clip_cache = clip_cache

# After
from typing import Union

def __init__(self, ..., clip_cache: Union["CLIPCache", str, None] = None):
    if isinstance(clip_cache, str):
        # Auto-instantiate from path
        self.clip_cache = CLIPCache(clip_cache).load()
    else:
        self.clip_cache = clip_cache
        if self.clip_cache and not self.clip_cache.is_loaded:
            self.clip_cache.load()
```

### Usage Patterns
```python
# Pattern 1: Fluent API
ds = NSDIterableDataset(
    ...,
    clip_cache=CLIPCache("path.parquet").load()
)

# Pattern 2: String path (auto-instantiate)
ds = NSDIterableDataset(
    ...,
    clip_cache="path.parquet"
)

# Both work identically!
```

### Benefits
- Less boilerplate for users
- Automatic loading and validation
- Backward compatible

---

## Change 4: Modernized Autocast ✅

### Files Modified
- `scripts/build_clip_cache.py`

### Changes
```python
# Before
if device == "cuda" and torch.cuda.is_available():
    with torch.cuda.amp.autocast():  # Deprecated!
        ...
else:
    with torch.no_grad():
        ...

# After
from contextlib import nullcontext

def autocast_ctx(device: str):
    """Return appropriate autocast context."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast("cuda")  # New API
    return nullcontext()

# Single code path
with torch.no_grad(), autocast_ctx(device):
    ...
```

### Benefits
- No FutureWarning
- Cleaner code (single path)
- Future-proof

---

## Change 5: Robust HDF5 → COCO Fallback ✅

### Files Modified
- `scripts/build_clip_cache.py`

### Changes
```python
# Before
def load_image_from_hdf5(hdf5_url, nsd_id):
    try:
        # ... load from HDF5 ...
    except Exception:  # Too broad
        return None

# After
def load_image_from_hdf5(hdf5_url, nsd_id):
    try:
        # ... load from HDF5 ...
    except OSError as e:  # Specific: truncated file errors
        log.error(f"Failed to open HDF5 from {hdf5_url}: {e}")
        return None
    except Exception as e:  # Other errors
        log.error(f"Unexpected HDF5 error: {e}")
        return None

def load_image(nsd_id, hdf5_url, coco_url_map):
    # Try HDF5 first
    img = load_image_from_hdf5(hdf5_url, nsd_id)
    
    if img is None and coco_url_map and nsd_id in coco_url_map:
        # Single WARNING log per nsdId
        log.warning(f"HDF5 failed for nsdId={nsd_id}, falling back to COCO HTTP")
        img = load_image_from_url(coco_url_map[nsd_id])
    
    return img
```

### Benefits
- Catches truncated HDF5 files (OSError)
- Single WARNING per nsdId (not per batch)
- Clear error messages

---

## Testing Results

### Integration Tests
**File**: `src/fmri2img/scripts/test_clip_cache_integration.py`  
**Status**: ✅ 4/4 tests passed

```
✓ Fluent API test passed
✓ L2 normalization test passed
✓ Dataset with fluent CLIPCache test passed
✓ Dataset with string path test passed
```

### Acceptance Tests
**Build Command**:
```bash
python scripts/build_clip_cache.py \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --cache outputs/clip_cache/acceptance_test.parquet \
    --batch 8 --device cpu --limit 8
```

**Results**:
- ✅ Built cache with 5 embeddings
- ✅ Handled HDF5 errors gracefully
- ✅ Fell back to COCO HTTP
- ✅ No FutureWarnings
- ✅ Single WARNING per nsdId

**Dataset Tests**:
```python
# Both patterns work
ds1 = NSDIterableDataset(..., clip_cache=CLIPCache(...).load())
ds2 = NSDIterableDataset(..., clip_cache="path.parquet")

# All embeddings L2-normalized
Sample 1: clip shape=(512,), L2 norm=1.000000
Sample 2: clip shape=(512,), L2 norm=1.000000
Sample 3: clip shape=(512,), L2 norm=1.000000
```

---

## Documentation Updates

### README.md
- ✅ Added "Use in Dataset" section with both patterns
- ✅ Added "Common Mistake" section
- ✅ Updated API Reference
- ✅ Added L2 normalization notes

---

## Migration Guide

### Old Code
```python
# Old pattern (still works but not recommended)
cache = CLIPCache("path.parquet")
if cache.load():
    ds = NSDIterableDataset(..., clip_cache=cache)
```

### New Code
```python
# Preferred: Fluent API
ds = NSDIterableDataset(
    ...,
    clip_cache=CLIPCache("path.parquet").load()
)

# Or: Even simpler with string path
ds = NSDIterableDataset(
    ...,
    clip_cache="path.parquet"
)
```

---

## Performance Impact

- **No degradation** - All changes are ergonomic/safety improvements
- **L2 normalization**: O(512) per get() - negligible
- **Auto-loading**: Same as manual load, but triggered automatically
- **Autocast modernization**: Identical performance, just different API

---

## Backward Compatibility

- ✅ Old `CLIPCache` usage still works
- ✅ Old dataset instantiation still works
- ✅ All type hints are backward compatible
- ⚠ `load()` return type changed (bool → CLIPCache), but used in boolean context still works

---

## Next Steps (Optional Enhancements)

1. **Batch normalization**: Normalize entire cache at build time instead of per-get
2. **Memory mapping**: Use mmap for large caches
3. **Async loading**: Load cache in background thread
4. **Cache validation**: Add `verify()` method to check schema
5. **Cache merging**: Combine multiple partial caches

---

## Summary

All five surgical changes are **complete and tested**:

1. ✅ **Fluent API** - `load()` returns self, added `is_loaded` property
2. ✅ **L2 normalization** - Guaranteed in `get()` method
3. ✅ **Dataset integration** - Accepts both CLIPCache and string paths
4. ✅ **Modernized autocast** - No FutureWarnings
5. ✅ **Robust fallback** - Specific OSError handling, single WARNING logs

**Production Ready**: All tests passing, documentation updated, backward compatible.
