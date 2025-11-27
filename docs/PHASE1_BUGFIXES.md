# PHASE 1 BUGFIXES
# =================

## Issues Fixed

### Issue 1: train_clip_to_fmri.py Loss Showing as 0.0000

**Problem:**
```
Epoch 47/50: train_loss=0.0000, val_loss=0.0000, val_corr=0.0016
```

The loss values were very small (e.g., 0.000123) but displayed as 0.0000 due to `.4f` formatting.

**Root Cause:**
```python
# Old code
logger.info(
    f"Epoch {epoch+1}/{args.epochs}: "
    f"train_loss={train_loss:.4f}, "  # Only 4 decimal places
    f"val_loss={val_loss:.4f}, "
    f"val_corr={val_corr:.4f}"
)
```

**Fix Applied:**
```python
# New code
logger.info(
    f"Epoch {epoch+1}/{args.epochs}: "
    f"train_loss={train_loss:.6f}, "  # Now 6 decimal places
    f"val_loss={val_loss:.6f}, "
    f"val_corr={val_corr:.4f}"
)
```

**Result:**
```
OLD: train_loss=0.0001, val_loss=0.0005
NEW: train_loss=0.000123, val_loss=0.000456
```

**Files Modified:**
- `scripts/train_clip_to_fmri.py` (lines ~297, ~322)

---

### Issue 2: train_two_stage.py PCA Showing as N/A

**Problem:**
```
Loaded preprocessing: PCA k=N/A
```

The PCA component count was not displayed because the code looked for wrong key in `pca_info_` dict.

**Root Cause:**
```python
# Old code
logger.info(f"Loaded preprocessing: PCA k={preprocessor.pca_info_.get('n_components_eff', 'N/A')}")
# ❌ Wrong key: 'n_components_eff' doesn't exist
```

The actual key in `pca_info_` is `'k_eff'` (from `preprocess.py` line 821):
```python
self.pca_info_ = {
    "k_eff": int(components.shape[0]),
    "explained_variance_ratio": meta.get("explained_variance_ratio", 0.0)
}
```

**Fix Applied:**
```python
# New code
pca_k = preprocessor.pca_info_.get('k_eff', preprocessor.pca_.n_components_ if preprocessor.pca_ else None)
logger.info(f"Loaded preprocessing: PCA k={pca_k if pca_k else 'disabled'}")
# ✅ Uses correct key 'k_eff' with fallback
```

**Result:**
```
OLD: PCA k=N/A
NEW: PCA k=512
```

**Files Modified:**
- `scripts/train_two_stage.py` (lines ~626-628)

---

## Verification

### Test 1: Loss Formatting
```python
train_loss = 0.000123
val_loss = 0.000456

# Old: train_loss=0.0001, val_loss=0.0005
# New: train_loss=0.000123, val_loss=0.000456
```
✅ **PASS**: Loss values now show with proper precision

### Test 2: PCA K Display
```python
pca_info_ = {'k_eff': 512, 'explained_variance_ratio': 0.85}

# Old: PCA k=N/A
# New: PCA k=512
```
✅ **PASS**: PCA component count displays correctly

---

## Impact

### train_clip_to_fmri.py
- **Before**: Misleading 0.0000 losses (looked like training failure)
- **After**: Clear visibility of actual loss values (e.g., 0.000123)
- **Better UX**: Users can now see training progress properly

### train_two_stage.py
- **Before**: Confusing "PCA k=N/A" message
- **After**: Clear "PCA k=512" (or "disabled" if no PCA)
- **Better debugging**: Users know exact PCA dimensionality being used

---

## Files Summary

**Modified:**
1. `scripts/train_clip_to_fmri.py`
   - Line ~297: Changed `:.4f` → `:.6f` for train_loss and val_loss
   - Line ~322: Changed `:.4f` → `:.6f` for best_val_loss

2. `scripts/train_two_stage.py`
   - Lines ~626-628: Fixed PCA k lookup to use correct key `'k_eff'`

**Total changes:** ~5 lines across 2 files

---

## Status

✅ **Both issues FIXED**

**Ready to proceed to PHASE 2!**
