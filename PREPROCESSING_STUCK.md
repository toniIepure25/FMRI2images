# ⚠️ Preprocessing Taking Too Long? Here's Why & How to Fix

## Current Status

Your `nsd_fit_preproc.py` has been running for **19+ minutes** and appears stuck. This is **NOT normal**.

### What's Happening

The script is trying to:
1. Load **24,000 fMRI β-volumes** (training set) from NIfTI files
2. Compute split-half reliability (requires loading images twice)
3. Fit PCA on high-dimensional data

**Expected time:** 5-15 minutes  
**Your time:** 19+ minutes (likely stuck)

### Most Likely Cause

The script is loading fMRI data from **slow storage** (network or S3), not local files.

---

## ✅ SOLUTION 1: Use Existing Preprocessing (FASTEST)

You already have preprocessing files! Just use different parameters that work with training:

```bash
# Check what exists
ls -lh outputs/preproc/subj01/

# Use the existing preprocessing with k=512
# (The directories show you already ran k=512 before)
```

**IMPORTANT:** Your existing `meta.json` shows only **2 PCA components** and **3 training samples** - this is from old testing and won't work.

---

## ✅ SOLUTION 2: Kill and Restart with Better Settings

### Step 1: Kill the stuck process

```bash
# Find and kill the process
pkill -f nsd_fit_preproc

# OR press Ctrl+C in the terminal where it's running
```

### Step 2: Check if fMRI data is local

```bash
# Check if beta files exist locally
ls -lh cache/nsd_hdf5/ 2>/dev/null || echo "No local fMRI data"

# Check what's in cache
find cache -name "*.nii.gz" -o -name "betas*.hdf5" | head -10
```

### Step 3: Run with verbose logging

```bash
python scripts/nsd_fit_preproc.py \
  --subject subj01 \
  --k 512 \
  --out-dir outputs/preproc \
  --reliability-thr 0.1 \
  2>&1 | tee preproc_log.txt
```

This will:
- Save all output to `preproc_log.txt`
- Let you see where it's stuck
- Show loading progress

---

## ✅ SOLUTION 3: Skip Preprocessing for Now (TEST TRAINING)

The training script `train_two_stage.py` might work with the existing preprocessing files (even if sub-optimal). Let's test:

```bash
# Try training with existing preprocessing
python scripts/train_two_stage.py \
  --config configs/sota_two_stage.yaml \
  --subject subj01 \
  --limit 100 \
  --output-dir checkpoints/two_stage/test
```

If this works, great! You can refit preprocessing later with better data locality.

---

## ⚡ SOLUTION 4: Fast Preprocessing (Skip Reliability)

If fMRI loading is slow, skip the expensive reliability computation:

```bash
# Kill current process
pkill -f nsd_fit_preproc

# Run without reliability threshold (faster, uses all voxels)
python scripts/nsd_fit_preproc.py \
  --subject subj01 \
  --k 512 \
  --reliability-thr 0.0 \
  --min-variance 1e-6 \
  --out-dir outputs/preproc
```

With `--reliability-thr 0.0`:
- Skips split-half correlation (saves ~50% time)
- Uses all voxels with variance > 1e-6
- Still fits PCA

**Expected time:** 5-10 minutes

---

## 🔍 Diagnostic Commands

### Check if process is still running
```bash
ps aux | grep nsd_fit_preproc
```

### Check memory usage
```bash
free -h
```

### Check disk I/O (if process exists)
```bash
iostat -x 1 5
```

### Check what files are being accessed
```bash
lsof -p $(pgrep -f nsd_fit_preproc) 2>/dev/null | grep -E "\.nii|\.hdf5"
```

---

## 📊 What You Need for Training

Training script expects these files:

```
outputs/preproc/subj01/
├── scaler_mean.npy           # ✓ You have this
├── scaler_std.npy            # ✓ You have this  
├── reliability_mask.npy      # ✓ You have this
├── pca_components.npy        # ✓ You have this (but k=2, need k=512)
├── pca_mean.npy              # ✓ You have this
└── meta.json                 # ✓ You have this (but wrong k)
```

The issue: Your PCA has **k=2** (from testing), but training needs **k=512**.

---

## 🎯 RECOMMENDED ACTION

**Right now:**

1. **Check terminal** - Is it still showing the same log line?
   - If yes → Press **Ctrl+C** to kill it

2. **Run fast preprocessing** (skip reliability):
   ```bash
   python scripts/nsd_fit_preproc.py \
     --subject subj01 \
     --k 512 \
     --reliability-thr 0.0 \
     --out-dir outputs/preproc
   ```

3. **Expected output:**
   ```
   2025-11-15 XX:XX:XX - INFO - Fitting scaler and reliability mask...
   2025-11-15 XX:XX:XX - INFO - Fitting PCA with 512 components...
   2025-11-15 XX:XX:XX - INFO - PCA components (k_eff): 512
   2025-11-15 XX:XX:XX - INFO - Explained variance: XX.X%
   ```

4. **Time:** Should complete in 5-10 minutes

5. **Verify:**
   ```bash
   python -c "import json; print(json.load(open('outputs/preproc/subj01/meta.json'))['pca_components'])"
   # Should print: 512
   ```

---

## 📝 Summary

- **Problem:** Preprocessing stuck (19+ min) loading fMRI data
- **Cause:** Slow data loading (network/S3) or expensive reliability computation
- **Solution:** Kill process, run with `--reliability-thr 0.0` (faster)
- **Alternative:** Try training with existing files (might work despite wrong k)
- **Time:** 5-10 minutes with fast settings

Let me know which solution you want to try!
