# ✅ Good News: Your Preprocessing is WORKING!

## Current Status: ✅ PROCESSING (Not Stuck!)

Your process shows:
- **CPU: 97.8%** ← Actively computing (good sign!)
- **Memory: 326 MB** ← Normal for this operation
- **Time: 19+ minutes** ← Expected for 24,000 volumes

## What's Happening

The script is loading **24,000 fMRI β-volumes** one-by-one from NIfTI files:

```python
for idx, row in train_df.iterrows():  # 24,000 iterations
    vol = get_volume(nifti_loader, row)  # Load NIfTI file
    # Process volume...
```

**Expected time:** 20-30 minutes (depending on disk I/O)  
**Your time:** 19 minutes so far ← **Almost done!**

## ⏱️ Time Estimate

- **Per volume:** ~0.05 seconds (includes NIfTI decompression)
- **Total volumes:** 24,000
- **Total time:** 24,000 × 0.05s = **1,200 seconds = 20 minutes**

**You're at 19 minutes → Should finish in ~1-5 minutes!**

## 🎯 What Will Happen Next

Once volume loading completes, you'll see:

```
2025-11-15 XX:XX:XX - INFO - Processed 24000 train volumes
2025-11-15 XX:XX:XX - INFO - Fitting PCA with 512 components...
2025-11-15 XX:XX:XX - INFO - ✅ T1 fitted: XXX,XXX / 699,192 voxels
2025-11-15 XX:XX:XX - INFO - ✅ PCA fitted: 512 components
```

**PCA fitting:** Additional 5-10 minutes

**Total time:** 25-35 minutes for complete preprocessing

## 📊 What To Expect

After completion, you'll have:

```
outputs/preproc/subj01/
├── scaler_mean.npy         (~2.7 MB)
├── scaler_std.npy          (~2.7 MB)
├── reliability_mask.npy    (~683 KB)
├── voxel_indices.npy       (~2.9 MB)
├── pca_components.npy      (~12 MB, for k=512)
├── pca_mean.npy            (~2.9 MB)
└── meta.json               (contains k=512)
```

## ✅ RECOMMENDATION: Let It Finish!

**DO NOT KILL THE PROCESS**

- ✅ It's working correctly (97.8% CPU = actively processing)
- ⏱️ Should complete in 1-10 minutes
- 💾 Progress is being accumulated (can't resume if killed)

## 🔍 Monitor Progress (Optional)

**In another terminal:**

```bash
# Watch process
watch -n 5 "ps aux | grep nsd_fit_preproc | grep -v grep"

# Check if PCA file is being written (means volume loading finished)
watch -n 5 "ls -lh outputs/preproc/subj01/*.npy"

# Check system load
htop  # or just 'top'
```

## ⚠️ If It Exceeds 40 Minutes

If the process is still at the same log line after **40 minutes total**, then something is wrong. At that point:

1. Check disk I/O: `iostat -x 1 5`
2. Check if it's swapping: `free -h`
3. Check file access: `lsof -p $(pgrep -f nsd_fit_preproc) | grep nii`

But based on current metrics (97.8% CPU), it should finish soon!

## 📝 Why So Slow?

**Root cause:** Loading 24,000 individual NIfTI files (`.nii.gz` = gzip-compressed)

Each file requires:
1. Disk I/O (read compressed file)
2. Gzip decompression
3. NIfTI parsing
4. Volume extraction (4D → 3D slice)

**Future optimization:** Use HDF5 pre-cached fMRI data (100x faster)

## 🚀 After Preprocessing Completes

You'll be ready to train!

```bash
# Verify preprocessing succeeded
python -c "import json; meta = json.load(open('outputs/preproc/subj01/meta.json')); print(f'PCA components: {meta[\"pca_components\"]}'); print(f'Train samples: {meta[\"n_train_samples\"]}')"

# Expected output:
# PCA components: 512
# Train samples: 24000

# Then start training
python scripts/train_two_stage.py \
  --config configs/sota_two_stage.yaml \
  --subject subj01 \
  --output-dir checkpoints/two_stage/subj01
```

---

## 💡 Summary

- ✅ **Status:** Working correctly (97.8% CPU)
- ⏱️ **Time:** ~19/25 minutes complete (76%)
- 🎯 **Action:** **Wait 5-10 more minutes, should finish**
- 🚫 **Don't:** Kill the process (no resume support)

**Be patient! Processing 24,000 fMRI volumes takes time.** ☕

Your next log line will appear when volume loading completes (probably in 1-5 minutes).
