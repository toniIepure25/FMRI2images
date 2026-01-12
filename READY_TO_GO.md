# ✅ YOU'RE READY TO RUN EXPERIMENTS!

## 🎉 What You Have

```
✓ All 37 fMRI beta files (~17GB)  ← COMPLETE!
✓ Built-in image loading system    ← AUTOMATIC!
✓ COCO API fallback                ← NO MANUAL DOWNLOAD NEEDED!
```

## 🧠 How Image Loading Works

Your code has a **smart 3-tier fallback system** (see `src/fmri2img/io/image_loader.py`):

### Tier 1: Local HDF5 (Optional - Fastest)
- **Path:** `cache/nsd_hdf5/nsd_stimuli.hdf5`
- **Size:** ~10GB
- **Speed:** Instant
- **Status:** ❌ Not downloaded (but not needed!)

### Tier 2: S3 HDF5 (Automatic - Moderate)
- **Path:** `s3://natural-scenes-dataset/.../nsd_stimuli.hdf5`
- **Size:** Streamed
- **Speed:** ~1-2 sec per image
- **Status:** ❌ S3 path issue (but not needed!)

### Tier 3: COCO HTTP API (Automatic - Cached)
- **URL:** `http://images.cocodataset.org/train2017/{cocoId}.jpg`
- **Cache:** `~/.cache/coco/{cocoId}_train2017.jpg`
- **Speed:** 
  - First time: ~1-2 sec per image (downloads)
  - Cached: Instant
- **Status:** ✅ **WORKS AUTOMATICALLY!**

## 🚀 What Happens When You Run Experiments

```python
# Your code automatically:
1. Loads fMRI beta from: /bigdata/.../data/nsd/nsddata_betas/ppdata/subj01/...
2. Tries to load stimulus image from local HDF5 → Not found
3. Tries to load stimulus image from S3 HDF5 → Path issue
4. Downloads stimulus from COCO API → SUCCESS! ✓
5. Caches image locally → Future runs are fast ✓
```

**No manual download needed!** The code handles everything.

## 📊 Expected Behavior

### First Run
```bash
python -m src.fmri2img.training.train_smoke --subject subj01 --session 1 --limit 8

# You'll see:
⚠️  No local HDF5 found (normal - will use COCO fallback)
✓ Loaded fMRI beta for session 1
✓ Loaded cocoId=123456 from COCO HTTP
✓ Loaded cocoId=789012 from COCO HTTP
...
# Takes ~1-2 sec per image to download + cache
```

### Second Run (Same Data)
```bash
python -m src.fmri2img.training.train_smoke --subject subj01 --session 1 --limit 8

# Now it's fast:
⚠️  No local HDF5 found (still shows warning)
✓ Loaded fMRI beta for session 1
✓ Loaded cocoId=123456 from cache  ← FAST!
✓ Loaded cocoId=789012 from cache  ← FAST!
...
# Instant - images are cached
```

## 🎯 Next Steps

### 1️⃣ Verify Your Setup (30 seconds)
```bash
cd ~/Bachelor_V2
chmod +x verify_data_ready.sh
bash verify_data_ready.sh
```

Expected output:
```
✓ Beta files: 37/37 (17G)
✓ All beta files present - READY FOR EXPERIMENTS!
[NOTE] No local stimulus images found
✓ This is OK! Your code will auto-download from COCO API
```

### 2️⃣ Run First REAL Experiment (1-2 minutes)
```bash
# Test with 8 real fMRI trials
python -m src.fmri2img.training.train_smoke \
    --subject subj01 \
    --session 1 \
    --limit 8

# What you'll see:
# - Loads real fMRI data ✓
# - Downloads 8 stimulus images from COCO ✓
# - Caches them locally ✓
# - Trains on REAL brain data! ✓
```

### 3️⃣ Run Full Baseline Experiment (5-10 minutes)
```bash
bash scripts/run_experiment_simple.sh experiments/baseline_subj01.yaml

# This will:
# - Use all training data from subj01
# - Download ~1000-2000 stimulus images (first time)
# - Cache them for future runs
# - Train real model on real brain data!
```

### 4️⃣ Monitor Cache Growth
```bash
# Watch cache directory grow as images download
watch -n 5 'du -sh ~/.cache/coco'

# Expected size after full experiment: ~2-3GB
```

## 💡 Why This is Better

**Option A: Manual Download (What We Tried)**
```
❌ Find correct S3 path (tricky)
❌ Download 73,000 images (~10GB)
❌ Wait 1-2 hours
❌ Use disk space for images you might never need
```

**Option B: Automatic COCO Fallback (What Your Code Does)**
```
✅ Zero configuration
✅ Downloads only images you actually use
✅ Automatic caching for reuse
✅ Start experiments immediately
✅ Saves disk space
```

## 📈 Cache Management

### Check Cache Size
```bash
du -sh ~/.cache/coco
```

### Clear Cache (if needed)
```bash
# If disk space gets low
rm -rf ~/.cache/coco/*

# Images will re-download automatically
```

### Expected Cache Sizes
- Smoke test (8 images): ~1MB
- Baseline experiment (~1000 images): ~500MB-1GB
- Full experiments (~2000 images): ~1-2GB

## 🔍 Troubleshooting

### "No local HDF5 found" Warning
**Status:** ✅ **Normal! Safe to ignore.**
- This warning appears every run
- Code automatically falls back to COCO
- Images download and cache normally

### Slow First Run
**Status:** ✅ **Normal! Expected behavior.**
- First run downloads images from COCO (~1-2 sec each)
- Subsequent runs use cached images (instant)
- Cache persists across experiments

### "Failed to load image" Error
**Status:** ⚠️ **Check internet connection**
```bash
# Test COCO API access
curl -I http://images.cocodataset.org/train2017/000000000139.jpg

# Should return: HTTP/2 200
```

## 🎓 Summary

### What You Downloaded
- ✅ 37 fMRI beta files (17GB) - **THE IMPORTANT DATA!**

### What You DON'T Need to Download
- ❌ 73,000 stimulus images (10GB) - **Code auto-downloads from COCO!**

### What Happens Automatically
1. Your code loads fMRI data from beta files ✓
2. Your code downloads stimulus images from COCO as needed ✓
3. Your code caches images locally for reuse ✓
4. You get real results from real brain data! ✓

---

## 🚀 Bottom Line

**You have everything you need to run real experiments right now!**

Just run:
```bash
cd ~/Bachelor_V2
bash verify_data_ready.sh
python -m src.fmri2img.training.train_smoke --subject subj01 --session 1 --limit 8
```

The rest happens automatically! 🎉
