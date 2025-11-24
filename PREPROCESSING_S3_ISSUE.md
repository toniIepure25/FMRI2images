# 🚨 CRITICAL: Preprocessing Downloading from S3!

## The Problem

Your preprocessing has been running for **80+ minutes** because it's downloading **24,000 fMRI volumes from S3** (Amazon cloud storage) over the internet!

**Current status:**
- ⚠️ Process consuming 3.5 GB RAM
- ⚠️ Downloading volumes one-by-one from S3
- ⚠️ Will take **many hours** (possibly 10+ hours)
- ❌ Not sustainable for local development

## 🛑 IMMEDIATE ACTION: Kill the Process

```bash
# In the terminal where it's running, press:
Ctrl+C

# OR from another terminal:
pkill -9 -f nsd_fit_preproc
```

## Why This Happened

Your NSD index points to S3 URLs:
```
beta_path: s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/...
```

But you don't have local copies of the beta files (only stimulus images).

## The Reality: NSD Beta Files are HUGE

- **Per subject:** ~100 GB of compressed NIfTI files
- **Download time:** 2-6 hours (depending on connection)
- **Storage:** Need 100+ GB free space
- **Processing:** Then another 30+ minutes to fit preprocessing

**This is not practical for quick experimentation!**

## ✅ SOLUTIONS (Ranked by Speed)

### Solution 1: Use Smaller Dataset (FASTEST - 2 minutes)

Work with a small subset that fits in memory:

```bash
# Kill current process
pkill -9 -f nsd_fit_preproc

# Run with tiny dataset (100 trials only)
python scripts/nsd_fit_preproc.py \
  --subject subj01 \
  --k 512 \
  --reliability-thr 0.0 \
  --out-dir outputs/preproc_test

# Wait for it to hit the same stuck point, then:
# Modify scripts/nsd_fit_preproc.py to limit to first 100 rows
```

Actually, let me create a better solution...

### Solution 2: Mock Preprocessing for Testing (INSTANT)

Create fake preprocessing files to test the training pipeline:

```bash
# I'll create a script for this
python scripts/create_mock_preprocessing.py --subject subj01 --k 512
```

This will create dummy preprocessing files so you can test the training script.

### Solution 3: Download NSD Beta Files (SLOW - hours)

If you need real data:

1. **Register and download from NSD:**
   - Visit: https://cvnlab.slite.page/p/NKalgWd_hc/NSD-Data-Manual
   - Download: `nsddata_betas/ppdata/subj01/*.nii.gz` (~100 GB)
   - Place in: `cache/nsd_betas/subj01/`

2. **Update index to point to local files:**
   - Rebuild index with local paths
   - Rerun preprocessing

**Estimated time:** 3-8 hours total

### Solution 4: Use Pre-trained Model (if available)

Skip all preprocessing and use an existing trained encoder:

```bash
# Check for existing models
ls -lh checkpoints/clip_adapter/subj01/
ls -lh checkpoints/mlp/subj01/
ls -lh checkpoints/ridge/subj01/

# If any exist, you can evaluate directly
```

## 🎯 RECOMMENDED: Create Mock Preprocessing

Let me create a script that generates fake but valid preprocessing files for testing:
