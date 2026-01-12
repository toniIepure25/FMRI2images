# 🔧 QUICK FIXES APPLIED

## Issues Fixed:

### 1. ✅ Compare Script - Empty Losses Error
**Problem:** `IndexError: list index out of range` when experiments have no training losses  
**Fix:** Added checks for empty losses before accessing indices  
**Commit:** d2f71af

### 2. ✅ Missing Tensorboard Dependency  
**Problem:** `ModuleNotFoundError: No module named 'tensorboard'`  
**Fix:** Added to requirements.txt + created simplified script without it  
**Solution:** Install tensorboard OR use simplified script

### 3. ✅ Wrong Data Path
**Problem:** Looking for `data/nsd/nsddata_betas/...` but should be `/bigdata/userhome/.../data/nsd/...`  
**Solution:** Use absolute paths

---

## 🚀 CORRECTED COMMANDS (COPY THESE!)

### On Server (after git pull):

```bash
cd ~/Bachelor_V2
git pull origin probabilistic-distribution

# ============================================================================
# OPTION A: Install tensorboard (recommended for full features)
# ============================================================================
pip install tensorboard

# Then use full script:
python scripts/train_real_nsd.py --config experiments/real_baseline_subj01.yaml


# ============================================================================
# OPTION B: Use simplified script (no tensorboard needed)
# ============================================================================
python scripts/train_real_nsd_simple.py --config experiments/real_baseline_subj01.yaml
```

---

## 📊 Fixed Compare Command:

```bash
# This now works even with experiments that have empty losses:
python scripts/compare_experiments.py \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191153_baseline_subj01 \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191216_novel_subj01 \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191239_ablation_subj01 \
  --output experiment_comparison

# Note: Your previous experiments appear to have empty losses
# The script will now show "No Data" instead of crashing
```

---

## 🧠 Corrected fMRI Sample Extraction:

```bash
# The NSD data is in /bigdata, not relative path:
python -c "
import nibabel as nib
import numpy as np

# Use ABSOLUTE path to your NSD data:
beta_path = '/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz'

print(f'Loading from: {beta_path}')
img = nib.load(beta_path)
data = img.get_fdata()
sample = data[:, :, :, 0]
np.save('sample_fmri.npy', sample)
print(f'✓ Saved sample fMRI shape: {sample.shape}')
"
```

---

## ✅ RECOMMENDED WORKFLOW (SIMPLIFIED):

### Step 1: Train on Real Data (Use Simplified Script)

```bash
cd ~/Bachelor_V2

# This script doesn't need tensorboard:
python scripts/train_real_nsd_simple.py --config experiments/real_baseline_subj01.yaml
```

**Expected Output:**
```
====================================================================================================
                         REAL NSD DATA TRAINING (SIMPLIFIED)
====================================================================================================

Experiment: real_baseline_subj01_session1
Subject: subj01
Session: 1
====================================================================================================

✓ Created run directory: /bigdata/.../runs/20260112_XXXXXX_real_baseline_subj01_session1
✓ Using device: cuda
  GPU: GRID A100D-20C
Creating DataLoader with REAL NSD data...
  Index path: data/indices/nsd_index/subject=subj01/index.parquet
  Subject: subj01
  Session: 1
✓ DataLoader created successfully
✓ Model created with 2,345,678 parameters
✓ Training setup complete
  Epochs: 20
  Batch size: 16
  Learning rate: 0.0001

====================================================================================================
STARTING TRAINING
====================================================================================================

Epoch 1: 100%|████████████| 47/47 [00:15<00:00, 3.12 batch/s, loss=1.2345, avg=1.2567]
Epoch 1/20 - Loss: 1.2567 - Time: 15.23s
  🌟 New best loss: 1.2567
...

====================================================================================================
✅ TRAINING COMPLETE!
====================================================================================================

🏆 Best Loss: 0.8534
📁 Results: /bigdata/.../runs/20260112_XXXXXX_real_baseline_subj01_session1

====================================================================================================
```

---

### Step 2: Compare (After Training Completes)

```bash
# After training, compare with new run:
python scripts/compare_experiments.py \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_XXXXXX_real_baseline_subj01_session1 \
  --output new_comparison
```

---

## 📝 Summary of Changes:

| File | Change |
|------|--------|
| `scripts/compare_experiments.py` | ✅ Fixed empty losses handling |
| `scripts/train_real_nsd_simple.py` | ✅ NEW: No tensorboard dependency |
| `requirements.txt` | ✅ Added tensorboard |

---

## 🎯 WHAT TO DO RIGHT NOW:

```bash
# 1. Pull latest fixes
cd ~/Bachelor_V2
git pull origin probabilistic-distribution

# 2. Train on REAL data (simplified version, no tensorboard needed)
python scripts/train_real_nsd_simple.py --config experiments/real_baseline_subj01.yaml

# 3. Wait ~5-10 minutes for training to complete

# 4. Extract sample and generate
python -c "
import nibabel as nib
import numpy as np
beta_path = '/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz'
img = nib.load(beta_path)
data = img.get_fdata()
sample = data[:, :, :, 0]
np.save('sample_fmri.npy', sample)
print(f'✓ Saved {sample.shape}')
"

# 5. Generate images (use YOUR run directory!)
python scripts/generate_images.py \
  --checkpoint /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/YOUR_RUN_DIR/checkpoints/best_model.pt \
  --fmri sample_fmri.npy \
  --output generation_results \
  --visualize
```

---

## 💡 Pro Tips:

1. **Use `train_real_nsd_simple.py`** - No external dependencies
2. **Check run directory** - Look in `/bigdata/.../runs/` for latest timestamp
3. **Use absolute paths** - Avoid `data/nsd/...`, use `/bigdata/.../data/nsd/...`

---

**All fixes pushed! Pull and run the simplified training script! 🚀**
