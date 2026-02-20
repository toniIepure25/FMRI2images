# Data Requirements Guide

> **What data/models you need and how to get them**

---

## 📋 Overview

This project requires:
1. **NSD Dataset** (~300GB) - fMRI brain data + stimulus images
2. **Pre-trained Models** (~10GB) - Stable Diffusion, CLIP
3. **Generated Caches** (~5-50GB) - CLIP embeddings, preprocessed data

---

## 1️⃣ NSD Dataset (Required)

### What is it?
The **Natural Scenes Dataset (NSD)** contains:
- fMRI brain scans from 8 subjects viewing 10,000 unique images
- Stimulus images (73,000 images from COCO)
- Beta maps (brain activity patterns)

### Size
- **Full dataset**: ~300GB
- **Single subject**: ~40GB

### How to get it?

#### Option A: Download Manually (Recommended)

1. **Register and download** from official source:
   ```
   https://naturalscenesdataset.org/
   ```

2. **Extract to**:
   ```bash
   /bigdata/userhome/students/$USER/data/nsd/
   ```

3. **Expected structure**:
   ```
   /bigdata/userhome/students/$USER/data/nsd/
   ├── nsddata/
   │   ├── ppdata/          # Preprocessed data
   │   ├── freesurfer/      # Brain anatomy
   │   └── experiments/     # Experiment info
   ├── nsddata_betas/
   │   └── ppdata/
   │       └── subj01/      # Subject-specific betas
   │           ├── func1mm/
   │           └── func1pt8mm/
   └── nsddata_stimuli/
       ├── stimuli/         # 73,000 images
       └── stimuli_metadata/
   ```

4. **Update .env**:
   ```bash
   NSD_DATA_ROOT=/bigdata/userhome/students/$USER/data/nsd
   ```

#### Option B: Use S3 Backend (If Available)

If you have AWS credentials and the data is on S3:

```bash
# In .env
ALLOW_S3_ONLY=1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

The pipeline can stream data from S3 without downloading everything.

#### Option C: Use Subset (For Testing)

For quick testing, you can:
1. Download only `subj01` beta files
2. Download first 1000 stimuli images
3. The pipeline will work with partial data

### Verify Dataset

```bash
python scripts/utils/verify_dataset.py
```

Expected output:
```
========================================
NSD dataset verification
========================================
[PASS] nsd_root: found: /bigdata/.../nsd
[PASS] betas_root: found: .../nsddata_betas/ppdata/subj01
[PASS] stimuli: found: .../nsddata_stimuli/stimuli
========================================
PASS: dataset looks usable
```

---

## 2️⃣ Pre-trained Models (Auto-downloaded)

### What models?

| Model | Size | Purpose | Auto-download? |
|-------|------|---------|----------------|
| Stable Diffusion 2.1 | ~5GB | Image generation | ✅ Yes |
| CLIP ViT-L/14 | ~2GB | Image embeddings | ✅ Yes |
| OpenCLIP models | ~2GB | Alternative embeddings | ✅ Yes |

### How to download?

**Automatic** (recommended):
```bash
make prepare
```

This will:
- Download Stable Diffusion 2.1 from HuggingFace
- Download CLIP models
- Cache them in `$CACHE_ROOT/hf/`

**Manual**:
```bash
python scripts/fetch_models.py
```

### Model Locations

After download:
```
/bigdata/userhome/students/$USER/cache/hf/
├── hub/                    # HuggingFace Hub cache
│   └── models--stabilityai--stable-diffusion-2-1/
├── transformers/           # CLIP text encoders
└── diffusers/              # Diffusion models
```

### Gated Models

Some models require HuggingFace authentication:

1. **Get token** from: https://huggingface.co/settings/tokens
2. **Add to .env**:
   ```bash
   HF_TOKEN=hf_YourTokenHere
   ```

### Skip Models (For CPU-only Testing)

If you just want to test the pipeline without diffusion:

```bash
make prepare --skip-models
```

---

## 3️⃣ CLIP Embeddings Cache (Auto-generated)

### What is it?

Pre-computed CLIP embeddings for NSD stimulus images. Instead of encoding each image every time, we cache them.

### Size
- ~5GB for all 73,000 NSD images
- ~500MB for subset of 10,000 images

### How to generate?

**Automatic** (on first run):
The pipeline automatically generates embeddings the first time you run an experiment.

**Manual**:
```bash
python scripts/build_clip_cache.py --subject subj01
```

### Cache Location

```
/bigdata/userhome/students/$USER/cache/clip_embeddings/
├── nsd_subj01_ViT-L-14.pkl
├── nsd_subj01_ViT-B-32.pkl
└── ...
```

---

## 🚀 Quick Setup: Step by Step

### Complete Setup (Fresh Start)

```bash
# 1. Clone repo
cd /bigdata/userhome/students/$USER
git clone <repo-url> Bachelor_V2
cd Bachelor_V2

# 2. Configure paths
cp .env.jupyterhub .env
nano .env  # Set USER and paths

# 3. Install Python environment
./setup.sh
source .venv/bin/activate

# 4. Download NSD dataset
# Do this manually from naturalscenesdataset.org
# Extract to /bigdata/userhome/students/$USER/data/nsd/

# 5. Download models and verify data
make prepare

# 6. Verify everything
make preflight

# 7. Run smoke test
python scripts/training/train.py --config configs/experiments/smoke_test.yaml --max_steps 1
```

### Minimal Setup (Testing Only)

If you just want to test the pipeline structure without full data:

```bash
# 1-3: Same as above (clone, configure, install)

# 4. Skip NSD for now
export ALLOW_S3_ONLY=1  # Or use dummy data

# 5. Download only models
make prepare --skip-models

# 6. Run tests with minimal data
python scripts/training/train.py --config configs/experiments/smoke_test.yaml --max_steps 1
```

---

## 📊 Disk Space Requirements

| Component | Size | Location |
|-----------|------|----------|
| NSD Dataset (full) | ~300GB | `$NSD_DATA_ROOT` |
| NSD Dataset (1 subject) | ~40GB | `$NSD_DATA_ROOT` |
| Pre-trained models | ~10GB | `$CACHE_ROOT/hf/` |
| CLIP embeddings | ~5GB | `$CACHE_ROOT/clip_embeddings/` |
| Experiment outputs | ~10-50GB | `$RUNS_DIR` |
| **Total (full)** | **~350GB** | |
| **Total (1 subject)** | **~100GB** | |

**Recommended**: Ensure at least **150GB free** on `/bigdata/`

Check available space:
```bash
df -h /bigdata
```

---

## 🔍 Verification Commands

### Check NSD Dataset
```bash
python scripts/utils/verify_dataset.py
```

### Check Models
```bash
python scripts/fetch_models.py
```

### Check All (Comprehensive)
```bash
python scripts/utils/doctor.py
```

### Quick Status
```bash
make prepare --verify-only
```

---

## ⚠️ Common Issues

### Issue 1: NSD Dataset Not Found

**Error**:
```
[FAIL] nsd_root: missing: /bigdata/.../data/nsd
```

**Solution**:
```bash
# Download from https://naturalscenesdataset.org/
# Then update .env:
NSD_DATA_ROOT=/path/to/your/nsd/data
```

### Issue 2: Models Download Failed

**Error**:
```
ERROR: failed to download model 'stabilityai/stable-diffusion-2-1'
```

**Solutions**:
1. **Check internet**: `ping huggingface.co`
2. **Check disk space**: `df -h`
3. **Try manual download**:
   ```bash
   python scripts/fetch_models.py
   ```
4. **Use HF token** (if gated):
   ```bash
   echo "HF_TOKEN=hf_your_token" >> .env
   ```

### Issue 3: Out of Disk Space

**Error**:
```
No space left on device
```

**Solutions**:
```bash
# Check space
df -h /bigdata

# Clean old runs
rm -rf runs/20260101_*

# Clean old caches
rm -rf cache/clip_embeddings/old_*

# Use smaller dataset (single subject only)
```

### Issue 4: Slow Downloads

**Solutions**:
1. **Use mirrors** (if available)
2. **Download during off-peak hours**
3. **Use tmux** so downloads persist:
   ```bash
   tmux new -s download
   make prepare
   # Ctrl+B, D to detach
   ```

---

## 📦 What Each Script Does

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `setup.sh` | Complete data setup | After initial install |
| `scripts/utils/verify_dataset.py` | Check NSD dataset | Anytime to verify |
| `fetch_models.py` | Download HF models | If models missing |
| `scripts/utils/doctor.py` | Full system check | Before important runs |

---

## 🎯 Recommended Workflow

**Day 1**: Initial Setup
```bash
./setup.sh
make prepare
```

**Day 2**: Verify and Test
```bash
source .venv/bin/activate
make preflight
python scripts/training/train.py --config configs/experiments/smoke_test.yaml --max_steps 1
```

**Day 3+**: Run Experiments
```bash
source .venv/bin/activate
python scripts/training/train.py --config configs/experiments/exp0_baseline.yaml
```

---

## 💡 Pro Tips

1. **Download NSD during off-hours** - it's 300GB
2. **Use tmux for downloads** - they take hours
3. **Download one subject first** - test before getting all
4. **Keep models cached** - don't delete `$CACHE_ROOT/hf/`
5. **Generate CLIP cache once** - reuse across experiments
6. **Monitor disk space** - set up alerts if possible

---

## 📞 Getting Help

**Dataset issues**:
```bash
python scripts/utils/verify_dataset.py --help
```

**Model issues**:
```bash
python scripts/fetch_models.py
cat logs/fetch_models.log
```

**Full diagnostic**:
```bash
python scripts/utils/doctor.py
```

**Verification only** (no downloads):
```bash
make prepare --verify-only
```

---

## ✅ Success Checklist

After running `make prepare`, you should have:

- [x] NSD dataset at `$NSD_DATA_ROOT`
- [x] Models in `$CACHE_ROOT/hf/`
- [x] CLIP cache directory created
- [x] `verify_dataset.py` shows PASS
- [x] `fetch_models.py` shows "already cached"
- [x] At least 50GB free disk space

**All checked? You're ready to run experiments!** 🚀
