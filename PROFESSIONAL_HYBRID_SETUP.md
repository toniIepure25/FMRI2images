# Professional Hybrid Setup: AWS S3 + MinIO

> **Optimal Strategy**: Use AWS S3 for NSD data + MinIO for your artifacts

---

## 🎯 **BEST Professional Approach**

### **Use BOTH - Hybrid Configuration**

```
AWS S3 (public, already there)
├── NSD Beta files (~50GB)          ← Use directly
├── NSD Stimuli images (~15GB)      ← Use directly
└── Metadata files                  ← Use directly

MinIO (your private bucket)
├── CLIP caches (~2GB)              ← Store here
├── Target CLIP cache (~4GB)        ← Store here
├── Preprocessors (<1MB)            ← Store here
├── Checkpoints (varies)            ← Store here
└── Experiment results              ← Store here
```

**Why this is optimal:**
- ✅ **No download/upload needed** for 65GB NSD data (saves 3-4 hours!)
- ✅ **AWS S3 is globally optimized** - very fast from anywhere
- ✅ **MinIO stores your generated data** - share with colleagues
- ✅ **Redundancy**: AWS for source data, MinIO for your work
- ✅ **Professional**: Separate concerns (public data vs your work)

---

## 📋 Complete Setup Instructions

### **Your MinIO Credentials**

```bash
Endpoint: https://ubbc1u.ro:9000
Access Key: cjChy0M8fUFIUWJ0SVd9
Secret Key: TeIDlHx94IJS8ACQPNsmyFqo8IfnKbmzheKYkfxf
Bucket: 6f628397-de73-4c99-b4b6-51e20859fea2
```

---

### **Step 1: Configure Hybrid Environment**

Create your `.env` file with hybrid configuration:

```bash
cd /path/to/Bachelor_V2

# Copy template
cp .env.jupyterhub .env

# Edit configuration
nano .env
```

Add this **optimized hybrid configuration**:

```bash
# =============================================================================
# HYBRID CONFIGURATION: AWS S3 (NSD) + MinIO (Your Artifacts)
# =============================================================================

# User
USER=${USER:-your_username_here}

# Local paths (for active work)
DATA_DIR=/bigdata/userhome/students/${USER}/data
CACHE_DIR=/bigdata/userhome/students/${USER}/.cache
CHECKPOINT_ROOT=/bigdata/userhome/students/${USER}/checkpoints
RUNS_DIR=/bigdata/userhome/students/${USER}/runs

# =============================================================================
# AWS S3 - NSD Dataset (Public, Use Directly)
# =============================================================================

# Use AWS S3 directly for NSD data (no credentials needed - public access)
NSD_S3_ROOT=s3://natural-scenes-dataset/nsddata
NSD_S3_BETAS=s3://natural-scenes-dataset/nsddata_betas
NSD_S3_STIMULI=s3://natural-scenes-dataset/nsddata_stimuli

# Enable S3 streaming for NSD data
USE_S3_FOR_NSD=true
S3_CACHE_DIR=${CACHE_DIR}/s3_cache

# =============================================================================
# MinIO - Your Generated Artifacts (Private Storage)
# =============================================================================

# MinIO endpoint and credentials
MINIO_ENDPOINT_URL=https://ubbc1u.ro:9000
MINIO_ACCESS_KEY_ID=cjChy0M8fUFIUWJ0SVd9
MINIO_SECRET_ACCESS_KEY=TeIDlHx94IJS8ACQPNsmyFqo8IfnKbmzheKYkfxf
MINIO_BUCKET=6f628397-de73-4c99-b4b6-51e20859fea2

# MinIO paths for your generated artifacts
MINIO_CLIP_CACHE=s3://6f628397-de73-4c99-b4b6-51e20859fea2/cache/clip_embeddings
MINIO_TARGET_CACHE=s3://6f628397-de73-4c99-b4b6-51e20859fea2/cache/target_clip_cache_sd21.h5
MINIO_PREPROCESSORS=s3://6f628397-de73-4c99-b4b6-51e20859fea2/cache/preprocessors
MINIO_CHECKPOINTS=s3://6f628397-de73-4c99-b4b6-51e20859fea2/checkpoints

# Enable MinIO for artifact storage
USE_MINIO_FOR_ARTIFACTS=true

# =============================================================================
# System Settings (Optimized for A100 40GB)
# =============================================================================

CUDA_VISIBLE_DEVICES=0
NUM_WORKERS=32
DEFAULT_SEED=42
LOG_LEVEL=INFO
```

---

### **Step 2: Install Python Environment**

```bash
# Install environment
bash scripts/setup_env.sh

# Activate
source activate_env.sh

# Install S3 support (if not already installed)
pip install s3fs fsspec aiobotocore
```

---

### **Step 3: Configure MinIO Client**

```bash
# Download MinIO client
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Configure MinIO alias
mc alias set uniminio https://ubbc1u.ro:9000 \
    cjChy0M8fUFIUWJ0SVd9 \
    TeIDlHx94IJS8ACQPNsmyFqo8IfnKbmzheKYkfxf

# Test connection
mc ls uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/
```

---

### **Step 4: Build CLIP Caches Locally (Stream from AWS S3)**

Your codebase already supports streaming from S3! Build caches directly:

```bash
# Build index (downloads small CSV from AWS S3)
python3 scripts/build_full_subj01_index.py

# Build CLIP cache (streams images from AWS S3)
# No download needed - processes directly from S3!
python3 scripts/build_target_clip_cache_robust.py \
    --cache_dir cache \
    --output cache/target_clip_cache_sd21.h5 \
    --batch_size 256 \
    --device cuda:0 \
    --use-s3  # This flag enables S3 streaming

# Build standard CLIP cache
bash scripts/build_clip_for_training.sh

# Build preprocessors
bash scripts/build_all_preprocessors.sh
```

**Time**: 2-3 hours (processes images directly from AWS S3, no download!)

---

### **Step 5: Upload Generated Artifacts to MinIO**

Upload your generated caches to MinIO for sharing:

```bash
# Upload CLIP caches
mc cp --recursive cache/clip_embeddings/ \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/clip_embeddings/

# Upload target cache
mc cp cache/target_clip_cache_sd21.h5 \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/

# Upload preprocessors
mc cp cache/center_pcr_k8.pkl \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/
mc cp cache/center_whiten.pkl \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/

# Verify upload
mc ls -r uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/
```

**Time**: 5-10 minutes (only ~6GB of artifacts, not 65GB of NSD data!)

---

### **Step 6: Create Hybrid Experiment Config**

Create optimized config for A100 with hybrid storage:

```bash
nano configs/experiments/a100_hybrid.yaml
```

```yaml
# A100 Hybrid Configuration - AWS S3 + MinIO
base: configs/base.yaml

experiment:
  name: "a100_hybrid_production"
  description: "Production run with AWS S3 + MinIO hybrid storage"
  tags: ["production", "a100", "hybrid-storage", "s3"]

# Use AWS S3 for NSD data (streams directly, no download)
dataset:
  subject: "subj01"
  index_file: "data/indices/nsd_index/subj01_full_train.csv"
  
  # AWS S3 paths (public access, no credentials needed)
  nsd_root: "s3://natural-scenes-dataset/nsddata"
  betas_root: "s3://natural-scenes-dataset/nsddata_betas"
  stimuli_root: "s3://natural-scenes-dataset/nsddata_stimuli"
  
  # Local cache for S3 downloads (speeds up repeated access)
  cache_dir: "/bigdata/userhome/students/${USER}/.cache/s3_cache"
  
  # CLIP cache (can use local or MinIO)
  clip_cache_path: "cache/clip_embeddings/clip_vitb32_embeddings.h5"
  
  # Enable S3 streaming
  use_s3: true
  s3_anon: true  # AWS NSD bucket is public

# Data loading optimized for S3 + A100 40GB
data_loading:
  batch_size: 512
  num_workers: 16  # Fewer workers for S3 I/O
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2  # Conservative for S3

# Training
training:
  max_epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01
  mixed_precision: true
  gradient_clip: 1.0

# Model
model:
  type: "ridge"
  ridge_alpha: 100000
  normalize: true

# Save checkpoints to MinIO
checkpoints:
  save_dir: "/bigdata/userhome/students/${USER}/checkpoints"
  upload_to_minio: true  # Auto-upload to MinIO after training
  minio_path: "s3://6f628397-de73-4c99-b4b6-51e20859fea2/checkpoints"

# System
system:
  device: "cuda:0"
  seed: 42
  num_threads: 40
```

---

### **Step 7: Run Experiments with Hybrid Setup**

```bash
# Activate environment
source activate_env.sh

# Run experiment (streams from AWS S3, saves to MinIO)
python3 -m fmri2img.train \
    --config configs/experiments/a100_hybrid.yaml \
    --gpu 0

# Or run all experiments
tmux new -s experiments
bash scripts/run_all_experiments.sh 0
```

---

## 🚀 **Performance Comparison**

| Approach | Setup Time | Disk Used | Speed | Professional |
|----------|------------|-----------|-------|--------------|
| **Download to local** | 3-4 hours | 65GB | Fast | ⭐⭐⭐ |
| **Download + Upload to MinIO** | 4-5 hours | 65GB | Medium | ⭐⭐ |
| **Hybrid (AWS + MinIO)** ⭐ | 2-3 hours | 6GB | Fast | ⭐⭐⭐⭐⭐ |

**Winner**: Hybrid approach!
- ✅ Saves 60+ hours of download/upload
- ✅ Uses only 6GB local disk (vs 65GB)
- ✅ AWS S3 is globally optimized
- ✅ Professional separation of concerns
- ✅ Easy to share your artifacts via MinIO

---

## 📊 **Storage Breakdown**

### **AWS S3 (Use directly - FREE, already there)**
```
s3://natural-scenes-dataset/
├── nsddata/                         # Metadata
├── nsddata_betas/                   # ~50GB fMRI data
│   └── ppdata/subj01/
│       └── func1pt8mm/
│           └── betas_fithrf_GLMdenoise_RR/
│               ├── betas_session01.nii.gz
│               ├── ...
│               └── betas_session40.nii.gz
└── nsddata_stimuli/                 # ~15GB images
    └── stimuli/nsd/
        ├── nsd00001.png
        ├── ...
        └── nsd73000.png
```

### **MinIO (Your private artifacts - ~6GB)**
```
s3://6f628397-de73-4c99-b4b6-51e20859fea2/
├── cache/
│   ├── clip_embeddings/             # ~2GB
│   ├── target_clip_cache_sd21.h5    # ~4GB
│   ├── center_pcr_k8.pkl           # <1MB
│   └── center_whiten.pkl           # <1MB
├── checkpoints/                     # Varies
│   ├── ridge/
│   ├── mlp/
│   └── clip_adapter/
└── results/                         # Your outputs
    └── experimental_results/
```

---

## 🔧 **Advanced: Automatic MinIO Upload Script**

Create a script to auto-upload checkpoints after training:

```bash
nano scripts/sync_to_minio.sh
```

```bash
#!/bin/bash
# Sync checkpoints and results to MinIO

MC_ALIAS="uniminio"
BUCKET="6f628397-de73-4c99-b4b6-51e20859fea2"

echo "Syncing to MinIO..."

# Sync checkpoints
mc mirror --overwrite \
    checkpoints/ \
    ${MC_ALIAS}/${BUCKET}/checkpoints/

# Sync results
mc mirror --overwrite \
    experimental_results/ \
    ${MC_ALIAS}/${BUCKET}/results/

echo "✅ Sync complete!"
```

Run after training:
```bash
bash scripts/sync_to_minio.sh
```

---

## ✅ **Verification**

### **Test AWS S3 Access**

```bash
# Test streaming from AWS S3
python3 -c "
import fsspec
fs = fsspec.filesystem('s3', anon=True)
files = fs.ls('natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/')
print(f'Found {len(files)} beta files on AWS S3')
"
```

**Expected**: `Found 40 beta files on AWS S3`

### **Test MinIO Access**

```bash
# Test MinIO connection
mc ls uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/
```

**Expected**: Lists your bucket contents (or empty if just created)

---

## 🎯 **Summary**

### **What You Do:**

1. ✅ Use AWS S3 directly for NSD data (no download!)
2. ✅ Build CLIP caches locally (streams from AWS S3)
3. ✅ Upload caches to MinIO (~6GB, 5-10 min)
4. ✅ Train models (streams from AWS S3)
5. ✅ Save checkpoints to MinIO

### **Benefits:**

- **Time saved**: 3-4 hours (no NSD download/upload)
- **Disk saved**: 60GB (only cache ~6GB locally)
- **Network**: AWS S3 is fast globally
- **Professional**: Proper separation of source data vs artifacts
- **Sharing**: Colleagues can use your MinIO caches

### **Total Setup Time:**

- Environment setup: 10 min
- Build caches (streaming from S3): 2-3 hours
- Upload to MinIO: 5-10 min
- **Total: ~3 hours** vs 6-7 hours for download+upload approach!

---

**🚀 Ready to start! Use the hybrid configuration above.**
