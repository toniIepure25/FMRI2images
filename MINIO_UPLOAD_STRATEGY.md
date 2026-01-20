# MinIO Setup Strategy - Using Your Private Bucket

> **Your Bucket**: `6f628397-de73-4c99-b4b6-51e20859fea2`  
> **Status**: Empty, ready to use ✅  
> **Access**: Private

---

## 🎯 Recommended Strategy (FASTEST)

### **Option 1: Download on Server → Upload to MinIO** ⭐ **BEST**

This is the fastest approach because:
- ✅ Server has high-speed internet (~1-10 Gbps)
- ✅ MinIO is on same network as server (very fast upload)
- ✅ Your PC doesn't need to download/upload anything (saves hours/days)
- ✅ You can do everything via terminal commands

**Estimated Time**:
- Download on server: 2-3 hours
- Upload to MinIO from server: 30-60 minutes
- **Total: 3-4 hours** (fully automated)

**vs downloading on PC**:
- Download on PC: 6-12 hours (depends on your home internet)
- Upload from PC to MinIO: 8-16 hours
- **Total: 14-28 hours** (requires your PC to stay on)

---

## 📋 Step-by-Step Instructions

### **Step 1: Get MinIO Access Keys**

You need to create access keys for your bucket. Click on **"Access Keys"** in the left menu, then create a new key.

**Or ask IT admin** to provide you with:
```
Access Key ID: YOUR_ACCESS_KEY
Secret Access Key: YOUR_SECRET_KEY
Endpoint: https://ubbc1u.ro:9000 (or the correct URL)
```

---

### **Step 2: SSH to JupyterHub Server**

```bash
# SSH to your JupyterHub instance
ssh your_username@jupyterhub.server.address

# Navigate to your work directory
cd /bigdata/userhome/students/$USER
```

---

### **Step 3: Clone Repository (if not already done)**

```bash
git clone https://github.com/toniIepure25/FMRI2images.git Bachelor_V2
cd Bachelor_V2
```

---

### **Step 4: Configure Environment with MinIO**

Create/update `.env` file with your MinIO credentials:

```bash
# Copy template
cp .env.jupyterhub .env

# Edit to add MinIO settings
nano .env
```

Add these lines at the end of `.env`:

```bash
# MinIO Configuration
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_HERE
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY_HERE
AWS_DEFAULT_REGION=us-east-1

# Your bucket name
MINIO_BUCKET=6f628397-de73-4c99-b4b6-51e20859fea2

# S3 paths in your bucket
NSD_S3_ROOT=s3://6f628397-de73-4c99-b4b6-51e20859fea2/nsddata/
NSD_S3_BETAS=s3://6f628397-de73-4c99-b4b6-51e20859fea2/nsddata_betas/
NSD_S3_STIMULI=s3://6f628397-de73-4c99-b4b6-51e20859fea2/nsddata_stimuli/

# Enable S3 mode
ALLOW_S3_ONLY=1
S3_CACHE_DIR=/bigdata/userhome/students/${USER}/.cache/s3_cache
```

---

### **Step 5: Install Python Environment**

```bash
bash scripts/setup_env.sh
source activate_env.sh
```

---

### **Step 6: Download Data on Server**

Download directly to server storage (not MinIO yet):

```bash
# Download full subject01 data (~50GB, 2-3 hours)
bash scripts/download_full_nsd_subj01.sh

# Download stimuli images (~15GB, 30 min)
mkdir -p data/nsd/nsddata_stimuli/stimuli/nsd
cd data/nsd/nsddata_stimuli/stimuli/nsd
wget -r -np -nH --cut-dirs=4 \
    https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/
cd ~/Bachelor_V2
```

---

### **Step 7: Build CLIP Caches on Server**

Build the caches while you have the data locally:

```bash
# Build index
python3 scripts/build_full_subj01_index.py

# Build CLIP cache (~1-2 hours)
bash scripts/build_clip_for_training.sh

# Build target CLIP cache (~1-2 hours)
python3 scripts/build_target_clip_cache_robust.py \
    --cache_dir cache \
    --output cache/target_clip_cache_sd21.h5 \
    --batch_size 256

# Build preprocessors
bash scripts/build_all_preprocessors.sh
```

---

### **Step 8: Install MinIO Client on Server**

```bash
# Download MinIO client
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Configure MinIO alias (replace with your credentials)
mc alias set uniminio https://ubbc1u.ro:9000 \
    YOUR_ACCESS_KEY \
    YOUR_SECRET_KEY

# Test connection
mc ls uniminio/
# Should show your bucket: 6f628397-de73-4c99-b4b6-51e20859fea2
```

---

### **Step 9: Upload Data to MinIO from Server**

Now upload everything to MinIO (fast because server → MinIO is local network):

```bash
# Upload betas (~50GB, 30-60 min)
mc cp --recursive \
    data/nsd/nsddata_betas/ \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/nsddata_betas/

# Upload stimuli (~15GB, 15-30 min)
mc cp --recursive \
    data/nsd/nsddata_stimuli/ \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/nsddata_stimuli/

# Upload CLIP caches (~2GB, 2-5 min)
mc cp --recursive \
    cache/clip_embeddings/ \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/clip_embeddings/

# Upload target cache (~4GB, 5-10 min)
mc cp cache/target_clip_cache_sd21.h5 \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/

# Upload preprocessors (<1MB, instant)
mc cp cache/*.pkl \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/

# Verify upload
mc ls -r uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/
```

---

### **Step 10: Optional - Delete Local Data (Save Space)**

After confirming upload is complete:

```bash
# Verify MinIO has everything
mc ls -r uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/

# If all good, free up disk space
rm -rf data/nsd/nsddata_betas/*
rm -rf data/nsd/nsddata_stimuli/*

# Keep caches local for faster training
# Or delete if you want to use MinIO exclusively
```

---

### **Step 11: Run Experiments Using MinIO**

Update experiment config to use S3/MinIO paths:

```yaml
# configs/experiments/minio_experiment.yaml

dataset:
  nsd_root: ${NSD_S3_ROOT}
  betas_root: ${NSD_S3_BETAS}
  stimuli_root: ${NSD_S3_STIMULI}
  cache_dir: ${S3_CACHE_DIR}  # Local cache for downloads
  use_s3: true

data_loading:
  num_workers: 16  # Fewer workers for network I/O
  prefetch_factor: 2
```

Run experiments:

```bash
python3 -m fmri2img.train \
    --config configs/experiments/minio_experiment.yaml \
    --allow-s3-only
```

---

## 🚫 Why NOT Download on Your PC?

**Don't do this** unless server approach fails:

```
Your PC → Download NSD (6-12 hours)
Your PC → Upload to MinIO (8-16 hours)
Total: 14-28 hours + your PC must stay on
```

**vs Server approach:**
```
Server → Download NSD (2-3 hours)
Server → Upload to MinIO (30-60 minutes)
Total: 3-4 hours + fully automated
```

**Server is 5-7x faster!** ⚡

---

## 📊 Data Size Breakdown

| Data | Size | Download Time (Server) | Upload Time (Server→MinIO) |
|------|------|------------------------|----------------------------|
| Beta files (40 sessions) | ~50GB | 2-3 hours | 30-60 min |
| Stimuli images (73k) | ~15GB | 30 min | 15-30 min |
| CLIP cache | ~2GB | N/A (built on server) | 2-5 min |
| Target cache | ~4GB | N/A (built on server) | 5-10 min |
| Preprocessors | <1MB | N/A | instant |
| **TOTAL** | **~71GB** | **2.5-3.5 hours** | **50-100 min** |

---

## ✅ Verification Checklist

After upload, verify:

```bash
# Check bucket contents
mc ls -r uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/

# Should see:
# nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/
#   ├── betas_session01.nii.gz
#   ├── betas_session02.nii.gz
#   ...
#   └── betas_session40.nii.gz
# nsddata_stimuli/stimuli/nsd/
#   ├── nsd00001.png
#   ├── nsd00002.png
#   ...
#   └── nsd73000.png
# cache/
#   ├── clip_embeddings/
#   ├── target_clip_cache_sd21.h5
#   └── *.pkl

# Get total size
mc du uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/
# Should show ~71GB
```

---

## 🆘 Troubleshooting

### **Issue: Access Keys Not Available**

In MinIO web UI:
1. Click "Access Keys" in left menu
2. Click "Create Access Key" 
3. Copy Access Key ID and Secret Key
4. Save them securely!

### **Issue: Upload Fails**

```bash
# Check connection
mc alias ls

# Test small upload
echo "test" > test.txt
mc cp test.txt uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/
rm test.txt

# If works, retry full upload
```

### **Issue: Slow Upload**

```bash
# Upload in parallel (faster)
mc cp --recursive --parallel 8 \
    data/nsd/nsddata_betas/ \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/nsddata_betas/
```

---

## 🎯 Summary

**Your bucket**: `6f628397-de73-4c99-b4b6-51e20859fea2` ✅  
**Strategy**: Download on server → Upload to MinIO ✅  
**Total time**: 3-4 hours (automated) ✅  
**Status**: Ready to proceed! ✅

---

**Next**: Get your MinIO access keys and start Step 2! 🚀
