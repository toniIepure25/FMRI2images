# MinIO Setup Guide for fMRI2IMG

> **How to use MinIO Object Store instead of local disk for the NSD dataset**

---

## 🎯 Should You Use MinIO?

### ✅ **YES, Use MinIO If:**

1. **You don't want to download 300GB** locally
2. **The technician already has NSD on MinIO** (saves re-uploading)
3. **Multiple students share the same data** (saves disk space)
4. **You have good network connection** to the MinIO server
5. **Your local disk is limited** (<350GB available)

### ❌ **NO, Use Local Disk If:**

1. **You have enough disk space** (350GB+)
2. **Network is slow/unreliable** (MinIO needs good connectivity)
3. **You need maximum performance** (local disk is faster)
4. **You're doing many experiments** (streaming overhead adds up)

---

## 🔍 What MinIO Is

**MinIO** = S3-compatible object storage (like AWS S3, but self-hosted)

Your codebase **already supports S3**, so it works with MinIO automatically! 🎉

The screenshot shows you have MinIO running at:
```
https://ubbc{1u}.ro/apps/s3hub1/browser
```

---

## 📦 What Your Codebase Already Has

Good news! Your project has **full S3/MinIO support built-in**:

✅ `src/fmri2img/io/s3.py` - S3 filesystem wrapper  
✅ `fsspec` + `s3fs` libraries - Handle S3 streaming  
✅ Local caching - Downloads files once, caches locally  
✅ Automatic fallback - Uses local if S3 fails  
✅ `--allow-s3-only` flag - Works without local data  

**You're already 90% ready to use MinIO!**

---

## 🚀 Setup Steps

### **Step 1: Get MinIO Credentials**

Ask your technician for:

```bash
# MinIO endpoint
MINIO_ENDPOINT=https://ubbc1u.ro:9000  # Or whatever the actual endpoint is

# Access credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Bucket name where NSD data is stored
MINIO_BUCKET=nsd-data  # Or whatever it's called
```

### **Step 2: Configure Environment**

Add to your `.env` file:

```bash
# MinIO/S3 Configuration
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000  # MinIO server
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1  # Doesn't matter for MinIO, but required

# Tell pipeline to allow S3-only mode (no local data required)
ALLOW_S3_ONLY=1

# S3 paths (adjust bucket/prefix as needed)
NSD_S3_ROOT=s3://nsd-data/nsddata/
NSD_S3_BETAS=s3://nsd-data/nsddata_betas/
NSD_S3_STIMULI=s3://nsd-data/nsddata_stimuli/

# Local cache for downloaded files (saves bandwidth)
S3_CACHE_DIR=/bigdata/userhome/students/${USER}/.cache/s3_cache
```

### **Step 3: Test Connection**

```bash
# Activate your environment
source activate_env.sh

# Test MinIO connection with Python
python -c "
import os
os.environ['AWS_ENDPOINT_URL'] = 'https://ubbc1u.ro:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'your_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret'

import s3fs
fs = s3fs.S3FileSystem(
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY']
)
print('Buckets:', fs.ls(''))
"
```

**Expected output**: List of buckets (should include your NSD data bucket)

### **Step 4: Verify MinIO NSD Data**

```bash
# Check if NSD data structure exists on MinIO
python scripts/verify_dataset.py --allow-s3-only
```

**Expected**: Should find data on S3/MinIO instead of locally

### **Step 5: Update Your Configs**

Edit experiment configs to use S3 paths:

```yaml
# configs/experiments/your_experiment.yaml

dataset:
  index_root: data/indices/nsd_index  # Still local (small files)
  nsd_root: ${NSD_S3_ROOT}  # Points to MinIO
  betas_root: ${NSD_S3_BETAS}  # Points to MinIO
  stimuli_root: ${NSD_S3_STIMULI}  # Points to MinIO
  cache_dir: ${S3_CACHE_DIR}  # Local cache for downloads
  
data_loading:
  num_workers: 2  # Fewer workers for network I/O
  prefetch_factor: 2  # Less aggressive prefetching
```

---

## 🎮 Running with MinIO

### **Option A: Modify Existing Scripts**

Your existing scripts will work automatically if `.env` is configured:

```bash
# Just run normally - it will use S3/MinIO automatically
source activate_env.sh
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml
```

The codebase detects S3 paths and handles them automatically!

### **Option B: Manual Python Usage**

```python
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
import os

# Set MinIO endpoint
os.environ['AWS_ENDPOINT_URL'] = 'https://ubbc1u.ro:9000'

# Get S3 filesystem
s3_fs = get_s3_filesystem(cache_storage=".cache/s3_cache")

# Load data from MinIO
loader = NIfTILoader(s3_fs)
beta_data = loader.load("s3://nsd-data/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz")

print(f"Beta shape: {beta_data.shape}")
```

---

## 🔧 How It Works

### **Automatic Caching**

1. **First access**: Downloads from MinIO → saves to local cache
2. **Subsequent access**: Reads from local cache (fast!)
3. **Cache location**: `$S3_CACHE_DIR` (configurable)

### **Cache Structure**

```
/bigdata/userhome/students/$USER/.cache/s3_cache/
├── <hash1>.nii.gz  # Cached beta file
├── <hash2>.nii.gz
└── <hash3>.hdf5    # Cached HDF5 file
```

**Cache size**: ~20-50GB (only caches what you use)

### **Network Optimization**

Your code already has:
- ✅ Chunked streaming (1MB blocks)
- ✅ Smart caching (downloads once)
- ✅ Retry logic (handles network hiccups)
- ✅ Memory-safe loading (no OOM errors)

---

## 📊 Performance Comparison

| Aspect | Local Disk | MinIO |
|--------|------------|-------|
| **Setup time** | 2-8 hours (download 300GB) | 5 minutes (just config) |
| **Disk usage** | 300GB | 20-50GB (cache only) |
| **First run** | Fast | Slower (downloads on-demand) |
| **Subsequent runs** | Fast | Fast (cached) |
| **Shared data** | No (each student downloads) | Yes (one copy for all) |
| **Network dependency** | None | Yes (needs good connection) |

### **Recommendation**

1. **Start with MinIO** (fast setup, see if it works)
2. **If slow**, download frequently-used subjects locally
3. **Hybrid approach**: Cache `subj01` locally, stream others from MinIO

---

## 🛠️ Configuration Examples

### **Full MinIO Setup**

```bash
# .env
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
ALLOW_S3_ONLY=1

# All data from MinIO
NSD_DATA_ROOT=s3://nsd-data/nsddata/
S3_CACHE_DIR=/bigdata/userhome/students/${USER}/.cache/s3_cache
```

### **Hybrid Setup (MinIO + Local)**

```bash
# .env
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Use local for subj01, MinIO for others
NSD_DATA_ROOT=/bigdata/userhome/students/${USER}/data/nsd
NSD_S3_ROOT=s3://nsd-data/nsddata/

# In your script: decide per subject
# subj01: Use $NSD_DATA_ROOT (local)
# subj02-08: Use $NSD_S3_ROOT (MinIO)
```

---

## 🔍 Checking What's on MinIO

### **List Buckets**

```bash
python -c "
import s3fs, os
fs = s3fs.S3FileSystem(
    endpoint_url='https://ubbc1u.ro:9000',
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY']
)
for bucket in fs.ls(''):
    print(f'Bucket: {bucket}')
"
```

### **List NSD Files**

```bash
python -c "
import s3fs, os
fs = s3fs.S3FileSystem(
    endpoint_url='https://ubbc1u.ro:9000',
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY']
)
# Adjust bucket name
files = fs.ls('nsd-data/nsddata_betas/ppdata/subj01/')
for f in files[:10]:  # First 10
    print(f)
"
```

---

## 🆘 Troubleshooting

### **Problem: "Connection refused"**

**Cause**: Wrong endpoint URL

**Solution**:
```bash
# Check with technician - MinIO might be at different port
# Common ports: 9000, 9001
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000  # Try this
AWS_ENDPOINT_URL=https://ubbc1u.ro:9001  # Or this
AWS_ENDPOINT_URL=http://ubbc1u.ro:9000   # HTTP instead of HTTPS
```

### **Problem: "Access denied"**

**Cause**: Wrong credentials or bucket policy

**Solution**:
```bash
# Verify credentials with technician
# Check bucket name is correct
# Ensure you have read permissions
```

### **Problem: "Slow first run"**

**Cause**: Downloading data on-demand

**Solution**:
```bash
# Normal! First run downloads to cache
# Subsequent runs will be fast
# Or pre-download commonly used files:
python -c "
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
import os
os.environ['AWS_ENDPOINT_URL'] = 'your_endpoint'
s3_fs = get_s3_filesystem()
loader = NIfTILoader(s3_fs)
# This will cache the file
loader.load('s3://nsd-data/path/to/frequently/used/file.nii.gz')
"
```

### **Problem: "SSL certificate error"**

**Cause**: Self-signed certificate on MinIO server

**Solution**:
```bash
# Add to .env
AWS_NO_VERIFY_SSL=1

# Or in Python:
import os
os.environ['AWS_NO_VERIFY_SSL'] = '1'
```

### **Problem: "Out of cache space"**

**Cause**: Cache directory filling up

**Solution**:
```bash
# Check cache size
du -sh $S3_CACHE_DIR

# Clear cache if needed
rm -rf $S3_CACHE_DIR/*

# Or move cache to larger disk
S3_CACHE_DIR=/path/to/larger/disk/s3_cache
```

---

## 📋 Checklist for MinIO Setup

- [ ] Got MinIO credentials from technician
- [ ] Added credentials to `.env`
- [ ] Set `ALLOW_S3_ONLY=1` in `.env`
- [ ] Tested connection with `s3fs` Python script
- [ ] Verified NSD data exists on MinIO
- [ ] Configured `S3_CACHE_DIR` with enough space (~50GB)
- [ ] Updated experiment configs to use S3 paths
- [ ] Ran test experiment successfully

---

## 🎓 Questions to Ask Your Technician

1. **What's the exact MinIO endpoint URL?**
   - Example: `https://ubbc1u.ro:9000`

2. **What are the access credentials?**
   - Access Key ID and Secret Access Key

3. **What's the bucket name for NSD data?**
   - Example: `nsd-data`, `fmri-datasets`, etc.

4. **What's the directory structure inside?**
   - Example: `nsd-data/nsddata/`, `nsd-data/betas/`, etc.

5. **Is SSL/TLS enabled?**
   - If self-signed cert, you need `AWS_NO_VERIFY_SSL=1`

6. **Do I have read-only or read-write access?**
   - You only need read access for this project

7. **Is there bandwidth/quota limit?**
   - Important for large-scale experiments

---

## 🚀 Quick Start with MinIO

**Complete setup in 5 minutes**:

```bash
# 1. Add to .env (get values from technician)
cat >> .env << 'EOF'
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
ALLOW_S3_ONLY=1
S3_CACHE_DIR=/bigdata/userhome/students/${USER}/.cache/s3_cache
EOF

# 2. Activate environment
source activate_env.sh

# 3. Test connection
python -c "import s3fs, os; fs = s3fs.S3FileSystem(endpoint_url=os.environ['AWS_ENDPOINT_URL']); print('Buckets:', fs.ls(''))"

# 4. Verify data
python scripts/verify_dataset.py --allow-s3-only

# 5. Run experiment!
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

---

## 💡 Pro Tips

1. **Cache warmup**: Run smoke test first to cache commonly-used files
2. **Monitor cache size**: `watch du -sh $S3_CACHE_DIR`
3. **Network-aware configs**: Reduce `num_workers` for S3 backends
4. **Hybrid strategy**: Keep small index files local, stream large betas
5. **Tmux sessions**: Network hiccups won't kill your experiment
6. **Check before long runs**: Ensure MinIO is accessible before overnight experiments

---

## 🎯 Final Recommendation

**For your bachelor thesis**:

1. ✅ **YES, use MinIO** if:
   - Technician already has NSD there (saves time)
   - You have <350GB free disk
   - Network is reliable

2. **Start with MinIO**, then decide:
   - If it works well → stick with it
   - If it's slow → download `subj01` locally (40GB)
   - If very slow → download full dataset (300GB)

**Your codebase is already MinIO-ready** - just configure and go! 🚀

---

## 📚 Related Documentation

- `DATA_REQUIREMENTS.md` - Alternative data acquisition methods
- `COMPLETE_SETUP_GUIDE.md` - Full setup workflow
- `src/fmri2img/io/s3.py` - S3 implementation details
- `scripts/verify_dataset.py` - Dataset verification tool

---

**Need help?** Ask your technician about:
1. MinIO endpoint URL
2. Access credentials
3. Bucket structure
4. Network bandwidth/quotas

**Then follow this guide and you're set!** ✅
