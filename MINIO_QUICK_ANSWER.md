# MinIO Integration Summary

> **Quick answer: YES, MinIO fits your approach perfectly!**

---

## ✅ The Good News

Your codebase **already has full MinIO/S3 support** built in:

1. ✅ **S3 filesystem layer** (`src/fmri2img/io/s3.py`)
2. ✅ **Automatic local caching** (downloads once, caches locally)
3. ✅ **Smart fallback** (tries S3, falls back to local)
4. ✅ **Memory-safe streaming** (handles large files efficiently)
5. ✅ **Works with all scripts** (no code changes needed)

**You just need to configure it!**

---

## 🎯 Should You Use MinIO?

### **YES, if:**
- ✅ Technician already has NSD on MinIO (saves 300GB download)
- ✅ You have limited disk space (<350GB)
- ✅ Multiple students sharing data (avoids duplicates)
- ✅ Network is reliable

### **NO, if:**
- ❌ Network is slow/unreliable
- ❌ You have plenty of disk (>500GB free)
- ❌ You need maximum performance (local is faster)
- ❌ Data not yet on MinIO (no point uploading yourself)

### **Recommendation: HYBRID**
Start with MinIO, then:
- If it works well → stick with it
- If slow → cache `subj01` locally (40GB)
- If very slow → download full dataset (300GB)

---

## 📦 What You Need from Technician

Ask for these 4 things:

```bash
1. MinIO endpoint URL
   Example: https://ubbc1u.ro:9000

2. Access credentials
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY

3. Bucket name
   Example: nsd-data

4. Directory structure
   Example: 
   - s3://nsd-data/nsddata/
   - s3://nsd-data/nsddata_betas/
   - s3://nsd-data/nsddata_stimuli/
```

---

## 🚀 Setup (5 Minutes)

```bash
# 1. Copy MinIO config template
cp .env.minio .env

# 2. Edit with your credentials (get from technician)
nano .env
# Set:
#   AWS_ENDPOINT_URL=https://ubbc1u.ro:9000
#   AWS_ACCESS_KEY_ID=your_key
#   AWS_SECRET_ACCESS_KEY=your_secret
#   MINIO_BUCKET=nsd-data

# 3. Test connection
source activate_env.sh
python -c "
import s3fs, os
fs = s3fs.S3FileSystem(
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY']
)
print('Buckets:', fs.ls(''))
"

# 4. Verify NSD data
python scripts/verify_dataset.py --allow-s3-only

# 5. Run experiment!
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**That's it!** Your scripts automatically detect S3 paths and handle everything.

---

## 🔧 How It Works

### **Automatic Caching**

```
First Access:  MinIO → Download → Local Cache
               [slow, ~1-2 min per file]

Later Access:  Local Cache → Read
               [fast, same as local disk]
```

**Cache location**: `/bigdata/userhome/students/$USER/.cache/s3_cache/`  
**Cache size**: ~20-50GB (only what you use)

### **Zero Code Changes**

```python
# Your code just works with S3 paths:
beta_path = "s3://nsd-data/nsddata_betas/ppdata/subj01/betas_session01.nii.gz"

# Existing code handles it automatically:
from fmri2img.io.s3 import NIfTILoader
loader = NIfTILoader()
data = loader.load(beta_path)  # Downloads + caches automatically
```

---

## 📊 Performance Impact

| Metric | Local Disk | MinIO (First Run) | MinIO (Cached) |
|--------|------------|-------------------|----------------|
| Setup time | 2-8 hours | 5 minutes | 5 minutes |
| Disk usage | 300GB | 20-50GB | 20-50GB |
| Data loading | Fast | Slower | Fast |
| Dependencies | None | Network | None |

**After caching frequently-used files, performance is identical to local!**

---

## 🎮 Daily Usage

### **With MinIO**
```bash
# Every session (same workflow!)
cd Bachelor_V2
source activate_env.sh
bash scripts/run_experiment_simple.sh configs/experiments/my_exp.yaml
```

**No difference from local workflow!**

### **Monitoring Cache**
```bash
# Check cache size
du -sh $S3_CACHE_DIR

# Clear cache if needed
rm -rf $S3_CACHE_DIR/*
```

---

## 🔍 Files Created for You

1. **`MINIO_SETUP_GUIDE.md`** ⭐ - Complete MinIO guide (read this!)
   - Detailed setup instructions
   - Configuration examples
   - Troubleshooting guide
   - Performance tips

2. **`.env.minio`** - MinIO configuration template
   - Copy to `.env` and customize
   - Pre-configured with best practices
   - Includes all necessary environment variables

3. **Updated `QUICK_REFERENCE.md`**
   - Added MinIO setup option
   - Added S3 verification commands
   - Added cache monitoring commands

4. **Updated `DATA_REQUIREMENTS.md`** (already had S3 section)
   - S3/MinIO backend option documented
   - `ALLOW_S3_ONLY=1` flag explained

---

## 💡 Pro Tips

1. **Start with smoke test** - caches commonly-used files
   ```bash
   bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
   ```

2. **Monitor cache growth**
   ```bash
   watch du -sh $S3_CACHE_DIR
   ```

3. **Use tmux for network resilience**
   ```bash
   bash scripts/tmux_run.sh configs/experiments/my_exp.yaml my_session
   ```

4. **Hybrid approach** - keep small files local
   ```bash
   # Keep indices local (small, fast):
   data/indices/nsd_index/
   
   # Stream betas from MinIO (large):
   s3://nsd-data/nsddata_betas/
   ```

---

## 🆘 Troubleshooting

### **"Connection refused"**
```bash
# Wrong endpoint - try different port
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000  # or 9001, or http instead of https
```

### **"SSL certificate error"**
```bash
# Add to .env:
AWS_NO_VERIFY_SSL=1
```

### **"Slow first run"**
```bash
# Normal! It's downloading + caching
# Second run will be fast
# Or pre-cache: run smoke test first
```

### **"Out of space"**
```bash
# Clear cache:
rm -rf $S3_CACHE_DIR/*

# Or move to larger disk:
S3_CACHE_DIR=/path/to/larger/disk/s3_cache
```

---

## 🎓 For Your Bachelor Thesis

### **Recommended Approach**

```
Phase 1: Setup (Day 1)
├── Get MinIO credentials from technician
├── Configure .env with MinIO settings
├── Test connection
└── Run smoke test (caches common files)

Phase 2: Development (Week 1-2)
├── Use MinIO for all experiments
├── Monitor cache size
└── Note any slow operations

Phase 3: Production (Week 3+)
├── If MinIO works well → continue using it
├── If slow → cache subj01 locally
└── Keep using MinIO for other subjects
```

### **Disk Space Budget**

With MinIO:
- Code: ~100MB
- Python env: ~5GB
- Models: ~10GB
- S3 cache: ~20-50GB
- Outputs: ~10-50GB
- **Total: ~100GB** (vs 350GB local)

---

## ✅ Final Checklist

- [ ] Read `MINIO_SETUP_GUIDE.md` (comprehensive guide)
- [ ] Ask technician for:
  - [ ] MinIO endpoint URL
  - [ ] Access credentials
  - [ ] Bucket name
  - [ ] Directory structure
- [ ] Copy `.env.minio` to `.env`
- [ ] Add credentials to `.env`
- [ ] Test connection with `s3fs`
- [ ] Verify data with `--allow-s3-only`
- [ ] Run smoke test
- [ ] Monitor cache during first run
- [ ] Decide: MinIO-only, hybrid, or local

---

## 📚 Next Steps

1. **Read the guide**: `MINIO_SETUP_GUIDE.md`
2. **Get credentials**: Talk to technician
3. **Test setup**: Follow 5-minute setup above
4. **Run smoke test**: See if performance is acceptable
5. **Make decision**: MinIO vs local vs hybrid

---

## 🎉 Summary

**MinIO Integration Status**: ✅ **FULLY SUPPORTED**

**What you need to do**: Configure credentials (5 minutes)

**Will it work?**: YES, your code is already S3-ready

**Should you use it?**: Probably YES (saves 300GB, setup is instant)

**How to start?**: Read `MINIO_SETUP_GUIDE.md` and follow the steps

**Risk level**: LOW (easy to switch to local if needed)

---

**Go ahead and try MinIO - it's the fastest way to get started!** 🚀

If it doesn't work well, you can always download locally later.
