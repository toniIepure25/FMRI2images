# Manual NSD Subj01 Download Guide

> If the automated script doesn't work, use this manual download guide

---

## 📋 What You Need to Download

### **Location on Server:**
```
/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/
```

### **Required Files (~40GB Total):**

1. **fMRI Beta Files** (~30GB)
   - 37 session files
   - Format: `betas_sessionXX.nii.gz`

2. **Stimulus Images** (~10GB)
   - 73,000 images
   - Format: `nsd_stimuli.hdf5` OR individual PNG files

---

## 🔗 Download Sources

### **Option 1: AWS S3 (Recommended)**

**NSD S3 Bucket:** `s3://natural-scenes-dataset` (public access)

**Using AWS CLI:**

```bash
# Install AWS CLI
pip install awscli

# Download beta files
aws s3 sync \
  s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/ \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/ \
  --no-sign-request

# Download stimuli HDF5
aws s3 cp \
  s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd_stimuli.hdf5 \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_stimuli/stimuli/nsd_stimuli.hdf5 \
  --no-sign-request
```

### **Option 2: NSD Website Direct Download**

1. **Register:** https://naturalscenesdataset.org/
2. **Go to Data Access**
3. **Select Downloads:**
   - `nsddata_betas` → `ppdata` → `subj01` → `func1pt8mm` → `betas_fithrf_GLMdenoise_RR`
   - Download all 37 `betas_sessionXX.nii.gz` files
4. **Download Stimuli:**
   - `nsddata_stimuli` → `stimuli` → `nsd_stimuli.hdf5`

### **Option 3: CRCNS (Alternative Mirror)**

Some NSD data is also on: http://crcns.org/

---

## 📦 Expected File Structure

After download, your structure should look like:

```
/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/
├── nsddata_betas/
│   └── ppdata/
│       └── subj01/
│           └── func1pt8mm/
│               └── betas_fithrf_GLMdenoise_RR/
│                   ├── betas_session01.nii.gz  (~800MB each)
│                   ├── betas_session02.nii.gz
│                   ├── betas_session03.nii.gz
│                   ├── ...
│                   └── betas_session37.nii.gz
│
└── nsddata_stimuli/
    └── stimuli/
        └── nsd_stimuli.hdf5  (~10GB)
        OR
        ├── nsd00000.png
        ├── nsd00001.png
        ├── ...
        └── nsd72999.png
```

---

## ✅ Verification

After downloading, verify with:

```bash
cd ~/Bachelor_V2

# Check beta files
ls -lh /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/*.nii.gz | wc -l
# Should show: 37

# Check stimulus file
ls -lh /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_stimuli/stimuli/

# Run verification script
bash scripts/prepare_data.sh --verify-only
```

---

## 📝 File List for Manual Download

If downloading individually from the website, here are all 37 beta files:

```
betas_session01.nii.gz
betas_session02.nii.gz
betas_session03.nii.gz
betas_session04.nii.gz
betas_session05.nii.gz
betas_session06.nii.gz
betas_session07.nii.gz
betas_session08.nii.gz
betas_session09.nii.gz
betas_session10.nii.gz
betas_session11.nii.gz
betas_session12.nii.gz
betas_session13.nii.gz
betas_session14.nii.gz
betas_session15.nii.gz
betas_session16.nii.gz
betas_session17.nii.gz
betas_session18.nii.gz
betas_session19.nii.gz
betas_session20.nii.gz
betas_session21.nii.gz
betas_session22.nii.gz
betas_session23.nii.gz
betas_session24.nii.gz
betas_session25.nii.gz
betas_session26.nii.gz
betas_session27.nii.gz
betas_session28.nii.gz
betas_session29.nii.gz
betas_session30.nii.gz
betas_session31.nii.gz
betas_session32.nii.gz
betas_session33.nii.gz
betas_session34.nii.gz
betas_session35.nii.gz
betas_session36.nii.gz
betas_session37.nii.gz
```

**Each file is ~750-850MB**

---

## 🚀 Using wget (Alternative)

If AWS CLI doesn't work and the NSD website provides direct URLs:

```bash
# Example (replace with actual URLs from NSD)
cd /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/

for i in {01..37}; do
    wget "https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session${i}.nii.gz"
done
```

---

## ⏱️ Download Time Estimates

| Method | Speed | Time for 40GB |
|--------|-------|---------------|
| AWS CLI (fast network) | 50-100 MB/s | 7-15 minutes |
| AWS CLI (normal) | 10-20 MB/s | 35-70 minutes |
| Website download | 5-10 MB/s | 1-2 hours |
| Slow connection | 1-5 MB/s | 2-10 hours |

**Recommendation:** Use AWS CLI if available. Run in `tmux` or `screen` if connection is unstable.

---

## 🆘 Troubleshooting

### **AWS CLI fails with "NoCredentialsError"**
```bash
# Add --no-sign-request flag
aws s3 cp s3://natural-scenes-dataset/... /path/to/local --no-sign-request
```

### **Connection timeouts**
```bash
# Use tmux to keep download running
tmux new -s nsd_download
bash download_nsd_subj01.sh
# Press Ctrl+B, then D to detach
# Reattach later: tmux attach -t nsd_download
```

### **Insufficient disk space**
```bash
# Check space
df -h /bigdata

# Clean up if needed
rm -rf ~/runs/old_runs
```

### **Partial downloads**
```bash
# Resume AWS sync
aws s3 sync ... --no-sign-request  # Will skip already downloaded files

# Check integrity
md5sum betas_session01.nii.gz  # Compare with NSD checksums if provided
```

---

## 📞 Need Help?

1. **NSD Support:** https://naturalscenesdataset.org/
2. **AWS S3 Docs:** https://docs.aws.amazon.com/cli/latest/reference/s3/
3. **Your technician:** Ask about MinIO/shared storage

---

## ✅ Success Checklist

- [ ] 37 beta files downloaded (~30GB)
- [ ] Stimulus file downloaded (~10GB)
- [ ] Total size ~40GB
- [ ] File structure matches expected layout
- [ ] `scripts/prepare_data.sh --verify-only` passes

**All checked? You're ready to run experiments!** 🚀
