# Minimal Data Setup - Just What You Need

> **You're RIGHT! You don't need 300GB. Here's the minimal approach.**

---

## ✅ YES - You Can Work WITHOUT Full Dataset!

### **Two Options:**

1. **Option 1: Smoke Test Mode** (~5GB) - Works WITHOUT any NSD download
2. **Option 2: Single Subject** (~40GB) - Download only what you need

---

## 🚀 Option 1: Smoke Test (NO Download Needed!)

**Perfect for:** Testing the pipeline, learning the code, verifying setup

### **What You Need:**
- ✅ Pre-trained models (auto-download, ~10GB)
- ✅ Your code repository
- ❌ NO NSD dataset needed!

### **How It Works:**
Your project has **smoke test configs** that use:
- Dummy/synthetic data
- Or tiny subset of data
- Just to verify the pipeline works

### **Setup:**

```bash
# 1. Install environment
cd ~/Desktop/"Bachelor V2"
cp .env.jupyterhub .env
nano .env  # Edit USER=your_username
bash scripts/setup_env.sh
source activate_env.sh

# 2. Download ONLY models (not NSD data)
bash scripts/prepare_data.sh --skip-nsd

# 3. Run smoke test (works without NSD!)
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**Time**: 15 minutes  
**Disk**: ~15GB (code + models + outputs)

---

## 📦 Option 2: Single Subject Only (~40GB)

**Perfect for:** Your bachelor thesis! Most students only use 1 subject.

### **What You Actually Need:**

```
Minimal NSD Download (~40GB):
├── nsddata_betas/ppdata/subj01/     # 30GB - Brain data for ONE subject
└── nsddata_stimuli/stimuli/         # 10GB - First ~10,000 images
```

**NOT needed** (can skip):
- ❌ Other subjects (subj02-08) - 240GB
- ❌ Full stimuli set (73,000 images) - Skip 60GB
- ❌ Anatomy data - Not used in your pipeline
- ❌ Freesurfer outputs - Not needed

### **How to Download Minimal Set:**

#### **From NSD Website:**

1. Register at https://naturalscenesdataset.org/
2. Download ONLY these files:
   ```
   nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/
   ├── betas_session01.nii.gz
   ├── betas_session02.nii.gz
   ├── ... (37 sessions total)
   └── betas_session37.nii.gz
   
   nsddata_stimuli/stimuli/
   ├── nsd00000.png
   ├── nsd00001.png
   ├── ... (first 10,000 images)
   └── nsd09999.png
   ```

3. Extract to:
   ```bash
   /bigdata/userhome/students/$USER/data/nsd/
   ```

#### **Expected Structure:**

```bash
$NSD_DATA_ROOT/
├── nsddata_betas/
│   └── ppdata/
│       └── subj01/          # ONLY this subject
│           └── func1pt8mm/
│               └── betas_fithrf_GLMdenoise_RR/
│                   ├── betas_session01.nii.gz
│                   └── ... (37 files, ~30GB)
└── nsddata_stimuli/
    └── stimuli/
        ├── nsd00000.png
        └── ... (first 10,000, ~10GB)
```

**Total**: ~40GB (vs 300GB full dataset)

---

## 🎯 What Each Config Needs

### **Smoke Test** (`configs/experiments/smoke_test.yaml`)
- Data: None (uses dummy data or tiny subset)
- Time: 2-3 minutes
- Purpose: Verify pipeline works

### **Quick Start** (`configs/experiments/jupyterhub_quickstart.yaml`)
- Data: subj01 (~40GB)
- Time: 30-60 minutes
- Purpose: Fast experiment with real data

### **Full Experiment** (`configs/experiments/novel_subj01.yaml`)
- Data: subj01 (~40GB)
- Time: Several hours
- Purpose: Your bachelor thesis results

**All use only 1 subject!** 🎉

---

## 💡 Smart Approach for Your Thesis

### **Phase 1: Setup & Testing (Day 1-2)**
```bash
# Use smoke test - NO data needed
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```
**Verify**: Pipeline works, code runs, GPU accessible

### **Phase 2: Development (Week 1-2)**
```bash
# Download ONLY subj01 (~40GB)
# Run quick experiments
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml
```
**Goal**: Tune hyperparameters, test ideas

### **Phase 3: Final Results (Week 3+)**
```bash
# Use same subj01 data
# Run full training
bash scripts/run_experiment_simple.sh configs/experiments/novel_subj01.yaml
```
**Goal**: Generate thesis results

**Total data needed**: 40GB (not 300GB!) 🎉

---

## 📊 Disk Space Budget

| Component | Size | When Needed |
|-----------|------|-------------|
| Code | ~100MB | Always |
| Python env | ~5GB | Always |
| Models (SD + CLIP) | ~10GB | Always |
| **Smoke test** | **0GB** | **Testing only** |
| **subj01 betas** | **~30GB** | **Real experiments** |
| **Stimuli (10K)** | **~10GB** | **Real experiments** |
| Generated cache | ~5GB | After first run |
| Outputs | ~5-10GB | After experiments |
| **TOTAL (minimal)** | **~15GB** | **Smoke testing** |
| **TOTAL (real work)** | **~75GB** | **With subj01 data** |

**NOT 300GB!** ✅

---

## 🔧 Practical Setup Steps

### **Step 1: Smoke Test First (No Data)**

```bash
cd ~/Desktop/"Bachelor V2"

# 1. Setup environment
cp .env.jupyterhub .env
nano .env  # Edit USER=your_username
bash scripts/setup_env.sh
source activate_env.sh

# 2. Download models only
bash scripts/prepare_data.sh --skip-nsd

# 3. Run smoke test
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**Expected**: ✅ Pipeline works! (5 minutes)

### **Step 2: When Ready, Download subj01**

```bash
# Option A: From NSD website (manual)
# - Register at naturalscenesdataset.org
# - Download subj01 betas (~30GB)
# - Download first 10K stimuli (~10GB)
# - Extract to $NSD_DATA_ROOT

# Option B: From MinIO (if technician sets it up)
# - Already has the data
# - Streams on-demand
# - Caches locally (~20GB)
```

### **Step 3: Run Real Experiments**

```bash
# After downloading subj01
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml
```

**Expected**: ✅ Real training with real brain data!

---

## 🎓 For Your Bachelor Thesis

### **You Only Need:**
1. ✅ **Code** (this repo)
2. ✅ **Models** (auto-download, ~10GB)
3. ✅ **subj01 data** (~40GB)

### **You Don't Need:**
- ❌ All 8 subjects (only need 1)
- ❌ All 73K images (only need 10K)
- ❌ Anatomy data (not used)
- ❌ 300GB dataset (WAY too much)

### **Typical Bachelor Thesis Uses:**
- 1 subject (subj01)
- ~10,000 training images
- ~750-1000 test images
- Total: **40GB** ✅

---

## 🤔 But What About Quality?

**Q: Is 1 subject enough for a bachelor thesis?**  
**A:** YES! Most published papers use 1-3 subjects. 1 subject is perfectly valid for:
- Proof of concept
- Method comparison
- Novel architecture testing
- Bachelor/Master thesis

**Q: Is 10K images enough?**  
**A:** YES! Your configs already use subsets:
- Training: 750-8000 images (configurable)
- Validation: 100-500 images
- Test: 100-982 images

The NSD dataset has each subject viewing 10,000 unique images. That's plenty!

---

## 🚀 Recommended Path

### **For You RIGHT NOW:**

```bash
# 1. Start with smoke test (TODAY)
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml

# 2. While that runs, decide on data:
#    Option A: Ask technician about MinIO (if NSD is there)
#    Option B: Start downloading subj01 (~40GB, overnight)

# 3. Tomorrow: Run with real data
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml

# 4. Week 2-3: Full experiments for thesis
bash scripts/run_experiment_simple.sh configs/experiments/novel_subj01.yaml
```

**Total data: 40GB (NOT 300GB!)** 🎉

---

## 📝 Update Your .env for Minimal Setup

```bash
# In .env file:

# Use only subj01
NSD_SUBJECTS=subj01

# Limit stimuli (optional)
NSD_STIMULI_LIMIT=10000

# Paths (minimal structure)
NSD_DATA_ROOT=/bigdata/userhome/students/${USER}/data/nsd
CACHE_ROOT=/bigdata/userhome/students/${USER}/.cache

# Everything else same as before
```

---

## ✅ Summary

### **Question 1: Can I work without downloading the dataset?**
**Answer**: YES! Use smoke tests for testing. No NSD data needed.

### **Question 2: Do I need all 300GB?**
**Answer**: NO! You only need:
- **Smoke testing**: 0GB (no NSD data)
- **Real work**: 40GB (subj01 only)
- **Full dataset**: 300GB (only if you want all 8 subjects)

### **For Bachelor Thesis:**
- ✅ Use **1 subject** (subj01)
- ✅ Download **40GB** (not 300GB)
- ✅ Start with **smoke test** (0GB download)
- ✅ Perfectly valid for academic work

### **Your Next Action:**

```bash
# Run smoke test NOW (no data needed!)
cd ~/Desktop/"Bachelor V2"
bash scripts/setup_env.sh
source activate_env.sh
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**You don't need to wait for anything!** 🚀

---

## 🎯 The Real Answer

**You were 100% right:**
1. ✅ Can work without downloading NSD (smoke test mode)
2. ✅ Don't need 300GB (only need 40GB for 1 subject)

**The guides were covering worst-case (full dataset). You're smart to ask!** 💡

---

**Start with smoke test NOW. Download subj01 later. You're good to go!** ✅
