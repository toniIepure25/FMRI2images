# 🚀 START HERE - Simple Setup (No MinIO)

> **Forget MinIO for now. Let's get you running experiments TODAY!**

---

## ✅ What You'll Do (3 Steps, 15 Minutes)

1. **Install Python environment** (5-10 min)
2. **Test with smoke test** (2 min) - No data download needed!
3. **Later: Download only what you need** (40GB, not 300GB)

---

## 📝 Step 1: Install Environment

Open terminal and run these commands:

```bash
# 1. Navigate to project
cd ~/Desktop/"Bachelor V2"

# 2. Copy configuration
cp .env.jupyterhub .env

# 3. Edit your username
nano .env
# Find line: USER=${USER:-student01}
# Change to: USER=${USER:-YOUR_ACTUAL_USERNAME}
# Save: Ctrl+X, then Y, then Enter

# 4. Install everything
bash scripts/setup_env.sh
```

**Wait for**: "✓ Setup complete!" (5-10 minutes)

---

## 🧪 Step 2: Run Smoke Test (No Data Needed!)

```bash
# 1. Activate environment
source activate_env.sh

# You should see (venv) in your terminal

# 2. Verify setup
bash scripts/preflight.sh

# 3. Run smoke test (works WITHOUT downloading NSD!)
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**Expected output**: 
```
✓ Pre-flight checks passed
✓ Training started
✓ Epoch 1/2 completed
✓ Epoch 2/2 completed
✓ Experiment completed successfully
```

**Congratulations!** 🎉 Your pipeline works!

---

## 📦 Step 3: Download Minimal Data (When Ready)

### **Option A: Just Test More (Recommended First)**

You can keep testing without any data:

```bash
# Run smoke test again with different configs
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

### **Option B: Download Real Data (Only 40GB)**

When you're ready for real experiments:

#### **What to Download:**
- ✅ **subj01 betas only** (~30GB) - ONE subject, not all 8
- ✅ **First 10,000 images** (~10GB) - Not all 73,000
- ❌ Skip everything else

#### **Where to Get It:**

1. **Register** at: https://naturalscenesdataset.org/

2. **Download only these folders:**
   ```
   nsddata_betas/ppdata/subj01/
   nsddata_stimuli/stimuli/ (first 10,000 images)
   ```

3. **Extract to:**
   ```bash
   /bigdata/userhome/students/$USER/data/nsd/
   ```

4. **Expected structure:**
   ```
   /bigdata/userhome/students/$USER/data/nsd/
   ├── nsddata_betas/
   │   └── ppdata/
   │       └── subj01/              # Only this subject!
   │           └── func1pt8mm/
   │               └── betas_fithrf_GLMdenoise_RR/
   │                   ├── betas_session01.nii.gz
   │                   ├── betas_session02.nii.gz
   │                   └── ... (37 sessions total)
   └── nsddata_stimuli/
       └── stimuli/
           ├── nsd00000.png
           ├── nsd00001.png
           └── ... (first 10,000 images)
   ```

5. **After downloading:**
   ```bash
   # Download models (auto-download, ~10GB)
   source activate_env.sh
   bash scripts/prepare_data.sh
   
   # Run real experiment
   bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml
   ```

---

## 🎯 Quick Decision Guide

### **TODAY (Right Now):**
✅ Install environment  
✅ Run smoke test  
✅ Verify everything works  
✅ **NO data download needed!**

### **THIS WEEK (When Ready):**
- Download subj01 only (40GB)
- OR ask technician about MinIO later
- Either way, you're not blocked!

### **NEVER NEED:**
❌ All 8 subjects (300GB)  
❌ Full stimuli set  
❌ MinIO setup (optional)

---

## 📊 Disk Space You Actually Need

| Component | Size | Status |
|-----------|------|--------|
| Code | ~100MB | ✅ You have it |
| Python env | ~5GB | Installing now |
| Models | ~10GB | Auto-downloads |
| **For smoke test** | **~15GB** | **Enough for today** |
| subj01 data (later) | ~40GB | Download when ready |
| **For real work** | **~60GB total** | **Not 300GB!** |

---

## ✅ Today's Checklist

Copy and run this entire block:

```bash
# ==================================================
# COMPLETE SETUP - RUN ALL AT ONCE
# ==================================================

# 1. Navigate
cd ~/Desktop/"Bachelor V2"

# 2. Configure
cp .env.jupyterhub .env
echo "Edit .env file now - update USER line"
echo "Press Enter after you've edited it..."
read

# 3. Install
bash scripts/setup_env.sh

# 4. Activate
source activate_env.sh

# 5. Verify
bash scripts/preflight.sh

# 6. Test!
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml

echo ""
echo "✅ DONE! Your pipeline is working!"
echo ""
```

**That's it!** You're running experiments without downloading 300GB!

---

## 🆘 If Something Goes Wrong

### **Problem: "setup_env.sh fails"**
```bash
# Try with force flag
bash scripts/setup_env.sh --force

# Check Python version
python --version  # Should be 3.11+
```

### **Problem: "preflight.sh shows warnings"**
- CUDA warnings are OK if you don't have GPU access yet
- Disk space warnings: need at least 20GB free
- Other checks: Read the message, usually tells you how to fix

### **Problem: "smoke_test.yaml fails"**
```bash
# Check logs
cat runs/*/logs/train.log

# Or ask for help with the error message
```

### **Problem: "I don't know my username"**
```bash
# Run this to find it:
whoami
```

---

## 📚 What to Read Next

1. ✅ This file (you're reading it!)
2. 📖 `MINIMAL_DATA_SETUP.md` - Details on minimal data approach
3. 📖 `COMPLETE_SETUP_GUIDE.md` - Full guide (optional, for later)
4. 📖 `QUICK_REFERENCE.md` - Command cheat sheet

**Skip all MinIO guides for now!**

---

## 🎉 Summary

**What you're doing:**
- ✅ Installing Python environment
- ✅ Running smoke tests (no data download)
- ✅ Testing with minimal disk usage (15GB)

**What you're NOT doing:**
- ❌ Setting up MinIO (forget it for now)
- ❌ Downloading 300GB (only need 40GB later)
- ❌ Configuring S3 (not needed)

**What happens next:**
- Today: Smoke tests work!
- This week: Download subj01 (40GB) when ready
- Next week: Run real experiments for thesis

**You're on the fast track!** 🚀

---

## 💡 Pro Tips

1. **Start with smoke test** - Verifies everything works
2. **Download data overnight** - 40GB takes time
3. **Use tmux for long experiments** - Don't lose progress
4. **One subject is enough** - Bachelor thesis doesn't need all 8
5. **Ask questions** - Better to clarify than waste time

---

## 🚀 Next Command

```bash
cd ~/Desktop/"Bachelor V2"
cp .env.jupyterhub .env
nano .env  # Edit USER line
bash scripts/setup_env.sh
```

**Let's get started!** 💪
