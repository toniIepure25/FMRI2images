# Complete Setup Guide - From Clone to Running Experiments

> **The definitive step-by-step guide with data preparation**

---

## 🎯 What You'll Do

1. ✅ Clone repository
2. ✅ Configure environment
3. ✅ Install Python dependencies
4. ✅ **Download/prepare datasets and models**
5. ✅ Verify everything works
6. ✅ Run your first experiment

---

## 📦 Prerequisites

Before starting, you need:
- Access to JupyterHub/HPC with GPU
- ~150GB free disk space (for NSD dataset + models)
- Internet connection (for downloading models)
- NSD dataset access (register at https://naturalscenesdataset.org/)

---

## 🚀 Complete Setup Steps

### **Step 1: Clone Repository**

```bash
cd /bigdata/userhome/students/$USER
git clone <your-repo-url> Bachelor_V2
cd Bachelor_V2
```

---

### **Step 2: Configure Environment**

```bash
# Copy the JupyterHub template
cp .env.jupyterhub .env

# Edit to set your username (replace 'your_username')
sed -i "s/USER=\${USER:-student01}/USER=\${USER:-your_username}/" .env
```

**Or edit manually**:
```bash
nano .env
# Change: USER=${USER:-student01}
# To:     USER=${USER:-your_actual_username}
# Save: Ctrl+X, Y, Enter
```

---

### **Step 3: Source Initial Setup** (if required)

If your JupyterHub provides `initialSetup.sh`:

```bash
run=true source initialSetup.sh
```

Skip if file doesn't exist.

---

### **Step 4: Install Python Environment**

```bash
bash scripts/setup_env.sh
```

**What it does**:
- Creates virtual environment in `./venv/`
- Installs PyTorch with CUDA 12.2
- Installs all dependencies (numpy, pandas, scikit-learn, etc.)
- Creates helper scripts

**Time**: 5-10 minutes

---

### **Step 5: Activate Environment**

```bash
source activate_env.sh
```

You should see `(venv)` in your prompt.

---

### **Step 6: Verify Python Setup**

```bash
bash scripts/preflight.sh
```

**Checks**:
- ✓ Python and packages installed
- ✓ GPU available and CUDA works
- ✓ Disk space sufficient
- ✓ Write permissions OK

**Expected**: "All critical checks passed! ✓"

---

### **Step 7: Download NSD Dataset** ⚠️ **MANUAL STEP**

The NSD dataset is ~300GB and must be downloaded manually:

1. **Register** at: https://naturalscenesdataset.org/
2. **Download** the dataset (takes hours/overnight)
3. **Extract** to: `/bigdata/userhome/students/$USER/data/nsd/`

**Expected structure**:
```
/bigdata/userhome/students/$USER/data/nsd/
├── nsddata/
├── nsddata_betas/
└── nsddata_stimuli/
```

**Alternative**: Download only `subj01` for testing (~40GB)

---

### **Step 8: Prepare Models and Cache**

```bash
bash scripts/prepare_data.sh
```

**What it does**:
- Verifies NSD dataset location
- Downloads Stable Diffusion 2.1 (~5GB)
- Downloads CLIP models (~2GB)
- Sets up cache directories

**Time**: 10-20 minutes (depending on internet)

**Expected output**:
```
========================================
📦 DATA PREPARATION
========================================

1. NSD Dataset Verification
[SUCCESS] NSD dataset verified!

2. Pre-trained Models
[SUCCESS] Models downloaded/verified!

3. CLIP Embeddings Cache
[SUCCESS] CLIP cache exists

📋 SUMMARY
  ✓ NSD dataset: /bigdata/.../data/nsd
  ✓ Models: /bigdata/.../cache/hf/
  ✓ CLIP cache: /bigdata/.../cache/clip_embeddings/
```

---

### **Step 9: Verify Complete Setup**

```bash
# Quick verification
bash scripts/preflight.sh

# Comprehensive check (includes data)
python scripts/doctor.py
```

**Expected**: All checks should PASS.

---

### **Step 10: Run Smoke Test**

```bash
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**What it does**:
- Runs 2-epoch minimal training
- Verifies entire pipeline works
- Creates output in `runs/`

**Time**: 1-2 minutes

**Expected**: "✓ Training completed successfully!"

---

### **Step 11: Run Your First Real Experiment**

#### **Option A: Short Run (Interactive)**
```bash
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml
```

#### **Option B: Long Run (tmux - Recommended)**
```bash
bash scripts/tmux_run.sh configs/experiments/jupyterhub_quickstart.yaml my_experiment

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t my_experiment
```

---

## 📋 Complete Command Sequence (Copy-Paste)

```bash
# ============================================
# COMPLETE SETUP - RUN ALL COMMANDS IN ORDER
# ============================================

# Step 1-2: Clone and configure
cd /bigdata/userhome/students/$USER
git clone <your-repo-url> Bachelor_V2
cd Bachelor_V2
cp .env.jupyterhub .env
sed -i "s/USER=\${USER:-student01}/USER=\${USER:-your_username}/" .env

# Step 3: Source initial setup (if exists)
[ -f initialSetup.sh ] && run=true source initialSetup.sh

# Step 4-5: Install and activate
bash scripts/setup_env.sh
source activate_env.sh

# Step 6: Verify Python environment
bash scripts/preflight.sh

# Step 7: Download NSD dataset (MANUAL - do this separately)
# Register at https://naturalscenesdataset.org/
# Download to /bigdata/userhome/students/$USER/data/nsd/

# Step 8: Prepare models
bash scripts/prepare_data.sh

# Step 9: Comprehensive verification
python scripts/doctor.py

# Step 10: Smoke test
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml

# Step 11: Real experiment
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml
```

---

## ⏱️ Time Estimates

| Step | Task | Time | Can Skip? |
|------|------|------|-----------|
| 1-3 | Clone & configure | 2 min | No |
| 4 | Install Python | 5-10 min | No |
| 5-6 | Activate & verify | 1 min | No |
| 7 | **Download NSD** | **2-8 hours** | Testing: Yes |
| 8 | Download models | 10-20 min | No |
| 9-10 | Verify & test | 2 min | Optional |
| 11 | First experiment | 5-60 min | No |

**Total active time**: ~20-30 minutes (excluding NSD download)

**Total wall time**: 3-9 hours (including NSD download in background)

---

## 🔄 Subsequent Sessions (Daily Use)

After initial setup, each time you log in:

```bash
# Navigate to project
cd /bigdata/userhome/students/$USER/Bachelor_V2

# Activate environment (ALWAYS!)
source activate_env.sh

# Run experiments
bash scripts/run_experiment_simple.sh configs/experiments/your_config.yaml
```

**Time**: 10 seconds

---

## 💾 Disk Space Requirements

| Component | Size | Required? |
|-----------|------|-----------|
| Code repository | ~100MB | Yes |
| Python environment | ~5GB | Yes |
| NSD (full dataset) | ~300GB | Yes* |
| NSD (single subject) | ~40GB | For testing |
| Pre-trained models | ~10GB | Yes |
| CLIP embeddings | ~5GB | Auto-generated |
| Experiment outputs | ~10-50GB | Grows over time |
| **Total (full)** | **~350GB** | |
| **Total (minimal)** | **~75GB** | |

\* Can use S3 backend as alternative

**Check space**:
```bash
df -h /bigdata
```

---

## 📚 Documentation Reference

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **This file** | Complete setup guide | Setup time |
| `QUICK_REFERENCE.md` | Quick commands | Keep handy |
| `README_JUPYTERHUB.md` | Detailed guide | For deep dives |
| `DATA_REQUIREMENTS.md` | Data/model details | Step 7-8 |
| `IMPLEMENTATION_SUMMARY.md` | Technical details | Optional |

---

## ✅ Verification Checklist

After completing all steps, verify:

- [ ] `(venv)` appears in terminal prompt
- [ ] `bash scripts/preflight.sh` shows all green
- [ ] NSD dataset exists at `$NSD_DATA_ROOT`
- [ ] `python scripts/verify_dataset.py` passes
- [ ] Models downloaded in `$CACHE_ROOT/hf/`
- [ ] `python scripts/fetch_models.py` shows "already cached"
- [ ] Smoke test completes successfully
- [ ] Output appears in `runs/` directory
- [ ] At least 50GB free disk space

**All checked? You're fully set up!** 🎉

---

## 🆘 Troubleshooting

### Problem: NSD dataset too large

**Solution**: Download only `subj01` for testing
```bash
# Download only:
# - nsddata_betas/ppdata/subj01/
# - nsddata_stimuli/ (first 10,000 images)
```

### Problem: Model download fails

**Solutions**:
```bash
# Check internet
ping huggingface.co

# Retry download
bash scripts/prepare_data.sh

# Use HF token (if needed)
echo "HF_TOKEN=hf_your_token" >> .env
bash scripts/prepare_data.sh
```

### Problem: Out of disk space

**Solutions**:
```bash
# Check space
df -h /bigdata

# Clean old runs
rm -rf runs/202601*

# Use smaller dataset
# Download only subj01 instead of full NSD
```

### Problem: CUDA not available

**Solution**:
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch
bash scripts/setup_env.sh --force --cuda-version 12.2

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🎓 Success Indicators

You're ready to start your research when:

1. ✅ All scripts run without errors
2. ✅ Smoke test completes successfully
3. ✅ You understand the workflow
4. ✅ Data is in place (or S3 configured)
5. ✅ Models are cached locally

---

## 🚀 Next Steps

After setup:

1. **Read experiment configs**: `configs/experiments/*.yaml`
2. **Understand outputs**: Check `runs/<timestamp>_<name>/`
3. **Customize configs**: Copy and modify for your experiments
4. **Use tmux for long runs**: `scripts/tmux_run.sh`
5. **Monitor progress**: `nvidia-smi`, `tail -f logs/*.log`

---

## 💡 Pro Tips

1. **Download NSD overnight** - it's 300GB
2. **Use tmux for everything** - persistent sessions are key
3. **Test with smoke test first** - catches issues early
4. **Monitor disk space** - `df -h /bigdata`
5. **Keep environment active** - always `source activate_env.sh`
6. **Read logs when stuck** - `cat runs/*/logs/train.log`
7. **Commit before experiments** - for reproducibility
8. **Use descriptive names** - `--name my_experiment_v1`

---

## 🎉 You're Ready!

If you've completed all steps, you now have:

- ✅ Fully configured environment
- ✅ All dependencies installed
- ✅ Dataset and models ready
- ✅ Verified working pipeline
- ✅ Production-ready workflow

**Start experimenting!** 🔬

```bash
bash scripts/run_experiment_simple.sh configs/experiments/your_experiment.yaml
```

**Good luck with your bachelor thesis!** 🎓
