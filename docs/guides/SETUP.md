# Complete Setup Guide

> **Unified guide for setting up the fMRI-to-Image reconstruction system on any environment**

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (6GB+ VRAM recommended)
- 50GB+ free disk space
- Git

### Automated Setup

```bash
# Clone repository
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images
git checkout probabilistic-distribution

# Run automated setup
chmod +x setup_enhanced.sh
./setup_enhanced.sh
```

**What it does:**
- ✅ Checks system requirements (Python, GPU, disk space)
- ✅ Configures environment variables
- ✅ Creates Python virtual environment
- ✅ Installs PyTorch with CUDA support
- ✅ Installs all dependencies
- ✅ Downloads NSD data (~17GB)
- ✅ Builds data index and CLIP embeddings cache
- ✅ Detects GPU memory and recommends batch size
- ✅ Runs health checks

**Time:** ~30-45 minutes (depending on network speed)

---

## 📋 Environment-Specific Setup

### JupyterHub/HPC Cluster

**System Specs:**
- CPU: 20 cores (Intel Xeon Platinum 8380 @ 2.30GHz)
- RAM: 125GB
- GPU: NVIDIA A100D-20C (20GB VRAM)
- CUDA: 12.2

**Setup:**

```bash
# 1. Configure environment
cd ~/Bachelor_V2
cp .env.jupyterhub .env

# Edit username in .env
nano .env
# Change: USER=${USER:-YOUR_USERNAME}

# 2. Set up Python environment
bash scripts/setup_env.sh

# 3. Activate environment
source activate_env.sh

# 4. Verify installation
bash scripts/preflight.sh
```

### Local Machine / Workstation

```bash
# 1. Copy example config
cp .env.example .env

# 2. Edit paths for your system
nano .env

# 3. Run setup
bash setup_enhanced.sh

# 4. Activate environment
conda activate fmri2img  # or: source venv/bin/activate
```

---

## 🔧 Manual Setup (If Automated Fails)

### Step 1: Environment Configuration

Create `.env` file:

```bash
# Project structure
PROJECT_ROOT=/path/to/Bachelor_V2
CACHE_DIR=${PROJECT_ROOT}/cache
DATA_DIR=${PROJECT_ROOT}/data
CHECKPOINT_DIR=${PROJECT_ROOT}/checkpoints

# Data paths
NSD_ROOT=${CACHE_DIR}/nsd_hdf5
STIM_INFO_CSV=${CACHE_DIR}/nsd_stim_info_merged.csv
CLIP_EMBEDDINGS_DIR=${CACHE_DIR}/clip_embeddings

# System
DEVICE=cuda
NUM_WORKERS=4
```

### Step 2: Create Python Environment

**Option A: Conda**
```bash
conda env create -f environment.yml
conda activate fmri2img
```

**Option B: Virtualenv**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Step 3: Download Data

```bash
# Download NSD data for subject 01 (~17GB)
bash download_nsd_subj01.sh

# Build data index
python build_minimal_index.py
```

### Step 4: Verify Installation

```bash
# Run preflight checks
bash scripts/preflight.sh

# Expected output:
# ✅ Python environment
# ✅ PyTorch + CUDA
# ✅ All dependencies
# ✅ GPU availability
# ✅ Data files present
```

---

## 🧪 Testing Your Setup

### Quick Smoke Test (No Data Download Needed)

```bash
# Run tests with synthetic data
pytest tests/ -v

# Expected: 53 passed ✅
```

### Test with Real Data

```bash
# Quick training test (1 batch)
python scripts/train.py \
  --config configs/experiments/exp0_baseline.yaml \
  --max_steps 1 \
  --debug

# Expected: No errors, checkpoint saved
```

---

## 📦 Data Overview

### Required Data (~17GB)

| Component | Size | Location | Purpose |
|-----------|------|----------|---------|
| **fMRI betas** | ~15GB | `cache/nsd_hdf5/subj01/*.hdf5` | Brain activity |
| **Stimulus info** | ~2MB | `cache/nsd_stim_info_merged.csv` | Metadata |
| **CLIP embeddings** | ~2GB | `cache/clip_embeddings/*.pt` | Image embeddings |

### Optional Data

| Component | Size | Location | Purpose |
|-----------|------|----------|---------|
| **Images (HDF5)** | ~10GB | `cache/nsd_hdf5/nsd_stimuli.hdf5` | Faster image loading |
| **Full NSD** | ~300GB | `bigdata/NSD/` | All subjects |

**Note:** Images are auto-downloaded from COCO API on-the-fly if not present locally.

---

## 🎯 Post-Setup: Next Steps

### 1. Verify Tests Pass
```bash
pytest tests/test_losses.py tests/test_soft_reliability.py tests/test_uncertainty.py -v
# Expected: 53 passed ✅
```

### 2. Run First Experiment
```bash
# Train baseline model
python scripts/train.py --config configs/experiments/exp0_baseline.yaml

# Monitor with tensorboard
tensorboard --logdir outputs/
```

### 3. Explore Documentation
- **Running experiments:** [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md)
- **Evaluation suite:** [EVALUATION_SUITE_GUIDE.md](EVALUATION_SUITE_GUIDE.md)
- **Novel contributions:** [NOVEL_CONTRIBUTIONS_PIPELINE.md](NOVEL_CONTRIBUTIONS_PIPELINE.md)

---

## 🐛 Troubleshooting

### Common Issues

**Issue: CUDA not available**
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Issue: Out of GPU memory**
```bash
# Reduce batch size in config
nano configs/experiments/exp0_baseline.yaml
# Change: batch_size: 32 → batch_size: 8
```

**Issue: Data files not found**
```bash
# Re-download NSD data
bash download_nsd_subj01.sh

# Rebuild index
python build_minimal_index.py
```

**Issue: Import errors**
```bash
# Reinstall package in editable mode
pip install -e .
```

### Still Having Issues?

See [SETUP_TROUBLESHOOTING.md](../../SETUP_TROUBLESHOOTING.md) for detailed debugging steps.

---

## 📖 Additional Resources

- **Quick reference:** [QUICK_START.md](QUICK_START.md)
- **Realistic workflow:** [REALISTIC_WORKFLOW.md](REALISTIC_WORKFLOW.md)
- **Remote setup:** [docs/REMOTE_QUICKSTART.md](../REMOTE_QUICKSTART.md)
- **Architecture overview:** [docs/architecture/WORKFLOW.md](../architecture/WORKFLOW.md)
