# Quick Setup Guide

## Automated Installation (Recommended)

### One-Command Setup

```bash
# Clone repository
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images

# Switch to probabilistic-distribution branch
git checkout probabilistic-distribution

# Run automated setup
chmod +x setup.sh
./setup.sh
```

This will:
- ✅ Check system requirements
- ✅ Configure environment variables
- ✅ Create Python virtual environment
- ✅ Install all dependencies (PyTorch, transformers, etc.)
- ✅ Download NSD data (~17GB)
- ✅ Build data index
- ✅ Build CLIP embeddings cache
- ✅ Verify installation

**Time**: ~30-45 minutes (depending on network speed)

**Disk**: ~60GB required

---

## Setup Options

### Minimal Setup (Environment Only)

If you already have data or want to download it later:

```bash
./setup.sh --minimal
```

This installs only Python environment and dependencies (~10 minutes).

### Skip Specific Steps

```bash
# Skip data download (if already present)
./setup.sh --skip-data

# Skip CLIP cache building (if already built)
./setup.sh --skip-clip-cache

# Skip model downloads
./setup.sh --skip-models
```

### Get Help

```bash
./setup.sh --help
```

---

## Manual Setup (Advanced)

If you prefer manual control:

### 1. Environment Variables

```bash
# For shared servers with /bigdata
export TORCH_HOME=/bigdata/userhome/$(whoami)/cache/torch
export HF_HOME=/bigdata/userhome/$(whoami)/cache/huggingface
export TMPDIR=/bigdata/userhome/$(whoami)/tmp

# For local machines
export TORCH_HOME=$(pwd)/cache/torch
export HF_HOME=$(pwd)/cache/huggingface
export TMPDIR=$(pwd)/tmp
```

Add to `~/.bashrc` for persistence.

### 2. Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Download Data

```bash
# Download NSD stimulus info
python scripts/fetch_models.py --component nsd-stim-info

# Download beta files (subject 01, session 1)
python scripts/fetch_models.py --component nsd-betas --subject subj01 --session 1

# Build index
python scripts/build_full_index.py --subject subj01 --session 1
```

### 4. Build CLIP Cache

```bash
python scripts/build_target_clip_cache_robust.py \
    --subject subj01 \
    --index-root data/indices/nsd_index \
    --model-id runwayml/stable-diffusion-v1-5 \
    --batch-size 100 \
    --inference-batch-size 64 \
    --output cache/clip_embeddings/nsd_clipvitl14.parquet
```

---

## Verification

After setup, verify installation:

```bash
# Activate environment
source venv/bin/activate
source .env

# Run smoke test
python scripts/smoke.py

# Check setup
python scripts/doctor.py
```

---

## Quick Start After Setup

### 1. Train Ultimate Model

```bash
source venv/bin/activate
source .env

python scripts/train_ultimate_novel.py \
    --config experiments/ultimate_novel_subj01.yaml
```

**Time**: 8-12 hours on A100 GPU

**Output**: Trained model with probabilistic inference, uncertainty estimation, and 7 novel contributions

### 2. Monitor Training

```bash
# View logs
tail -f logs/ultimate_novel_subj01.log

# TensorBoard
tensorboard --logdir outputs/runs
```

### 3. Generate Images

After training:

```bash
python scripts/run_reconstruct_and_eval.py \
    --checkpoint checkpoints/ultimate_novel_subj01/best.pt \
    --subject subj01 \
    --session 1
```

---

## Troubleshooting

### Disk Space Issues

If you get "No space left on device":

```bash
# Check disk usage
df -h

# Clean pip cache
pip cache purge

# Clean temporary files
rm -rf /tmp/tmp*.h5
rm -rf ~/.cache/torch/*
```

### CUDA Out of Memory

Reduce batch size in config:

```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Import Errors

Reinstall package:

```bash
pip install -e . --force-reinstall --no-cache-dir
```

### Data Download Slow

Use faster mirror or download directly:

```bash
# Set AWS region
export AWS_REGION=us-east-1

# Or download from backup
# (Contact repository maintainer)
```

---

## System Requirements

### Minimum

- **OS**: Linux (Ubuntu 20.04+, CentOS 7+)
- **Python**: 3.11+
- **RAM**: 32GB
- **Disk**: 60GB free
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Recommended

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11
- **RAM**: 64GB+
- **Disk**: 100GB+ free (SSD preferred)
- **GPU**: NVIDIA A100 (20GB) or RTX 4090 (24GB)
- **CUDA**: 12.1+

### Tested Configurations

| Environment | GPU | RAM | Status |
|------------|-----|-----|--------|
| Local Ubuntu 22.04 | RTX 3090 (24GB) | 64GB | ✅ Working |
| Remote Server | A100 (20GB) | 125GB | ✅ Working |
| Google Colab | T4 (16GB) | 12GB | ⚠️ Requires batch_size=8 |
| AWS EC2 p3.2xlarge | V100 (16GB) | 61GB | ✅ Working |

---

## Next Steps

After successful setup:

1. **Read Documentation**
   - `START_HERE.md` - Project overview
   - `ULTIMATE_TRAINING_GUIDE.md` - Training the ultimate model
   - `ARCHITECTURE.md` - System architecture

2. **Run Experiments**
   - Smoke test: `python scripts/smoke.py`
   - Baseline: `python scripts/train_ultimate_novel.py --config experiments/baseline_subj01.yaml`
   - Ultimate: `python scripts/train_ultimate_novel.py --config experiments/ultimate_novel_subj01.yaml`

3. **Contribute**
   - Report issues on GitHub
   - Submit pull requests
   - Share your results!

---

## Support

- **Issues**: https://github.com/toniIepure25/FMRI2images/issues
- **Discussions**: https://github.com/toniIepure25/FMRI2images/discussions
- **Email**: [Your Email]

---

**Happy brain decoding! 🧠→🖼️**
