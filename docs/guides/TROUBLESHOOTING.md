# Setup Script Troubleshooting Guide

This guide helps resolve common issues encountered during the automated setup process.

---

## Table of Contents

1. [System Requirements Issues](#system-requirements-issues)
2. [Network/Download Issues](#networkdownload-issues)
3. [Disk Space Issues](#disk-space-issues)
4. [Permission Issues](#permission-issues)
5. [Python/Package Issues](#pythonpackage-issues)
6. [GPU/CUDA Issues](#gpucuda-issues)
7. [Data Issues](#data-issues)
8. [Environment Issues](#environment-issues)

---

## System Requirements Issues

### Python Version Too Old

**Error**: `Python 3.X detected (requires 3.11+)`

**Solution**:
```bash
# Install Python 3.11 on Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Make it default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

### Missing Git

**Error**: `✗ git is not installed`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install git

# CentOS/RHEL
sudo yum install git

# macOS
brew install git
```

### Insufficient Disk Space

**Error**: `Only XGB available. Recommended: 60GB+`

**Solution**:
1. **Check disk usage**:
   ```bash
   df -h
   du -sh ~/* | sort -h | tail -20
   ```

2. **Clean up space**:
   ```bash
   # Clean pip cache
   pip cache purge
   
   # Clean old Python packages
   sudo apt autoremove
   sudo apt clean
   
   # Remove old Docker images (if applicable)
   docker system prune -a
   
   # Clean tmp
   sudo rm -rf /tmp/*
   ```

3. **Use external storage**:
   ```bash
   # Point caches to external drive
   export TORCH_HOME=/mnt/external/cache/torch
   export HF_HOME=/mnt/external/cache/huggingface
   ./setup.sh
   ```

---

## Network/Download Issues

### Slow Download Speed

**Issue**: NSD data download taking too long

**Solutions**:

1. **Use faster network**:
   ```bash
   # Download on faster connection first, then rsync to server
   rsync -avz --progress nsd_data/ server:/path/to/FMRI2images/data/
   ```

2. **Resume interrupted downloads**:
   ```bash
   # The script automatically resumes - just re-run
   ./setup.sh
   ```

3. **Skip data download and transfer manually**:
   ```bash
   ./setup.sh --skip-data
   # Then manually copy data files
   ```

### SSL Certificate Errors

**Error**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**:
```bash
# Temporarily disable SSL verification (not recommended for production)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Or install certificates
pip install --upgrade certifi
```

### Connection Timeout

**Error**: `Connection timed out`

**Solution**:
```bash
# Increase timeout
export PIP_TIMEOUT=300

# Use mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## Disk Space Issues

### No Space Left on Device (During Setup)

**Error**: `[Errno 28] No space left on device`

**Root Cause**: Usually `/tmp` or system root partition is full

**Solution**:

1. **Check which filesystem is full**:
   ```bash
   df -h
   ```

2. **Clean system disk**:
   ```bash
   # Remove large temp files
   sudo du -sh /tmp/* | sort -h | tail -20
   sudo rm -rf /tmp/tmp*.h5  # Large HDF5 temp files
   sudo rm -rf /tmp/torchinductor_*
   
   # Clean user cache
   rm -rf ~/.cache/pip/*
   rm -rf ~/.cache/torch/*
   ```

3. **Use /bigdata or external storage**:
   ```bash
   export TMPDIR=/bigdata/userhome/$(whoami)/tmp
   export TORCH_HOME=/bigdata/userhome/$(whoami)/cache/torch
   export HF_HOME=/bigdata/userhome/$(whoami)/cache/huggingface
   mkdir -p $TMPDIR
   ./setup.sh
   ```

4. **Clean up after setup**:
   ```bash
   # Setup script creates cache directories
   # After successful setup, you can move them
   mv cache /bigdata/userhome/$(whoami)/FMRI2images_cache
   ln -s /bigdata/userhome/$(whoami)/FMRI2images_cache cache
   ```

### Cache Directories Full

**Issue**: HuggingFace or PyTorch cache filling disk

**Solution**:
```bash
# Check cache sizes
du -sh ~/.cache/huggingface
du -sh ~/.cache/torch

# Clean old models
rm -rf ~/.cache/huggingface/hub/models--*/snapshots/*/  # Keep latest only
rm -rf ~/.cache/torch/hub/checkpoints/*  # Remove if not needed

# Or point to larger disk
export HF_HOME=/bigdata/cache/huggingface
export TORCH_HOME=/bigdata/cache/torch
```

---

## Permission Issues

### Permission Denied on /tmp

**Error**: `cannot remove '/tmp/...': Operation not permitted`

**Solution**:
```bash
# Use your own tmp directory
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
./setup.sh
```

### Cannot Write to Installation Directory

**Error**: `Permission denied: '/usr/local/lib/...'`

**Solution**:
```bash
# Don't use sudo with pip
# Use virtual environment instead (setup script does this automatically)
python3 -m venv venv
source venv/bin/activate
./setup.sh
```

### Cannot Execute setup.sh

**Error**: `Permission denied: ./setup.sh`

**Solution**:
```bash
chmod +x setup.sh
./setup.sh
```

---

## Python/Package Issues

### Conflicting Package Versions

**Error**: `ERROR: pip's dependency resolver...`

**Solution**:
```bash
# Use fresh virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
./setup.sh
```

### PyTorch CUDA Version Mismatch

**Error**: `CUDA error: no kernel image is available`

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 12.1)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors After Installation

**Error**: `ModuleNotFoundError: No module named 'fmri2img'`

**Solution**:
```bash
# Reinstall in development mode
cd /path/to/FMRI2images
pip install -e . --force-reinstall
```

### Jupyter Notebook Kernel Issues

**Error**: Kernel not finding installed packages

**Solution**:
```bash
# Install ipykernel in venv
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=fmri2img

# Select kernel in Jupyter: Kernel -> Change Kernel -> fmri2img
```

---

## GPU/CUDA Issues

### No GPU Detected

**Warning**: `No GPU detected (nvidia-smi not available)`

**Check**:
```bash
# Verify GPU
nvidia-smi

# Check CUDA installation
nvcc --version
```

**Solution** (if NVIDIA driver not installed):
```bash
# Ubuntu 22.04
sudo apt update
sudo apt install nvidia-driver-535

# Reboot
sudo reboot

# Verify
nvidia-smi
```

### CUDA Out of Memory During Setup

**Error**: `CUDA out of memory`

**Solution**:
```bash
# Reduce batch size in CLIP cache building
# Edit setup.sh line ~385:
--inference-batch-size 32  # Reduce from 64
```

### Mixed CUDA Versions

**Error**: Multiple CUDA versions causing conflicts

**Solution**:
```bash
# Use conda for clean environment
conda create -n fmri python=3.11
conda activate fmri
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

---

## Data Issues

### NSD Data Download Fails

**Error**: Failed to download from S3

**Solutions**:

1. **Check AWS credentials** (if using private data):
   ```bash
   aws configure
   ```

2. **Manual download**:
   ```bash
   ./setup.sh --skip-data
   # Then manually download from https://naturalscenesdataset.org
   ```

3. **Resume partial download**:
   ```bash
   # Script automatically resumes - just re-run
   ./setup.sh
   ```

### Index Building Fails

**Error**: `Failed to build data index`

**Solution**:
```bash
# Check beta files are present
ls -lh data/nsddata_betas/ppdata/subj01/

# Rebuild manually
python scripts/build_full_index.py --subject subj01 --session 1 --force
```

### CLIP Cache Building Fails

**Error**: Various errors during CLIP cache building

**Solutions**:

1. **Memory error**:
   ```bash
   # Reduce batch size
   python scripts/build_target_clip_cache_robust.py \
       --subject subj01 \
       --index-root data/indices/nsd_index \
       --model-id runwayml/stable-diffusion-v1-5 \
       --batch-size 50 \
       --inference-batch-size 32 \
       --output cache/clip_embeddings/nsd_clipvitl14.parquet
   ```

2. **Model download fails**:
   ```bash
   # Pre-download CLIP model
   python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-large-patch14')"
   # Then re-run setup
   ```

3. **Timeout**:
   ```bash
   # Increase timeout
   export HF_HUB_DOWNLOAD_TIMEOUT=600
   ./setup.sh
   ```

---

## Environment Issues

### Environment Variables Not Persisting

**Issue**: Variables reset after closing terminal

**Solution**:
```bash
# Add to ~/.bashrc (for bash)
echo 'source /path/to/FMRI2images/.env' >> ~/.bashrc

# Or ~/.zshrc (for zsh)
echo 'source /path/to/FMRI2images/.env' >> ~/.zshrc

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Virtual Environment Not Activating

**Error**: `venv/bin/activate: No such file or directory`

**Solution**:
```bash
# Recreate venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Wrong Python Version in Venv

**Issue**: Venv using old Python version

**Solution**:
```bash
# Specify Python version explicitly
python3.11 -m venv venv
source venv/bin/activate
python --version  # Verify
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check logs**:
   ```bash
   # Setup creates detailed logs
   cat /tmp/fmri2img_setup.log
   ```

2. **Run diagnostics**:
   ```bash
   source venv/bin/activate
   python scripts/doctor.py
   ```

3. **Verbose mode**:
   ```bash
   # Re-run setup with verbose output
   bash -x setup.sh 2>&1 | tee setup_debug.log
   ```

4. **Report issue**:
   - GitHub Issues: https://github.com/toniIepure25/FMRI2images/issues
   - Include: OS, Python version, error message, setup_debug.log

---

## Common Setup Patterns

### Minimal Setup (Development)

```bash
# Just environment + dependencies
./setup.sh --minimal

# Manually download only what you need
python scripts/fetch_models.py --component nsd-stim-info
```

### Full Setup (Production)

```bash
# Complete automated setup
./setup.sh

# Verify everything
python scripts/doctor.py
python scripts/smoke.py
```

### Server Setup (Shared Resources)

```bash
# Use /bigdata for caches
export TORCH_HOME=/bigdata/userhome/$(whoami)/cache/torch
export HF_HOME=/bigdata/userhome/$(whoami)/cache/huggingface
export TMPDIR=/bigdata/userhome/$(whoami)/tmp

# Run setup
./setup.sh

# Add to ~/.bashrc for persistence
echo "export TORCH_HOME=/bigdata/userhome/$(whoami)/cache/torch" >> ~/.bashrc
echo "export HF_HOME=/bigdata/userhome/$(whoami)/cache/huggingface" >> ~/.bashrc
echo "export TMPDIR=/bigdata/userhome/$(whoami)/tmp" >> ~/.bashrc
```

---

## Pro Tips

1. **Save disk space**: Use `--minimal` first, then selectively download data
2. **Speed up setup**: Pre-download data on fast connection, then rsync to server
3. **Avoid re-downloads**: Point `TORCH_HOME`/`HF_HOME` to existing caches
4. **Test before full setup**: Run `python scripts/doctor.py` to check environment
5. **Use screen/tmux**: For long-running setups on remote servers
   ```bash
   screen -S fmri_setup
   ./setup.sh
   # Ctrl+A, D to detach
   ```

---

**Still stuck? Open an issue with full error output!**
