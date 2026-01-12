# Machine Setup Guide for Bachelor Thesis

This guide provides professional scripts for setting up the fMRI-to-Image project on any machine, whether local or remote.

## Quick Start on New Machine

### Option 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images
git checkout probabilistic-distribution

# Run enhanced setup script
chmod +x setup_enhanced.sh
./setup_enhanced.sh
```

**What it does:**
- ✅ Checks system requirements (Python 3.11+, GPU, disk space)
- ✅ Configures environment variables intelligently
- ✅ Creates Python virtual environment
- ✅ Installs all dependencies (PyTorch with CUDA)
- ✅ Downloads NSD data (~17GB)
- ✅ Downloads model checkpoints
- ✅ Builds CLIP embedding cache
- ✅ Runs health checks
- ✅ **Detects GPU memory and recommends optimal batch size!**

**Time:** ~45 minutes total

---

## GPU Memory Management

### Checking GPU Availability

Use the GPU monitor script to wait for available memory:

```bash
# Wait until 5GB is free, then notify
./wait_and_train.sh --min-memory 5 --notify

# Wait and auto-start training when ready
./wait_and_train.sh --min-memory 5 --auto-start

# Check every 2 minutes instead of 5
./wait_and_train.sh --check-interval 120
```

**Features:**
- 🔍 Monitors GPU memory in real-time
- ⏰ Waits for sufficient memory (configurable)
- 📊 Shows current GPU processes
- 💡 Recommends optimal batch size based on available memory
- 🔔 Desktop notifications when ready
- 🚀 Can auto-start training

---

## Script Options

### setup_enhanced.sh Options

```bash
# Minimal setup (no data download)
./setup_enhanced.sh --minimal

# Skip specific components
./setup_enhanced.sh --skip-data        # If data already exists
./setup_enhanced.sh --skip-models      # If models already cached
./setup_enhanced.sh --skip-clip-cache  # If CLIP cache exists

# Non-interactive (for automation)
./setup_enhanced.sh --auto-yes

# Health check only (no installation)
./setup_enhanced.sh --check-only
```

### wait_and_train.sh Options

```bash
--min-memory GB       # Minimum free GPU memory (default: 5)
--check-interval SEC  # Seconds between checks (default: 300)
--max-wait MIN        # Maximum wait time (default: 480 = 8 hours)
--notify              # Send desktop notification
--auto-start          # Start training automatically
```

---

## Troubleshooting

### Setup Fails

```bash
# Check log file (created automatically)
ls -lth setup_*.log | head -1
cat setup_YYYYMMDD_HHMMSS.log

# Run diagnostics
python scripts/doctor.py

# Check environment
bash scripts/check_setup.sh
```

### GPU Out of Memory

The enhanced setup script automatically detects your GPU and recommends batch size:

**Example output:**
```
GPU: NVIDIA A100-SXM4-20GB
    Total Memory: 20480 MB
    Used Memory:  13840 MB (70% utilization)
    Free Memory:  6640 MB
    
⚠ GPU is heavily used (13840MB / 20480MB)
ℹ    Consider waiting for GPU to free up
ℹ    Recommended batch_size: 8 (moderate)
```

**Action:** Edit `experiments/ultimate_novel_subj01.yaml`:
```yaml
training:
  batch_size: 8  # Use recommended value

advanced:
  gradient_accumulation_steps: 4  # = 32 / batch_size
```

---

## Environment Detection

The setup script automatically detects your environment:

### Shared Server (e.g., GPU20C-N5)
- **Detected by:** Presence of `/bigdata` directory
- **Cache location:** `/bigdata/userhome/$(whoami)/cache/`
- **Why:** Prevents filling up system disk
- **Effect:** All large caches (PyTorch, HuggingFace, temp files) go to `/bigdata`

### Local Machine
- **Cache location:** Project directory
- **Standard setup for personal workstations**

---

## Resume Capability

If setup is interrupted, it can resume from where it stopped:

```bash
# Setup was interrupted after downloading data
./setup_enhanced.sh

# Output:
# ✓ System checks already completed. Skipping...
# ✓ Environment already configured. Skipping...
# ✓ Python environment already set up. Skipping...
# ✓ NSD data already downloaded. Skipping...
# ▶ Downloading Stable Diffusion model...
```

The script tracks completed steps in `.setup_state` file.

**Force full re-run:**
```bash
rm .setup_state
./setup_enhanced.sh
```

---

## Comparison: Original vs Enhanced

| Feature | setup.sh | setup_enhanced.sh |
|---------|----------|-------------------|
| System checks | Basic | Detailed with diagnostics |
| GPU detection | Simple | Memory analysis + batch size recommendation |
| Resume capability | ❌ | ✅ |
| Logging | ❌ | ✅ (timestamped log file) |
| Health checks | Basic | Comprehensive |
| Error handling | Basic | Advanced with recovery |
| Progress tracking | Minimal | Detailed with ETA |
| Internet check | ❌ | ✅ |
| Disk space check | Basic | Detailed breakdown |
| GPU wait script | ❌ | ✅ (separate script) |

---

## Professional Features

### 1. **Intelligent Environment Configuration**
- Auto-detects shared server vs local
- Configures cache directories to prevent disk full issues
- Sets up environment variables persistently

### 2. **Detailed System Diagnostics**
```
GPU: NVIDIA A100-SXM4-20GB
    Total Memory: 20480 MB
    Used Memory:  13840 MB (70% utilization)
    Free Memory:  6640 MB
✓ GPU has sufficient free memory (6640MB available)
ℹ    Recommended batch_size: 8 (moderate)
```

### 3. **Resume Capability**
- Tracks completed steps
- Skips already-done work
- Can restart after interruption

### 4. **Comprehensive Logging**
- All output saved to timestamped log file
- Useful for debugging and verification
- Can share log with advisor/collaborators

### 5. **Health Checks**
- Verifies Python imports
- Checks data file presence
- Tests GPU functionality
- Reports disk usage

---

## Example Workflow: New Remote Machine

```bash
# 1. SSH into remote machine
ssh user@gpu-server

# 2. Clone repository
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images
git checkout probabilistic-distribution

# 3. Run setup (will detect GPU memory)
./setup_enhanced.sh

# Output shows:
# ⚠ GPU is heavily used (13840MB / 20480MB)
# ℹ    Recommended batch_size: 4

# 4. Wait for GPU to free up
./wait_and_train.sh --min-memory 5 --notify --auto-start

# Script monitors GPU and auto-starts training when ready!
```

---

## Tips for Bachelor Thesis

### 1. **Document Your Setup**
Save the setup log as proof of reproducibility:
```bash
cp setup_YYYYMMDD_HHMMSS.log ~/thesis/logs/
```

### 2. **Track Configurations**
The enhanced script logs:
- System specs (GPU, Python version, disk space)
- Recommended settings (batch size, memory)
- Timestamps for all operations

### 3. **Share Setup Instructions**
In your thesis methodology section:
```
"The complete experimental setup can be reproduced by running
./setup_enhanced.sh on any machine with Python 3.11+ and CUDA 11.8+.
The script automatically configures optimal parameters based on
available GPU memory."
```

### 4. **Verify Reproducibility**
Before thesis submission, test on a fresh machine:
```bash
# Clean machine
./setup_enhanced.sh
./wait_and_train.sh --auto-start

# Should work identically to your development machine
```

---

## Support

**Issues during setup?**
1. Check log file: `setup_YYYYMMDD_HHMMSS.log`
2. Run diagnostics: `python scripts/doctor.py`
3. See troubleshooting: `SETUP_TROUBLESHOOTING.md`
4. Enhanced documentation: `docs/guides/SETUP_SCRIPT_DOCS.md`

**GPU memory issues?**
1. Use GPU monitor: `./wait_and_train.sh --notify`
2. Check current usage: `nvidia-smi`
3. Use recommended batch size from setup script
4. See: Memory optimization section in `QUICKSTART.md`

---

## Next Steps After Setup

1. **Verify Installation:**
   ```bash
   source venv/bin/activate
   source .env
   python scripts/smoke.py
   ```

2. **Configure Training:**
   - Edit `experiments/ultimate_novel_subj01.yaml`
   - Use recommended batch size from setup

3. **Start Training:**
   ```bash
   python scripts/train_ultimate_novel.py --config experiments/ultimate_novel_subj01.yaml
   ```

4. **Monitor:**
   ```bash
   tensorboard --logdir outputs/runs
   ```

Good luck with your bachelor thesis! 🧠→🖼️
