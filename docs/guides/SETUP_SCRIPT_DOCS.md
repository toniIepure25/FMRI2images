# Setup Script - Complete Documentation

## Overview

The **automated setup script** (`setup.sh`) provides a professional, one-command installation for the fMRI-to-Image brain decoding project. It handles everything from environment configuration to data downloads.

---

## Quick Reference

### Basic Usage

```bash
# Standard installation (recommended)
./setup.sh

# Minimal setup (environment only)
./setup.sh --minimal

# Skip specific steps
./setup.sh --skip-data          # Skip NSD data download
./setup.sh --skip-models        # Skip model checkpoints
./setup.sh --skip-clip-cache    # Skip CLIP cache building

# Get help
./setup.sh --help
```

### Time & Space Requirements

| Component | Time | Disk Space |
|-----------|------|------------|
| Environment Setup | 5-10 min | ~5GB |
| NSD Data Download | 15-30 min | ~17GB |
| CLIP Cache Build | 5-10 min | ~200MB |
| **Total (Full Setup)** | **30-45 min** | **~60GB** |

---

## What It Does

### 1. System Checks ✓
- Verifies Python 3.11+
- Checks Git installation  
- Detects GPU availability
- Validates disk space (60GB+ recommended)

### 2. Environment Configuration 🔧
- **Auto-detects** shared server vs local environment
- Creates `.env` file with optimal settings
- **Configures cache directories**:
  - On shared servers: Uses `/bigdata` to avoid system disk issues
  - On local machines: Uses project directory
- Sets environment variables:
  - `TORCH_HOME` - PyTorch models
  - `HF_HOME` - HuggingFace models
  - `TMPDIR` - Temporary files (prevents disk full errors!)
- Updates shell rc file (`.bashrc`/`.zshrc`)

### 3. Python Environment 🐍
- Creates virtual environment
- Upgrades pip/setuptools
- **Installs PyTorch** (auto-detects CUDA version)
- Installs all dependencies from `requirements.txt`
- Installs project in development mode (`pip install -e .`)

### 4. NSD Data Download 📊
- Downloads stimulus info CSV (~11MB)
- Downloads beta files for subject 01, session 1 (~17GB)
- Builds data index (750 trials)
- Handles resume if interrupted

### 5. Model Checkpoints 🤖
- Downloads Stable Diffusion v1.5
- Caches CLIP models
- Downloads preprocessing models

### 6. CLIP Cache Building ⚡
- Extracts CLIP embeddings for 750 NSD images
- Saves to optimized parquet format
- Enables fast training (no re-computation)

### 7. Verification ✅
- Tests Python imports
- Checks data files presence
- Verifies GPU accessibility
- Reports disk usage

### 8. Next Steps Guide 📖
- Activation instructions
- Training commands
- Documentation links
- Troubleshooting tips

---

## Architecture

```
setup.sh
├── System Checks
│   ├── Python version (3.11+)
│   ├── Git installation
│   ├── GPU detection (nvidia-smi)
│   └── Disk space (60GB+)
│
├── Environment Setup
│   ├── Detect server type (/bigdata exists?)
│   ├── Create .env file
│   │   ├── TORCH_HOME=/bigdata/.../cache/torch
│   │   ├── HF_HOME=/bigdata/.../cache/huggingface
│   │   └── TMPDIR=/bigdata/.../tmp
│   ├── Update ~/.bashrc
│   └── Create directory structure
│
├── Python Environment
│   ├── Create venv
│   ├── Upgrade pip
│   ├── Install PyTorch (CUDA 12.1)
│   ├── Install requirements.txt
│   └── pip install -e .
│
├── Data Download (if not --skip-data)
│   ├── Stimulus info CSV
│   ├── Beta files (subj01, session 1)
│   └── Build index
│
├── Models (if not --skip-models)
│   └── Stable Diffusion v1.5
│
├── CLIP Cache (if not --skip-clip-cache)
│   ├── Load CLIP model
│   ├── Process 750 images (batched)
│   └── Save embeddings (parquet)
│
└── Verification
    ├── Import tests
    ├── File checks
    └── Usage report
```

---

## Key Features

### 🎯 Smart Environment Detection

```bash
# Automatically detects shared server
if [[ -d "/bigdata" ]]; then
    CACHE_BASE="/bigdata/userhome/$(whoami)"
else
    CACHE_BASE="$SCRIPT_DIR"
fi
```

**Why**: Prevents "No space left on device" errors by using the larger `/bigdata` partition on shared servers.

### 🎨 Professional Output

- Color-coded messages (errors, warnings, success)
- Progress indicators
- Clear section headers
- Detailed logging

### 🔄 Resume Support

- Checks existing installations
- Skips completed steps
- Offers re-download/rebuild options

### 🛡️ Error Handling

- `set -e`: Exits on any error
- `set -u`: Exits on undefined variables
- Comprehensive error messages
- Validation at each step

### 📝 Configuration Persistence

- Creates `.env` file (project-specific)
- Updates shell rc file (permanent)
- No manual configuration needed

---

## Example Output

```
═══════════════════════════════════════════════════════════════════
  fMRI-TO-IMAGE BRAIN DECODING - AUTOMATED SETUP
═══════════════════════════════════════════════════════════════════

Project: fMRI2img
Location: /home/user/FMRI2images
User: user
Date: Sun Jan 12 22:00:00 UTC 2026

Continue with setup? (Y/n): y

═══════════════════════════════════════════════════════════════════
  SYSTEM REQUIREMENTS CHECK
═══════════════════════════════════════════════════════════════════

▶ Checking Python installation...
✓ Python 3.11.2 detected

▶ Checking Git installation...
✓ git is installed

▶ Checking GPU availability...
✓ GPU detected: NVIDIA A100-SXM4-20GB

▶ Checking disk space...
✓ 94GB available

✓ All system checks passed!

═══════════════════════════════════════════════════════════════════
  CONFIGURING ENVIRONMENT VARIABLES
═══════════════════════════════════════════════════════════════════

▶ Detected shared server environment (/bigdata exists)
▶ Creating environment configuration: /home/user/FMRI2images/.env
▶ Adding environment sourcing to /home/user/.bashrc
✓ Shell configuration updated
▶ Creating cache directories...
✓ Environment configured successfully!

═══════════════════════════════════════════════════════════════════
  SETTING UP PYTHON ENVIRONMENT
═══════════════════════════════════════════════════════════════════

▶ Creating virtual environment...
✓ Virtual environment created
▶ Activating virtual environment...
▶ Upgrading pip...
▶ Installing Python dependencies...
⚠ This may take 5-10 minutes...
▶ Installing PyTorch with CUDA support...
▶ Installing from requirements.txt...
▶ Installing fmri2img package...
✓ Python environment configured successfully!

[... continued ...]

═══════════════════════════════════════════════════════════════════
  SETUP COMPLETE!
═══════════════════════════════════════════════════════════════════

✓ Installation successful!

Next Steps:

1. Activate the environment:
   source venv/bin/activate
   source .env

2. Run a quick smoke test:
   python scripts/smoke.py

3. Train the ultimate model (8-12 hours on A100):
   python scripts/train_ultimate_novel.py --config experiments/ultimate_novel_subj01.yaml

[...]

Happy brain decoding! 🧠→🖼️
```

---

## Files Created

```
FMRI2images/
├── .env                              # Environment configuration (auto-generated)
├── venv/                            # Python virtual environment
├── cache/
│   ├── torch/                       # PyTorch model cache
│   ├── huggingface/                 # HuggingFace model cache
│   ├── clip_embeddings/
│   │   └── nsd_clipvitl14.parquet  # CLIP cache (750 embeddings)
│   └── nsd_stim_info_merged.csv    # Stimulus metadata
├── data/
│   ├── nsddata_betas/               # fMRI beta files (~17GB)
│   └── indices/
│       └── nsd_index/               # Processed index
└── tmp/                             # Temporary files (prevents system disk fill)
```

---

## Integration with Existing Workflows

### With Makefile

The setup script complements the existing Makefile:

```bash
# Automated setup (new)
./setup.sh

# Or traditional make workflow
make setup
make doctor
make prepare
```

### With Manual Setup

Can be used partially:

```bash
# Just environment
./setup.sh --minimal

# Then manual steps
python scripts/fetch_models.py --component nsd-betas
python scripts/build_full_index.py
```

---

## Advanced Usage

### Custom Cache Locations

```bash
# Set before running setup
export TORCH_HOME=/mnt/ssd/cache/torch
export HF_HOME=/mnt/ssd/cache/huggingface
./setup.sh
```

### Silent/Automated Mode

```bash
# Accept all defaults (for CI/CD)
yes | ./setup.sh
```

### Debugging

```bash
# Verbose mode
bash -x setup.sh 2>&1 | tee setup_debug.log

# Check what would be done (dry-run)
# [Feature can be added if needed]
```

---

## Comparison with Other Methods

| Method | Time | Complexity | Reliability | Resume Support |
|--------|------|------------|-------------|----------------|
| **setup.sh** | 30-45 min | Low (1 command) | High | Yes |
| Manual | 1-2 hours | High (many steps) | Medium | Partial |
| Makefile | 45-60 min | Medium | Medium | No |
| Docker | 20-30 min | Low | High | N/A |

---

## Troubleshooting

See **[SETUP_TROUBLESHOOTING.md](SETUP_TROUBLESHOOTING.md)** for comprehensive solutions to common issues.

Quick fixes:

```bash
# Permission issues
chmod +x setup.sh

# Disk space issues
export TMPDIR=/bigdata/tmp
export TORCH_HOME=/bigdata/cache/torch

# Re-run failed step
./setup.sh  # Automatically resumes

# Clean start
rm -rf venv .env
./setup.sh
```

---

## Related Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup guide with examples
- **[SETUP_TROUBLESHOOTING.md](SETUP_TROUBLESHOOTING.md)** - Detailed troubleshooting
- **[README.md](README.md)** - Project overview
- **[ULTIMATE_TRAINING_GUIDE.md](ULTIMATE_TRAINING_GUIDE.md)** - Training guide

---

## Future Enhancements

Potential improvements (contributions welcome!):

- [ ] Docker setup option
- [ ] Multi-subject download
- [ ] Progress bars for long operations
- [ ] Setup state serialization (pause/resume)
- [ ] Configuration presets (minimal/standard/full)
- [ ] Automatic rollback on failure
- [ ] Setup verification tests
- [ ] Conda environment support
- [ ] Windows WSL support
- [ ] Installation size estimator

---

## Maintainer Notes

### Script Structure

- **Modular functions**: Each setup step is a separate function
- **Helper functions**: Color printing, checks, utilities
- **Error handling**: Exits on error, clear messages
- **Documentation**: Inline comments explain complex logic

### Testing

Before releasing changes:

```bash
# Test on clean machine
docker run -it ubuntu:22.04 bash
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images
./setup.sh --minimal  # Fast test

# Test full setup
./setup.sh  # On machine with 60GB+ space
```

### Debugging

Add to script:

```bash
set -x  # Print commands as executed
# ... your code ...
set +x  # Stop printing
```

---

**Script Version**: 1.0  
**Last Updated**: 2026-01-12  
**Maintainer**: [Your Name/Team]
