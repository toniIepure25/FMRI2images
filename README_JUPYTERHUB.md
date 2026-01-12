# JupyterHub/HPC Quick Start Guide

> **Complete guide for running experiments on JupyterHub GPU environments (HPC-style)**

This guide provides everything you need to set up and run experiments on a JupyterHub/HPC GPU server, where you don't have `sudo` access and `systemctl` doesn't work.

---

## 📋 Table of Contents

- [Environment Overview](#environment-overview)
- [Initial Setup](#initial-setup)
- [Running Experiments](#running-experiments)
- [Long-Running Jobs](#long-running-jobs)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## 🖥️ Environment Overview

### What We're Working With

- **Platform**: JupyterHub terminal on GPU server
- **Home Directory**: `/bigdata/userhome/students/<USER>/`
- **Python**: Pre-installed Python 3.11.2, pip 23.x
- **GPU**: NVIDIA A100 (20GB), CUDA 12.2
- **Constraints**:
  - ❌ No `sudo` access
  - ❌ No `systemctl` (no systemd)
  - ✅ Virtual environments work
  - ✅ GPU available via `nvidia-smi`

### Key Directories

```
/bigdata/userhome/students/<USER>/
├── Bachelor_V2/          # This project (git clone here)
├── venv/                 # Python virtual environment
├── data/                 # Large datasets
│   └── nsd/             # Natural Scenes Dataset
├── runs/                 # Experiment outputs
├── cache/               # Embeddings, preprocessed data
└── checkpoints/         # Model checkpoints
```

---

## 🚀 Initial Setup

### Step 1: Clone Repository

```bash
cd /bigdata/userhome/students/$USER
git clone <your-repo-url> Bachelor_V2
cd Bachelor_V2
```

### Step 2: Configure Environment

Copy the JupyterHub-specific environment template:

```bash
cp .env.jupyterhub .env
```

Edit `.env` to set your username:

```bash
nano .env
# Change: USER=student01
# To:     USER=your_actual_username
```

The `.env` file sets up paths for:
- `DATA_DIR`: Where datasets live
- `RUNS_DIR`: Where experiment outputs go
- `CACHE_DIR`: For embeddings and preprocessed data
- `CHECKPOINT_ROOT`: For model checkpoints

### Step 3: Source Initial Setup (If Required)

If your JupyterHub provides an `initialSetup.sh` script, source it:

```bash
run=true source initialSetup.sh
```

**Important**: Use `source`, not `bash`, because it sets environment variables in your current shell.

### Step 4: Create Virtual Environment and Install Dependencies

Run the automated setup script:

```bash
bash scripts/setup_env.sh
```

This script will:
- Create a Python virtual environment at `./venv` (or path from `.env`)
- Upgrade pip, setuptools, wheel
- Install PyTorch with CUDA 12.2 support
- Install all other dependencies (numpy, pandas, scikit-learn, etc.)
- Create necessary directories
- Generate an `activate_env.sh` helper script

**What if setup fails?**

If CUDA version detection fails or you need a specific version:

```bash
bash scripts/setup_env.sh --cuda-version 11.8
```

To force rebuild from scratch:

```bash
bash scripts/setup_env.sh --force
```

### Step 5: Activate Environment

For subsequent sessions, activate the environment:

```bash
source activate_env.sh
# or
source venv/bin/activate
```

You should see `(venv)` in your prompt.

### Step 6: Run Preflight Checks

Verify everything is working:

```bash
bash scripts/preflight.sh
```

This checks:
- ✓ Python and pip are accessible
- ✓ Virtual environment is active
- ✓ Required packages are installed
- ✓ GPU is detected and CUDA works
- ✓ PyTorch can use GPU
- ✓ Disk space is sufficient
- ✓ Write permissions to output directories

**Expected output:**
```
========================================
🚀 PRE-FLIGHT CHECKS
========================================

1. Environment and Paths
[✓ PASS] Project root: /bigdata/userhome/students/user/Bachelor_V2
...
[✓ PASS] PyTorch CUDA available
[✓ PASS] CUDA tensor operations working

📋 SUMMARY
[✓ PASS] All critical checks passed! ✓
System is ready for experiments!
```

---

## 🧪 Running Experiments

### Quick Start: Smoke Test

Test the pipeline with a minimal configuration:

```bash
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

This runs a 2-epoch dummy training to verify everything works.

### Running a Real Experiment

```bash
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml
```

**With custom options:**

```bash
# Custom experiment name
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml --name my_experiment

# Custom random seed
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml --seed 123

# Resume from checkpoint
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml --resume

# Skip preflight checks (if you already ran them)
bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml --skip-preflight
```

### What Happens During a Run?

1. **Environment Loading**: Loads `.env` file
2. **Preflight Checks**: Validates GPU, disk space, permissions
3. **Run Directory Creation**: Creates `runs/<timestamp>_<expname>/`
4. **Provenance Tracking**:
   - Saves config copy
   - Captures git commit hash
   - Takes environment snapshot (Python packages, system info, GPU info)
5. **Training**: Runs your model
6. **Checkpointing**: Saves model checkpoints periodically
7. **Metrics Logging**: Logs to `metrics.jsonl`
8. **Success/Failure Marker**: Creates `SUCCESS` or `FAILED` file

### Understanding Run Output

After running, you'll find:

```
runs/20260112_143022_my_experiment/
├── config.yaml                 # Your config (copy)
├── git_info.json              # Git commit, branch, changes
├── environment/               # Complete environment snapshot
│   ├── environment.json
│   ├── requirements_frozen.txt
│   ├── system_info.json
│   └── gpu_info.json
├── checkpoints/               # Model checkpoints
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_5.pt
│   └── best_model.pt
├── logs/
│   └── train.log             # Full training log
├── metrics/
│   ├── metrics.jsonl         # Per-step metrics
│   └── summary.json          # Final summary
├── outputs/                   # Generated outputs (if any)
└── SUCCESS                    # Or FAILED
```

---

## ⏱️ Long-Running Jobs

For experiments that take hours/days, you need to keep them running after you log out.

### Option 1: tmux (Recommended)

**tmux** creates persistent terminal sessions that survive disconnection.

**Start experiment in tmux:**

```bash
bash scripts/tmux_run.sh configs/experiments/jupyterhub_quickstart.yaml
```

Or with custom session name:

```bash
bash scripts/tmux_run.sh configs/experiments/jupyterhub_quickstart.yaml my_long_exp
```

**Common tmux commands:**

```bash
# List all sessions
tmux ls

# Attach to a session
tmux attach -t my_long_exp

# Detach from session (while inside): Ctrl+B, then D

# Kill a session
tmux kill-session -t my_long_exp
```

**Workflow:**
1. Start experiment in tmux
2. Detach (Ctrl+B, D) or close terminal
3. Log out, go home, sleep
4. Log back in next day
5. Reattach: `tmux attach -t my_long_exp`

### Option 2: nohup

**nohup** runs commands that ignore hangup signals (continues after logout).

```bash
bash scripts/nohup_run.sh configs/experiments/jupyterhub_quickstart.yaml
```

This creates:
- Log file: `logs/nohup_<expname>_<timestamp>.log`
- PID file: `logs/nohup_<expname>_<timestamp>.pid`

**Monitor the log:**

```bash
tail -f logs/nohup_jupyterhub_quickstart_20260112_143022.log
```

**Check if running:**

```bash
# Get PID from file
PID=$(cat logs/nohup_jupyterhub_quickstart_20260112_143022.pid)
ps -p $PID
```

**Stop the job:**

```bash
kill $(cat logs/nohup_jupyterhub_quickstart_20260112_143022.pid)
```

### Comparison: tmux vs nohup

| Feature | tmux | nohup |
|---------|------|-------|
| Reattach to see live output | ✅ Yes | ❌ No (log file only) |
| Interactive control | ✅ Yes | ❌ No |
| Simpler setup | ❌ Slightly complex | ✅ Very simple |
| Multiple experiments | ✅ Multiple sessions | ✅ Multiple PIDs |

**Recommendation**: Use **tmux** for interactive monitoring, **nohup** for fire-and-forget jobs.

---

## 🔁 Running Multiple Experiments (Sweeps)

Run a batch of experiments with one command:

```bash
bash scripts/run_sweep.sh configs/experiments/smoke_test.yaml \
                          configs/experiments/jupyterhub_quickstart.yaml \
                          configs/experiments/full_training_example.yaml
```

**With options:**

```bash
# Custom sweep name
bash scripts/run_sweep.sh --sweep-name my_ablation configs/experiments/ablation_*.yaml

# Continue even if one experiment fails
bash scripts/run_sweep.sh --continue-on-error configs/experiments/*.yaml

# Skip preflight for each (if you already checked once)
bash scripts/run_sweep.sh --skip-preflight configs/experiments/*.yaml
```

**Sweep creates:**

```
runs/sweep_20260112_143022/
├── sweep_manifest.json      # Sweep metadata
├── sweep_results.txt        # Summary of all experiments
└── <individual run dirs>/   # In main runs/ directory
```

---

## 🐛 Troubleshooting

### Problem: Virtual environment not found

**Symptom:**
```
[ERROR] Virtual environment not active!
```

**Solution:**
```bash
source venv/bin/activate
# or
source activate_env.sh
```

### Problem: CUDA not available in PyTorch

**Symptom:**
```
[✗ FAIL] PyTorch CUDA not available
```

**Solutions:**

1. **Check GPU visibility:**
   ```bash
   nvidia-smi
   ```

2. **Check PyTorch installation:**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

3. **Reinstall PyTorch with correct CUDA version:**
   ```bash
   bash scripts/setup_env.sh --force --cuda-version 12.2
   ```

### Problem: Out of memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** in your config:
   ```yaml
   batch_size: 16  # Instead of 64
   ```

2. **Enable mixed precision** (if not already):
   ```yaml
   mixed_precision: true
   ```

3. **Check GPU memory usage:**
   ```bash
   nvidia-smi
   ```

4. **Kill other processes using GPU** (if any):
   ```bash
   # Find processes
   nvidia-smi
   # Kill specific PID
   kill <PID>
   ```

### Problem: Disk space full

**Symptom:**
```
[✗ FAIL] Runs directory has less than 50 GB free
```

**Solutions:**

1. **Check disk usage:**
   ```bash
   df -h /bigdata
   ```

2. **Clean up old runs:**
   ```bash
   # List runs by size
   du -sh runs/* | sort -h
   
   # Remove specific run
   rm -rf runs/20260101_*
   ```

3. **Clean up cache:**
   ```bash
   rm -rf cache/clip_embeddings/*
   ```

### Problem: Permission denied

**Symptom:**
```
[✗ FAIL] No write permission for Runs directory
```

**Solutions:**

1. **Check directory ownership:**
   ```bash
   ls -ld runs/
   ```

2. **Create with correct permissions:**
   ```bash
   mkdir -p runs
   chmod 755 runs
   ```

3. **Check `.env` paths** - make sure they point to writeable locations.

### Problem: Import errors for custom modules

**Symptom:**
```
ModuleNotFoundError: No module named 'fmri2img'
```

**Solution:**

Install the package in development mode:

```bash
pip install -e .
```

---

## ✨ Best Practices

### 1. Always Use Virtual Environments

```bash
# Activate before every session
source activate_env.sh
```

### 2. Commit Before Running

For full reproducibility:

```bash
git add .
git commit -m "Experiment config for run X"
bash scripts/run_experiment_simple.sh configs/experiments/my_exp.yaml
```

The run will capture the git commit hash. If you have uncommitted changes, you'll get a warning.

### 3. Use Descriptive Experiment Names

```yaml
# configs/experiments/my_exp.yaml
experiment_name: ablation_lr0001_bs64_dropout02
```

Or override on command line:

```bash
bash scripts/run_experiment_simple.sh configs/experiments/base.yaml --name ablation_lr0001
```

### 4. Start with Smoke Tests

Before a long run, test with minimal config:

```bash
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

### 5. Use Fixed Seeds for Reproducibility

```yaml
seed: 42
```

Or override:

```bash
bash scripts/run_experiment_simple.sh configs/experiments/my_exp.yaml --seed 42
```

### 6. Monitor Disk Space

```bash
# Check before starting
df -h /bigdata

# Set in .env
PREFLIGHT_MIN_FREE_GB=100
```

### 7. Clean Up Regularly

```bash
# Remove old/failed runs
rm -rf runs/20260101_*

# Remove old checkpoints (keep only best)
find runs/*/checkpoints -name 'checkpoint_epoch_*.pt' -not -name 'best_model.pt' -delete
```

### 8. Use tmux for Long Runs

```bash
# Start in tmux
bash scripts/tmux_run.sh configs/experiments/long_training.yaml

# Detach: Ctrl+B, D
# Log out safely
```

### 9. Document Your Experiments

Create a `RUN_LOG.md`:

```markdown
# Experiment Log

## 2026-01-12: Baseline Run
- Config: `configs/experiments/baseline.yaml`
- Run ID: `20260112_143022_baseline`
- Notes: First complete run on JupyterHub
- Result: Val loss 0.123
```

### 10. Review Environment Snapshots

After a run, check what was captured:

```bash
cat runs/20260112_143022_my_exp/environment/requirements_frozen.txt
cat runs/20260112_143022_my_exp/git_info.json
```

This helps reproduce results later.

---

## 📚 Command Reference

### Essential Commands

```bash
# Setup (first time)
bash scripts/setup_env.sh

# Activate environment (every session)
source activate_env.sh

# Preflight checks
bash scripts/preflight.sh

# Run experiment
bash scripts/run_experiment_simple.sh configs/experiments/my_exp.yaml

# Run in tmux
bash scripts/tmux_run.sh configs/experiments/my_exp.yaml

# Run with nohup
bash scripts/nohup_run.sh configs/experiments/my_exp.yaml

# Run sweep
bash scripts/run_sweep.sh configs/experiments/*.yaml
```

### Monitoring Commands

```bash
# Check GPU
nvidia-smi

# Monitor log (nohup)
tail -f logs/nohup_*.log

# Attach to tmux
tmux attach -t <session_name>

# List tmux sessions
tmux ls

# Check running processes
ps aux | grep python

# Check disk space
df -h /bigdata
```

---

## 🎓 Example Workflow

Here's a complete workflow from clone to results:

```bash
# 1. Clone and setup
cd /bigdata/userhome/students/$USER
git clone <repo> Bachelor_V2
cd Bachelor_V2

# 2. Configure
cp .env.jupyterhub .env
nano .env  # Edit USER=your_username

# 3. Source initial setup (if required)
run=true source initialSetup.sh

# 4. Install dependencies
bash scripts/setup_env.sh

# 5. Activate environment
source activate_env.sh

# 6. Verify setup
bash scripts/preflight.sh

# 7. Test with smoke test
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml

# 8. Run real experiment in tmux
bash scripts/tmux_run.sh configs/experiments/jupyterhub_quickstart.yaml my_exp

# 9. Detach from tmux
# Press: Ctrl+B, then D

# 10. Log out, come back later

# 11. Reattach to check progress
tmux attach -t my_exp

# 12. View results
ls -lh runs/
cat runs/20260112_*/metrics/summary.json
```

---

## 🆘 Getting Help

1. **Check the logs:**
   ```bash
   cat runs/<your_run>/logs/train.log
   ```

2. **Run preflight checks:**
   ```bash
   bash scripts/preflight.sh
   ```

3. **Check system resources:**
   ```bash
   nvidia-smi
   df -h
   free -h
   ```

4. **Review environment snapshot:**
   ```bash
   cat runs/<your_run>/environment/environment.txt
   ```

5. **Check for uncommitted changes:**
   ```bash
   git status
   ```

---

## 📝 Summary

**One-time setup:**
```bash
cp .env.jupyterhub .env       # Configure paths
bash scripts/setup_env.sh      # Install dependencies
```

**Every session:**
```bash
source activate_env.sh         # Activate environment
```

**Run experiments:**
```bash
bash scripts/run_experiment_simple.sh <config.yaml>  # Short runs
bash scripts/tmux_run.sh <config.yaml>              # Long runs
```

**That's it!** 🎉

Your experiments are now:
- ✅ Reproducible (seeds, git tracking, environment snapshots)
- ✅ Logged (metrics, checkpoints, full provenance)
- ✅ Resumable (checkpoint support)
- ✅ Production-ready (error handling, validation, cleanup)

Happy experimenting! 🚀
