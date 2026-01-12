# 🚀 JupyterHub Workflow - Getting Started

> **Start here for running experiments on JupyterHub/HPC GPU environments**

---

## 📖 Documentation Index

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 🎯 **START HERE**
   - Quick reference card with essential commands
   - Print and keep handy!

2. **[README_JUPYTERHUB.md](README_JUPYTERHUB.md)** - 📚 **Complete Guide**
   - Detailed setup instructions
   - Troubleshooting section
   - Best practices
   - Example workflows

3. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - 📋 **Technical Overview**
   - What's been implemented
   - Architecture decisions
   - File structure
   - Educational value

---

## ⚡ Quick Start (3 Commands)

```bash
# 1. Setup (first time only)
cp .env.jupyterhub .env && bash scripts/setup_env.sh

# 2. Activate (every session)
source activate_env.sh

# 3. Run experiment
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**That's it!** 🎉

---

## 📁 Key Files

### Configuration
- `.env.jupyterhub` - JupyterHub-specific environment settings
- `configs/experiments/*.yaml` - Experiment configurations

### Scripts (All in `scripts/`)
- `setup_env.sh` - Setup virtual environment and install dependencies
- `preflight.sh` - Validate environment before running
- `run_experiment_simple.sh` - Run a single experiment
- `run_sweep.sh` - Run multiple experiments
- `tmux_run.sh` - Run in persistent tmux session
- `nohup_run.sh` - Run in background with nohup

### Training
- `src/train.py` - Main training entrypoint

---

## 🎯 Common Tasks

### First Time Setup
```bash
cp .env.jupyterhub .env
nano .env  # Edit USER=your_username
bash scripts/setup_env.sh
source activate_env.sh
bash scripts/preflight.sh
```

### Run a Test
```bash
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

### Run Long Experiment (tmux)
```bash
bash scripts/tmux_run.sh configs/experiments/jupyterhub_quickstart.yaml
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t <session_name>
```

### Monitor
```bash
nvidia-smi                    # Check GPU
tail -f logs/train.log       # Monitor training
tmux ls                      # List tmux sessions
```

---

## 🆘 Help!

### Problem: Environment not active
```bash
source activate_env.sh
```

### Problem: CUDA not available
```bash
bash scripts/preflight.sh
bash scripts/setup_env.sh --force --cuda-version 12.2
```

### Problem: Need more details
See [README_JUPYTERHUB.md](README_JUPYTERHUB.md) - Troubleshooting section

---

## ✅ What You Get

Every experiment run creates:
- ✅ Full provenance (git commit, environment snapshot)
- ✅ Structured logs (console + file)
- ✅ Metrics tracking (JSONL + summary)
- ✅ Checkpoints (periodic + best model)
- ✅ Config snapshot (exact parameters)
- ✅ Success/failure marker

**This is publication-grade reproducibility!**

---

## 🎓 Learn More

- **Quick commands**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Detailed guide**: [README_JUPYTERHUB.md](README_JUPYTERHUB.md)
- **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## 💡 Pro Tips

1. Always activate environment: `source activate_env.sh`
2. Test with smoke test first: `configs/experiments/smoke_test.yaml`
3. Use tmux for long runs: `scripts/tmux_run.sh`
4. Check preflight before important runs: `bash scripts/preflight.sh`
5. Commit before running for full reproducibility: `git commit -am "Exp X"`

---

**Ready to go? 🚀**

```bash
source activate_env.sh
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**Good luck with your bachelor thesis!** 🎓
