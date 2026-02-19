# JupyterHub Quick Reference Card

> **Print this or keep it handy while working on JupyterHub!**

---

## 🚀 First Time Setup (Run Once)

### Option A: Local Storage (300GB download)
```bash
# 1. Configure environment
cp .env.jupyterhub .env
nano .env  # Edit USER=your_username

# 2. Source initialSetup.sh if required
run=true source initialSetup.sh

# 3. Install dependencies
bash scripts/setup_env.sh

# 4. Verify setup
source activate_env.sh
bash scripts/preflight.sh

# 5. Download NSD dataset (manual - see DATA_REQUIREMENTS.md)
# 6. Prepare models
bash scripts/prepare_data.sh
```

### Option B: MinIO Object Storage (No download!)
```bash
# 1. Configure for MinIO
cp .env.minio .env
nano .env  # Add MinIO credentials (see MINIO_SETUP_GUIDE.md)

# 2-4. Same as Option A (install & verify)

# 5. Test MinIO connection
python -c "import s3fs, os; fs = s3fs.S3FileSystem(endpoint_url=os.environ['AWS_ENDPOINT_URL']); print('OK')"

# 6. Verify S3 data
python scripts/verify_dataset.py --allow-s3-only
```

---

## 📝 Every Session

```bash
# Activate environment
source activate_env.sh
```

---

## 🧪 Running Experiments

### Quick Test
```bash
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

### Short Run (interactive)
```bash
bash scripts/run_experiment_simple.sh configs/experiments/my_exp.yaml
```

### Long Run (tmux - recommended)
```bash
# Start in tmux
bash scripts/tmux_run.sh configs/experiments/my_exp.yaml my_session

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t my_session
```

### Long Run (nohup - fire and forget)
```bash
bash scripts/nohup_run.sh configs/experiments/my_exp.yaml
tail -f logs/nohup_*.log
```

### Multiple Experiments (sweep)
```bash
bash scripts/run_sweep.sh configs/experiments/*.yaml
```

---

## 🔧 Common Options

```bash
# Custom name
--name my_experiment

# Custom seed
--seed 42

# Resume from checkpoint
--resume

# Skip preflight checks
--skip-preflight
```

**Example:**
```bash
bash scripts/run_experiment_simple.sh configs/experiments/my_exp.yaml \
  --name ablation_lr001 \
  --seed 42
```

---

## 📊 Monitoring

```bash
# Check GPU
nvidia-smi

# Watch GPU continuously
watch -n 1 nvidia-smi

# Monitor log file
tail -f logs/train.log

# List tmux sessions
tmux ls

# Attach to tmux
tmux attach -t session_name

# Disk space
df -h /bigdata

# Running processes
ps aux | grep python
```

---

## 🛠️ Troubleshooting

### Virtual environment not active
```bash
source activate_env.sh
```

### CUDA not available
```bash
nvidia-smi  # Check GPU
python -c "import torch; print(torch.cuda.is_available())"
bash scripts/setup_env.sh --force --cuda-version 12.2
```

### Out of memory
```yaml
# In config: reduce batch_size
batch_size: 16  # instead of 64
```

### Disk full
```bash
df -h /bigdata
rm -rf runs/old_run_*
```

---

## 📁 Important Paths

```bash
# Project structure
./venv/              # Virtual environment
./configs/           # Experiment configs
./scripts/           # Helper scripts
./runs/              # Experiment outputs
./logs/              # Log files
./checkpoints/       # Model checkpoints

# After a run
runs/<timestamp>_<name>/
  ├── config.yaml              # Config used
  ├── git_info.json           # Git state
  ├── environment/            # Full snapshot
  ├── checkpoints/            # Model files
  ├── logs/train.log          # Training log
  ├── metrics/metrics.jsonl   # Per-step metrics
  └── SUCCESS or FAILED       # Status
```

---

## 🔑 Key Commands Cheatsheet

| Task | Command |
|------|---------|
| Activate env | `source activate_env.sh` |
| Check setup | `bash scripts/preflight.sh` |
| Prepare data | `bash scripts/prepare_data.sh` |
| Test MinIO | `python scripts/verify_dataset.py --allow-s3-only` |
| Run experiment | `bash scripts/run_experiment_simple.sh <config>` |
| Run in tmux | `bash scripts/tmux_run.sh <config>` |
| Run in background | `bash scripts/nohup_run.sh <config>` |
| Run multiple | `bash scripts/run_sweep.sh <configs...>` |
| List tmux | `tmux ls` |
| Attach tmux | `tmux attach -t <name>` |
| Detach tmux | `Ctrl+B, then D` |
| Monitor log | `tail -f logs/*.log` |
| Check GPU | `nvidia-smi` |
| Check disk | `df -h /bigdata` |
| Check S3 cache | `du -sh $S3_CACHE_DIR` |

---

## 💡 Pro Tips

1. **Always commit before running**: `git commit -am "Config for exp X"`
2. **Use descriptive names**: `--name ablation_dropout02_lr001`
3. **Test with smoke test first**: `smoke_test.yaml` (2 epochs)
4. **Long runs = tmux**: Don't lose progress on disconnect
5. **Check disk space first**: `df -h /bigdata`
6. **Clean up old runs**: `rm -rf runs/failed_*`
7. **Review snapshots**: Each run captures full environment
8. **Monitor actively**: `watch -n 1 nvidia-smi`

---

## 📞 Emergency Commands

```bash
# Kill tmux session
tmux kill-session -t session_name

# Kill process by PID
kill <PID>

# Kill all Python processes (careful!)
pkill -f python

# Force clean and restart
bash scripts/setup_env.sh --force
source activate_env.sh
bash scripts/preflight.sh
```

---

## ✅ Pre-Run Checklist

- [ ] Environment activated: `source activate_env.sh`
- [ ] Preflight passed: `bash scripts/preflight.sh`
- [ ] Config ready: `configs/experiments/my_exp.yaml`
- [ ] Git committed: `git status` shows clean
- [ ] Disk space OK: `df -h /bigdata` shows >50GB
- [ ] Using tmux for long runs: `scripts/tmux_run.sh`

---

**Ready to run? 🚀**

```bash
bash scripts/run_experiment_simple.sh configs/experiments/my_exp.yaml
```

---

**Need help?** Check `README_JUPYTERHUB.md` for full documentation.
