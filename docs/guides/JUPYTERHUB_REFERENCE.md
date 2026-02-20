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

# 3. Run automated setup
./setup.sh

# 4. Verify setup
source .venv/bin/activate
make preflight

# 5. Download NSD dataset (manual - see DATA_REQUIREMENTS.md)
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
source .venv/bin/activate
```

---

## 🧪 Running Experiments

### Quick Test

```bash
python scripts/training/train.py --config configs/experiments/smoke_test.yaml --max_steps 1
```

### Single Experiment

```bash
python scripts/training/train.py --config configs/experiments/exp0_baseline.yaml
```

### Long Run (tmux - recommended)

```bash
# Start in tmux
tmux new -s my_session
python scripts/training/train.py --config configs/experiments/exp0_baseline.yaml

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t my_session
```

### Long Run (nohup)

```bash
nohup python scripts/training/train.py --config configs/experiments/exp0_baseline.yaml &
tail -f nohup.out
```

### All Experiments (batch)

```bash
bash scripts/orchestration/run_all_experiments.sh 0
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
python scripts/training/train.py \
  --config configs/experiments/exp0_baseline.yaml \
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
source .venv/bin/activate
```

### CUDA not available
```bash
nvidia-smi  # Check GPU
python -c "import torch; print(torch.cuda.is_available())"
./setup.sh
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
| Activate env | `source .venv/bin/activate` |
| Check setup | `make preflight` |
| Test MinIO | `python scripts/utils/verify_dataset.py --allow-s3-only` |
| Run experiment | `python scripts/training/train.py --config <config>` |
| Run all experiments | `bash scripts/orchestration/run_all_experiments.sh 0` |
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

## Emergency Commands

```bash
# Kill tmux session
tmux kill-session -t session_name

# Kill process by PID
kill <PID>

# Kill all Python processes (careful!)
pkill -f python

# Force clean and restart
rm -rf .venv
./setup.sh
source .venv/bin/activate
make preflight
```

---

## ✅ Pre-Run Checklist

- [ ] Environment activated: `source .venv/bin/activate`
- [ ] Preflight passed: `make preflight`
- [ ] Config ready: `configs/experiments/exp0_baseline.yaml`
- [ ] Git committed: `git status` shows clean
- [ ] Disk space OK: `df -h /bigdata` shows >50GB
- [ ] Using tmux for long runs: `scripts/tmux_run.sh`

---

**Ready to run? 🚀**

```bash
python scripts/training/train.py --config configs/experiments/exp0_baseline.yaml
```
