# A100 Quick Commands - Copy & Paste

> **Essential commands after cloning on fresh JupyterHub**

---

## 🚀 Complete Setup (Run in Order)

```bash
# 1. Clone and navigate
cd /bigdata/userhome/students/$USER
git clone https://github.com/toniIepure25/FMRI2images.git Bachelor_V2
cd Bachelor_V2

# 2. Configure
cp .env.jupyterhub .env
sed -i "s/student01/$USER/g" .env

# 3. Install environment (5-10 min)
bash scripts/setup_env.sh
source activate_env.sh

# 4. Verify
bash scripts/preflight.sh

# 5. Download full NSD subject01 data (2-3 hours)
bash scripts/download_full_nsd_subj01.sh

# 6. Download stimuli (30 min)
mkdir -p data/nsd/nsddata_stimuli/stimuli/nsd
cd data/nsd/nsddata_stimuli/stimuli/nsd
wget -r -np -nH --cut-dirs=4 \
    https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/
cd ~/Bachelor_V2

# 7. Build index (5 min)
python3 scripts/build_full_subj01_index.py

# 8. Build CLIP cache (1-2 hours)
bash scripts/build_clip_for_training.sh

# 9. Build target CLIP cache (1-2 hours)
python3 scripts/build_target_clip_cache_robust.py \
    --cache_dir cache \
    --output cache/target_clip_cache_sd21.h5 \
    --batch_size 256 \
    --device cuda:0

# 10. Build preprocessors (2 min)
bash scripts/build_all_preprocessors.sh

# 11. Final verification
bash scripts/preflight.sh
python3 scripts/verify_dataset.py
```

**Total time**: 4-6 hours  
**Result**: Fully configured system ready for experiments

---

## 🎯 Run Experiments

```bash
# Every login - activate environment
cd ~/Bachelor_V2
source activate_env.sh

# Run all experiments in background with tmux
tmux new -s experiments
bash scripts/run_all_experiments.sh 0
# Detach: Ctrl+B then D

# Monitor progress (in another terminal)
tmux new -s monitor
tail -f experimental_results/logs/run_all_*.log
# Detach: Ctrl+B then D

# Reattach to sessions
tmux attach -t experiments
tmux attach -t monitor

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## 📤 MinIO Upload (After Getting Credentials)

```bash
# Install MinIO client
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Configure (replace with your credentials)
mc alias set uniminio https://ubbc1u.ro:9000 YOUR_ACCESS_KEY YOUR_SECRET_KEY

# Create bucket
mc mb uniminio/nsd-data

# Upload data (2-3 hours)
mc cp --recursive data/nsd/nsddata_betas/ uniminio/nsd-data/nsddata_betas/
mc cp --recursive data/nsd/nsddata_stimuli/ uniminio/nsd-data/nsddata_stimuli/
mc cp --recursive cache/clip_embeddings/ uniminio/nsd-data/cache/clip_embeddings/

# Verify
mc ls -r uniminio/nsd-data/

# Add to .env (for future use)
echo "" >> .env
echo "# MinIO Configuration" >> .env
echo "AWS_ENDPOINT_URL=https://ubbc1u.ro:9000" >> .env
echo "AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY" >> .env
echo "AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY" >> .env
echo "ALLOW_S3_ONLY=1" >> .env
```

---

## 🔍 Monitoring Commands

```bash
# GPU usage
nvidia-smi
watch -n 1 nvidia-smi

# Disk usage
df -h

# Memory usage  
free -h

# CPU usage
htop

# Experiment logs
tail -f experimental_results/logs/run_all_*.log

# Check running processes
ps aux | grep python

# Experiment status
bash scripts/check_experiment_status.sh
```

---

## 🛠️ Useful Shortcuts

```bash
# Kill all Python processes (if stuck)
pkill -9 python

# Clear CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()"

# List tmux sessions
tmux ls

# Kill tmux session
tmux kill-session -t experiments

# Reactivate environment after login
cd ~/Bachelor_V2 && source activate_env.sh

# Quick data check
ls -lh data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/

# Quick cache check
ls -lh cache/
```

---

## 🚨 Emergency Commands

```bash
# System running out of space
du -sh * | sort -h
rm -rf logs/old_experiments

# GPU stuck
sudo fuser -v /dev/nvidia*
sudo kill -9 PID_NUMBER

# Python environment corrupted
rm -rf venv/
bash scripts/setup_env.sh

# Clear all caches
rm -rf cache/clip_embeddings/*
rm -rf cache/target_clip_cache*.h5
# Then rebuild with build commands

# Reset experiments
rm -rf experimental_results/
rm -rf checkpoints/
bash scripts/run_all_experiments.sh 0
```

---

## 📋 Daily Workflow

```bash
# Morning routine
cd ~/Bachelor_V2
source activate_env.sh
nvidia-smi
tmux attach -t experiments || tmux new -s experiments

# Run experiments
bash scripts/run_all_experiments.sh 0

# Check progress
tail -f experimental_results/logs/run_all_*.log

# Before leaving
tmux detach  # Ctrl+B then D
# Experiments keep running!
```

---

## 💡 Optimization for A100 40GB

```yaml
# Add to experiment configs for maximum performance

data_loading:
  batch_size: 512          # Use full VRAM
  num_workers: 32          # Use most cores
  persistent_workers: true
  prefetch_factor: 4       # Aggressive prefetching

training:
  mixed_precision: true    # fp16 for speed
  gradient_clip: 1.0
  
system:
  num_threads: 40          # All cores
```

---

## 🎯 Time Estimates on A100 40GB

| Task | Time |
|------|------|
| Environment setup | 5-10 min |
| NSD data download | 2-3 hours |
| Stimuli download | 30 min |
| CLIP cache build | 1-2 hours |
| Target CLIP cache | 1-2 hours |
| Preprocessors | 2 min |
| Ridge training | 5-10 min |
| MLP training (50 epochs) | 30-45 min |
| Full ablation suite | 6-12 hours |

**Total initial setup**: 4-6 hours  
**Total experiments**: 6-12 hours

---

## 📞 Quick Help

- Setup issues: `JUPYTERHUB_A100_SETUP.md`
- MinIO help: `MINIO_SETUP_GUIDE.md`
- Experiment help: `EXPERIMENT_GUIDE.md`
- All commands: `COMMANDS_AFTER_CLONE.md`

---

**🚀 Ready to go! Start with the Complete Setup commands above.**
