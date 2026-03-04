# JupyterHub Quick Reference Card

> **Print this or keep it handy while working on JupyterHub!**

---

## Hardware

| Resource | Value |
|----------|-------|
| GPU | **NVIDIA H100 80GB HBM3** |
| Driver | 570.211.01 |
| CUDA | 12.8 |
| System RAM | ~100 GB |
| CPU cores | 32 |
| Storage | PVC at `/home/jovyan/work` (~877 GB, NFS-backed) |
| Python | 3.13 via `conda base` |

---

## First Time Setup

```bash
# 1. Configure environment
cp .env.jupyterhub .env
nano .env  # Set HF_TOKEN, verify paths

# 2. Install package
pip install -e ".[train,diffusion]"

# 3. Verify setup
make preflight

# 4. Pre-extract fMRI features
make preextract SUBJECT=subj01

# 5. Build CLIP cache (if not already present)
make clip-cache

# 6. Build multi-layer CLIP cache (for v7+ fused targets)
python3 scripts/build/build_multilayer_clip_cache.py \
    --subjects subj01 subj02 subj05 subj07 \
    --layers 12 18 24 --fuse-alpha 0.3
```

---

## Every Session

```bash
# Source environment
set -a && source .env && set +a

# Re-install if pod restarted (editable install lost on restart)
pip install -e ".[train,diffusion]"
```

---

## Running Experiments

### V9 Ablation (Current)

```bash
# All 4 N-series V9 experiments on subj01
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N9 > ablation_v9.log 2>&1 &
tail -f ablation_v9.log

# Without checkpoints (saves ~5 GB disk)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N9 SAVE_CKPT=0 > ablation_v9.log 2>&1 &
```

### Single Experiment

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1v9_vmf_nce.yaml --gpu 0
```

### All N-series (v5 through v9)

```bash
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N > ablation_all_n.log 2>&1 &
```

---

## Monitoring

```bash
# GPU utilization
nvidia-smi
watch -n 1 nvidia-smi

# Training log
tail -f ablation_v9.log

# Disk space
df -h /home/jovyan/work

# Running processes
ps aux | grep python
```

---

## Key Paths

```bash
# Project
/home/jovyan/work/FMRI2images/          # Git repo root (branch: dirbrain-vmf-uacfg)

# Data
/home/jovyan/work/data/nsd/              # NSD raw data
cache/preextracted/subject=subj01/       # Pre-extracted fMRI (~1.8 GB)
outputs/clip_cache/clip.parquet          # Standard CLIP cache
outputs/clip_cache/clip_multilayer.parquet  # Multi-layer CLIP cache

# Configs
configs/experiments/N1v9_vmf_nce.yaml    # V9 configs (current)
configs/experiments/N2v9_roi_transformer.yaml
configs/experiments/N3v9_roi_dcf.yaml
configs/experiments/N4v9_full_system.yaml

# Results
experimental_results/{experiment}/{subject}/
  metrics/summary.json                   # Key metrics (R@1, loss, wall time)
  metrics/training_log.csv               # Per-epoch training history
  checkpoints/best.pt, last.pt           # Model checkpoints
  logs/train.log                         # Full training log
```

---

## Troubleshooting

### Module not found (`fmri2img`)
```bash
pip install -e ".[train,diffusion]"
```

### Out of GPU memory
Reduce `batch_size` in config (e.g., 64 instead of 128) and increase `gradient_accumulation_steps` proportionally.

### Disk full
```bash
# Check usage
du -sh /home/jovyan/work/FMRI2images/experimental_results/*/subj*/checkpoints/

# Remove unused subject features
rm cache/preextracted/subject=subj02/fmri_features.npy
rm cache/preextracted/subject=subj05/fmri_features.npy
rm cache/preextracted/subject=subj07/fmri_features.npy
```

### CUDA/Driver mismatch
```bash
nvidia-smi  # Check driver version
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## Key Commands Cheatsheet

| Task | Command |
|------|---------|
| Setup env | `set -a && source .env && set +a` |
| Install package | `pip install -e ".[train,diffusion]"` |
| Check GPU | `nvidia-smi` |
| Run V9 ablation | `make ablation SUBJECTS="subj01" GPU=0 ONLY=N9` |
| Run single exp | `python3 scripts/training/train_unified.py --config <config>` |
| Aggregate results | `python3 scripts/evaluation/aggregate_ablation.py --subjects subj01` |
| Check disk | `df -h /home/jovyan/work` |
| Monitor log | `tail -f ablation_v9.log` |
