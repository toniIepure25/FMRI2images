# JupyterHub A100 Complete Setup Guide

> **After Fresh Clone - Full Production Setup for A100 GPU (40GB VRAM, 200GB RAM, 40 Cores)**

---

## 🎯 Overview

This guide covers complete setup after JupyterHub cleanup:
- ✅ Environment setup
- ✅ Full NSD subject01 data download (~50GB)
- ✅ CLIP cache building
- ✅ Stable Diffusion model download
- ✅ MinIO data upload for university sharing
- ✅ Optimized configs for A100 40GB

**Total Time**: 3-5 hours  
**Disk Space Needed**: ~150GB  
**Your Hardware**: A100 40GB VRAM, 200GB RAM, 40 CPU cores

---

## 📋 Quick Start (Copy-Paste All)

```bash
# Navigate to your user directory
cd /bigdata/userhome/students/$USER

# Clone the repository
git clone https://github.com/toniIepure25/FMRI2images.git Bachelor_V2
cd Bachelor_V2

# Configure environment
cp .env.jupyterhub .env
sed -i "s/student01/$USER/g" .env

# Install Python environment (5-10 minutes)
bash scripts/setup_env.sh

# Activate environment
source activate_env.sh

# Verify setup
bash scripts/preflight.sh
```

**Status**: Environment ready! ✅ Now continue with data preparation below.

---

## 📊 Step-by-Step Setup

### **1. Environment Setup** (10 minutes)

Already done above! Skip if you ran the quick start.

### **2. Download Full NSD Subject01 Data** (2-3 hours)

Download all 40 sessions of subject01 beta files (~50GB):

```bash
cd ~/Bachelor_V2
source activate_env.sh

# Run the full download script
bash scripts/download_full_nsd_subj01.sh
```

**What it downloads**:
- 40 sessions of preprocessed fMRI data (betas)
- Each session: ~1.2GB
- Total: ~50GB

**Alternative - Use MinIO** (if already uploaded by others):
- See section "Using MinIO" below
- Skip download if data already on MinIO server

---

### **3. Download Stimuli Images** (30 minutes)

Download the NSD stimulus images (73k images, ~15GB):

```bash
# Download stimuli
python3 scripts/download_nsd_stimuli.py --output data/nsd/nsddata_stimuli
```

**Or use wget** (faster):
```bash
mkdir -p data/nsd/nsddata_stimuli/stimuli/nsd
cd data/nsd/nsddata_stimuli/stimuli/nsd

# Download all stimulus images
wget -r -np -nH --cut-dirs=4 \
    https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/
```

---

### **4. Build Full Subject01 Index** (5 minutes)

Create the training index for all 40 sessions:

```bash
cd ~/Bachelor_V2
python3 scripts/build_full_subj01_index.py
```

**Output**: `data/indices/nsd_index/subj01_full_train.csv` (~30k trials)

---

### **5. Build CLIP Cache** (1-2 hours)

Pre-compute CLIP embeddings for all stimuli:

```bash
# Build CLIP cache for training (ViT-B/32, 512-D)
bash scripts/build_clip_for_training.sh
```

**What it does**:
- Loads CLIP ViT-B/32 model
- Processes 73k images
- Saves embeddings to `cache/clip_embeddings/`
- Uses your 40GB VRAM efficiently

**Progress**: Shows progress bar, ~1000 images/minute on A100

---

### **6. Build Target CLIP Cache for SD-2.1** (1-2 hours)

Build 1024-D CLIP embeddings for Stable Diffusion:

```bash
python3 scripts/build_target_clip_cache_robust.py \
    --cache_dir cache \
    --output cache/target_clip_cache_sd21.h5 \
    --batch_size 256 \
    --device cuda:0
```

**Uses**: OpenCLIP ViT-H/14 (1024-D) for Stable Diffusion 2.1

---

### **7. Build Embedding Preprocessors** (2 minutes)

Build PCA and whitening preprocessors:

```bash
bash scripts/build_all_preprocessors.sh
```

**Creates**:
- `cache/center_pcr_k8.pkl` - PCA with 8 components
- `cache/center_whiten.pkl` - Whitening transform

---

### **8. Verify Everything** (1 minute)

```bash
bash scripts/preflight.sh
python3 scripts/verify_dataset.py
```

**Expected**: All checks pass ✅

---

## 🚀 Running Optimized Experiments on A100

### **Create A100-Optimized Config**

```bash
nano configs/experiments/a100_production.yaml
```

Add this content:

```yaml
# A100 40GB Optimized Configuration
base: configs/base.yaml

# Experiment identification
experiment:
  name: "a100_production_run"
  description: "Full production run optimized for A100 40GB"
  tags: ["production", "a100", "full-data"]

# Dataset configuration  
dataset:
  subject: "subj01"
  index_file: "data/indices/nsd_index/subj01_full_train.csv"
  nsd_root: "/bigdata/userhome/students/${USER}/data/nsd/nsddata"
  betas_root: "/bigdata/userhome/students/${USER}/data/nsd/nsddata_betas"
  stimuli_root: "/bigdata/userhome/students/${USER}/data/nsd/nsddata_stimuli"
  cache_dir: "cache"
  clip_cache_path: "cache/clip_embeddings/clip_vitb32_embeddings.h5"

# Data loading optimized for 40 cores
data_loading:
  batch_size: 512          # Large batch for 40GB VRAM
  num_workers: 32          # Use 32 of 40 cores
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4       # Aggressive prefetching with 200GB RAM

# Training configuration
training:
  max_epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01
  gradient_clip: 1.0
  mixed_precision: true    # fp16 for faster training

# Model configuration
model:
  type: "ridge"
  ridge_alpha: 100000
  normalize: true

# Evaluation
evaluation:
  eval_every_n_epochs: 5
  retrieval_k: [1, 5, 10]
  save_best_model: true

# System
system:
  device: "cuda:0"
  seed: 42
  num_threads: 40
  
# Logging
logging:
  log_every_n_steps: 10
  checkpoint_dir: "checkpoints/a100_production"
```

---

### **Run Production Experiments**

```bash
# Activate environment
cd ~/Bachelor_V2
source activate_env.sh

# Single experiment
python3 -m fmri2img.train \
    --config configs/experiments/a100_production.yaml \
    --gpu 0

# Or use the automated runner for all experiments
bash scripts/run_all_experiments.sh 0
```

**In background with tmux**:
```bash
tmux new -s experiments
source activate_env.sh
bash scripts/run_all_experiments.sh 0
# Detach: Ctrl+B then D
# Reattach: tmux attach -t experiments
```

---

## 📤 MinIO Setup (For University Sharing)

### **1. Get MinIO Credentials**

Contact your IT admin:
```
Hi,

I need MinIO access to upload NSD dataset for shared access.
Can you provide:
- Access Key ID
- Secret Access Key  
- Bucket name (or create "nsd-data")

Thanks!
```

---

### **2. Configure MinIO**

Once you have credentials:

```bash
# Add to .env file
nano .env
```

Add these lines:
```bash
# MinIO Configuration
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# S3 paths
NSD_S3_ROOT=s3://nsd-data/nsddata/
NSD_S3_BETAS=s3://nsd-data/nsddata_betas/
NSD_S3_STIMULI=s3://nsd-data/nsddata_stimuli/

# Enable S3
ALLOW_S3_ONLY=1
S3_CACHE_DIR=/bigdata/userhome/students/${USER}/.cache/s3_cache
```

---

### **3. Upload Data to MinIO**

```bash
# Install MinIO client
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Configure MinIO alias
mc alias set uniminio https://ubbc1u.ro:9000 \
    your_access_key your_secret_key

# Create bucket (if doesn't exist)
mc mb uniminio/nsd-data

# Upload data (takes 2-3 hours for 50GB)
mc cp --recursive data/nsd/nsddata_betas/ uniminio/nsd-data/nsddata_betas/
mc cp --recursive data/nsd/nsddata_stimuli/ uniminio/nsd-data/nsddata_stimuli/
mc cp --recursive cache/clip_embeddings/ uniminio/nsd-data/cache/clip_embeddings/

# Verify upload
mc ls -r uniminio/nsd-data/
```

---

### **4. Use MinIO in Experiments**

Update experiment config:

```yaml
dataset:
  nsd_root: ${NSD_S3_ROOT}
  betas_root: ${NSD_S3_BETAS}
  stimuli_root: ${NSD_S3_STIMULI}
  cache_dir: ${S3_CACHE_DIR}
  use_s3: true
```

---

## 🔍 Monitoring & Optimization

### **Monitor GPU Usage**

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or with more detail
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv'
```

### **Monitor Training Progress**

```bash
# Follow logs
tail -f experimental_results/logs/run_all_*.log

# Or use monitoring script
bash scripts/monitor_experiments.sh
```

### **Check Experiment Status**

```bash
bash scripts/check_experiment_status.sh
```

---

## 📊 Performance Expectations on A100 40GB

| Task | Time (A100) | Previous (A100 20GB) |
|------|-------------|----------------------|
| CLIP cache build | 1-2 hours | 2-3 hours |
| Ridge training (30k samples) | 5-10 min | 10-15 min |
| MLP training (50 epochs) | 30-45 min | 60-90 min |
| Full ablation suite | 6-12 hours | 12-24 hours |

**Optimizations enabled**:
- ✅ Batch size 512 (vs 256)
- ✅ 32 data workers (vs 16)
- ✅ Mixed precision training
- ✅ Persistent workers
- ✅ 4x prefetch factor

---

## 🎯 Recommended Batch Sizes for A100 40GB

| Model | Batch Size | VRAM Usage |
|-------|------------|------------|
| Ridge | 512 | ~8GB |
| MLP (4 layers) | 512 | ~12GB |
| CLIP Adapter | 256 | ~20GB |
| Two-Stage | 256 | ~25GB |

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Environment activated: `(venv)` in prompt
- [ ] GPU visible: `nvidia-smi` shows A100 40GB
- [ ] All 40 beta sessions downloaded
- [ ] Stimuli images present (73k images)
- [ ] CLIP cache built (~2GB file)
- [ ] Target CLIP cache built (~4GB file)
- [ ] Preprocessors built (2 .pkl files)
- [ ] Index file created (~30k rows)
- [ ] Preflight checks pass
- [ ] Dataset verification passes

---

## 🆘 Troubleshooting

### **Issue: Out of Memory (OOM)**

```bash
# Reduce batch size in config
batch_size: 256  # Instead of 512

# Or reduce workers
num_workers: 16  # Instead of 32
```

### **Issue: Slow Data Loading**

```bash
# Check I/O wait
iostat -x 1

# If high I/O wait, reduce prefetching
prefetch_factor: 2  # Instead of 4
```

### **Issue: MinIO Connection Fails**

```bash
# Test connection
python3 -c "
import s3fs
fs = s3fs.S3FileSystem(
    endpoint_url='https://ubbc1u.ro:9000',
    key='your_key',
    secret='your_secret'
)
print(fs.ls(''))
"
```

### **Issue: CUDA Out of Memory**

```bash
# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Restart kernel if in notebook
# Or kill process and restart
```

---

## 📝 Daily Workflow

Each time you log in:

```bash
cd ~/Bachelor_V2
source activate_env.sh

# Check GPU
nvidia-smi

# Run experiments
python3 -m fmri2img.train --config configs/experiments/your_config.yaml
```

---

## 🎓 Next Steps

After setup:

1. **Read experiment guide**: `EXPERIMENT_GUIDE.md`
2. **Review configs**: `configs/experiments/`
3. **Run ablation suite**: `bash scripts/run_all_experiments.sh 0`
4. **Monitor results**: `bash scripts/monitor_experiments.sh`
5. **Analyze outputs**: `experimental_results/`

---

## 📚 Key Documentation Files

- `COMMANDS_AFTER_CLONE.md` - Quick command reference
- `COMPLETE_SETUP_GUIDE.md` - Detailed setup guide
- `JUPYTERHUB_GUIDE.md` - Running experiments on cluster
- `MINIO_SETUP_GUIDE.md` - MinIO detailed configuration
- `EXPERIMENT_GUIDE.md` - Running experiments
- `EVALUATION_GUIDE.md` - Evaluating results

---

## 🎉 Ready to Run!

You're now set up with:
- ✅ Full NSD subject01 data (30k trials)
- ✅ All CLIP caches pre-built
- ✅ Environment optimized for A100 40GB
- ✅ MinIO configured for data sharing
- ✅ Production-ready configs

**Start training**:
```bash
tmux new -s training
source activate_env.sh
bash scripts/run_all_experiments.sh 0
```

**Good luck with your bachelor thesis! 🚀**
