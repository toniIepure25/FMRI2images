#!/bin/bash
# QUICK COMMANDS FOR SERVER - Copy and paste these!
# Run after: cd ~/Bachelor_V2 && git pull

# =============================================================================
# 1️⃣ COMPARE EXISTING EXPERIMENTS (synthetic data runs)
# =============================================================================
python scripts/compare_experiments.py \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191153_baseline_subj01 \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191216_novel_subj01 \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191239_ablation_subj01 \
  --output experiment_comparison

# View results:
cat experiment_comparison/comparison_report.md


# =============================================================================
# 2️⃣ TRAIN ON REAL NSD DATA (MOST IMPORTANT!)
# =============================================================================
python scripts/train_real_nsd.py --config experiments/real_baseline_subj01.yaml

# This will create a new run directory like:
# /bigdata/.../runs/20260112_XXXXXX_real_baseline_subj01_session1/


# =============================================================================
# 3️⃣ GENERATE IMAGES FROM TRAINED MODEL
# =============================================================================
# First, extract a sample fMRI:
python -c "
import nibabel as nib
import numpy as np
img = nib.load('data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz')
data = img.get_fdata()
sample = data[:, :, :, 0]
np.save('sample_fmri.npy', sample)
print(f'✓ Saved sample fMRI shape: {sample.shape}')
"

# Then generate (REPLACE WITH YOUR RUN DIRECTORY!):
python scripts/generate_images.py \
  --checkpoint /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/YYYYMMDD_HHMMSS_real_baseline_subj01_session1/checkpoints/best_model.pt \
  --fmri sample_fmri.npy \
  --output generation_results \
  --visualize


# =============================================================================
# 4️⃣ EVALUATE MODEL PERFORMANCE
# =============================================================================
# REPLACE WITH YOUR RUN DIRECTORY!
python scripts/evaluate_model.py \
  --checkpoint /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/YYYYMMDD_HHMMSS_real_baseline_subj01_session1/checkpoints/best_model.pt \
  --output evaluation_results


# =============================================================================
# 📊 VIEW RESULTS
# =============================================================================
# View comparison report:
cat experiment_comparison/comparison_report.md

# View evaluation metrics:
cat evaluation_results/evaluation_metrics.json

# List all runs:
ls -lht /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/

# Check latest run logs:
tail -f /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/*/logs/train.log


# =============================================================================
# 🔧 TROUBLESHOOTING
# =============================================================================
# Check CUDA:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Check disk space:
df -h /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/

# Check GPU usage:
nvidia-smi

# Check index exists:
ls -lh data/indices/nsd_index/subject=subj01/index.parquet
