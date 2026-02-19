# Scripts Directory

**67 production-ready executable scripts organized into 7 functional categories.**

---

## 📂 Directory Structure

```
scripts/
├── training/          # Model training scripts (11 files)
├── evaluation/        # Evaluation and metrics (10 files)
├── reconstruction/    # Image generation (5 files)
├── build/            # Data preparation & caching (13 files)
├── analysis/         # Results analysis & comparison (13 files)
├── orchestration/    # Experiment orchestration (8 files)
├── utils/            # System utilities & validation (17 files)
└── README.md         # This file
```

**Total**: 67 executable scripts (CLI tools) + 7 READMEs = 74 files

---

## 📁 Categories

### 🏋️ `training/` - Model Training (11 scripts)

**Main entry point:**
- **`train.py`** ⭐ - Wrapper forwarding to `train_unified.py`

**Training implementations:**
- **`train_unified.py`** ⭐ - Main unified training (all features, all losses)
- `train_ridge.py` - Ridge regression baseline
- `train_mlp.py` - MLP encoder baseline
- `train_clip_adapter.py` - CLIP adapter training (512D→768/1024D)
- `train_two_stage.py` - Two-stage training pipeline
- `train_real_nsd.py` - Training on real NSD data
- `train_clip_to_fmri.py` - Reverse direction training

### 📊 `evaluation/` - Evaluation & Metrics (10 scripts)

**Main entry point:**
- **`evaluate.py`** ⭐ - Wrapper forwarding to `evaluate_experiment.py`

**Evaluation implementations:**
- **`evaluate_experiment.py`** ⭐ - Main experiment evaluation (all metrics)
- **`eval_stage1_embeddings.py`** ⭐ - Stage 1 embedding-space evaluation
- `eval_stage1_probabilistic.py` - Probabilistic/uncertainty evaluation
- `eval_comprehensive.py` - Comprehensive metrics suite
- `eval_shared1000_full.py` - Shared1000 benchmark evaluation
- `eval_reconstruction.py` - Reconstruction quality evaluation
- `eval_retrieval.py` - Retrieval-specific metrics
- `summarize_shared1000.py` - Shared1000 result summarization

### 🎨 `reconstruction/` - Image Generation (5 scripts)

**Main entry point:**
- **`reconstruct.py`** ⭐ - Wrapper forwarding to `generate_images.py`

**Implementation:**
- **`generate_images.py`** ⭐ - Generate reconstructed images from fMRI
- `decode_diffusion.py` - Diffusion model decoding
- `decode_two_stage.py` - Two-stage reconstruction pipeline
- `reconstruct_nn.py` - Nearest neighbor reconstruction

### 🔧 `build/` - Data Preparation & Caching (13 scripts)

**Key scripts:**
- **`build_target_clip_cache_robust.py`** ⭐ - Robust CLIP cache builder
- **`fit_preprocessing.py`** ⭐ - Fit preprocessing pipeline (PCA/PCR)
- `build_full_index.py` - Build full NSD data index
- `build_full_subj01_index.py` - Build subj01-specific index
- `build_clip_cache.py` - Standard CLIP cache builder
- `build_multilayer_clip_cache.py` - Multi-layer CLIP embeddings
- `build_text_clip_cache.py` - Text CLIP embeddings
- `build_embedding_preproc.py` - Build embedding preprocessing
- `nsd_index_builder.py` - NSD index builder utility
- `download_stimuli_hdf5.py` - Download NSD stimuli
- `fast_preproc.py` - Fast preprocessing implementation
- `build_all_preprocessors.sh` - Build all preprocessors (batch script)

### 🔬 `analysis/` - Results Analysis (13 scripts)

- `analyze_run.py` - Analyze single experiment run
- `compare_experiments.py` - Compare multiple experiments
- `compare_evals.py` - Compare evaluation results
- `compare_methods.py` - Method comparison
- `ablate_preproc_and_ridge.py` - Preprocessing ablation study
- `ablation_driver.py` - Ablation study orchestration
- `generate_comparison_gallery.py` - Visual comparison galleries
- `generate_report.py` - Report generation
- `plot_metrics.py` - Metric visualization
- `report_ablation.py` - Ablation report generation
- `summarize_reports.py` - Report summarization
- `build_paper_artifacts.py` - Generate paper figures and tables

### 🚀 `orchestration/` - Experiment Orchestration (8 scripts)

**Key scripts:**
- **`run_all_experiments.sh`** ⭐ - Run all experiments (ablation study)
- `run_experiment.py` - Single experiment runner
- `run_full_pipeline.py` - Full end-to-end pipeline
- `run_reconstruct_and_eval.py` - Reconstruction + evaluation
- `run_stage34_recon_eval.py` - Stage 3/4 reconstruction evaluation
- `monitor_experiments.sh` - Monitor running experiments
- `check_experiment_status.sh` - Check experiment status

### ✅ `utils/` - System Utilities (17 scripts)

**Health checks:**
- **`preflight.py`** ⭐ - Quick system preflight checks
- **`doctor.py`** ⭐ - Comprehensive health diagnostics
- `smoke.py` - Quick smoke test
- `train_smoke.py` - Training smoke test

**Validation:**
- **`verify_dataset.py`** ⭐ - Dataset verification
- `validate_checkpoints.py` - Validate checkpoint files
- `validate_index.py` - Validate index files
- `check_index_headers.py` - Validate index file headers

**Other utilities:**
- **`fetch_models.py`** ⭐ - Fetch HuggingFace models
- `archive_experiment.py` - Archive experiment results
- `write_run_manifest.py` - Write experiment manifests
- `quick_status.py` - Quick project status check
- `download_sd_model.py` - Download Stable Diffusion models

---

## 🎯 Quick Start

### Setup
```bash
# From root directory
./setup.sh
```

### Training
```bash
# Using wrapper (recommended)
python scripts/training/train.py --config configs/experiments/exp0_baseline.yaml

# Direct (more options)
python scripts/training/train_unified.py --config configs/experiments/exp0_baseline.yaml
```

### Evaluation
```bash
# Using wrapper
python scripts/evaluation/evaluate.py --checkpoint outputs/exp0/best.pt

# Direct
python scripts/evaluation/evaluate_experiment.py --checkpoint outputs/exp0/best.pt
```

### Reconstruction
```bash
# Using wrapper
python scripts/reconstruction/reconstruct.py --checkpoint outputs/exp0/best.pt --n_images 100

# Direct
python scripts/reconstruction/generate_images.py --checkpoint outputs/exp0/best.pt --n_images 100
```

### Health Checks
```bash
python scripts/utils/preflight.py  # Quick system check
python scripts/utils/doctor.py      # Comprehensive check
python scripts/utils/smoke.py       # Quick import test
```

---

## 📋 Common Workflows

### Full Ablation Study
```bash
# 1. Build preprocessors (once)
bash scripts/build/build_all_preprocessors.sh

# 2. Run all experiments
bash scripts/orchestration/run_all_experiments.sh 0

# 3. Compare results
python scripts/analysis/compare_experiments.py experimental_results/*
```

### Single Experiment
```bash
# 1. Check system
python scripts/utils/preflight.py

# 2. Train
python scripts/training/train.py --config configs/experiments/exp4_gaussian_nce.yaml

# 3. Evaluate
python scripts/evaluation/evaluate.py --checkpoint outputs/exp4/best.pt

# 4. Generate images
python scripts/reconstruction/reconstruct.py --checkpoint outputs/exp4/best.pt
```

### Build Data Caches
```bash
# Build index
python scripts/build/build_full_index.py --subject subj01

# Fit preprocessing
python scripts/build/fit_preprocessing.py \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --output-dir cache/preproc/subject=subj01

# Build CLIP cache
python scripts/build/build_target_clip_cache_robust.py
```

---

## 🔍 Script Details

### Training Options

**Unified Training** (`training/train_unified.py`) supports:
- Multiple architectures (Ridge, MLP, Unified, Gaussian)
- Multiple losses (MSE, InfoNCE, Gaussian NLL, Gaussian-NCE, KL)
- Preprocessing pipelines (PCA/PCR, soft reliability)
- Memory queue (MoCo-style contrastive learning)
- Comprehensive logging and checkpointing

**Specialized Training:**
- `training/train_ridge.py` - Ridge regression (fast baseline)
- `training/train_mlp.py` - MLP encoder
- `training/train_clip_adapter.py` - Adapter for different CLIP dimensions
- `training/train_two_stage.py` - Two-stage fMRI→latent→CLIP

### Evaluation Options

**Experiment Evaluation** (`evaluation/evaluate_experiment.py`) includes:
- Retrieval metrics (R@K, MRR, nDCG)
- Identification (2AFC, AUC)
- Representational similarity (RSA, CKA)
- Probabilistic metrics (calibration, conformal prediction)
- Comprehensive plots and reports

**Specialized Evaluation:**
- `evaluation/eval_stage1_embeddings.py` - Embedding-space evaluation
- `evaluation/eval_stage1_probabilistic.py` - Probabilistic/uncertainty evaluation
- `evaluation/eval_comprehensive.py` - All metrics combined
- `evaluation/eval_shared1000_full.py` - Benchmark on Shared1000 test set

---

## ⭐ Most Used Scripts

1. **`training/train.py`** / `train_unified.py` - Main training
2. **`evaluation/evaluate.py`** / `evaluate_experiment.py` - Main evaluation
3. **`utils/preflight.py`** - System checks
4. **`utils/doctor.py`** - Health check before experiments
5. **`build/build_target_clip_cache_robust.py`** - Build CLIP cache
6. **`orchestration/run_all_experiments.sh`** - Run ablation study
7. **`utils/fetch_models.py`** - Download models
8. **`reconstruction/reconstruct.py`** / `generate_images.py` - Image generation

---

## 📚 Documentation

For more details, see:
- [RUNNING_EXPERIMENTS.md](../docs/guides/RUNNING_EXPERIMENTS.md) - Comprehensive guide
- [QUICK_REFERENCE.md](../docs/QUICK_REFERENCE.md) - Command cheat sheet
- Main [README.md](../README.md) - Project overview

---

## 🧹 Organization & Maintenance

This directory follows **professional research-grade structure**:

### ✅ What Scripts Are
- **Executable CLI tools only** - all have `if __name__ == "__main__"` 
- **Thin wrappers** - import functionality from `src/fmri2img/` library
- **User-facing entry points** - for researchers and users
- **Organized by function** - training, evaluation, build, analysis, etc.

### ✅ What Scripts Are NOT
- ❌ NOT library code (that belongs in `src/fmri2img/`)
- ❌ NOT test code (that belongs in `tests/`)
- ❌ NOT configuration (that belongs in `configs/`)

### 📊 Current Status
- **67 executable scripts** organized into 7 categories
- **0 duplicate files** - all consolidated
- **100% functional** - all tested and working
- **Fully documented** - README in each subdirectory

### 🔗 Related Structure
- **Library code**: `src/fmri2img/` (79 modules)
- **Tests**: `tests/` (29 test files)
- **Documentation**: `docs/` + `ARCHITECTURE.md`
- **Configs**: `configs/` (YAML experiment configs)

See [ARCHITECTURE.md](../ARCHITECTURE.md) for complete project structure details.

**Last updated:** February 20, 2026
