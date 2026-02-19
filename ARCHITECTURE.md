# Project Architecture

**Professional Research-Grade Python Project Structure**

---

## 📁 Repository Structure

```
FMRI2images/
├── src/fmri2img/          ← Python package (library code only)
├── scripts/               ← Executable CLI scripts
├── tests/                 ← Test suite
├── configs/               ← Configuration files (YAML)
├── docs/                  ← Documentation
├── data/                  ← Data indices and datasets
├── cache/                 ← Cached embeddings and preprocessed data
├── checkpoints/           ← Model checkpoints
├── outputs/               ← Experiment outputs
├── experimental_results/  ← Published experiment results
├── setup.py              ← Package installation
├── pyproject.toml        ← Modern Python packaging
├── requirements.txt      ← Dependencies
└── README.md             ← Main documentation
```

---

## 🔧 Core Philosophy

### Clear Separation of Concerns

1. **`src/fmri2img/`** - **LIBRARY CODE ONLY**
   - Importable modules, classes, functions
   - NO `if __name__ == "__main__"` blocks
   - NO argparse CLI interfaces
   - Pure Python library that can be imported anywhere

2. **`scripts/`** - **EXECUTABLES ONLY**
   - Command-line interfaces
   - Thin wrappers that import from `src/fmri2img/`
   - Entry points for users and researchers
   - Organized by function (training, evaluation, build, etc.)

3. **`tests/`** - **TEST CODE ONLY**
   - Unit tests, integration tests
   - Test utilities and fixtures
   - Organized test suite for CI/CD

---

## 📚 Library Structure (`src/fmri2img/`)

**79 library modules organized by functionality:**

```
src/fmri2img/
│
├── 🧠 Core Models (models/)
│   ├── unified_model.py       ← Main fMRI→CLIP model
│   ├── encoders.py            ← Various encoder architectures
│   ├── ridge.py               ← Ridge regression baseline
│   ├── mlp.py                 ← MLP encoder
│   └── clip_adapter.py        ← CLIP dimension adapter
│
├── 📉 Loss Functions (losses/)
│   ├── gaussian_nll.py        ← Gaussian negative log-likelihood
│   ├── gaussian_nce.py        ← Gaussian Noise Contrastive Estimation
│   └── infonce_queue.py       ← InfoNCE with memory queue
│
├── 💾 Data Loading (data/)
│   ├── torch_dataset.py       ← PyTorch Dataset implementations
│   ├── loaders.py             ← Data loaders and batching
│   ├── preprocess.py          ← fMRI preprocessing (PCA, PCR)
│   ├── clip_cache.py          ← CLIP embedding cache interface
│   ├── nsd_index.py           ← NSD data indexing
│   └── reliability.py         ← Soft reliability weighting
│
├── 📊 Evaluation (eval/)
│   ├── embedding_metrics.py   ← Retrieval, RSA, CKA metrics
│   ├── probabilistic_metrics.py ← Calibration, uncertainty
│   ├── retrieval.py           ← R@K, MRR, nDCG
│   ├── embedding_eval.py      ← Stage 1 embedding evaluation
│   └── report_generation.py   ← Report generation utilities
│
├── 🎨 Generation (generation/)
│   ├── diffusion_utils.py     ← Diffusion model utilities
│   └── advanced_diffusion.py  ← Advanced generation methods
│
├── 🏋️ Training Utilities (training/)
│   ├── trainer.py             ← Training loops
│   ├── callbacks.py           ← Training callbacks
│   └── distributed.py         ← Distributed training utils
│
├── 🔄 Contrastive Learning (contrastive/)
│   └── queue.py               ← Memory queue (MoCo-style)
│
├── 🔮 Inference (inference/)
│   ├── pipeline.py            ← Inference pipeline
│   ├── allocator.py           ← Memory allocation
│   └── selector.py            ← Model selection
│
├── 🔬 Reliability (reliability/)
│   └── noise_ceiling.py       ← Noise ceiling estimation
│
├── 📁 I/O (io/)
│   ├── nsd_images.py          ← NSD image loading
│   ├── nsd_layout.py          ← NSD dataset layout
│   ├── s3.py                  ← HDF5/S3 data access
│   └── image_loader.py        ← Robust image loading
│
├── 📈 Statistics (stats/)
│   └── [statistical utilities]
│
└── 🛠️ Utils (utils/)
    └── [helper functions, config loading, etc.]
```

---

## 🚀 Scripts Structure (`scripts/`)

**67 executable scripts organized into 7 categories:**

### 1. **`training/`** (11 scripts)
Main training scripts and specialized trainers.

```bash
training/
├── train.py ⭐                  # Wrapper → train_unified.py
├── train_unified.py ⭐          # Main unified training (all architectures/losses)
├── train_ridge.py              # Ridge regression baseline
├── train_mlp.py                # MLP encoder training
├── train_clip_adapter.py       # CLIP adapter training
├── train_two_stage.py          # Two-stage pipeline
├── train_real_nsd.py           # Real NSD data training
├── train_clip_to_fmri.py       # Reverse direction (CLIP→fMRI)
└── README.md
```

### 2. **`evaluation/`** (10 scripts)
Evaluation metrics, benchmarks, and analysis.

```bash
evaluation/
├── evaluate.py ⭐               # Wrapper → evaluate_experiment.py
├── evaluate_experiment.py ⭐    # Main evaluation (all metrics)
├── eval_comprehensive.py       # Comprehensive evaluation suite
├── eval_stage1_embeddings.py   # Embedding-space evaluation
├── eval_stage1_probabilistic.py # Probabilistic evaluation
├── eval_shared1000_full.py     # Shared1000 benchmark
├── eval_reconstruction.py      # Reconstruction quality
├── eval_retrieval.py           # Retrieval metrics
├── summarize_shared1000.py     # Shared1000 summarization
└── README.md
```

### 3. **`build/`** (13 scripts)
Data preparation, caching, and preprocessing.

```bash
build/
├── build_target_clip_cache_robust.py ⭐  # Robust CLIP cache
├── fit_preprocessing.py ⭐               # Fit PCA/PCR preprocessing
├── build_full_index.py                  # Full NSD index
├── build_clip_cache.py                  # Standard CLIP cache
├── build_multilayer_clip_cache.py       # Multi-layer CLIP
├── build_text_clip_cache.py             # Text CLIP embeddings
├── build_embedding_preproc.py           # Embedding preprocessing
├── nsd_index_builder.py                 # Index builder
├── download_stimuli_hdf5.py             # Download stimuli
├── fast_preproc.py                      # Fast preprocessing
├── build_all_preprocessors.sh           # Batch build all
└── README.md
```

### 4. **`reconstruction/`** (5 scripts)
Image generation and reconstruction.

```bash
reconstruction/
├── reconstruct.py ⭐            # Wrapper → generate_images.py
├── generate_images.py ⭐        # Main image generation
├── decode_diffusion.py         # Diffusion decoding
├── decode_two_stage.py         # Two-stage reconstruction
├── reconstruct_nn.py           # Nearest neighbor reconstruction
└── README.md
```

### 5. **`analysis/`** (13 scripts)
Result analysis, comparison, and visualization.

```bash
analysis/
├── compare_experiments.py      # Compare multiple experiments
├── compare_evals.py            # Compare evaluations
├── compare_methods.py          # Method comparison
├── analyze_run.py              # Analyze single run
├── ablate_preproc_and_ridge.py # Preprocessing ablation
├── ablation_driver.py          # Ablation study driver
├── generate_comparison_gallery.py # Visual comparison
├── generate_report.py          # Report generation
├── plot_metrics.py             # Metric visualization
├── report_ablation.py          # Ablation reporting
├── summarize_reports.py        # Report summarization
├── build_paper_artifacts.py    # Paper figures/tables
└── README.md
```

### 6. **`orchestration/`** (8 scripts)
Experiment orchestration and monitoring.

```bash
orchestration/
├── run_all_experiments.sh ⭐   # Run ablation study
├── run_experiment.py           # Single experiment
├── run_full_pipeline.py        # End-to-end pipeline
├── run_reconstruct_and_eval.py # Reconstruction + eval
├── run_stage34_recon_eval.py   # Multi-stage recon
├── monitor_experiments.sh      # Monitor progress
├── check_experiment_status.sh  # Status check
└── README.md
```

### 7. **`utils/`** (17 scripts)
System utilities, validation, and helpers.

```bash
utils/
├── preflight.py ⭐              # Quick system check
├── doctor.py ⭐                 # Comprehensive diagnostics
├── smoke.py                    # Smoke test
├── verify_dataset.py ⭐         # Dataset verification
├── validate_checkpoints.py     # Checkpoint validation
├── validate_index.py           # Index validation
├── check_index_headers.py      # Header validation
├── fetch_models.py ⭐           # Download HuggingFace models
├── archive_experiment.py       # Archive results
├── write_run_manifest.py       # Write manifests
├── quick_status.py             # Quick status check
├── download_sd_model.py        # Download Stable Diffusion
├── train_smoke.py              # Training smoke test
└── README.md
```

---

## 🧪 Tests Structure (`tests/`)

**29 test files:**

```
tests/
├── conftest.py                          # Pytest configuration
├── test_*.py                            # Integration tests (22 files)
└── unit/                                # Unit tests
    ├── test_compare_evals.py
    ├── test_preprocess.py
    ├── test_reliability.py
    ├── nsd_index_reader.py
    └── nsd_sanity_check.py
```

---

## 🎯 Usage Patterns

### Pattern 1: Using Library Code (Programmatic)

```python
# Import from library
from fmri2img.models import UnifiedModel
from fmri2img.data import NSDDataset, preprocess_fmri
from fmri2img.eval import compute_retrieval_metrics

# Use in your code
model = UnifiedModel(architecture='gaussian', loss='gaussian_nce')
dataset = NSDDataset(subject='subj01')
metrics = compute_retrieval_metrics(embeddings, targets)
```

### Pattern 2: Using CLI Scripts (Command-line)

```bash
# Train model
python scripts/training/train.py --config configs/experiments/exp0.yaml

# Evaluate
python scripts/evaluation/evaluate.py --checkpoint outputs/exp0/best.pt

# Build data cache
python scripts/build/build_target_clip_cache_robust.py

# Run full pipeline
python scripts/orchestration/run_all_experiments.sh 0
```

### Pattern 3: Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_losses.py

# Run unit tests only
pytest tests/unit/
```

---

## 📦 Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/toniIepure25/FMRI2images.git
cd "Bachelor V2"

# 2. Run setup (idempotent, safe to re-run)
./setup.sh

# 3. Verify installation
python scripts/utils/preflight.py

# 4. Run comprehensive health check
python scripts/utils/doctor.py
```

---

## 🔄 Development Workflow

### Adding New Functionality

1. **Add library code** to `src/fmri2img/[module]/`
   - Write classes, functions
   - Add docstrings and type hints
   - NO CLI code

2. **Add CLI script** to `scripts/[category]/`
   - Import from `src/fmri2img/`
   - Handle argparse, logging, errors
   - Keep thin and focused

3. **Add tests** to `tests/`
   - Unit tests for library code
   - Integration tests for workflows

4. **Update documentation**
   - Update relevant README files
   - Add to this ARCHITECTURE.md if needed

### Code Quality Standards

- ✅ Type hints on all functions
- ✅ Docstrings (Google style)
- ✅ Tests for new functionality
- ✅ Logging instead of print statements
- ✅ Config files over hard-coded values
- ✅ Error handling and validation

---

## 📊 Metrics

- **Library modules**: 79 (pure Python, importable)
- **CLI scripts**: 67 (organized into 7 categories)
- **Test files**: 29 (unit + integration)
- **Configuration files**: 20+ YAML configs
- **Documentation files**: 15+ markdown docs

---

## 🏆 Best Practices Followed

### ✅ Professional Python Structure
- Clear separation: library vs scripts vs tests
- No CLI code in library modules
- Thin CLI wrappers that import from library

### ✅ Research Reproducibility
- YAML configs for all experiments
- Comprehensive logging and checkpointing
- Seed management for reproducibility
- Version tracking in manifests

### ✅ Code Organization
- Logical directory structure by functionality
- README files in each major directory
- Consistent naming conventions
- Clear dependency management

### ✅ Quality Assurance
- Comprehensive test suite
- Preflight checks before experiments
- Health diagnostics (doctor.py)
- Validation utilities

### ✅ Documentation
- Architecture overview (this file)
- Setup instructions (README.md)
- Per-directory READMEs
- Inline code documentation

---

## 🔗 Related Documentation

- [README.md](README.md) - Main project documentation
- [scripts/README.md](scripts/README.md) - Scripts overview
- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Command cheat sheet
- [docs/guides/RUNNING_EXPERIMENTS.md](docs/guides/RUNNING_EXPERIMENTS.md) - Experiment guide

---

## 🔮 Future Improvements

While the current structure is professional and functional, some scripts could be further refined:

### Script Complexity Refinement (Optional)

Some complex scripts (1000+ lines) contain logic that could be extracted to library modules:

- `train_two_stage.py` (1876 lines) → Extract to `src/fmri2img/training/two_stage.py`
- `run_reconstruct_and_eval.py` (1467 lines) → Extract to `src/fmri2img/pipelines/recon_eval.py`
- `decode_diffusion.py` (1375 lines) → Extract to `src/fmri2img/generation/decoder.py`
- `run_full_pipeline.py` (1114 lines) → Extract to `src/fmri2img/pipelines/full_pipeline.py`
- `build_target_clip_cache.py` (946 lines) → Extract to `src/fmri2img/data/cache_builder.py`

**Recommendation**: Refactor incrementally when modifying these scripts. Current structure is functional and follows best practices.

---

**Last Updated**: February 20, 2026  
**Maintainer**: Research Team  
**Status**: Production-ready, research-grade structure  
**Next Refactoring**: Incremental (as-needed basis)
