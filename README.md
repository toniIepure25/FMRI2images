# Brain-to-Image: Neural Decoding of Visual Perception from fMRI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-53%2F53%20passing-brightgreen.svg)]()

> **A comprehensive framework for reconstructing visual stimuli from brain activity using the Natural Scenes Dataset (NSD), CLIP embeddings, and Stable Diffusion.**

This repository implements state-of-the-art approaches for decoding visual information from fMRI signals, featuring multiple encoder architectures, robust preprocessing pipelines, and comprehensive evaluation metrics. The system achieves high-quality image reconstruction by bridging the gap between neural representations and generative models.

---

## 📚 Documentation

### Quick Navigation

| Category | Document | Description |
|----------|----------|-------------|
| **🚀 Getting Started** | [SETUP.md](docs/guides/SETUP.md) | Complete setup for any environment |
| **🎯 Running Experiments** | [RUNNING_EXPERIMENTS.md](docs/guides/RUNNING_EXPERIMENTS.md) | Training and evaluation guide |
| **📊 Implementation Status** | [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) | What's ready to use |
| **🔧 Troubleshooting** | [TROUBLESHOOTING.md](docs/guides/TROUBLESHOOTING.md) | Common issues and solutions |
| **📖 Documentation Index** | [docs/README.md](docs/README.md) | All guides and technical docs |
| **📝 Paper/Thesis** | [docs/paper/README.md](docs/paper/README.md) | Academic documentation |
| **⚡ Quick Reference** | [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Command cheat sheet |

---

## 🚀 Quick Start

### Automated Setup (5 Minutes)

```bash
# Clone repository
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images
git checkout probabilistic-distribution

# Run automated setup (30-45 minutes)
chmod +x setup_enhanced.sh
./setup_enhanced.sh
```

**What it does:**
- ✅ Checks system requirements (Python, GPU, disk space)
- ✅ Creates Python environment and installs dependencies
- ✅ Downloads required data (~17GB)
- ✅ Builds data indices and CLIP embeddings cache
- ✅ Detects GPU memory and recommends batch size
- ✅ Runs comprehensive health checks

### First Experiment (10 Minutes)

```bash
# Activate environment
source activate_env.sh  # or: conda activate fmri2img

# Train baseline model
python scripts/train.py --config configs/experiments/exp0_baseline.yaml

# Monitor with TensorBoard
tensorboard --logdir outputs/
```

**📖 Full details:** See [docs/guides/SETUP.md](docs/guides/SETUP.md) for comprehensive setup instructions.

---

## 🎯 Key Features

### Novel Contributions (Research-Ready)

1. **✅ Soft Reliability Weighting** - Continuous voxel importance instead of binary thresholding
2. **✅ InfoNCE Contrastive Loss** - Direct ranking optimization for improved retrieval  
3. **✅ MC Dropout Uncertainty** - Bayesian confidence estimation with calibration analysis

**Status:** All implementations tested (53/53 tests passing ✅) and documented. Ready for experiments and thesis writing.

### System Features

- **🧠 Multiple Encoder Architectures**: Ridge regression, MLP, Unified models (deterministic + Gaussian)
- **🔬 Robust Preprocessing**: Z-score normalization, PCA/PCR, soft reliability weighting
- **📊 Comprehensive Evaluation**: Standard metrics (retrieval, 2AFC, RSA) + Bayesian metrics (calibration, conformal prediction)
- **🎨 High-Quality Reconstruction**: Integration with Stable Diffusion for photorealistic outputs
- **⚡ Production-Ready**: Professional logging, configuration management, checkpoint handling
- **✅ Extensively Tested**: 53/53 automated tests with comprehensive coverage

### Research Context

This work builds upon recent advances in neural decoding and generative modeling:
- Allen et al. (2022) - [Natural Scenes Dataset (NSD)](https://www.nature.com/articles/s41593-021-00962-x)
- Radford et al. (2021) - [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- Rombach et al. (2022) - [Stable Diffusion: Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

---

## 🏗️ Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  fMRI Data  │────▶│Preprocessing │────▶│   Encoder   │────▶│CLIP Embedding│
│  (15k-70k   │     │ (Z-score +   │     │  (Ridge/    │     │   (512/768/  │
│   voxels)   │     │  PCA/PCR)    │     │   MLP/2S)   │     │   1024-D)    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│Reconstructed│◀────│   Diffusion  │◀────│   Stable    │◀────│Predicted CLIP│
│   Image     │     │   Sampling   │     │  Diffusion  │     │  Embedding   │
│  (512×512)  │     │ (50 steps)   │     │  (v1.5/2.1) │     └──────────────┘
└─────────────┘     └──────────────┘     └─────────────┘
```

**See:** [docs/architecture/PIPELINE_ARCHITECTURE.md](docs/architecture/PIPELINE_ARCHITECTURE.md) for detailed architecture documentation.

---

## 📦 Experiment Configurations

### Ablation Study (7 Experiments)

| Experiment | Description | Novel Feature | Config File |
|------------|-------------|---------------|-------------|
| **EXP0** | Baseline | None | [exp0_baseline.yaml](configs/experiments/exp0_baseline.yaml) |
| **EXP1** | + Preprocessing | Center + PCR (k=8) | [exp1_preproc.yaml](configs/experiments/exp1_preproc.yaml) |
| **EXP2** | + Memory queue | MoCo-style (Q=8192) | [exp2_queue.yaml](configs/experiments/exp2_queue.yaml) |
| **EXP3** | + Gaussian NLL | Heteroscedastic regression | [exp3_gaussian_nll.yaml](configs/experiments/exp3_gaussian_nll.yaml) |
| **EXP4** | + Gaussian-NCE ⭐ | Distribution-aware loss | [exp4_gaussian_nce.yaml](configs/experiments/exp4_gaussian_nce.yaml) |
| **EXP5** | + KL annealing | Free-bits regularization | [exp5_kl_anneal.yaml](configs/experiments/exp5_kl_anneal.yaml) |
| **EXP6** | Ablation | Whitening vs PCR | [exp6_whiten.yaml](configs/experiments/exp6_whiten.yaml) |

⭐ = Novel contribution

### Running the Full Ablation Study

```bash
# Build preprocessors (once, 1-2 hours)
bash scripts/build_all_preprocessors.sh

# Run all experiments (3-7 days on A100)
bash scripts/run_all_experiments.sh 0

# Results saved to: experimental_results/
```

**See:** [docs/guides/RUNNING_EXPERIMENTS.md](docs/guides/RUNNING_EXPERIMENTS.md) for comprehensive usage guide.

---

## 📊 Evaluation Metrics

### Standard Metrics (All Experiments)

- **Retrieval@K**: Top-1, Top-5, Top-10 accuracy
- **Ranking**: Mean rank, Median rank, MRR
- **2AFC**: Two-way identification with bootstrap CI
- **Separability**: ROC AUC, Cohen's d
- **RSA**: Representational Similarity Analysis
- **Collapse diagnostics**: Per-dim std, pairwise similarity

### Bayesian Metrics (EXP3-6 Only) ⭐

- **Proper scoring**: Gaussian NLL, Energy Score
- **Bayesian retrieval**: Distribution-aware ranking (log q(c|x))
- **Probabilistic 2AFC**: Uncertainty propagation via MC sampling
- **Calibration analysis**: Reliability diagrams, Mahalanobis + Chi-square
- **Risk-coverage curves**: Selective prediction with AURC metric
- **Conformal prediction**: Distribution-free uncertainty quantification

**See:** [docs/guides/EVALUATION_SUITE_GUIDE.md](docs/guides/EVALUATION_SUITE_GUIDE.md) for detailed metrics documentation.

---

## 🧪 Testing

All tests passing: **53/53 ✅**

```bash
# Run full test suite
pytest tests/ -v

# Quick smoke test
pytest tests/test_smoke.py -v

# Test with real data
python test_real_data.py
```

**Test Coverage:**
- ✅ All encoder architectures (Ridge, MLP, Unified, Gaussian)
- ✅ All loss functions (InfoNCE, Gaussian NLL, Gaussian-NCE, KL)
- ✅ All preprocessing pipelines (PCA/PCR, whitening, soft reliability)
- ✅ Data loading and CLIP caching
- ✅ Training loops and checkpointing
- ✅ Evaluation metrics (embedding + probabilistic)

---

## 📁 Repository Structure

```
FMRI2images/
├── configs/                    # Configuration files
│   ├── base.yaml              # Base configuration
│   ├── experiments/           # Experiment configs (EXP0-6)
│   └── ...
├── src/fmri2img/              # Core library
│   ├── models/                # Encoder architectures
│   ├── losses/                # Loss functions
│   ├── preprocessing/         # Preprocessing pipelines
│   ├── eval/                  # Evaluation metrics
│   ├── training/              # Training infrastructure
│   └── data/                  # Data loading
├── scripts/                   # Executable scripts
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation script
│   └── ...
├── tests/                     # Test suite (53 tests)
├── docs/                      # Documentation
│   ├── guides/               # User guides
│   ├── architecture/         # Architecture docs
│   ├── technical/            # Technical docs
│   └── paper/                # Paper/thesis documentation
├── cache/                     # Data cache (created by setup)
├── checkpoints/              # Model checkpoints
├── outputs/                  # Training outputs
├── experimental_results/     # Experiment results
├── README.md                 # This file
├── environment.yml           # Conda environment
└── setup_enhanced.sh         # Automated setup script
```

---

## 🔧 Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **GPU** | 6GB VRAM | 20GB+ VRAM (A100) |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 50GB free | 200GB+ free |
| **OS** | Linux/macOS | Linux (Ubuntu 20.04+) |

### Key Dependencies

- PyTorch 2.0+
- transformers (HuggingFace)
- diffusers (Stable Diffusion)
- scikit-learn (preprocessing)
- h5py (data loading)
- pandas, numpy, scipy

**Full environment:** See [environment.yml](environment.yml)

---

## 💡 Usage Examples

### Train Single Experiment

```bash
python scripts/train.py \
  --config configs/experiments/exp4_gaussian_nce.yaml \
  --output_dir outputs/exp4 \
  --device cuda
```

### Evaluate Model

```bash
python scripts/evaluate.py \
  --checkpoint outputs/exp4/best.pt \
  --test_split test \
  --output_dir experimental_results/exp4
```

### Generate Reconstructions

```bash
python scripts/reconstruct.py \
  --checkpoint outputs/exp4/best.pt \
  --n_images 100 \
  --diffusion_model stabilityai/stable-diffusion-2-1 \
  --output_dir outputs/reconstructions
```

---

## 🎓 For Thesis/Paper Writers

### Ready to Use

- **✅ Complete ablation study design** (7 experiments)
- **✅ Comprehensive evaluation protocol** with statistical significance tests
- **✅ Novel contributions** fully implemented and tested
- **✅ Paper-grade documentation** with reproducibility tracking
- **✅ Visualization pipeline** for figures and tables

### Documentation for Writing

- **Paper structure:** [docs/paper/README.md](docs/paper/README.md)
- **Methodology:** [docs/paper/method.md](docs/paper/method.md)
- **Experiments design:** [docs/paper/experiments.md](docs/paper/experiments.md)
- **Evaluation protocol:** [docs/paper/evaluation_protocol.md](docs/paper/evaluation_protocol.md)
- **Results template:** [docs/paper/results.md](docs/paper/results.md)

**See:** [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for complete status overview.

---

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run tests before committing
pytest tests/ -v

# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{fmri2img2025,
  title={Brain-to-Image: Neural Decoding of Visual Perception from fMRI},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/toniIepure25/FMRI2images}},
}
```

**Related datasets and methods:**
- Allen et al. (2022) - Natural Scenes Dataset (NSD)
- Radford et al. (2021) - CLIP
- Rombach et al. (2022) - Stable Diffusion

See [REFERENCES.bib](REFERENCES.bib) for complete citations.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Natural Scenes Dataset**: Emily Allen and team at the University of Minnesota
- **OpenAI CLIP**: Alec Radford and team at OpenAI
- **Stable Diffusion**: Robin Rombach and team at Stability AI
- **HuggingFace**: For the excellent `diffusers` and `transformers` libraries

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/toniIepure25/FMRI2images/issues)
- **Documentation**: [docs/](docs/)
- **Quick Reference**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

---

<div align="center">

**[Documentation](docs/) • [Setup Guide](docs/guides/SETUP.md) • [Running Experiments](docs/guides/RUNNING_EXPERIMENTS.md) • [Paper Docs](docs/paper/README.md)**

Made with ❤️ for neuroscience and AI research

</div>
