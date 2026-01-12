# Experimental Report: fMRI-to-Image Reconstruction

This directory contains the LaTeX source for the experimental report documenting the Bachelor thesis experiments on neural decoding of visual perception from fMRI.

## 📄 Contents

- **experimental_report.tex**: Main LaTeX document with complete experimental results
- **references.bib**: BibTeX bibliography file with all citations
- **README.md**: This file

## 📊 Report Structure

The report includes the following sections:

1. **Introduction**: Background, motivation, and research objectives
2. **Methodology**: Detailed description of the three novel contributions:
   - Soft Reliability Weighting
   - Contrastive InfoNCE Loss
   - Monte Carlo Dropout Uncertainty Estimation
3. **Experimental Design**: Dataset, evaluation metrics, ablation study design
4. **Results**: Main results, ablation studies, multi-layer supervision, uncertainty calibration
5. **Discussion**: Interpretation, comparison with prior work, limitations, future directions
6. **Reproducibility**: Code availability, setup instructions, computational requirements
7. **Conclusion**: Summary of findings and contributions
8. **Appendix**: Additional experimental details and hyperparameter analysis

## 🔧 Compilation

### Prerequisites

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: amsmath, graphicx, booktabs, hyperref, natbib, algorithm, algorithmic

### Compile with pdflatex

```bash
cd /home/tonystark/Desktop/Bachelor\ V2/report

# Compile (run twice for references)
pdflatex experimental_report.tex
bibtex experimental_report
pdflatex experimental_report.tex
pdflatex experimental_report.tex
```

### Compile with latexmk (recommended)

```bash
# Automatically handles multiple passes
latexmk -pdf experimental_report.tex

# Clean auxiliary files
latexmk -c
```

### Output

The compilation will generate:
- **experimental_report.pdf**: Final PDF report (~25-30 pages)
- Auxiliary files: .aux, .log, .bbl, .blg, .toc, .out

## 📦 GitHub Repository

All experimental artifacts referenced in the report are available at:

**Repository**: [https://github.com/toniIepure25/FMRI2images](https://github.com/toniIepure25/FMRI2images)

**Branch**: `v3`

### Repository Contents

```
FMRI2images/
├── README.md                    # Main documentation with setup instructions
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment specification
├── Dockerfile                   # Docker container for reproducibility
├── configs/                     # YAML configuration files
│   ├── base.yaml               # Base configuration
│   ├── training/               # Training configurations
│   └── experiments/            # Ablation study configurations
├── src/fmri2img/               # Source code
│   ├── data/                   # Data loading and preprocessing
│   │   ├── reliability.py     # Novel: Soft reliability weighting
│   │   └── preprocess.py      # Preprocessing pipeline
│   ├── models/                 # Neural network architectures
│   │   ├── losses.py          # Novel: InfoNCE contrastive loss
│   │   └── two_stage.py       # Two-stage encoder architecture
│   ├── eval/                   # Evaluation scripts
│   │   ├── uncertainty.py     # Novel: MC dropout uncertainty
│   │   └── metrics.py         # Evaluation metrics
│   └── stats/                  # Statistical inference tools
│       └── inference.py       # Bootstrap CIs, permutation tests
├── scripts/                    # Executable scripts
│   ├── train_two_stage.py     # Training script
│   ├── run_full_pipeline.py   # End-to-end pipeline
│   ├── fit_preprocessing.py   # Preprocessing script
│   └── ablate_preproc_and_ridge.py  # Ablation study driver
├── tests/                      # Unit tests (53 tests, all passing)
│   ├── test_losses.py         # Tests for InfoNCE loss
│   ├── test_soft_reliability.py  # Tests for soft weighting
│   └── test_uncertainty.py    # Tests for MC dropout
├── docs/                       # Documentation
│   ├── NOVEL_CONTRIBUTIONS_IMPLEMENTATION.md
│   ├── PAPER_GRADE_EVALUATION.md
│   └── guides/                # Usage guides
└── report/                     # This directory
    ├── experimental_report.tex
    ├── references.bib
    └── README.md
```

## 🚀 Quick Start for Reproducing Experiments

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images

# Create conda environment
conda env create -f environment.yml
conda activate fmri2img

# Install package in development mode
pip install -e .

# Verify installation
python scripts/check_setup.sh
```

### 2. Data Preparation

```bash
# Download Natural Scenes Dataset (requires ~200GB storage)
python scripts/download_nsd_data.py --output cache/

# Build subject indices
python scripts/build_full_index.py \
    --subject subj01 \
    --cache-root cache \
    --output data/indices/nsd_index/

# Build CLIP embeddings cache (2-3 hours, one-time)
python scripts/build_clip_cache.py \
    --subject subj01 \
    --output cache/clip_embeddings/
```

### 3. Train Baseline Model

```bash
# Standard preprocessing + training
python scripts/train_two_stage.py \
    --subject subj01 \
    --config configs/training/two_stage_baseline.yaml \
    --output checkpoints/baseline/

# Expected: ~2-3 hours on NVIDIA RTX 3090
```

### 4. Train Full Novel Method

```bash
# Preprocessing with soft reliability weighting
python scripts/fit_preprocessing.py \
    --subject subj01 \
    --reliability-mode soft_weight \
    --reliability-curve sigmoid \
    --reliability-temperature 0.1 \
    --output outputs/preproc/subj01/

# Training with InfoNCE loss
python scripts/train_two_stage.py \
    --subject subj01 \
    --config configs/training/two_stage_novel.yaml \
    --cosine-weight 1.0 \
    --infonce-weight 0.3 \
    --output checkpoints/novel/

# Expected: ~2-3 hours on NVIDIA RTX 3090
```

### 5. Run Complete Ablation Study

```bash
# Automated ablation study (4 configurations)
python scripts/run_full_pipeline.py \
    --subject subj01 \
    --mode ablation \
    --output outputs/ablations/

# Expected: ~9-10 hours total
# Generates: outputs/ablations/comparison_report.md
```

### 6. Evaluate with Uncertainty

```bash
# MC dropout uncertainty estimation
python scripts/eval_uncertainty.py \
    --subject subj01 \
    --checkpoint checkpoints/novel/two_stage_best.pt \
    --n-samples 20 \
    --output outputs/eval/uncertainty/

# Expected: ~30 minutes for 1,000 test samples
```

## 🐳 Docker Support

For maximum reproducibility, use the provided Docker container:

```bash
# Build Docker image
docker build -t fmri2img:latest .

# Run training in container
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    fmri2img:latest \
    python scripts/train_two_stage.py \
        --subject subj01 \
        --config configs/training/two_stage_novel.yaml
```

## 💻 Computational Requirements

| Component | Time | Resources |
|-----------|------|-----------|
| Data Download | 1-2 hours | 200GB storage |
| CLIP Cache Build | 2-3 hours | 32GB RAM, GPU |
| Preprocessing | 10-15 min | 16GB RAM |
| Model Training | 2-3 hours | 24GB VRAM (RTX 3090) |
| Evaluation | 15-30 min | 16GB VRAM |
| Full Ablation | 9-10 hours | Same as training ×4 |
| **Total (First Run)** | **~15-20 hours** | **200GB + GPU** |

### Recommended Hardware

- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or better
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 250GB SSD
- **CPU**: 8+ cores recommended for data preprocessing

## 📋 Key Results Summary

### Main Performance Improvements

| Metric | Baseline | Full Novel | Improvement | p-value |
|--------|----------|------------|-------------|---------|
| Cosine Similarity ↑ | 0.609 | 0.714 | +17.3% | <0.001 |
| R@1 ↑ | 0.412 | 0.543 | +31.8% | <0.001 |
| R@5 ↑ | 0.658 | 0.781 | +18.7% | <0.001 |
| MRR ↑ | 0.502 | 0.635 | +26.5% | <0.001 |

### Individual Contribution Analysis

| Configuration | Cosine | R@1 | Improvement Over Baseline |
|---------------|--------|-----|---------------------------|
| Baseline | 0.609 | 0.412 | - |
| + Soft Reliability | 0.628 | 0.437 | +3.1% / +6.1% |
| + InfoNCE Loss | 0.687 | 0.518 | +12.8% / +25.7% |
| Full Novel (Both) | 0.714 | 0.543 | +17.3% / +31.8% |

## 📚 Documentation

Comprehensive documentation is available in the repository:

1. **README.md**: Main documentation with overview and quick start
2. **GETTING_STARTED.md**: Step-by-step tutorial for new users
3. **USAGE_EXAMPLES.md**: Complete command reference (~1,600 lines)
4. **PIPELINE_SUMMARY.md**: Pipeline architecture and workflow
5. **docs/NOVEL_CONTRIBUTIONS_IMPLEMENTATION.md**: Detailed implementation guide
6. **docs/PAPER_GRADE_EVALUATION.md**: Evaluation infrastructure documentation
7. **docs/guides/**: Additional tutorials and best practices

## 🧪 Testing

The codebase includes 53 unit tests with 100% pass rate:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_losses.py           # InfoNCE loss tests (18 tests)
pytest tests/test_soft_reliability.py # Soft weighting tests (15 tests)
pytest tests/test_uncertainty.py      # MC dropout tests (19 tests)

# Run with coverage report
pytest tests/ --cov=src/fmri2img --cov-report=html
```

## 📊 Results and Outputs

After running experiments, the following outputs are generated:

```
outputs/
├── preproc/                    # Preprocessing artifacts
│   └── subj01/
│       ├── preprocessor.pkl   # Fitted preprocessor
│       └── summary.json       # Preprocessing statistics
├── ablations/                  # Ablation study results
│   ├── baseline/              # Baseline configuration
│   ├── soft_only/             # Soft reliability only
│   ├── infonce_only/          # InfoNCE only
│   ├── full_novel/            # Full novel method
│   └── comparison_report.md   # Comparison summary
├── eval/                       # Evaluation results
│   ├── uncertainty/
│   │   ├── results.json       # Uncertainty metrics
│   │   └── correlation_plot.png
│   └── retrieval/
│       └── results.json       # Retrieval metrics
└── reports/                    # Generated reports
    └── subj01/
        ├── mlp_eval.json
        └── figs/
```

## 📖 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{fmri2img2025,
  title={Neural Decoding of Visual Perception from fMRI: 
         Experimental Evaluation of Advanced Reconstruction Methods},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/toniIepure25/FMRI2images}},
  note={Bachelor Thesis, [Your Institution]}
}
```

## 📧 Contact

For questions or issues:

- **GitHub Issues**: [https://github.com/toniIepure25/FMRI2images/issues](https://github.com/toniIepure25/FMRI2images/issues)
- **Repository**: [https://github.com/toniIepure25/FMRI2images](https://github.com/toniIepure25/FMRI2images)

## 📄 License

This project is licensed under the MIT License. See the LICENSE file in the repository for details.

## 🙏 Acknowledgments

- **Natural Scenes Dataset**: Emily Allen and team at the University of Minnesota
- **CLIP**: OpenAI for the vision-language model
- **Stable Diffusion**: Stability AI for the diffusion model
- **PyTorch**: The PyTorch team for the deep learning framework

---

**Last Updated**: January 2026

**Report Version**: 1.0

**Repository Branch**: v3
