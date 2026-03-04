# Brain-to-Image: Neural Decoding of Visual Perception from fMRI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research framework for reconstructing perceived images from 7T fMRI brain activity using the Natural Scenes Dataset (NSD). The system maps voxel activations to CLIP ViT-L/14 embeddings (768-D) via probabilistic von Mises-Fisher decoders, then generates images through Stable Diffusion 2.1.

This is a bachelor thesis project targeting publication at NeurIPS/MICCAI venues. The codebase implements a progressive ablation study (9 config versions, 4 novel experiments) comparing deterministic, Gaussian, and vMF decoding approaches with novel contributions in probabilistic hyperspherical decoding, ROI-aware brain encoding, and uncertainty-driven image reconstruction.

---

## Project Status (March 2026)

| Version | Experiments | Best R@1 (subj01) | Status |
|---------|-------------|-------------------|--------|
| B-series (baselines, v4) | B0v4, B1v4 | ~22% | Complete |
| N-series v5 | N1-N4 | 39.9% | Complete |
| N-series v7 | N1-N4 | **49%** | Complete |
| N-series v8 | N3-N4 | **50.8%** | Complete |
| **N-series v9 (current)** | **N1-N4** | **Pending** | **Ready to run** |

**V9 target:** 65-70% R@1 via anti-overfitting regularization, projection head separation, CSLS retrieval correction, and MC-Dropout test-time augmentation. Runs on **NVIDIA H100 80GB** with bf16 and effective batch 1024.

**SOTA reference:** MindEye2 achieves 93.0% R@1 on the same dataset (Scotti et al., 2024).

---

## Novel Contributions

| ID | Contribution | Description |
|----|-------------|-------------|
| N1 | **vMF-NCE Loss** | Bessel-free contrastive loss on the unit hypersphere. Softplus kappa parameterization (v7+) with R-Drop consistency regularization (v9). |
| N2 | **ROI Transformer Encoder** | Brain-topology-aware tokenization: 17 ROI tokens + multi-subject joint training (v7+) with Stochastic Depth (v9). |
| N3 | **Directional Consensus Fusion** | Per-ROI vMF expert heads fused via spherical weighted averaging. Dual uncertainty: kappa (concentration) + delta (ROI disagreement). |
| N4 | **Full System + DUA-CFG** | All innovations + Self-Paced Curriculum Learning (SPCL) + Decomposed Uncertainty-Aware CFG for diffusion guidance. |

**V9 architectural innovations:** ContrastiveProjectionHead (separates retrieval from contrastive representations), R-Drop for vMF distributions (symmetric KL between dropout masks), CSLS hubness correction, MC-Dropout TTA with kappa-weighted averaging.

---

## Pipeline Architecture

```
fMRI Voxels        Encoder              Decoder            Reconstruction
(~15k voxels)                           (768-D output)

+-----------+    +----------------+    +-------------+    +----------------+
| ROI-masked|    | MLP (N1)       |    | vMF         |    |                |
| beta maps |--->| or Multi-Subj  |--->| (mu, kappa) |--->| CLIP ViT-L/14  |
| float32   |    | ROI Transformer|    | or vMF-DCF  |    | embedding      |
|           |    | (N2/N3/N4)     |    | (mu,k,delta)|    | (768-D, L2)    |
+-----------+    +----------------+    +------+------+    +-------+--------+
                                              |                   |
                                    Projection Head          CSLS + TTA
                                    (contrastive)           (retrieval)
```

---

## Version Progression

| Version | Key Innovation | Best R@1 |
|---------|---------------|----------|
| v4 | nsdId fix, per-session z-scoring, MixCo, SoftCLIP | 36% |
| v5 | tau=1.0 (kappa collapse fix), EMA, noise augmentation | 39.9% |
| v6 | Delta-SPCL, vMF-SoftCLIP, Slerp MixCo | ~40% |
| v7 | Softplus kappa, multi-layer CLIP targets, multi-subject training | **49%** |
| v8 | Hierarchical CLIP alignment, CKA loss | **50.8%** |
| **v9** | **DropPath, projection head, R-Drop, CSLS, MC-TTA, bf16, batch 1024** | **Pending** |

---

## Running Experiments

### Prerequisites

- NVIDIA H100 80GB (or A100 40GB with reduced batch_size)
- Python 3.10+, PyTorch 2.0+ with CUDA
- NSD dataset pre-downloaded to `/home/jovyan/work/data/nsd/`

### Quick Start

```bash
# Install and configure
pip install -e ".[train,diffusion]"
cp .env.jupyterhub .env
set -a && source .env && set +a

# Pre-extract features (once per subject)
make preextract SUBJECT=subj01

# Run V9 ablation (4 experiments)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N9 > ablation_v9.log 2>&1 &
tail -f ablation_v9.log

# Aggregate results
python3 scripts/evaluation/aggregate_ablation.py \
    --results-dir experimental_results --subjects subj01
```

### Single Experiment

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1v9_vmf_nce.yaml --subject subj01
```

---

## Repository Structure

```
FMRI2images/
├── src/fmri2img/                  # Core library (pip install -e .)
│   ├── models/                    # UnifiedModel, ROITransformer, MultiSubjectEncoder,
│   │                              #   VonMisesFisherDecoder, ROI-DCF, ProjectionHead
│   ├── losses/                    # vMF-NCE, R-Drop, SoftCLIP, MixCo, CKA, HierarchicalCLIP
│   ├── eval/                      # Retrieval (+ CSLS), embedding, kappa calibration
│   ├── data/                      # NSD loading, multi-subject dataset, ROI index
│   ├── contrastive/               # Memory queue (16384 entries)
│   ├── inference/                 # DUA-CFG, vMF mixture sampling
│   └── ...
├── scripts/
│   ├── training/                  # train_unified.py, run_ablation_ladder.sh
│   ├── build/                     # build_clip_cache.py, build_multilayer_clip_cache.py,
│   │                              #   preextract_fmri.py
│   ├── evaluation/                # aggregate_ablation.py, eval_reconstruction.py
│   └── reconstruction/            # decode_diffusion.py
├── configs/experiments/           # N1v9-N4v9 (current), plus all prior versions
├── tests/                         # 27+ test files (pytest)
├── docs/
│   ├── EXPERIMENT_CONTEXT.md      # Comprehensive technical context (v1-v9)
│   ├── IMPLEMENTATION_STATUS.md   # Module readiness overview
│   └── guides/                    # RUNNING_EXPERIMENTS, VMF_UACFG_GUIDE
├── experimental_results/          # Per-experiment outputs
├── Makefile                       # Build targets
└── pyproject.toml                 # Package definition
```

---

## Evaluation Metrics

### Retrieval (Primary)

| Metric | Description |
|--------|-------------|
| R@1, R@5, R@10 | Top-K retrieval accuracy on validation gallery (~986 images) |
| CSLS R@1/5/10 | Cross-domain Similarity Local Scaling corrected retrieval (V9) |
| MRR | Mean reciprocal rank |

### Reconstruction (Post-Diffusion)

| Metric | Description | Direction |
|--------|-------------|-----------|
| PixCorr | Pixel-level correlation | Higher is better |
| SSIM | Structural similarity | Higher is better |
| AlexNet(2), AlexNet(5) | Perceptual similarity | Higher is better |

### SOTA Benchmarks

| Method | PixCorr | SSIM | Alex(2) | Alex(5) |
|--------|---------|------|---------|---------|
| MindEye (Scotti et al., 2024) | 0.309 | 0.323 | 0.947 | 0.978 |
| MindEye2 (Scotti et al., 2024) | 0.320 | 0.341 | 0.960 | 0.983 |
| Brain Diffuser (Ozcelik & VanRullen, 2023) | 0.254 | 0.356 | 0.942 | 0.962 |

---

## Hardware

| Component | Recommended |
|-----------|-------------|
| GPU | NVIDIA H100 80GB HBM3 |
| Python | 3.13 (conda) |
| System RAM | 100 GB |
| Storage | 500 GB+ |
| CUDA | 12.8 |

---

## Key References

- Allen, K. et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25, 116-126.
- Scotti, P. et al. (2024). MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data. *ICML 2024*.
- Ozcelik, F. and VanRullen, R. (2023). Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion. *arXiv:2303.05334*.
- Banerjee, A. et al. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *JMLR*, 6, 1345-1382.
- Davidson, T. et al. (2018). Hyperspherical variational auto-encoders. *UAI 2018*.

---

## Citation

```bibtex
@misc{fmri2img2026,
  title={Brain-to-Image: Neural Decoding of Visual Perception from fMRI
         via Probabilistic Hyperspherical Embeddings},
  author={Iepure, Toni},
  year={2026},
  howpublished={\url{https://github.com/toniIepure25/FMRI2images}},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Natural Scenes Dataset: Allen et al. at the University of Minnesota
- OpenAI CLIP: Radford et al. at OpenAI
- Stable Diffusion: Rombach et al. at Stability AI
- HuggingFace for the `diffusers` and `transformers` libraries

---

[Documentation](docs/) | [Running Experiments](docs/guides/RUNNING_EXPERIMENTS.md) | [Experiment Context](docs/EXPERIMENT_CONTEXT.md) | [vMF Guide](docs/guides/VMF_UACFG_GUIDE.md)
