# Brain-to-Image: Neural Decoding of Visual Perception from fMRI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A framework for reconstructing perceived images from 7T fMRI brain activity using the Natural Scenes Dataset (NSD). The system maps voxel activations to CLIP ViT-L/14 embeddings (768-D) via probabilistic decoders, then generates images through Stable Diffusion 2.1.

This is a bachelor thesis project targeting publication at NeurIPS/MICCAI venues. The codebase implements a progressive ablation study comparing deterministic, Gaussian, and von Mises-Fisher (vMF) decoding approaches, with novel contributions in probabilistic hyperspherical decoding and ROI-aware brain encoding.

---

## Project Status (March 2026)

| Series | Experiments | R@1 (subj01, val) | Status |
|--------|-------------|-------------------|--------|
| B-series (baselines) | B0v4, B1v4 | 20-25% | Complete |
| N-series (novel, v4) | N1v4-N4v4 | 30-36% | Complete (kappa collapse identified) |
| N-series (novel, v5) | N1v5-N4v5 | Pending | Fix deployed, awaiting re-run |

**SOTA reference:** MindEye achieves 93.2% R@1 on the same dataset (Scotti et al., 2024).

Two critical bugs were identified and fixed:
1. **nsdId off-by-one** (v1-v3): fMRI trials paired with wrong CLIP embeddings. Fixed in v4.
2. **Kappa collapse** (v4 N-series): `tau=0.07` in vMF-NCE capped kappa at ~5.6, zeroing concentration gradients. Fixed in v5 (`tau=1.0`).

See [docs/EXPERIMENT_CONTEXT.md](docs/EXPERIMENT_CONTEXT.md) for full technical context.

---

## Novel Contributions

This work introduces four novel components for fMRI-to-image decoding:

| ID | Contribution | Description | Experiment |
|----|-------------|-------------|------------|
| N1 | vMF-NCE Loss | Bessel-free contrastive loss on the unit hypersphere with bounded-sigmoid kappa parameterization | N1v5 |
| N2 | ROI Transformer Encoder | Brain-topology-aware tokenization: 17 ROI tokens processed by a Transformer encoder | N2v5 |
| N3 | Directional Consensus Fusion | Per-ROI vMF expert heads fused via spherical weighted averaging with attention-derived alphas | N3v5 |
| N4 | Full System + DUA-CFG | All innovations combined with Decomposed Uncertainty-Aware Classifier-Free Guidance for diffusion | N4v5 |

**Key insight:** vMF distributions on the hypersphere are a natural fit for CLIP embeddings (which are L2-normalized). The concentration parameter kappa provides a built-in, calibrated uncertainty measure without auxiliary heads.

---

## Pipeline Architecture

```
fMRI Voxels        Encoder              Decoder            Reconstruction
(~15k voxels)                           (768-D output)

+-----------+    +----------------+    +-------------+    +----------------+
| ROI-masked|    | MLP (B0/B1/N1) |    | Deterministic|    |                |
| beta maps |--->| or             |--->| Gaussian    |--->| CLIP ViT-L/14  |
| float32   |    | ROI Transformer|    | or vMF      |    | embedding      |
|           |    | (N2/N3/N4)     |    | (mu, kappa) |    | (768-D, L2)    |
+-----------+    +----------------+    +-------------+    +-------+--------+
                                                                  |
                                                                  v
+-----------+    +----------------+    +-------------+    +----------------+
| Recon     |    | Stable         |    | CLIP text   |    | Predicted CLIP |
| image     |<---| Diffusion 2.1  |<---| encoder     |<---| embedding      |
| (768x768) |    | (150 steps)    |    | (IP-Adapter)|    | + UA-CFG       |
+-----------+    +----------------+    +-------------+    +----------------+
```

---

## Ablation Study

The ablation ladder progressively adds complexity. B-series uses v4 configs; N-series uses v5 (kappa collapse fix).

| Experiment | Model Type | Encoder | Loss | Key Feature |
|------------|-----------|---------|------|-------------|
| B0v4 | Deterministic | MLP [8192, 4096, 2048] | MSE + InfoNCE + SoftCLIP | Baseline |
| B1v4 | Gaussian | MLP [8192, 4096, 2048] | Gaussian NLL + NCE + KL | Probabilistic baseline |
| N1v5 | vMF | MLP [8192, 4096, 2048] | vMF-NCE (tau=1.0) | Hyperspherical decoding |
| N2v5 | vMF | ROI Transformer (d=768, 6L) | vMF-NCE (tau=1.0) | Brain-topology encoding |
| N3v5 | vMF-DCF | ROI Transformer (d=768, 6L) | vMF-NCE + MultiTask | Directional consensus fusion |
| N4v5 | vMF-DCF | ROI Transformer (d=768, 6L) | vMF-NCE-SPCL + MultiTask | Full system + DUA-CFG |

### Running the Ablation

```bash
# On JupyterHub (A100):
set -a && source .env && set +a
pip install -e ".[train,diffusion]"

# Run full ablation (all 6 experiments x N subjects)
nohup make ablation SUBJECTS="subj01 subj02 subj05 subj07" GPU=0 > ablation.log 2>&1 &
tail -f ablation.log

# Aggregate results
python scripts/evaluation/aggregate_ablation.py \
  --results-dir experimental_results \
  --subjects subj01 subj02 subj05 subj07
```

### Training a Single Experiment

```bash
python scripts/training/train_unified.py \
  --config configs/experiments/N1v5_vmf_nce.yaml \
  --subject subj01
```

---

## Repository Structure

```
FMRI2images/
├── src/fmri2img/                  # Core library (pip install -e .)
│   ├── models/                    # UnifiedModel, ROITransformerEncoder, VonMisesFisherDecoder, ROI-DCF
│   ├── losses/                    # InfoNCE, Gaussian NCE/NLL, vMF-NCE, SoftCLIP, MixCo
│   ├── eval/                      # Retrieval, embedding, reconstruction, kappa calibration metrics
│   ├── contrastive/               # Memory queue (MoCo-style, 16384 entries)
│   ├── data/                      # NSD data loading, ROI index, CLIP cache, torch datasets
│   ├── training/                  # KL schedule, training utilities
│   ├── generation/                # Diffusion utilities (SD 2.1)
│   ├── inference/                 # Probabilistic pipeline, DUA-CFG, vMF mixture
│   ├── reliability/               # Noise ceiling, NCSNR
│   ├── io/                        # NSD layout, image loading, S3 access
│   ├── stats/                     # Statistical inference
│   ├── utils/                     # Config loading, logging, CLIP utilities
│   └── embedding_preproc.py       # PCR, whitening, centering
├── scripts/
│   ├── training/                  # train_unified.py, run_ablation_ladder.sh
│   ├── build/                     # build_clip_cache.py, preextract_fmri.py, nsd_index_builder.py
│   ├── evaluation/                # aggregate_ablation.py, eval_reconstruction.py, eval_retrieval.py
│   ├── reconstruction/            # decode_diffusion.py (UA-CFG, UA-Steps)
│   ├── diagnostics/               # pipeline_diagnostic.py, training_signal_diagnostic.py
│   ├── orchestration/             # run_full_pipeline.py, run_all_experiments.sh
│   ├── analysis/                  # compare_experiments.py, generate_report.py
│   └── utils/                     # preflight.py, doctor.py, smoke.py
├── configs/
│   ├── base.yaml                  # Global defaults
│   ├── experiments/               # B0v4, B1v4, N1v5, N2v5, N3v5, N4v5 (+ all prior versions)
│   ├── system/                    # clip.yaml, data.yaml, logging.yaml
│   ├── training/                  # Training recipes
│   └── inference/                 # Inference recipes (fast, quality, production)
├── tests/                         # 26 test files (pytest)
├── docs/
│   ├── EXPERIMENT_CONTEXT.md      # Comprehensive experiment briefing
│   ├── IMPLEMENTATION_STATUS.md   # Module readiness overview
│   ├── guides/                    # RUNNING_EXPERIMENTS, VMF_UACFG_GUIDE, SETUP
│   ├── paper/                     # Thesis outline, method, ablation plan
│   └── technical/                 # NSD dataset guide, data models
├── experimental_results/          # Per-experiment outputs (metrics, checkpoints, logs)
├── Makefile                       # Build targets (ablation, train, preextract, clip-cache)
├── pyproject.toml                 # Package definition
└── .env.jupyterhub                # Environment template for A100 pod
```

---

## Evaluation Metrics

### Retrieval (Primary)

| Metric | Description |
|--------|-------------|
| R@1, R@5, R@10 | Top-K retrieval accuracy on validation gallery (~986 images) |
| MRR | Mean reciprocal rank |
| Median Rank | Median position of ground truth in ranked gallery |

### Reconstruction (Post-Diffusion)

| Metric | Description | Direction |
|--------|-------------|-----------|
| PixCorr | Pixel-level correlation | Higher is better |
| SSIM | Structural similarity | Higher is better |
| AlexNet(2), AlexNet(5) | Perceptual similarity at layers 2 and 5 | Higher is better |
| LPIPS | Learned perceptual distance | Lower is better |
| CLIPScore | CLIP embedding similarity | Higher is better |

### Uncertainty (vMF models)

| Metric | Description |
|--------|-------------|
| Kappa statistics | Mean, std, min, max of concentration parameter |
| Calibration curves | Kappa vs actual retrieval accuracy |
| Risk-coverage | Selective prediction with AURC |

### SOTA Benchmarks

| Method | PixCorr | SSIM | Alex(2) | Alex(5) |
|--------|---------|------|---------|---------|
| MindEye (Scotti et al., 2024) | 0.309 | 0.323 | 0.947 | 0.978 |
| MindEye2 (Scotti et al., 2024) | 0.320 | 0.341 | 0.960 | 0.983 |
| Brain Diffuser (Ozcelik and VanRullen, 2023) | 0.254 | 0.356 | 0.942 | 0.962 |

---

## Data

This project uses the **Natural Scenes Dataset** (Allen et al., 2022):
- 8 subjects, 7T fMRI, 1.8mm resolution
- 10,000 unique COCO images, 3 repetitions each (30,000 trials per subject)
- Primary ROI: `nsdgeneral` (~15,724 voxels for subj01)
- 17 ROI tokens for Transformer encoder: V1v/d, V2v/d, V3v/d, V3A, V3B, V4, FFA1, FFA2, PPA, EBA, OFA, OPA, RSC, nsdgeneral_other

Pre-extracted features (`fmri_features.npy`, ~1.8 GB per subject) are used for training instead of raw NIfTI files to avoid memory issues.

CLIP embeddings are pre-computed using ViT-L/14 and stored in `outputs/clip_cache/clip.parquet` (9,999 unique images, 768-D, L2-normalized).

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.13 (conda) |
| GPU | 16 GB VRAM | A100 40 GB |
| System RAM | 32 GB | 100 GB |
| Storage | 100 GB | 500 GB+ |
| OS | Linux | Ubuntu 20.04+ |

### Installation

```bash
# Clone and install
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images
git checkout dirbrain-vmf-uacfg
pip install -e ".[train,diffusion]"

# Configure environment
cp .env.jupyterhub .env
# Edit .env: set HF_TOKEN, verify paths
set -a && source .env && set +a

# Verify setup
make preflight
```

### Key Dependencies

- PyTorch 2.0+ (with CUDA)
- transformers, diffusers (HuggingFace)
- open_clip_torch (CLIP ViT-L/14)
- scikit-learn, scipy, numpy, pandas
- h5py (NSD stimulus loading)
- nibabel (NIfTI processing)

---

## Testing

26 test files covering models, losses, evaluation, and data pipelines:

```bash
pytest tests/ -v
```

---

## Make Targets

| Target | Usage | Description |
|--------|-------|-------------|
| `make setup` | `make setup` | Install package in editable mode |
| `make preflight` | `make preflight` | System readiness check |
| `make index` | `make index SUBJECT=subj01` | Build trial index from NSD behavioral data |
| `make preprocess` | `make preprocess SUBJECT=subj01` | Fit preprocessing artifacts (scaler, PCA) |
| `make preextract` | `make preextract SUBJECT=subj01` | Extract ROI-masked fMRI features to .npy |
| `make clip-cache` | `make clip-cache` | Build CLIP embedding cache (768-D) |
| `make train` | `make train CONFIG=configs/experiments/N1v5_vmf_nce.yaml` | Train single experiment |
| `make ablation` | `make ablation SUBJECTS="subj01 subj02 subj05 subj07" GPU=0` | Run full ablation ladder |
| `make aggregate` | `make aggregate SUBJECTS="subj01 ..."` | Aggregate results to CSV/TeX/PNG |
| `make full-pipeline` | `make full-pipeline SUBJECTS="..." GPU=0` | Data + train + reconstruct + evaluate |

---

## Key References

- Allen, K. et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25, 116-126.
- Scotti, P. et al. (2024). MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data. *ICML 2024*.
- Ozcelik, F. and VanRullen, R. (2023). Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion. *arXiv:2303.05334*.
- Banerjee, A. et al. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *JMLR*, 6, 1345-1382.
- Davidson, T. et al. (2018). Hyperspherical variational auto-encoders. *UAI 2018*.
- Naselaris, T. et al. (2011). Encoding and decoding in fMRI. *NeuroImage*, 56(2), 400-410.

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

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Natural Scenes Dataset: Allen et al. at the University of Minnesota
- OpenAI CLIP: Radford et al. at OpenAI
- Stable Diffusion: Rombach et al. at Stability AI
- HuggingFace for the `diffusers` and `transformers` libraries

---

[Documentation](docs/) | [Setup Guide](docs/guides/SETUP.md) | [Running Experiments](docs/guides/RUNNING_EXPERIMENTS.md) | [Experiment Context](docs/EXPERIMENT_CONTEXT.md) | [vMF and UA-CFG Guide](docs/guides/VMF_UACFG_GUIDE.md)
