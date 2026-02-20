# Documentation

Project documentation for the fMRI-to-Image neural decoding pipeline.
Phase 2: CLIP ViT-L/14 (768-D), vMF-NCE, ROI-DCF, Decomposed UA-CFG.

## Quick Navigation

| I want to... | Read |
|--------------|------|
| Set up the environment | [guides/SETUP.md](guides/SETUP.md) |
| Run experiments (B0-N4) | [guides/RUNNING_EXPERIMENTS.md](guides/RUNNING_EXPERIMENTS.md) |
| Understand vMF and UA-CFG | [guides/VMF_UACFG_GUIDE.md](guides/VMF_UACFG_GUIDE.md) |
| Run on JupyterHub | [guides/JUPYTERHUB_REFERENCE.md](guides/JUPYTERHUB_REFERENCE.md) |
| Evaluate results | [guides/EVALUATION_SUITE_GUIDE.md](guides/EVALUATION_SUITE_GUIDE.md) |
| Generate reconstructions | [architecture/DIFFUSION_DECODER.md](architecture/DIFFUSION_DECODER.md) |
| Check implementation status | [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) |
| Write the paper | [paper/outline.md](paper/outline.md) |

## Directory Structure

```
docs/
  README.md                          This file
  IMPLEMENTATION_STATUS.md           Phase 2 implementation tracker

  architecture/
    DIFFUSION_DECODER.md             Diffusion model generation pipeline
    WORKFLOW.md                      HPC / JupyterHub workflow

  guides/
    SETUP.md                         Environment setup and data download
    RUNNING_EXPERIMENTS.md           How to run B0-N4 experiments
    VMF_UACFG_GUIDE.md              vMF training, kappa calibration, UA-CFG
    JUPYTERHUB_REFERENCE.md          JupyterHub cheat sheet
    EVALUATION_SUITE_GUIDE.md        Comprehensive evaluation guide
    REPORTING_RECONSTRUCTION.md      Image reconstruction reporting
    RIDGE_BASELINE.md                Ridge regression baseline
    TROUBLESHOOTING.md               Common issues and fixes

  overview/
    overview.md                      Project overview (Romanian)
    PREZENTARE_COORDONATOR.md        Coordinator presentation (Romanian)

  paper/
    README.md                        Paper writing guide
    outline.md                       Full paper outline (publication-ready)
    ablation_plan.md                 B0-N4 ablation plan
    evaluation_protocol.md           Evaluation protocol and metrics
    method.md                        Method description
    PAPER_GRADE_EVALUATION.md        Paper-grade evaluation workflow

  research/
    README.md                        Multi-agent research workspace

  technical/
    DATA_REQUIREMENTS.md             Data requirements and NSD paths
    NSD_Dataset_Guide.md             NSD dataset reference
    DATA_MODELS.md                   Data model definitions
```

## Project Phases

**Phase 1** (completed): CLIP ViT-B/32 (512-D), Soft Reliability Weighting, InfoNCE, MC Dropout.
Results in `experimental_results/exp001_baseline_ultimate/` and `RaportPaper3/`.

**Phase 2** (current): CLIP ViT-L/14 (768-D), vMF-NCE, ROI Transformer, ROI-DCF, Decomposed UA-CFG.
6-experiment ablation ladder: B0, B1, N1, N2, N3, N4.
Configs in `configs/experiments/`. Training via `scripts/training/train_unified.py`.
