# Documentation Index

Complete documentation for the Brain-to-Image fMRI reconstruction system.

---

## Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[guides/SETUP.md](guides/SETUP.md)** | Complete installation guide | New users |
| **[guides/RUNNING_EXPERIMENTS.md](guides/RUNNING_EXPERIMENTS.md)** | Training and evaluation | Researchers |
| **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** | What's ready to use | All users |
| **[guides/JUPYTERHUB_REFERENCE.md](guides/JUPYTERHUB_REFERENCE.md)** | Command cheat sheet (HPC) | All users |
| **[paper/README.md](paper/README.md)** | Thesis/publication materials | Writers |

---

## Documentation Structure

### 1. Getting Started

- **[guides/SETUP.md](guides/SETUP.md)** -- Complete setup for any environment (automated + manual, JupyterHub/local)
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** -- Current system status, what's implemented and tested
- **[guides/QUICK_START.md](guides/QUICK_START.md)** -- Rapid setup for experienced users

---

### 2. User Guides (`guides/`)

#### Setup and Configuration

- **[SETUP.md](guides/SETUP.md)** -- Complete setup guide (all environments)
- **[TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)** -- Common issues and solutions
- **[JUPYTERHUB_REFERENCE.md](guides/JUPYTERHUB_REFERENCE.md)** -- JupyterHub/HPC quick reference

#### Training and Experiments

- **[RUNNING_EXPERIMENTS.md](guides/RUNNING_EXPERIMENTS.md)** -- Training guide (EXP0-EXP6, GPU management, resume)
- **[ADAPTER_TRAINING_GUIDE.md](guides/ADAPTER_TRAINING_GUIDE.md)** -- CLIP adapter training
- **[RIDGE_BASELINE.md](guides/RIDGE_BASELINE.md)** -- Ridge regression baseline
- **[MLP_IMPLEMENTATION.md](guides/MLP_IMPLEMENTATION.md)** -- MLP encoder details
- **[REALISTIC_WORKFLOW.md](guides/REALISTIC_WORKFLOW.md)** -- End-to-end workflow from zero to results

#### Evaluation and Analysis

- **[EVALUATION_SUITE_GUIDE.md](guides/EVALUATION_SUITE_GUIDE.md)** -- Comprehensive metrics (retrieval, 2AFC, Bayesian, calibration)
- **[REPORTING_RECONSTRUCTION.md](guides/REPORTING_RECONSTRUCTION.md)** -- Generate evaluation reports
- **[GALLERY_SUPPORT.md](guides/GALLERY_SUPPORT.md)** -- Create image comparison galleries

#### Novel Contributions

- **[NOVEL_CONTRIBUTIONS_PIPELINE.md](guides/NOVEL_CONTRIBUTIONS_PIPELINE.md)** -- Pipeline and implementation guide (soft reliability, InfoNCE, MC dropout)
- **[NOVEL_CONTRIBUTIONS_QUICK_REF.md](guides/NOVEL_CONTRIBUTIONS_QUICK_REF.md)** -- Quick reference cheat sheet

#### Diffusion and Pipeline

- **[PIPELINE_SCRIPT_GUIDE.md](guides/PIPELINE_SCRIPT_GUIDE.md)** -- `run_full_pipeline.py` documentation
- **[GETTING_STARTED_DIFFUSION.md](guides/GETTING_STARTED_DIFFUSION.md)** -- Diffusion models introduction

---

### 3. Architecture Documentation (`architecture/`)

- **[PIPELINE_ARCHITECTURE.md](architecture/PIPELINE_ARCHITECTURE.md)** -- Complete system architecture, component interactions, data flow
- **[WORKFLOW.md](architecture/WORKFLOW.md)** -- JupyterHub/HPC workflow, job management
- **[DIFFUSION_DECODER.md](architecture/DIFFUSION_DECODER.md)** -- Diffusion model integration, robustness, CLIP adapter

---

### 4. Technical Documentation (`technical/`)

#### Data

- **[DATA_REQUIREMENTS.md](technical/DATA_REQUIREMENTS.md)** -- Data requirements and specifications
- **[NSD_Dataset_Guide.md](technical/NSD_Dataset_Guide.md)** -- Natural Scenes Dataset reference
- **[DATA_PIPELINE.md](technical/DATA_PIPELINE.md)** -- Data validation and 30K sample upgrade
- **[DATA_MODELS.md](technical/DATA_MODELS.md)** -- Data structures, schemas, HuggingFace models

#### Model and Infrastructure

- **[ADAPTER_METADATA_SUMMARY.md](technical/ADAPTER_METADATA_SUMMARY.md)** -- CLIP adapter metadata
- **[MANUAL_MODEL_DOWNLOAD.md](technical/MANUAL_MODEL_DOWNLOAD.md)** -- Manual HuggingFace model download
- **[PREVENTING_MODEL_DOWNLOAD_BLOCKING.md](technical/PREVENTING_MODEL_DOWNLOAD_BLOCKING.md)** -- Model pre-download tips

---

### 5. Paper/Thesis Documentation (`paper/`)

- **[README.md](paper/README.md)** -- Paper documentation index and conventions
- **[outline.md](paper/outline.md)** -- Complete paper outline and structure
- **[evaluation_protocol.md](paper/evaluation_protocol.md)** -- Detailed evaluation protocol
- **[ablation_plan.md](paper/ablation_plan.md)** -- Ablation study design (EXP0-EXP6)
- **[PAPER_GRADE_EVALUATION.md](paper/PAPER_GRADE_EVALUATION.md)** -- Publication-quality evaluation setup
- **[method.md](paper/method.md)**, **[claims.md](paper/claims.md)**, **[experiments.md](paper/experiments.md)**, **[results.md](paper/results.md)**, **[limitations.md](paper/limitations.md)**, **[reproducibility.md](paper/reproducibility.md)** -- Section drafts
- **[refs.md](paper/refs.md)** -- Bibliography references
- **[experiments/TEMPLATE.md](paper/experiments/TEMPLATE.md)** -- Experiment card template

---

### 6. Overview/Presentations (`overview/`)

- **[overview.md](overview/overview.md)** -- Project overview (Romanian)
- **[PREZENTARE_COORDONATOR.md](overview/PREZENTARE_COORDONATOR.md)** -- Coordinator presentation (Romanian)

---

## Documentation by Task

**Set up the system:** [guides/SETUP.md](guides/SETUP.md) then [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md) if issues

**Run first experiment:** [guides/RUNNING_EXPERIMENTS.md](guides/RUNNING_EXPERIMENTS.md) -- start with EXP0

**Evaluate a trained model:** [guides/EVALUATION_SUITE_GUIDE.md](guides/EVALUATION_SUITE_GUIDE.md)

**Write thesis/paper:** [paper/README.md](paper/README.md), then section drafts in `paper/`

**Understand architecture:** [architecture/PIPELINE_ARCHITECTURE.md](architecture/PIPELINE_ARCHITECTURE.md)

**Debug an issue:** [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md), [guides/JUPYTERHUB_REFERENCE.md](guides/JUPYTERHUB_REFERENCE.md)

---

**[Main README](../README.md) | [Setup](guides/SETUP.md) | [Experiments](guides/RUNNING_EXPERIMENTS.md) | [Paper](paper/README.md)**
