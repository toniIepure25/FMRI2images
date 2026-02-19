# Documentation Index

> **Complete documentation for the Brain-to-Image fMRI reconstruction system**

This directory contains comprehensive documentation organized by purpose and audience. For paper/thesis writing, see [docs/paper/README.md](paper/README.md).

---

## 📚 Quick Navigation

### Essential Documents

| Document                                                    | Purpose                      | Audience    |
| ----------------------------------------------------------- | ---------------------------- | ----------- |
| **[SETUP.md](guides/SETUP.md)**                             | Complete installation guide  | New users   |
| **[RUNNING_EXPERIMENTS.md](guides/RUNNING_EXPERIMENTS.md)** | Training and evaluation      | Researchers |
| **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)**    | What's ready to use          | All users   |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**                | Command cheat sheet          | All users   |
| **[Paper Docs](paper/README.md)**                           | Thesis/publication materials | Writers     |

---

## 📖 Documentation Structure

### 1. **Getting Started**

**New to the project? Start here:**

- **[guides/SETUP.md](guides/SETUP.md)** - Complete setup for any environment
  - Automated setup scripts
  - Environment-specific instructions (JupyterHub, local, cluster)
  - Manual setup steps
  - Data download and preparation
  - Troubleshooting

- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Current system status
  - What's implemented and tested
  - Experiment configurations ready to run
  - Novel contributions status
  - Documentation completeness

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference
  - Common commands
  - Useful snippets
  - Configuration tips

---

### 2. **User Guides** (`guides/`)

**Step-by-step tutorials for common tasks:**

#### Setup & Configuration

- **[SETUP.md](guides/SETUP.md)** - Complete setup guide (all environments)
- **[SETUP_VERIFICATION.md](guides/SETUP_VERIFICATION.md)** - Verify installation
- **[COMMANDS_AFTER_CLONE.md](guides/COMMANDS_AFTER_CLONE.md)** - Post-clone steps
- **[TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)** - Common issues and solutions

#### Training & Experiments

- **[RUNNING_EXPERIMENTS.md](guides/RUNNING_EXPERIMENTS.md)** - Complete training guide
  - Single experiment training
  - Full ablation study
  - GPU memory management
  - Resume training
- **[ADAPTER_TRAINING_GUIDE.md](guides/ADAPTER_TRAINING_GUIDE.md)** - CLIP adapter training
- **[RIDGE_BASELINE.md](guides/RIDGE_BASELINE.md)** - Ridge regression baseline
- **[MLP_IMPLEMENTATION.md](guides/MLP_IMPLEMENTATION.md)** - MLP encoder details

#### Evaluation & Analysis

- **[EVALUATION_SUITE_GUIDE.md](guides/EVALUATION_SUITE_GUIDE.md)** - Comprehensive metrics
  - Standard metrics (retrieval, 2AFC, RSA)
  - Bayesian metrics (calibration, conformal prediction)
  - Statistical significance tests
- **[REPORTING_RECONSTRUCTION.md](guides/REPORTING_RECONSTRUCTION.md)** - Generate reports
- **[GALLERY_SUPPORT.md](guides/GALLERY_SUPPORT.md)** - Create image galleries

#### Advanced Topics

- **[NOVEL_CONTRIBUTIONS_PIPELINE.md](guides/NOVEL_CONTRIBUTIONS_PIPELINE.md)** - Novel features guide
  - Soft reliability weighting
  - InfoNCE contrastive loss
  - MC Dropout uncertainty
- **[NOVEL_CONTRIBUTIONS_IMPLEMENTATION.md](guides/NOVEL_CONTRIBUTIONS_IMPLEMENTATION.md)** - Detailed implementation
- **[REALISTIC_WORKFLOW.md](guides/REALISTIC_WORKFLOW.md)** - End-to-end workflow
- **[PIPELINE_SCRIPT_GUIDE.md](guides/PIPELINE_SCRIPT_GUIDE.md)** - Script documentation
- **[SETUP_SCRIPT_DOCS.md](guides/SETUP_SCRIPT_DOCS.md)** - Setup script details
- **[GETTING_STARTED_DIFFUSION.md](guides/GETTING_STARTED_DIFFUSION.md)** - Diffusion models intro
- **[QUICK_START.md](guides/QUICK_START.md)** - Rapid setup for experienced users

---

### 3. **Architecture Documentation** (`architecture/`)

**System design and component specifications:**

- **[PIPELINE_ARCHITECTURE.md](architecture/PIPELINE_ARCHITECTURE.md)** - Complete system architecture
  - Overall pipeline design
  - Component interactions
  - Data flow
- **[WORKFLOW.md](architecture/WORKFLOW.md)** - JupyterHub/HPC workflow
  - Environment setup
  - Job submission
  - Resource management
- **[DIFFUSION_DECODER.md](architecture/DIFFUSION_DECODER.md)** - Diffusion model integration
  - Architecture details
  - Sampling strategies
- **[DIFFUSION_ROBUSTNESS.md](architecture/DIFFUSION_ROBUSTNESS.md)** - Robustness analysis
- **[MODULARIZATION_COMPLETE.md](architecture/MODULARIZATION_COMPLETE.md)** - Module organization

---

### 4. **Technical Documentation** (`technical/`)

**Implementation details and advanced topics:**

#### Data & Preprocessing

- **[DATA_REQUIREMENTS.md](technical/DATA_REQUIREMENTS.md)** - Data requirements and specs
- **[NSD_Dataset_Guide.md](technical/NSD_Dataset_Guide.md)** - Natural Scenes Dataset guide
- **[DATA_VALIDATION_REAL_VS_FALLBACK.md](technical/DATA_VALIDATION_REAL_VS_FALLBACK.md)** - Data validation
- **[UPGRADE_TO_30K_SAMPLES.md](technical/UPGRADE_TO_30K_SAMPLES.md)** - Scaling to full dataset

#### Model & Training

- **[OPTIMAL_CONFIGURATION_GUIDE.md](technical/OPTIMAL_CONFIGURATION_GUIDE.md)** - Best practices
- **[ADAPTER_METADATA_SUMMARY.md](technical/ADAPTER_METADATA_SUMMARY.md)** - CLIP adapter metadata
- **[CHECKPOINT_RESUME.md](technical/CHECKPOINT_RESUME.md)** - Checkpoint management
- **[PREVENTING_MODEL_DOWNLOAD_BLOCKING.md](technical/PREVENTING_MODEL_DOWNLOAD_BLOCKING.md)** - Model download tips
- **[MANUAL_MODEL_DOWNLOAD.md](technical/MANUAL_MODEL_DOWNLOAD.md)** - Manual model download

#### Advanced Features

- **[GET_ALL_SAMPLES_GUIDE.md](technical/GET_ALL_SAMPLES_GUIDE.md)** - Complete sample handling

---

### 5. **Paper/Thesis Documentation** (`paper/`)

**Materials for writing and publishing:**

- **[README.md](paper/README.md)** - Paper documentation index
- **[outline.md](paper/outline.md)** - Complete paper outline and structure
- **[method.md](paper/method.md)** - Methodology section template
- **[experiments.md](paper/experiments.md)** - Experiment design
- **[evaluation_protocol.md](paper/evaluation_protocol.md)** - Detailed evaluation protocol
- **[results.md](paper/results.md)** - Results section template
- **[claims.md](paper/claims.md)** - Research claims
- **[limitations.md](paper/limitations.md)** - Limitations and future work
- **[reproducibility.md](paper/reproducibility.md)** - Reproducibility guidelines
- **[experiments/](paper/experiments/)** - Experiment-specific documentation

---

### 6. **Reference Documentation**

**Quick reference materials:**

- **[DATA_MODELS.md](DATA_MODELS.md)** - Data structures and schemas
- **[EVAL_QUICK_REFERENCE.md](EVAL_QUICK_REFERENCE.md)** - Evaluation metrics reference
- **[NOVEL_CONTRIBUTIONS_QUICK_REF.md](NOVEL_CONTRIBUTIONS_QUICK_REF.md)** - Novel features summary
- **[PAPER_GRADE_EVALUATION.md](PAPER_GRADE_EVALUATION.md)** - Publication-quality evaluation
- **[ablation_plan.md](ablation_plan.md)** - Ablation study plan and design
- **[refs.md](refs.md)** - References and citations

---

## 🎯 Documentation by Task

### I want to...

#### **Set up the system**

1. Read [guides/SETUP.md](guides/SETUP.md)
2. Follow automated setup instructions
3. Verify with [guides/SETUP_VERIFICATION.md](guides/SETUP_VERIFICATION.md)
4. If issues: [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)

#### **Run my first experiment**

1. Complete setup (see above)
2. Read [guides/RUNNING_EXPERIMENTS.md](guides/RUNNING_EXPERIMENTS.md)
3. Start with EXP0 baseline
4. Monitor with TensorBoard

#### **Understand what's implemented**

1. Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
2. Check experiment configs in `configs/experiments/`
3. Review test results (53/53 passing)

#### **Evaluate a trained model**

1. Read [guides/EVALUATION_SUITE_GUIDE.md](guides/EVALUATION_SUITE_GUIDE.md)
2. Choose appropriate metrics for your experiments
3. Run evaluation scripts
4. Generate reports with [guides/REPORTING_RECONSTRUCTION.md](guides/REPORTING_RECONSTRUCTION.md)

#### **Write my thesis/paper**

1. Read [paper/README.md](paper/README.md)
2. Use [paper/method.md](paper/method.md) for methodology section
3. Use [paper/experiments.md](paper/experiments.md) for experiment design
4. Use [paper/results.md](paper/results.md) for results section
5. Follow [paper/reproducibility.md](paper/reproducibility.md) for reproducibility

#### **Understand the architecture**

1. Read [architecture/PIPELINE_ARCHITECTURE.md](architecture/PIPELINE_ARCHITECTURE.md)
2. For JupyterHub: [architecture/WORKFLOW.md](architecture/WORKFLOW.md)
3. For diffusion models: [architecture/DIFFUSION_DECODER.md](architecture/DIFFUSION_DECODER.md)

#### **Debug an issue**

1. Check [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)
2. Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common commands
3. Check [technical/DATA_VALIDATION_REAL_VS_FALLBACK.md](technical/DATA_VALIDATION_REAL_VS_FALLBACK.md) for data issues

---

## 📊 Documentation Status

### ✅ Complete and Current

- **Setup guides** - Up-to-date for all environments
- **Experiment guides** - All 7 experiments documented
- **Evaluation documentation** - Comprehensive metrics coverage
- **Architecture docs** - Complete system documentation
- **Paper templates** - Ready for thesis writing

### 🔄 Actively Maintained

- **Implementation status** - Updated with each milestone
- **Troubleshooting guide** - Updated with new issues/solutions
- **Technical docs** - Updated as features are added

---

## 🤝 Contributing to Documentation

When adding or updating documentation:

1. **Follow the organization** - Place docs in appropriate subdirectories
2. **Update this index** - Add new docs to relevant sections
3. **Cross-reference** - Link to related documents
4. **Keep it current** - Update dates and status when content changes
5. **Use templates** - Follow existing formatting conventions

---

## 📞 Getting Help

- **Quick answers**: Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Setup issues**: See [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)
- **General questions**: Read [guides/SETUP.md](guides/SETUP.md)
- **Paper writing**: Start with [paper/README.md](paper/README.md)
- **GitHub Issues**: For bugs or feature requests

---

<div align="center">

**[Main README](../README.md) • [Setup](guides/SETUP.md) • [Running Experiments](guides/RUNNING_EXPERIMENTS.md) • [Paper Docs](paper/README.md)**

</div>
