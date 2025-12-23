# Documentation Structure

**fMRI-to-Image Reconstruction Project**  
*Research-Level Neural Decoding Documentation*

---

## 📚 Documentation Organization

This directory contains comprehensive documentation for the fMRI-to-Image reconstruction pipeline, organized by purpose and audience.

### **Quick Navigation**

| Document | Purpose | Audience |
|----------|---------|----------|
| **[Quick Start Guide](../START_HERE.md)** | Get started in <5 minutes | New users, researchers |
| **[Usage Examples](../USAGE_EXAMPLES.md)** | Comprehensive command reference | All users |
| **[README](../README.md)** | Project overview & setup | GitHub visitors |

---

## 📖 Documentation Categories

### **1. User Guides** (`guides/`)
*Step-by-step tutorials for common tasks*

- **Getting Started**
  - `GETTING_STARTED_DIFFUSION.md` - Introduction to diffusion-based reconstruction
  - `QUICK_START.md` - Rapid setup and first experiments
  
- **Training Guides**
  - `ADAPTER_TRAINING_GUIDE.md` - CLIP adapter training workflow
  - `RIDGE_BASELINE.md` - Ridge regression baseline setup
  - `MLP_IMPLEMENTATION.md` - MLP encoder implementation
  
- **Evaluation & Analysis**
  - `EVALUATION_SUITE_GUIDE.md` - Comprehensive evaluation tools
  - **[`PAPER_GRADE_EVALUATION.md`](PAPER_GRADE_EVALUATION.md)** - ⭐ **NEW**: Publication-quality evaluation suite
  - `REPORTING_RECONSTRUCTION.md` - Generating reports and visualizations
  - `GALLERY_SUPPORT.md` - Creating image galleries

### **2. Architecture Documentation** (`architecture/`)
*System design and component specifications*

- **Overview Documents**
  - `MODULARIZATION_COMPLETE.md` - Module organization and structure
  - `DIFFUSION_DECODER.md` - Diffusion model integration architecture
  
- **Component Specifications**
  - Model architectures (Ridge, MLP, Two-Stage, Adapter)
  - Data pipeline design
  - Preprocessing modules
  - Evaluation framework

### **3. Technical Documentation** (`technical/`)
*Implementation details and troubleshooting*

- **Configuration**
  - `OPTIMAL_CONFIGURATION_GUIDE.md` - Best practices for hyperparameters
  - `ADAPTER_METADATA_SUMMARY.md` - Adapter configuration reference
  
- **Data Management**
  - `NSD_Dataset_Guide.md` - Natural Scenes Dataset structure
  - `DATA_VALIDATION_REAL_VS_FALLBACK.md` - Data validation procedures
  - `UPGRADE_TO_30K_SAMPLES.md` - Scaling to full dataset
  - `GET_ALL_SAMPLES_GUIDE.md` - Complete sample retrieval
  
- **Advanced Topics**
  - `DIFFUSION_ROBUSTNESS.md` - Robustness techniques for diffusion models
  - `MANUAL_MODEL_DOWNLOAD.md` - Manual model weight management
  - `PREVENTING_MODEL_DOWNLOAD_BLOCKING.md` - Offline model usage

---

## 🎯 Documentation by Task

### **I want to train a model**
1. Start with `RIDGE_BASELINE.md` for simplest approach
2. Progress to `MLP_IMPLEMENTATION.md` for neural encoders
3. See `ADAPTER_TRAINING_GUIDE.md` for diffusion integration

### **I want to evaluate my models**
1. Read `EVALUATION_SUITE_GUIDE.md` for overview
2. Use `REPORTING_RECONSTRUCTION.md` for generating reports
3. Check `GALLERY_SUPPORT.md` for visual comparisons

### **I want to understand the architecture**
1. Start with `MODULARIZATION_COMPLETE.md` for structure
2. Review `DIFFUSION_DECODER.md` for reconstruction pipeline
3. See component-specific docs in `architecture/`

### **I'm troubleshooting an issue**
1. Check `NSD_Dataset_Guide.md` for data problems
2. See `OPTIMAL_CONFIGURATION_GUIDE.md` for config issues
3. Review `DATA_VALIDATION_REAL_VS_FALLBACK.md` for validation

---

## 📊 Documentation Standards

### **Style Guide**
- Use clear, concise language appropriate for research audience
- Include code examples with expected outputs
- Provide performance benchmarks where applicable
- Cross-reference related documentation

### **Structure Template**
```markdown
# Title

**Brief description (1-2 sentences)**

## Overview
[High-level explanation]

## Prerequisites
[Required knowledge/setup]

## Detailed Guide
[Step-by-step instructions]

## Examples
[Concrete usage examples]

## Troubleshooting
[Common issues and solutions]

## References
[Related docs and citations]
```

### **Code Examples**
- Use syntax highlighting
- Include expected outputs
- Provide performance metrics (runtime, memory)
- Show both minimal and production examples

---

## 🔄 Documentation Maintenance

### **Versioning**
- Major changes: Increment version in doc header
- Keep deprecated sections with deprecation notices
- Archive outdated docs in `docs/archive/`

### **Review Process**
- Technical accuracy: Test all code examples
- Clarity: Ensure understandability by target audience
- Completeness: Cover prerequisites, steps, and troubleshooting
- Currency: Update with API/implementation changes

---

## 📚 Related Resources

### **External Documentation**
- [Natural Scenes Dataset (NSD)](http://naturalscenesdataset.org/)
- [CLIP by OpenAI](https://github.com/openai/CLIP)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [PyTorch Documentation](https://pytorch.org/docs/)

### **Academic Papers**
- Allen et al. (2022) - Natural Scenes Dataset (NSD)
- Radford et al. (2021) - CLIP: Learning Transferable Visual Models
- Rombach et al. (2022) - High-Resolution Image Synthesis with Latent Diffusion

### **Internal Resources**
- [Project README](../README.md)
- [Configuration Guide](../configs/README.md)
- [Source Code](../src/fmri2img/)

---

## 💡 Contributing to Documentation

### **Adding New Documentation**
1. Choose appropriate category (`guides/`, `architecture/`, `technical/`)
2. Follow the structure template above
3. Add entry to this README
4. Cross-reference in related documents
5. Test all code examples

### **Updating Existing Documentation**
1. Maintain backward compatibility notes
2. Update cross-references
3. Increment version number
4. Note changes in commit message

### **Quality Checklist**
- [ ] Code examples tested and working
- [ ] Cross-references verified
- [ ] Follows style guide
- [ ] Appropriate for target audience
- [ ] Performance metrics included (if applicable)
- [ ] Troubleshooting section complete

---

## 📞 Support

For questions not covered in documentation:
1. Check [GitHub Issues](https://github.com/toniIepure25/FMRI2images/issues)
2. Review [Usage Examples](../USAGE_EXAMPLES.md)
3. Open a new issue with reproducible example

---

**Last Updated**: December 7, 2025  
**Maintainer**: Bachelor Thesis Project  
**Status**: Active Development
