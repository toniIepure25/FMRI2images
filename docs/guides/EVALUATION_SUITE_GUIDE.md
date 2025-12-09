# Complete Evaluation Suite - User Guide

## Overview

The evaluation suite provides comprehensive tools for assessing fMRI reconstruction models:

1. **NSD Shared 1000 Evaluation** - Standard benchmark with 3 fMRI repetitions
2. **Comparison Galleries** - Visual side-by-side comparisons
3. **Ablation Studies** - Systematic hyperparameter sweeps
4. **Automated Reporting** - LaTeX tables and summaries

## Quick Start

### 1. Comprehensive Evaluation on NSD Shared 1000

Evaluate your model on the standard benchmark:

```bash
python scripts/eval_comprehensive.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/eval_shared1000 \
    --strategies single best_of_8 boi_lite \
    --clip-cache outputs/clip_cache/clip.parquet
```

**What it does:**
- Loads 1000 shared images (seen by all 8 subjects)
- Averages fMRI across 3 repetitions for higher SNR
- Predicts CLIP embeddings from fMRI
- Computes retrieval metrics (R@K, mean/median rank, MRR)
- Generates images with multiple strategies (TODO: next phase)
- Computes perceptual metrics (CLIPScore, SSIM, LPIPS)
- Measures brain alignment (encoding model correlation)

**Current Status:** ✅ Retrieval metrics working, ⏳ image generation integration pending

**Expected output:**
```
outputs/eval_shared1000/
├── eval_comprehensive.log
├── retrieval_metrics.json      # R@1, R@5, R@10, etc.
├── generation_metrics.json     # CLIPScore, SSIM, LPIPS (TODO)
└── brain_alignment.json        # Correlation with true fMRI (TODO)
```

### 2. Generate Comparison Galleries

Create visual comparisons of different generation strategies:

```bash
python scripts/generate_comparison_gallery.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/galleries/subj01 \
    --num-samples 16 \
    --strategies single best_of_8 boi_lite \
    --grid-cols 4
```

**What it does:**
- Loads test fMRI and generates images with all strategies
- Creates side-by-side comparison strips (GT | single | best-of-8 | BOI-lite)
- Arranges into grid for easy visual assessment
- Saves individual images and combined grid

**Output:**
```
outputs/galleries/subj01/
├── single/
│   ├── sample_000.png
│   ├── sample_001.png
│   └── ...
├── best_of_8/
│   └── ...
├── boi_lite/
│   └── ...
└── comparison_grid.png         # Combined grid visualization
```

### 3. Run Ablation Studies

Systematically test different hyperparameters:

#### PCA Dimensionality Ablation

```bash
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type pca_dims \
    --output-dir outputs/ablations/pca_dims \
    --base-config configs/sota_two_stage.yaml
```

Tests: k ∈ {128, 256, 512, 768, 1024}

**Expected trend:** Higher k → better performance (diminishing returns after 512)

#### InfoNCE Weight Ablation

```bash
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type infonce_weight \
    --output-dir outputs/ablations/infonce \
    --base-config configs/sota_two_stage.yaml
```

Tests: weight ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}

**Expected trend:** Optimal around 0.3-0.4

#### Architecture Depth Ablation

```bash
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type arch_depth \
    --output-dir outputs/ablations/depth \
    --base-config configs/sota_two_stage.yaml
```

Tests: n_blocks ∈ {2, 3, 4, 6, 8}

**Expected trend:** Deeper = better up to ~4 blocks, then overfitting

#### Best-of-N Ablation (Generation-only)

```bash
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type best_of_n \
    --output-dir outputs/ablations/best_of_n \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt
```

Tests: N ∈ {1, 2, 4, 8, 16, 32}

**Expected trend:** Logarithmic improvement, plateau at N=16

**Output:**
```
outputs/ablations/<ablation_type>/
├── value_<val1>/
│   ├── config.yaml
│   ├── two_stage_best.pt
│   └── eval_results.json
├── value_<val2>/
│   └── ...
├── results.csv                 # Combined results
├── results.json
└── results.tex                 # LaTeX table
```

### 4. Generate Reports

Create publication-ready reports from evaluation results:

```bash
python scripts/generate_report.py \
    --results-dir outputs/eval_shared1000 \
    --output-dir outputs/reports \
    --report-type full
```

**Output:**
```
outputs/reports/
├── summary.md                  # Markdown summary
├── retrieval_table.tex         # LaTeX table for paper
├── generation_table.tex
└── brain_alignment_table.tex
```

For ablation studies:

```bash
python scripts/generate_report.py \
    --results-dir outputs/ablations/infonce \
    --output-dir outputs/reports/ablation_infonce \
    --report-type ablation
```

**Output:**
```
outputs/reports/ablation_infonce/
├── ablation_table.tex          # LaTeX comparison table
├── ablation_plot.png           # Performance vs parameter plot
└── summary.md
```

## Available Metrics

### Retrieval Metrics

Computed by `eval_comprehensive.py` and `eval_retrieval.py`:

- **R@K**: Percentage of samples where true image is in top-K predictions
  - R@1: Top-1 accuracy (most strict)
  - R@5, R@10, R@20, R@50: Increasingly permissive
- **Mean Rank**: Average rank of true image (lower is better)
- **Median Rank**: Median rank (more robust to outliers)
- **MRR**: Mean Reciprocal Rank = average of 1/(rank+1)
- **Top-1 Cosine**: Mean cosine similarity with true CLIP embedding

### Perceptual Metrics

Computed from generated images:

- **CLIPScore**: Semantic similarity in CLIP space [0, 1]
  - Measures: High-level semantic content
  - Higher is better (typically 0.3-0.7 for reconstructions)
- **SSIM**: Structural Similarity Index [0, 1]
  - Measures: Structural content (edges, textures)
  - Higher is better
  - Requires: `pip install torchmetrics`
- **LPIPS**: Learned Perceptual Image Patch Similarity [0, ∞)
  - Measures: Perceptual distance using deep features
  - Lower is better (typically 0.2-0.6)
  - Requires: `pip install lpips`

### Brain Alignment Metrics

Measures neural fidelity:

- **Brain Correlation**: Correlation between:
  - Predicted fMRI from generated image (using encoding model)
  - True fMRI from stimulus
- **Range**: [-1, 1], higher is better
- **Interpretation**: How well does the generated image evoke similar brain activity?

## Expected Performance Ranges

Based on MindEye2, Brain-Diffuser, and our implementations:

### Encoder Performance (Validation)

| Configuration | Val Cosine | Test Cosine | R@1 (Test) | R@5 (Test) |
|--------------|------------|-------------|------------|------------|
| Baseline MLP | 0.52-0.54 | 0.50-0.52 | 2.5-3.5% | 10-12% |
| Two-Stage (k=256) | 0.54-0.56 | 0.52-0.54 | 4-5% | 14-16% |
| Two-Stage (k=512) | 0.56-0.58 | 0.54-0.56 | 5-7% | 18-22% |
| + InfoNCE | 0.59-0.61 | 0.57-0.59 | 7-9% | 22-26% |
| + SSL Pretrain | 0.60-0.62 | 0.58-0.60 | 8-10% | 24-28% |

### Generation Quality

| Strategy | CLIPScore | SSIM | LPIPS | Time (rel) |
|----------|-----------|------|-------|------------|
| Single | 0.45-0.50 | 0.20-0.23 | 0.45-0.55 | 1.0× |
| Best-of-4 | 0.52-0.56 | 0.22-0.25 | 0.38-0.48 | 4.0× |
| Best-of-8 | 0.56-0.60 | 0.23-0.26 | 0.35-0.45 | 8.0× |
| Best-of-16 | 0.58-0.62 | 0.24-0.27 | 0.33-0.43 | 16.0× |
| BOI-lite (3 steps) | 0.50-0.54 | 0.26-0.29 | 0.40-0.50 | 3.5× |
| Best-of-8 + BOI | 0.58-0.62 | 0.28-0.31 | 0.32-0.42 | 11.5× |

### Brain Alignment

| Method | Correlation | Interpretation |
|--------|-------------|----------------|
| Random images | 0.05-0.10 | Chance level |
| Baseline MLP | 0.20-0.25 | Weak neural similarity |
| SOTA Encoder | 0.25-0.30 | Moderate similarity |
| + Best-of-N | 0.27-0.32 | Improved selection |
| + BOI-lite | 0.32-0.38 | Strong neural fidelity |

## Troubleshooting

### Common Issues

1. **Missing CLIP embeddings**
   ```
   Error: Missing CLIP embedding for nsdId=12345
   ```
   **Solution:** Build CLIP cache first:
   ```bash
   python scripts/build_clip_cache.py --subject subj01
   ```

2. **Out of memory during generation**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution:** Reduce batch size or enable memory optimizations:
   ```bash
   --batch-size 8  # Reduce from 32
   ```

3. **Diffusion model not cached**
   ```
   ERROR: Diffusion model not cached
   ```
   **Solution:** Pre-download model:
   ```bash
   python scripts/download_sd_model.py --model-id stabilityai/stable-diffusion-2-1
   ```

4. **Missing encoding model for BOI-lite**
   ```
   WARNING: Encoding model not found, skipping BOI-lite
   ```
   **Solution:** Train encoding model first:
   ```bash
   python scripts/train_encoding_model.py --subject subj01
   ```

## Next Steps

After running evaluations:

1. **Analyze Results**
   - Check `summary.md` for overview
   - Look at `comparison_grid.png` for visual quality
   - Review ablation plots for insights

2. **Compare with Baselines**
   - Run same evaluation on Ridge/MLP baselines
   - Use `generate_report.py` to create comparison tables

3. **Generate Paper Figures**
   - Use LaTeX tables from reports
   - Create custom visualizations from saved metrics
   - Extract best/worst examples from galleries

4. **Run Statistical Tests**
   - Use `scripts/generate_report.py` with multiple runs
   - Compute significance tests (paired t-tests)
   - Report effect sizes (Cohen's d)

## File Structure Reference

```
scripts/
├── eval_comprehensive.py       # Main evaluation script (NSD shared 1000)
├── eval_retrieval.py           # Retrieval-only evaluation
├── generate_comparison_gallery.py  # Visual comparisons
├── ablation_driver.py          # Systematic ablations
└── generate_report.py          # Automated reporting

src/fmri2img/eval/
├── retrieval.py                # Retrieval metrics
├── image_metrics.py            # Perceptual metrics
└── __init__.py

src/fmri2img/generation/
├── advanced_diffusion.py       # Best-of-N, BOI-lite
├── diffusion_utils.py          # Pipeline utilities
└── __init__.py
```

## Citation

If you use this evaluation suite in your research, please cite:

```bibtex
@software{fmri2img_eval_suite,
  title = {Comprehensive Evaluation Suite for fMRI Reconstruction},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/fmri2img}
}
```

And relevant papers:
- Allen et al. (2022) for NSD dataset
- Scotti et al. (2024) for MindEye2 baseline
- Ozcelik & VanRullen (2023) for Brain-Diffuser baseline
