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

---

## Paper-Grade Shared1000 Evaluation Pipeline (NEW)

### Overview

The new **paper-grade evaluation framework** provides publication-ready infrastructure with:

1. **Statistical Rigor**: Bootstrap CIs, permutation tests, multiple comparison correction
2. **Noise Ceiling Normalization**: Neuroscience-correct comparison across ROIs/subjects
3. **Brain Alignment Metrics**: Direct neural fidelity quantification
4. **Multi-Seed Evaluation**: Proper uncertainty quantification
5. **Full Reproducibility**: Manifests track git commits, package versions, input hashes
6. **Standardized Outputs**: JSON/CSV/Markdown/LaTeX for easy aggregation

### Key Features

#### 1. Rep-Mode Support

Evaluate using different fMRI repetitions:

- **`avg`**: Average across 3 repetitions (highest SNR) [DEFAULT]
- **`rep1`, `rep2`, `rep3`**: Use specific single repetition
- **`all`**: Compute repeat consistency (novel metric)

**Why it matters**: Averaging gives best performance, but single-rep evaluation tests robustness. Repeat consistency quantifies decoder reliability.

#### 2. Noise Ceiling Normalization (NOVEL)

Raw scores aren't comparable across ROIs/subjects due to different noise levels:

```python
# Raw correlation (NOT comparable!)
raw_corr = 0.45  # Is this good? Depends on ceiling!

# Ceiling-normalized (FAIR comparison)
ceiling = 0.60  # From NCSNR data
normalized = raw_corr / ceiling  # = 0.75 → 75% of theoretical max
```

**Interpretation**:
- `normalized < 0.5`: Decoder is poor
- `0.5 < normalized < 0.7`: Decoder is moderate
- `normalized > 0.7`: Decoder is strong (close to noise limit)

#### 3. Brain Alignment (Encoding Model Evaluation)

Direct neural fidelity metric complementing perceptual scores:

```
Generated Image → Encoding Model → Predicted fMRI → Correlation with True fMRI
```

**Why it matters**: High CLIPScore doesn't guarantee neural similarity. Brain alignment measures if generated images evoke similar brain activity.

#### 4. Multi-Seed Statistical Testing

```bash
# Run with 3 seeds
python scripts/eval_shared1000_full.py \
    --subject subj01 \
    --seeds 0 1 2 \
    ...

# Output: mean ± bootstrap CI for each metric
# Plus: pairwise tests with Holm-Bonferroni correction
```

### Quick Start

#### Single Subject Evaluation

```bash
make eval-shared1000 \
    SUBJECT=subj01 \
    ENCODER_CKPT=checkpoints/mlp/subj01/mlp.pt \
    ENCODER_TYPE=mlp
```

**Output**: `outputs/eval_shared1000/subj01/`
- `manifest.json`: Full reproducibility info
- `shared1000_avg_metrics_single.json`: Aggregate metrics
- `shared1000_avg_per_sample_single.csv`: Per-sample details
- `shared1000_avg_summary_single.md`: Human-readable report

#### Multi-Strategy Comparison

```bash
make eval-shared1000 \
    SUBJECT=subj01 \
    ENCODER_CKPT=checkpoints/two_stage/subj01/two_stage_best.pt \
    ENCODER_TYPE=two_stage \
    STRATEGIES="single best_of_8 boi_lite" \
    USE_CEILING=1
```

Evaluates:
- **single**: Single generation per sample
- **best_of_8**: Best of 8 generations (highest CLIPScore)
- **boi_lite**: Best-of-Initialization 3-step refinement

#### With Brain Alignment + Ceiling Normalization

```bash
make eval-shared1000 \
    SUBJECT=subj01 \
    ENCODER_CKPT=checkpoints/two_stage/subj01/two_stage_best.pt \
    ENCODER_TYPE=two_stage \
    USE_CEILING=1 \
    ENCODING_CKPT=checkpoints/encoding/subj01/encoding.pt \
    ROI=nsdgeneral
```

**New metrics added**:
- `brain_alignment_raw`: Raw voxel-wise correlation
- `brain_alignment_normalized`: Ceiling-normalized correlation
- `repeat_consistency`: Decoder reliability (when using rep-mode=all)

#### Cross-Subject Aggregation

```bash
# Run evaluation for all subjects
for subj in subj01 subj02 subj03; do
    make eval-shared1000 SUBJECT=$subj ENCODER_CKPT=checkpoints/mlp/$subj/mlp.pt
done

# Aggregate results
make summarize-shared1000 SUBJECTS="subj01 subj02 subj03"
```

**Output**: `outputs/eval_shared1000/`
- `SUMMARY.csv`: Cross-subject table
- `SUMMARY.tex`: LaTeX table for paper
- `SUMMARY.md`: Markdown table
- `stats_r1.json`: Statistical tests (pairwise with correction)
- `figures/`: Bar charts with error bars

#### Smoke Test (Fast Validation)

```bash
make test-pipeline  # 8 samples, single strategy, no generation
```

**Use cases**:
- Quick sanity check after code changes
- CI/CD testing
- Verify environment setup

### Advanced Usage

#### Python Direct Usage

```python
from fmri2img.stats import bootstrap_ci, paired_permutation_test, holm_bonferroni_correction
from fmri2img.reliability import load_ncsnr, compute_voxel_noise_ceiling_from_ncsnr
from fmri2img.eval.brain_alignment import compute_brain_alignment_with_ceiling

# Statistical inference
scores_a = [0.65, 0.68, 0.72, 0.70]
scores_b = [0.70, 0.73, 0.75, 0.74]

lower, upper = bootstrap_ci(scores_a, n_boot=2000)
print(f"Strategy A: {np.mean(scores_a):.3f} [{lower:.3f}, {upper:.3f}]")

p_value = paired_permutation_test(scores_a, scores_b, n_perm=10000)
print(f"Significance: p = {p_value:.4f}")

# Multiple comparison correction
p_values = [0.01, 0.03, 0.05, 0.15]
rejected = holm_bonferroni_correction(p_values, alpha=0.05)
print(f"Rejected: {rejected}")  # [True, True, False, False]

# Noise ceiling
ncsnr = load_ncsnr("subj01", roi="nsdgeneral")
ceiling = compute_voxel_noise_ceiling_from_ncsnr(
    ncsnr, 
    method="correlation",  # or "standard", "linear"
    aggregation="mean"     # or "median", "rms"
)
print(f"ROI ceiling: {ceiling:.3f}")

# Brain alignment
alignment = compute_brain_alignment_with_ceiling(
    encoding_model,
    generated_images,
    fmri_targets,
    noise_ceiling_map=ceiling
)
print(f"Raw: {alignment['voxelwise_corr_mean']:.3f}")
print(f"Normalized: {alignment['voxelwise_corr_mean_normalized']:.3f}")
```

#### Custom Evaluation Script

```python
from fmri2img.eval.shared1000_io import (
    write_metrics_json,
    write_per_sample_csv,
    write_summary_markdown,
    plot_metrics_comparison
)

# Your evaluation logic here
metrics = {
    "retrieval": {"r1": 0.08, "r5": 0.24, ...},
    "perceptual": {"clip_score": 0.55, ...},
    "brain": {"alignment_raw": 0.30, "alignment_normalized": 0.75},
    "consistency": {"repeat_cosine_mean": 0.82}
}

per_sample = pd.DataFrame({
    "sample_id": [0, 1, 2, ...],
    "r1": [1, 0, 1, ...],
    "clip_score": [0.55, 0.60, 0.52, ...]
})

# Write standardized outputs
write_metrics_json(metrics, "outputs/eval/metrics.json")
write_per_sample_csv(per_sample, "outputs/eval/per_sample.csv")
write_summary_markdown(metrics, per_sample, "outputs/eval/summary.md")

# Generate comparison plot
plot_metrics_comparison(
    metrics_dict={"Strategy A": metrics_a, "Strategy B": metrics_b},
    metric_key="r1",
    output_path="outputs/eval/r1_comparison.png"
)
```

### Output Structure

```
outputs/eval_shared1000/
├── subj01/
│   ├── manifest.json                           # Reproducibility manifest
│   │   ├── environment: {python, torch, cuda, git_commit, ...}
│   │   ├── packages: {numpy: "1.24.3", scipy: "1.11.2", ...}
│   │   ├── input_hashes: {encoder: "abc123...", clip_cache: "def456..."}
│   │   └── timestamp: "2025-01-15T10:30:00"
│   │
│   ├── shared1000_avg_metrics_single.json      # Strategy: single
│   │   ├── retrieval: {r1, r5, r10, mrr, mean_rank, ...}
│   │   ├── perceptual: {clip_score, ssim, lpips, ...}
│   │   ├── brain: {alignment_raw, alignment_normalized, ceiling, ...}
│   │   └── consistency: {repeat_cosine_mean, repeat_cosine_std, ...}
│   │
│   ├── shared1000_avg_per_sample_single.csv    # Per-sample breakdown
│   │   Columns: sample_id, nsd_id, r1, r5, clip_score, ssim, brain_corr, ...
│   │
│   ├── shared1000_avg_summary_single.md        # Human-readable report
│   │
│   ├── shared1000_avg_metrics_best_of_8.json   # Strategy: best_of_8
│   └── ...
│
├── subj02/
│   └── ...
│
├── SUMMARY.csv                                  # Cross-subject table
│   Columns: subject, strategy, r1, r5, clip_score, brain_alignment, ...
│
├── SUMMARY.tex                                  # LaTeX table for paper
│   \begin{table}
│   \caption{NSD Shared1000 Results}
│   ...
│
├── SUMMARY.md                                   # Markdown table
│
├── stats_r1.json                                # Statistical tests
│   ├── pairwise_tests: [{strategy_a, strategy_b, p_value, cohens_d}, ...]
│   ├── holm_bonferroni_rejected: [...]
│   └── summary: "Strategy B significantly better (p<0.001, d=0.82)"
│
└── figures/
    ├── r1_comparison.png                        # Bar chart: R@1 across strategies
    ├── r5_comparison.png
    ├── clip_score_comparison.png
    └── brain_alignment_comparison.png
```

### Metrics Explained

#### Standard Retrieval Metrics

- **R@1, R@5, R@10**: Top-K accuracy
- **MRR**: Mean Reciprocal Rank = mean(1 / rank)
- **Mean/Median Rank**: Average position of true image

#### Novel Metrics (This Implementation)

1. **Repeat Consistency** (when rep-mode=all)
   ```python
   # Predict CLIP embedding from each fMRI repetition
   pred_rep0, pred_rep1, pred_rep2 = model(fmri_rep0), model(fmri_rep1), model(fmri_rep2)
   
   # Compute pairwise cosine similarities
   consistency = mean([
       cosine(pred_rep0, pred_rep1),
       cosine(pred_rep0, pred_rep2),
       cosine(pred_rep1, pred_rep2)
   ])
   ```
   
   **Interpretation**:
   - High consistency + high R@1 = reliable and accurate decoder
   - High consistency + low R@1 = reliable but inaccurate (systematic bias)
   - Low consistency = noisy decoder

2. **Ceiling-Normalized Brain Alignment**
   ```python
   raw_corr = correlate(predicted_fmri, true_fmri)
   normalized_corr = raw_corr / noise_ceiling
   ```
   
   **Interpretation**:
   - Accounts for SNR differences across ROIs/subjects
   - Normalized = 1.0 → perfect prediction given noise
   - Enables fair comparison across conditions

### Example Workflows

#### Workflow 1: Compare Two Strategies

```bash
# Strategy A: MLP baseline
make eval-shared1000 \
    SUBJECT=subj01 \
    ENCODER_CKPT=checkpoints/mlp/subj01/mlp.pt \
    ENCODER_TYPE=mlp \
    STRATEGIES=single

# Strategy B: Two-Stage
make eval-shared1000 \
    SUBJECT=subj01 \
    ENCODER_CKPT=checkpoints/two_stage/subj01/two_stage.pt \
    ENCODER_TYPE=two_stage \
    STRATEGIES=single

# Compare
python scripts/summarize_shared1000.py \
    --eval-dir outputs/eval_shared1000 \
    --subjects subj01 \
    --output-dir outputs/eval_shared1000

# Check: outputs/eval_shared1000/stats_r1.json for significance
```

#### Workflow 2: Multi-Subject Paper Table

```bash
# Train models for all subjects
for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    python scripts/train_two_stage.py --subject $subj
done

# Evaluate all
for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    make eval-shared1000 \
        SUBJECT=$subj \
        ENCODER_CKPT=checkpoints/two_stage/$subj/two_stage_best.pt \
        ENCODER_TYPE=two_stage \
        STRATEGIES="single best_of_8" \
        USE_CEILING=1
done

# Aggregate
make summarize-shared1000 SUBJECTS="subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08"

# Copy to paper: outputs/eval_shared1000/SUMMARY.tex
```

#### Workflow 3: Ablation with Statistics

```bash
# Run ablation with 3 seeds
for weight in 0.0 0.2 0.4 0.6; do
    for seed in 0 1 2; do
        python scripts/train_two_stage.py \
            --subject subj01 \
            --infonce-weight $weight \
            --seed $seed \
            --output-dir checkpoints/ablation/infonce_${weight}/seed_${seed}
        
        make eval-shared1000 \
            SUBJECT=subj01 \
            ENCODER_CKPT=checkpoints/ablation/infonce_${weight}/seed_${seed}/two_stage_best.pt \
            ENCODER_TYPE=two_stage
    done
done

# Aggregate with statistics
python scripts/summarize_shared1000.py \
    --eval-dir outputs/eval_shared1000 \
    --subjects subj01 \
    --group-by infonce_weight \
    --output-dir outputs/ablations/infonce

# Result: mean ± CI for each weight, statistical tests
```

### Troubleshooting

#### Issue: Missing NCSNR data

```
FileNotFoundError: NCSNR data not found for subj01
```

**Solution**: NCSNR is part of NSD dataset. Ensure full NSD download:
```bash
# Check for: cache/nsd_hdf5/ncsnr/
ls cache/nsd_hdf5/ncsnr/
# Should contain: nsd_ncsnr_betas.npy, nsd_ncsnr_rois.npy, etc.
```

#### Issue: Encoding model not found

```
WARNING: Encoding model not found, skipping brain alignment
```

**Solution**: Train encoding model first:
```bash
python scripts/train_encoding_model.py --subject subj01 --roi nsdgeneral
```

Or skip brain alignment:
```bash
make eval-shared1000 SUBJECT=subj01 ENCODING_CKPT=""
```

#### Issue: Smoke test fails

```
ERROR: Smoke test failed with exit code 1
```

**Solution**: Run manually to see full error:
```bash
python scripts/eval_shared1000_full.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/mlp/subj01/mlp.pt \
    --encoder-type mlp \
    --smoke
```

### Performance Expectations

#### Typical Runtimes (on A100 GPU)

| Task | Time | Notes |
|------|------|-------|
| Retrieval-only (1000 samples) | ~2 min | No generation |
| Single strategy (full) | ~15 min | With image generation |
| Best-of-8 | ~2 hours | 8× generations per sample |
| Multi-strategy (single + best-of-8 + boi_lite) | ~2.5 hours | All strategies |
| Cross-subject summary | ~1 min | Post-processing only |

#### Expected Metric Ranges

**Retrieval (averaged reps, SOTA encoder)**:
- R@1: 8-12%
- R@5: 24-32%
- R@10: 38-48%
- MRR: 0.15-0.25

**Brain Alignment (ceiling-normalized)**:
- Random: 0.0-0.1
- Weak decoder: 0.3-0.5
- Strong decoder: 0.6-0.8
- Near-ceiling: 0.8-0.95

**Repeat Consistency**:
- Poor: < 0.70
- Moderate: 0.70-0.85
- Strong: > 0.85

### Statistical Guidelines

1. **Sample Size**: Use full 1000 shared images (not subsets) for reproducibility
2. **Seeds**: Minimum 3 seeds for statistical testing
3. **Significance**: Report p-values with Holm-Bonferroni correction
4. **Effect Size**: Always report Cohen's d alongside p-values
5. **Confidence Intervals**: Use bootstrap (2000 iterations) for non-parametric CIs

### Implementation Details

**New Modules**:
- `src/fmri2img/stats/inference.py`: Bootstrap, permutation tests, multiple comparison correction
- `src/fmri2img/reliability/noise_ceiling.py`: NCSNR loading, ceiling computation, repeat consistency
- `src/fmri2img/eval/brain_alignment.py`: Encoding model evaluation with ceiling normalization
- `src/fmri2img/eval/shared1000_io.py`: Standardized I/O (JSON/CSV/Markdown/LaTeX)
- `src/fmri2img/utils/manifest.py`: Reproducibility tracking (git, packages, hashes)

**Testing**:
```bash
# Run all tests
pytest tests/ -v

# Specific modules
pytest tests/test_stats.py -v
pytest tests/test_reliability.py -v
pytest tests/test_brain_alignment.py -v
```

### See Also

- [IMPLEMENTATION_SUMMARY.md](../../IMPLEMENTATION_SUMMARY.md): Detailed technical overview
- [src/fmri2img/stats/README.md](../../src/fmri2img/stats/README.md): Statistical methods API
- [src/fmri2img/reliability/README.md](../../src/fmri2img/reliability/README.md): Noise ceiling documentation
- [scripts/eval_shared1000_full.py](../../scripts/eval_shared1000_full.py): Main evaluation script

---

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
- Efron & Tibshirani (1993) for Bootstrap methods
- Holm (1979) for multiple comparison correction
- Schoppe et al. (2016) for noise ceiling normalization
