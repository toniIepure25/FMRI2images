# 🚀 Complete Usage Examples - SOTA fMRI Reconstruction System

This guide provides **ready-to-run commands** for all components of the system.

---

## Prerequisites

```bash
# Ensure you're in the project directory with virtual environment activated
cd /path/to/Bachelor\ V2
source .venv/bin/activate  # or activate your conda env

# Check CUDA is available (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 1️⃣ Data Preparation

### Build CLIP Cache (Required for Training)

```bash
# Build CLIP embeddings for all NSD stimuli
python scripts/build_clip_cache.py \
    --cache-root cache \
    --output outputs/clip_cache/clip.parquet \
    --batch-size 256
```

**Time**: ~2-3 hours for 73K images  
**Output**: `outputs/clip_cache/clip.parquet`

### Build Index (Usually Already Done)

```bash
# Create training/val/test split indices
python scripts/build_full_index.py \
    --subject subj01 \
    --output data/indices/nsd_index
```

---

## 2️⃣ Training Models

### Option A: Train with Config File (Recommended)

```bash
# Train SOTA two-stage encoder using config
python scripts/train_two_stage.py \
    --config configs/sota_two_stage.yaml \
    --subject subj01 \
    --output-dir checkpoints/two_stage/subj01

# The config file specifies:
# - PCA k=512
# - Latent dim=768
# - 4 residual blocks
# - InfoNCE weight=0.4
# - Batch size=128
# - All other hyperparameters
```

### Option B: Train with CLI Arguments

```bash
# Same as above but with manual parameters
python scripts/train_two_stage.py \
    --subject subj01 \
    --use-preproc --pca-k 512 \
    --latent-dim 768 --n-blocks 4 \
    --head-type mlp --head-hidden 512 \
    --mse-weight 0.3 --cosine-weight 0.3 --info-nce-weight 0.4 \
    --temperature 0.05 \
    --batch-size 128 --epochs 50 \
    --lr 0.001 --wd 0.0001 \
    --checkpoint-dir checkpoints/two_stage/subj01
```

### Option C: With Self-Supervised Pretraining

```bash
# Enable SSL pretraining before supervised training
python scripts/train_two_stage.py \
    --config configs/sota_two_stage.yaml \
    --subject subj01 \
    --self-supervised \
    --ssl-objective masked \
    --ssl-epochs 20 \
    --output-dir checkpoints/two_stage_ssl/subj01
```

### Train Baseline MLP (for Comparison)

```bash
# Simple 1-layer MLP baseline
python scripts/train_mlp.py \
    --subject subj01 \
    --use-preproc --pca-k 100 \
    --batch-size 64 --epochs 50 \
    --output-dir checkpoints/mlp/subj01
```

### Train Encoding Model (for BOI-lite)

```bash
# Train Image → fMRI encoding model
python scripts/train_encoding_model.py \
    --subject subj01 \
    --backbone clip_vit_l14 \
    --use-preproc --pca-k 512 \
    --batch-size 64 --epochs 30 \
    --output-dir checkpoints/encoding_model/subj01
```

**Time Estimates**:
- Baseline MLP: ~30 minutes
- Two-stage encoder: ~2-3 hours
- With SSL pretraining: ~4-5 hours
- Encoding model: ~1-2 hours

---

## 3️⃣ Evaluation

### Comprehensive Evaluation on NSD Shared 1000

```bash
# Full evaluation with all metrics
python scripts/eval_comprehensive.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/eval/subj01_shared1000 \
    --clip-cache outputs/clip_cache/clip.parquet \
    --average-reps  # Average fMRI across 3 repetitions

# Quick test with fewer samples
python scripts/eval_comprehensive.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/eval/test \
    --max-samples 100
```

**What it computes**:
- R@1, R@5, R@10, R@20, R@50, R@100
- Mean/median rank, MRR
- Top-1 cosine similarity
- (Future) CLIPScore, SSIM, LPIPS
- (Future) Brain alignment correlation

### Retrieval-Only Evaluation

```bash
# Evaluate retrieval on test set
python scripts/eval_retrieval.py \
    --subject subj01 \
    --encoder-type two_stage \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --split test \
    --gallery test \
    --clip-cache outputs/clip_cache/clip.parquet \
    --output-json outputs/eval/retrieval_results.json

# Compare against full gallery (harder)
python scripts/eval_retrieval.py \
    --subject subj01 \
    --encoder-type two_stage \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --split test \
    --gallery full \
    --output-json outputs/eval/retrieval_full_gallery.json
```

### Compare Multiple Models

```bash
# Evaluate baseline
python scripts/eval_retrieval.py \
    --subject subj01 \
    --encoder-type mlp \
    --checkpoint checkpoints/mlp/subj01/mlp.pt \
    --split test --gallery test \
    --output-json outputs/eval/mlp_retrieval.json

# Evaluate SOTA
python scripts/eval_retrieval.py \
    --subject subj01 \
    --encoder-type two_stage \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --split test --gallery test \
    --output-json outputs/eval/sota_retrieval.json

# Compare results
python scripts/generate_report.py \
    --results-dir outputs/eval \
    --output-dir outputs/reports/comparison \
    --report-type summary
```

---

## 4️⃣ Image Generation & Galleries

### Generate Comparison Galleries

```bash
# Generate side-by-side comparisons with multiple strategies
python scripts/generate_comparison_gallery.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/galleries/subj01 \
    --num-samples 16 \
    --strategies single best_of_8 \
    --grid-cols 4 \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --seed 42

# Quick test (fewer samples, faster generation)
python scripts/generate_comparison_gallery.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/galleries/test \
    --num-samples 4 \
    --strategies single \
    --num-inference-steps 25
```

**Note**: Requires Stable Diffusion model (~5GB). Download first:
```bash
python scripts/download_sd_model.py --model-id stabilityai/stable-diffusion-2-1
```

**Time Estimates**:
- Single strategy (16 samples): ~5-10 minutes
- With best-of-8: ~40-80 minutes
- With BOI-lite: ~15-30 minutes

### Generate with BOI-lite Refinement

```bash
# Requires trained encoding model
python scripts/generate_comparison_gallery.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/galleries/subj01_boi \
    --num-samples 16 \
    --strategies single boi_lite \
    --encoding-model-checkpoint checkpoints/encoding_model/subj01/encoding_model.pt
```

---

## 5️⃣ Ablation Studies

### PCA Dimensionality Ablation

```bash
# Sweep PCA k: 128, 256, 512, 768, 1024
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type pca_dims \
    --base-config configs/sota_two_stage.yaml \
    --output-dir outputs/ablations/pca_dims

# Results will be in:
# - outputs/ablations/pca_dims/results.csv
# - outputs/ablations/pca_dims/results.tex (LaTeX table)
```

### InfoNCE Weight Ablation

```bash
# Sweep InfoNCE weight: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type infonce_weight \
    --base-config configs/sota_two_stage.yaml \
    --output-dir outputs/ablations/infonce
```

### Architecture Depth Ablation

```bash
# Sweep n_blocks: 2, 3, 4, 6, 8
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type arch_depth \
    --base-config configs/sota_two_stage.yaml \
    --output-dir outputs/ablations/depth
```

### Best-of-N Ablation (Generation-Only)

```bash
# Sweep N: 1, 2, 4, 8, 16, 32
# No training required - just generation
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type best_of_n \
    --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --output-dir outputs/ablations/best_of_n
```

### Dry Run (Test Without Execution)

```bash
# See what would be run without actually running
python scripts/ablation_driver.py \
    --subject subj01 \
    --ablation-type pca_dims \
    --base-config configs/sota_two_stage.yaml \
    --output-dir outputs/ablations/test \
    --dry-run
```

**Time Estimates**:
- PCA dims (5 values): ~10-15 hours total
- InfoNCE weight (7 values): ~14-21 hours total
- Architecture depth (5 values): ~10-15 hours total
- Best-of-N (6 values, generation-only): ~2-4 hours total

---

## 6️⃣ Reporting & Visualization

### Generate Full Report

```bash
# Create comprehensive report with LaTeX tables
python scripts/generate_report.py \
    --results-dir outputs/eval/subj01_shared1000 \
    --output-dir outputs/reports/subj01 \
    --report-type full

# Output files:
# - summary.md (Markdown summary)
# - retrieval_table.tex (LaTeX table)
# - generation_table.tex (LaTeX table)
```

### Generate Ablation Report

```bash
# Create report for ablation study
python scripts/generate_report.py \
    --results-dir outputs/ablations/infonce \
    --output-dir outputs/reports/ablation_infonce \
    --report-type ablation

# Output files:
# - ablation_table.tex (LaTeX comparison table)
# - ablation_plot.png (Performance vs parameter plot)
# - summary.md
```

### Quick Summary

```bash
# Just create markdown summary
python scripts/generate_report.py \
    --results-dir outputs/eval/subj01_shared1000 \
    --output-dir outputs/reports/summary \
    --report-type summary
```

---

## 7️⃣ Complete Pipeline Example

Here's a complete end-to-end workflow:

```bash
#!/bin/bash
# Complete SOTA fMRI Reconstruction Pipeline

SUBJECT="subj01"
BASE_DIR="/home/tonystark/Desktop/Bachelor V2"

cd "$BASE_DIR"

echo "=== 1. Build CLIP Cache ==="
python scripts/build_clip_cache.py \
    --output outputs/clip_cache/clip.parquet

echo "=== 2. Train SOTA Encoder ==="
python scripts/train_two_stage.py \
    --config configs/sota_two_stage.yaml \
    --subject $SUBJECT \
    --output-dir checkpoints/two_stage/$SUBJECT

echo "=== 3. Evaluate on Shared 1000 ==="
python scripts/eval_comprehensive.py \
    --subject $SUBJECT \
    --encoder-checkpoint checkpoints/two_stage/$SUBJECT/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/eval/${SUBJECT}_shared1000

echo "=== 4. Generate Comparison Gallery ==="
python scripts/generate_comparison_gallery.py \
    --subject $SUBJECT \
    --encoder-checkpoint checkpoints/two_stage/$SUBJECT/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/galleries/$SUBJECT \
    --num-samples 16 \
    --strategies single best_of_8

echo "=== 5. Run Ablation Study ==="
python scripts/ablation_driver.py \
    --subject $SUBJECT \
    --ablation-type infonce_weight \
    --base-config configs/sota_two_stage.yaml \
    --output-dir outputs/ablations/infonce

echo "=== 6. Generate Reports ==="
python scripts/generate_report.py \
    --results-dir outputs/eval/${SUBJECT}_shared1000 \
    --output-dir outputs/reports/$SUBJECT \
    --report-type full

python scripts/generate_report.py \
    --results-dir outputs/ablations/infonce \
    --output-dir outputs/reports/ablation_infonce \
    --report-type ablation

echo "=== Pipeline Complete! ==="
echo "Results in: outputs/reports/"
```

Save this as `run_full_pipeline.sh`, make it executable, and run:

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

---

## 8️⃣ Troubleshooting

### Issue: Missing CLIP embeddings

```bash
# Error: "Missing CLIP embedding for nsdId=12345"
# Solution: Build CLIP cache first
python scripts/build_clip_cache.py --output outputs/clip_cache/clip.parquet
```

### Issue: CUDA out of memory

```bash
# Solution 1: Reduce batch size
python scripts/train_two_stage.py \
    --config configs/sota_two_stage.yaml \
    --batch-size 64  # Reduce from 128

# Solution 2: Use CPU (slower)
python scripts/train_two_stage.py \
    --config configs/sota_two_stage.yaml \
    --device cpu
```

### Issue: Diffusion model not cached

```bash
# Error: "Model not cached"
# Solution: Pre-download model
python scripts/download_sd_model.py --model-id stabilityai/stable-diffusion-2-1
```

### Issue: Config not found

```bash
# Error: "Config file not found"
# Solution: Use absolute path or check current directory
python scripts/train_two_stage.py \
    --config /full/path/to/configs/sota_two_stage.yaml
```

---

## 9️⃣ Expected Outputs

### After Training
```
checkpoints/two_stage/subj01/
├── two_stage_best.pt           # Best checkpoint by validation
├── two_stage_last.pt           # Last epoch checkpoint
├── training_log.json           # Per-epoch metrics
└── config.yaml                 # Saved configuration
```

### After Evaluation
```
outputs/eval/subj01_shared1000/
├── eval_comprehensive.log      # Detailed log
├── retrieval_metrics.json      # R@K, ranking metrics
├── generation_metrics.json     # CLIPScore, SSIM, LPIPS (TODO)
└── brain_alignment.json        # Correlation metrics (TODO)
```

### After Gallery Generation
```
outputs/galleries/subj01/
├── single/                     # Single sample generations
│   ├── sample_000.png
│   ├── sample_001.png
│   └── ...
├── best_of_8/                  # Best-of-8 generations
│   └── ...
└── comparison_grid.png         # Combined grid visualization
```

### After Ablation
```
outputs/ablations/infonce/
├── value_0.0/
│   ├── config.yaml
│   ├── two_stage_best.pt
│   └── eval_results.json
├── value_0.2/
│   └── ...
├── results.csv                 # Combined results
├── results.json
└── results.tex                 # LaTeX table
```

### After Reporting
```
outputs/reports/subj01/
├── summary.md                  # Markdown summary
├── retrieval_table.tex         # LaTeX table for paper
└── generation_table.tex
```

---

## 🎯 Quick Reference Card

| Task | Command | Time |
|------|---------|------|
| **Build CLIP cache** | `python scripts/build_clip_cache.py` | 2-3h |
| **Train baseline** | `python scripts/train_mlp.py` | 30min |
| **Train SOTA** | `python scripts/train_two_stage.py --config configs/sota_two_stage.yaml` | 2-3h |
| **Evaluate** | `python scripts/eval_comprehensive.py` | 10min |
| **Generate gallery** | `python scripts/generate_comparison_gallery.py` | 5-10min |
| **Run ablation** | `python scripts/ablation_driver.py --ablation-type pca_dims` | 10-15h |
| **Generate report** | `python scripts/generate_report.py --report-type full` | <1min |

---

## 📚 Next Steps

1. **Start with evaluation** if you already have trained models
2. **Run ablations** to understand what matters
3. **Generate galleries** for visual assessment
4. **Create reports** for papers/presentations

For more details, see:
- `SOTA_QUICK_START.md` - Detailed usage guide
- `docs/EVALUATION_SUITE_GUIDE.md` - Evaluation documentation
- `SOTA_IMPLEMENTATION_SUMMARY.md` - Technical details
