# Reporting & Reconstruction Scripts

## Implementation Summary

Two surgical scripts added for ablation analysis and NN reconstruction baseline:

### 1. `scripts/report_ablation.py` - Mixed-Model Ablation Reporter

**Purpose**: Analyzes ablation CSV (Ridge vs MLP across reliability × PCA grid) and generates summary + plots.

**Features**:

- ✅ Backward compatible: treats missing `model` column as "Ridge"
- ✅ Generates markdown summary with top-10 table
- ✅ Identifies best settings per model
- ✅ Creates per-model line plots: test_cosine vs k_eff (one line per reliability)
- ✅ Creates per-model heatmaps: test_cosine across reliability × k_eff grid

**Usage**:

```bash
# Basic usage
python scripts/report_ablation.py --subject subj01

# Custom paths
python scripts/report_ablation.py \
    --csv outputs/reports/subj01/ablation_ridge.csv \
    --out outputs/reports/subj01/ablation_summary.md \
    --fig-dir outputs/reports/subj01/figs
```

**Outputs**:

- `outputs/reports/{subject}/ablation_summary.md` - Markdown summary with:
  - Top 10 settings by test_cosine
  - Best settings per model (Ridge/MLP)
  - Tie notes and model comparison
- `outputs/reports/{subject}/figs/test_cosine_vs_k_by_rel_{MODEL}.png` - Line plots
- `outputs/reports/{subject}/figs/heatmap_test_cosine_{MODEL}.png` - Heatmaps

**Scientific Notes**:

- Automatically handles mixed-model ablation tables (Ridge + MLP rows)
- Highlights ties in top performers (within 0.0001 cosine)
- Compares best Ridge vs best MLP for model selection

---

### 2. `scripts/reconstruct_nn.py` - Nearest-Neighbor Reconstruction Baseline

**Purpose**: Evaluates fMRI → image reconstruction via NN retrieval in CLIP space (strong baseline without diffusion).

**Features**:

- ✅ Loads Ridge or MLP encoder from checkpoint
- ✅ Reuses preprocessing pipeline from training (apples-to-apples)
- ✅ Builds gallery (excludes test samples, configurable limit)
- ✅ Retrieves top-K nearest neighbors for each test sample
- ✅ Computes retrieval metrics: R@K, mean/median rank, MRR
- ✅ Computes cosine similarity between predictions and ground truth
- ✅ Visualizes reconstruction grids (GT + top-K neighbors)
- ✅ Warns if gallery is tiny (R@K becomes trivial)

**Usage**:

```bash
# Ridge encoder
python scripts/reconstruct_nn.py \
    --subject subj01 \
    --encoder ridge \
    --ckpt checkpoints/ridge/subj01/ridge.pkl \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --gallery-limit 5000 \
    --topk 10 \
    --limit 256

# MLP encoder
python scripts/reconstruct_nn.py \
    --subject subj01 \
    --encoder mlp \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --gallery-limit 5000 \
    --topk 10 \
    --limit 256
```

**Outputs**:

- `outputs/reports/{subject}/nn_eval.csv` - Retrieval metrics:
  - mean_cosine, std_cosine
  - R@1, R@5, R@10, R@20
  - mean_rank, median_rank, mrr
  - gallery_size, n_test
- `outputs/reports/{subject}/nn_figs/reconstruction_nsd{ID}.png` - Reconstruction grids:
  - Each row: one test sample
  - Columns: GT + top-K retrieved neighbors
  - Green title = correct retrieval (GT in top-K)
  - Shows NSD IDs and cosine similarities

**Scientific Context**:

- Standard baseline for neural decoding (Ozcelik & VanRullen 2023)
- Uses encoder's learned mapping without additional training
- Evaluates semantic alignment in CLIP space (same as training objective)
- Gallery size matters: larger = harder retrieval, more realistic R@K

**CLIP Adapter Note**:

When using a CLIP adapter (512→768/1024D), NN retrieval can be done in either space:

- **512-D space** (default): Uses encoder output directly, matches training objective
- **Adapted space** (768/1024-D): Apply adapter before retrieval, matches diffusion conditioning

**Recommendation**: Keep NN retrieval in the **same space** as your comparison baseline:

- If comparing to encoder training metrics → use 512-D (pre-adapter)
- If comparing to diffusion-generated images → optionally use adapted space
- Always report which space is used for reproducibility

**Implementation**: To use adapted embeddings for NN retrieval, add `--clip-adapter` flag to `reconstruct_nn.py` (requires updating script to support adapters).

---

## 3. `scripts/eval_reconstruction.py` - Reconstruction Quality Evaluation

**Purpose**: Evaluates reconstructed images using CLIPScore and retrieval metrics in CLIP embedding space.

**Scientific Context**:

- **CLIPScore** (Hessel et al. 2021): Semantic similarity between generated and GT images in CLIP space
- **Retrieval@K**: How often generated image retrieves correct GT from gallery
- Standard metrics for image generation quality without pixel-level matching
- Supports both 512-D (ViT-B/32) and target-D (768/1024) evaluation

**Features**:

- ✅ Computes CLIPScore (per-sample cosine between generated and GT embeddings)
- ✅ Computes Retrieval@K (K=1,5,10) where query=generated, gallery=GT embeddings
- ✅ Ranking metrics: mean/median rank, MRR
- ✅ Supports 512-D (ViT-B/32) and target-D (768/1024 with `--use-adapter`)
- ✅ Automatic filename pattern matching (*_nsd{ID}.* or CSV mapping)
- ✅ Visualization grids: GT | Nearest Neighbor | Generated
- ✅ Per-sample CSV + aggregate JSON reports
- ✅ Graceful handling of missing images

**Usage**:

```bash
# Evaluate in 512-D space (ViT-B/32)
python scripts/eval_reconstruction.py \
    --subject subj01 \
    --recon-dir outputs/recon/subj01/mlp_diffusion/images \
    --clip-cache outputs/clip_cache/clip.parquet \
    --out-csv outputs/reports/subj01/recon_eval.csv \
    --out-fig outputs/reports/subj01/recon_grid.png

# Evaluate in 1024-D target space (SD 2.1)
python scripts/eval_reconstruction.py \
    --subject subj01 \
    --recon-dir outputs/recon/subj01/mlp_diffusion/images \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-adapter \
    --model-id stabilityai/stable-diffusion-2-1 \
    --out-csv outputs/reports/subj01/recon_eval_1024.csv \
    --out-fig outputs/reports/subj01/recon_grid_1024.png

# Or use Makefile targets
make eval-recon RECON_DIR=outputs/recon/subj01/run_001
make eval-recon-adapter RECON_DIR=outputs/recon/subj01/run_001
```

**Outputs**:

- **Per-sample CSV** (`recon_eval.csv`):
  ```
  nsdId, clipscore, rank, r@1, r@5, r@10
  12345, 0.723, 1, 1, 1, 1
  12346, 0.612, 3, 0, 1, 1
  ...
  ```

- **Aggregate JSON** (`recon_eval.json`):
  ```json
  {
    "subject": "subj01",
    "clip_space": "1024-D (target)",
    "n_samples": 256,
    "clipscore": {
      "mean": 0.654,
      "std": 0.092,
      "min": 0.412,
      "max": 0.891
    },
    "retrieval": {
      "R@1": 0.543,
      "R@5": 0.812,
      "R@10": 0.891
    },
    "ranking": {
      "mean_rank": 3.21,
      "median_rank": 2.0,
      "mrr": 0.612
    }
  }
  ```

- **Visualization Grid** (`recon_grid.png`):
  - 3 columns: Ground Truth | Nearest Neighbor | Generated
  - Up to 16 rows
  - CLIPScore overlay (green/orange/red based on threshold)
  - Rank overlay on NN column

**Metrics Explained**:

- **CLIPScore**: Cosine similarity ∈ [-1, 1] (typically [0, 1] for reasonable reconstructions)
  - Higher = better semantic preservation
  - >0.7: Excellent semantic match
  - 0.5-0.7: Good semantic match
  - 0.3-0.5: Moderate semantic match
  - <0.3: Poor semantic match

- **Retrieval@K**: Proportion of samples where GT appears in top-K retrievals
  - R@1 = 1.0: Perfect (generated always retrieves own GT as top-1)
  - R@5 > 0.8: Very good semantic alignment
  - R@10 > 0.9: Good semantic alignment

- **Mean Rank**: Average position of GT in ranked retrieval list
  - 1.0: Perfect (always top-1)
  - <5.0: Very good
  - <10.0: Good
  - >20.0: Poor

**Space Consistency**:

**CRITICAL**: Evaluate in the **same CLIP space** used for generation/conditioning:

- If generated with 512-D embeddings (no adapter) → evaluate in 512-D
- If generated with 768/1024-D adapted embeddings → evaluate in target space with `--use-adapter`

**Why?** Dimension mismatch creates unfair comparisons. A 512-D generated image should be evaluated against 512-D GTs for apples-to-apples comparison.

**Filename Matching**:

Script automatically detects filenames with patterns:

- `*_nsd{ID}.*` (e.g., `generated_nsd12345.png`)
- `*_{ID}.*` (e.g., `output_12345.jpg`)

Or provide CSV mapping:
```bash
--map-csv mapping.csv  # columns: nsdId, path
```

**Guardrails**:

- ✅ Embeddings are L2-normalized before metrics
- ✅ Warns if gallery size < test size (partial runs)
- ✅ Skips missing images (doesn't crash)
- ✅ Logs which CLIP space was used for reproducibility

---

## Reconstruction Metrics Summary

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **CLIPScore** | cos(gen_emb, gt_emb) | [-1, 1] | Semantic similarity (higher = better) |
| **R@1** | GT in top-1? | [0, 1] | Perfect retrieval rate |
| **R@5** | GT in top-5? | [0, 1] | Good retrieval rate |
| **R@10** | GT in top-10? | [0, 1] | Reasonable retrieval rate |
| **Mean Rank** | Avg position of GT | [1, ∞) | Lower = better |
| **MRR** | Mean(1/rank) | [0, 1] | Higher = better |

**Expected Performance** (based on literature):

- **Strong baseline** (NN retrieval): CLIPScore ~0.9-0.95, R@1 ~0.7-0.8
- **Diffusion-based** (with adapter): CLIPScore ~0.6-0.8, R@1 ~0.3-0.6
- **Trade-off**: Diffusion generates novel images (lower scores) but better perceptual quality

---

## Dependencies

Added to `requirements.txt`:

- `matplotlib` - Plotting
- `seaborn` - Heatmaps and styling
- `scikit-learn` - Already needed for Ridge

**Install**:

```bash
pip install matplotlib seaborn scikit-learn
```

---

## Testing

### Test `report_ablation.py`

1. **Check existing ablation CSV**:

   ```bash
   head -n 3 outputs/reports/subj01/ablation_ridge.csv
   ```

   - If no `model` column → backward compatibility kicks in
   - If has `model` column → plots per model

2. **Generate report**:

   ```bash
   python scripts/report_ablation.py --subject subj01
   ```

3. **Verify outputs**:
   ```bash
   cat outputs/reports/subj01/ablation_summary.md
   ls outputs/reports/subj01/figs/
   ```

**Expected**:

- Markdown with top-10 table and best settings
- Line plots: `test_cosine_vs_k_by_rel_{Ridge,MLP}.png`
- Heatmaps: `heatmap_test_cosine_{Ridge,MLP}.png`

---

### Test `reconstruct_nn.py`

1. **Check Ridge checkpoint exists**:

   ```bash
   ls checkpoints/ridge/subj01/ridge.pkl
   ```

2. **Run NN reconstruction** (small limit for quick test):

   ```bash
   python scripts/reconstruct_nn.py \
       --subject subj01 \
       --encoder ridge \
       --ckpt checkpoints/ridge/subj01/ridge.pkl \
       --clip-cache outputs/clip_cache/clip.parquet \
       --use-preproc \
       --gallery-limit 1000 \
       --topk 10 \
       --limit 50
   ```

3. **Verify outputs**:
   ```bash
   cat outputs/reports/subj01/nn_eval.csv
   ls outputs/reports/subj01/nn_figs/
   ```

**Expected**:

- CSV with retrieval metrics (R@1/5/10, mean_rank, cosine)
- PNG grids showing GT + top-K neighbors for each test sample
- Warning if gallery is tiny (< 100 samples)

---

## Implementation Notes

### Surgical Design Principles

**report_ablation.py**:

- ✅ No training code changes
- ✅ Reads existing ablation CSV
- ✅ Backward compatible with old CSV format (no `model` column)
- ✅ Handles mixed Ridge/MLP rows automatically
- ✅ Uses pandas pivot for heatmaps (robust to missing cells)

**reconstruct_nn.py**:

- ✅ No training code changes
- ✅ Reuses existing utilities:
  - `train_val_test_split()` from `train_utils` (same splits as training)
  - `NSDPreprocessor.load_artifacts()` (same preprocessing as training)
  - `cosine_sim()`, `retrieval_at_k()`, `compute_ranking_metrics()` from `eval.retrieval`
- ✅ Works with both Ridge and MLP (common predict() interface)
- ✅ Warns if gallery is tiny (R@K trivial)
- ✅ Visualizes reconstruction quality (GT + top-K)

### Code Reuse

**From existing modules**:

- `fmri2img.data.nsd_index_reader` - Index loading
- `fmri2img.data.preprocess` - Preprocessing pipeline
- `fmri2img.data.clip_cache` - CLIP embeddings
- `fmri2img.io.s3` - NIfTI loading
- `fmri2img.models.ridge` - Ridge encoder
- `fmri2img.models.mlp` - MLP encoder (load_mlp)
- `fmri2img.models.train_utils` - Split logic
- `fmri2img.eval.retrieval` - Retrieval metrics

**New code only**:

- Report generation (markdown table, summary)
- Visualization (line plots, heatmaps, reconstruction grids)
- NN retrieval orchestration
- MLPWrapper for common predict() interface

---

## Acceptance Checklist

### `report_ablation.py`

✅ **Reads CSV**: `outputs/reports/{subject}/ablation_ridge.csv`  
✅ **Handles mixed models**: `model` column with "Ridge" and "MLP"  
✅ **Backward compatible**: Treats missing `model` column as "Ridge"  
✅ **Generates markdown**: `ablation_summary.md` with top-10 table + best settings  
✅ **Line plots**: `test_cosine_vs_k_by_rel_{MODEL}.png` (one per model)  
✅ **Heatmaps**: `heatmap_test_cosine_{MODEL}.png` (one per model)  
✅ **No training code changes**: Pure analysis script

### `reconstruct_nn.py`

✅ **Loads Ridge encoder**: From `checkpoints/ridge/{subject}/ridge.pkl`  
✅ **Loads MLP encoder**: From `checkpoints/mlp/{subject}/mlp.pt`  
✅ **Reuses preprocessing**: Same artifacts as training (`--use-preproc`)  
✅ **Reuses splits**: Same `train_val_test_split()` as training  
✅ **Builds gallery**: Excludes test samples, configurable limit  
✅ **Retrieves top-K**: NN retrieval via cosine similarity  
✅ **Computes metrics**: R@K, mean_rank, cosine, etc.  
✅ **Saves CSV**: `outputs/reports/{subject}/nn_eval.csv`  
✅ **Visualizes grids**: PNG grids with GT + top-K neighbors  
✅ **Warns if gallery tiny**: Logs warning if gallery < 100  
✅ **No training code changes**: Pure evaluation script

---

## Example Workflows

### Workflow 1: Analyze Ablation Results

```bash
# After running ablation study (Ridge and/or MLP)
python scripts/ablate_preproc_and_ridge.py --model ridge ...
python scripts/ablate_preproc_and_ridge.py --model mlp ...

# Generate report and plots
python scripts/report_ablation.py --subject subj01

# View results
cat outputs/reports/subj01/ablation_summary.md
open outputs/reports/subj01/figs/*.png
```

### Workflow 2: Evaluate NN Reconstruction Baseline

```bash
# Using best Ridge checkpoint
python scripts/reconstruct_nn.py \
    --subject subj01 \
    --encoder ridge \
    --ckpt checkpoints/ridge/subj01/ridge.pkl \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --gallery-limit 5000 \
    --topk 10

# Using best MLP checkpoint
python scripts/reconstruct_nn.py \
    --subject subj01 \
    --encoder mlp \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --gallery-limit 5000 \
    --topk 10

# Compare results
cat outputs/reports/subj01/nn_eval.csv
```

### Workflow 3: Paper Figures

```bash
# 1. Run ablation for both models
make ablate            # Ridge
make ablate-mlp        # MLP

# 2. Generate report and plots
python scripts/report_ablation.py --subject subj01

# 3. Evaluate NN reconstruction
python scripts/reconstruct_nn.py \
    --subject subj01 \
    --encoder mlp \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --gallery-limit 5000 \
    --topk 10

# 4. Collect figures for paper
cp outputs/reports/subj01/figs/heatmap_test_cosine_MLP.png paper/figures/
cp outputs/reports/subj01/nn_figs/*.png paper/figures/
```

---

## Scientific Notes

### Ablation Report

**Top-10 Table**:

- Ranks all (model, rel, k) settings by test_cosine
- Includes best_alpha for Ridge (N/A for MLP)
- Shows retrieval metrics (R@1/5/10)
- Shows data split sizes (train/val/test)

**Best Settings Per Model**:

- Identifies optimal hyperparameters for each model
- Enables fair Ridge vs MLP comparison
- Reports checkpoint paths for reproducibility

**Line Plots**:

- Shows effect of PCA dimensionality (k_eff) on test_cosine
- One line per reliability threshold
- Helps identify sweet spot for k

**Heatmaps**:

- 2D view of reliability × k_eff grid
- Color intensity = test_cosine
- Quickly identifies optimal region

### NN Reconstruction

**Gallery Construction**:

- Excludes test samples (no leakage)
- Configurable size (larger = harder retrieval)
- Typical: 1000-10000 for realistic evaluation

**Retrieval Metrics**:

- **R@K**: How often GT appears in top-K (higher is better)
- **Mean rank**: Average position of GT (lower is better)
- **MRR**: Mean reciprocal rank (higher is better)
- **Cosine**: Direct similarity between pred and GT

**Visualization**:

- Shows reconstruction quality qualitatively
- Green title = correct retrieval (GT in top-K)
- Helps diagnose semantic vs perceptual alignment

**Warning on Gallery Size**:

- If gallery < 100: R@K becomes trivial (easy to retrieve)
- Realistic evaluation needs gallery ≥ 1000
- NSD has ~73k unique images (use large gallery)

---

## Files Created

1. **scripts/report_ablation.py** (~350 lines)

   - CSV loading with backward compatibility
   - Markdown summary generation
   - Line plots (test_cosine vs k_eff)
   - Heatmaps (reliability × k_eff)

2. **scripts/reconstruct_nn.py** (~500 lines)

   - Encoder loading (Ridge/MLP)
   - Feature extraction (reuses train pipeline)
   - Gallery building
   - NN retrieval
   - Metrics computation
   - Reconstruction grid visualization

3. **requirements.txt** (updated)

   - Added: matplotlib, seaborn, scikit-learn

4. **docs/REPORTING_RECONSTRUCTION.md** (this file)
   - Scikit-learn dependency fix

4. **docs/REPORTING_RECONSTRUCTION.md** (this file)
   - Complete documentation
   - Usage examples
   - Testing procedures
   - Acceptance checklist

---

## 4. scripts/run_reconstruct_and_eval.py - One-Click Orchestrator

**Purpose**: End-to-end workflow that generates reconstructions and evaluates them in the correct CLIP space, producing a thesis-ready Markdown summary.

### Features

**✅ Complete Pipeline:**
- Checks SD model cache before starting
- Generates images via `decode_diffusion.py`
- Evaluates in matching CLIP space via `eval_reconstruction.py`
- Creates thesis-ready Markdown summary with all metrics

**✅ Space Consistency Guarantee:**
- No adapter → 512-D evaluation (ViT-B/32)
- With adapter → 768/1024-D evaluation (target CLIP)
- Automatically matches generation to evaluation space

**✅ Thesis-Ready Output:**
- Configuration table (subject, encoder, adapter, model)
- Metrics table (CLIPScore, R@K, ranking stats)
- Quality assessment with interpretation
- Baseline comparison table
- Output file paths

**✅ Robust Error Handling:**
- Validates all checkpoints exist
- Checks SD cache before generating
- Exits cleanly on any step failure
- Propagates exit codes properly

### Usage

**No Adapter (512-D):**
```bash
# Using defaults
make recon-eval

# Custom settings
make recon-eval \
    SUBJECT=subj01 \
    ENCODER=mlp \
    CKPT=checkpoints/mlp/subj01/mlp.pt \
    LIMIT=64

# Direct invocation
python scripts/run_reconstruct_and_eval.py \
    --subject subj01 \
    --encoder mlp \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --clip-cache outputs/clip_cache/clip.parquet \
    --output-dir outputs/recon/subj01/auto \
    --report-dir outputs/reports/subj01 \
    --limit 64
```

**With Adapter (768/1024-D):**
```bash
# Using defaults (1024-D for SD-2.1)
make recon-eval-adapter

# Custom settings
make recon-eval-adapter \
    SUBJECT=subj01 \
    ENCODER=mlp \
    CKPT=checkpoints/mlp/subj01/mlp.pt \
    ADAPTER=checkpoints/clip_adapter/subj01/adapter.pt \
    MODEL=stabilityai/stable-diffusion-2-1 \
    LIMIT=64

# Direct invocation
python scripts/run_reconstruct_and_eval.py \
    --subject subj01 \
    --encoder mlp \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --use-adapter \
    --adapter checkpoints/clip_adapter/subj01/adapter.pt \
    --model-id stabilityai/stable-diffusion-2-1 \
    --clip-cache outputs/clip_cache/clip.parquet \
    --output-dir outputs/recon/subj01/auto_adapter \
    --report-dir outputs/reports/subj01 \
    --limit 64
```

### Workflow Steps

**Step 1: Check SD Cache**
- Runs `scripts/check_hf_cache.py` to verify model is downloaded
- If missing, prompts user to run `make download-sd`
- Does NOT auto-download (avoids blocking on large downloads)

**Step 2: Generate Images**
- Shells out to `scripts/decode_diffusion.py`
- Passes all relevant flags (encoder, ckpt, adapter, model-id, etc.)
- Ensures filenames embed `nsd{ID}` for matching
- Propagates `--limit` to control test set size

**Step 3: Evaluate**
- Shells out to `scripts/eval_reconstruction.py`
- Matches CLIP space to generation:
  - No adapter → default 512-D evaluation
  - With adapter → `--use-adapter --model-id` for target-D
- Produces CSV (per-sample), JSON (aggregate), PNG (grid)

**Step 4: Generate Markdown Summary**
- Parses evaluation JSON
- Creates `recon_eval_summary.md` with:
  - Configuration header
  - Metrics table with interpretations
  - Quality assessment
  - Baseline comparison
  - Output file paths
- Prints summary to console for quick review

### Output Structure

```
outputs/
├── recon/
│   └── subj01/
│       ├── auto_no_adapter/       # Images from make recon-eval
│       │   ├── generated_nsd12345.png
│       │   ├── generated_nsd12346.png
│       │   └── ...
│       └── auto_with_adapter/     # Images from make recon-eval-adapter
│           ├── generated_nsd12345.png
│           └── ...
└── reports/
    └── subj01/
        ├── recon_eval.csv         # Per-sample metrics
        ├── recon_eval.json        # Aggregate metrics
        ├── recon_grid.png         # Visualization grid
        └── recon_eval_summary.md  # Thesis-ready summary ⭐
```

### Markdown Summary Example

```markdown
# Reconstruction Evaluation Summary

**Generated:** 2025-10-25 14:32:10

## Configuration

- **Subject:** `subj01`
- **Encoder:** `mlp`
- **Checkpoint:** `checkpoints/mlp/subj01/mlp.pt`
- **Adapter:** Yes (`checkpoints/clip_adapter/subj01/adapter.pt`)
- **Model ID:** `stabilityai/stable-diffusion-2-1`
- **CLIP Space:** **1024-D** (target CLIP)
- **Test Samples:** 64 / 982
- **Output Directory:** `outputs/recon/subj01/auto_with_adapter`

---

**Note:** Evaluated in **1024-D CLIP space (target for SD-2.1)** — matched to generation space.

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **CLIPScore** | 0.654 ± 0.092 | Good |
| **R@1** | 0.543 | 54.3% top-1 correct |
| **R@5** | 0.812 | 81.2% in top-5 |
| **R@10** | 0.891 | 89.1% in top-10 |
| **Mean Rank** | 3.21 | Avg position in gallery |
| **Median Rank** | 2.0 | Median position |
| **MRR** | 0.612 | Mean reciprocal rank |

### Quality Assessment

**Good** — reasonable semantic preservation

## Output Files

- **CSV (per-sample):** `outputs/reports/subj01/recon_eval.csv`
- **JSON (aggregate):** `outputs/reports/subj01/recon_eval.json`
- **Visualization Grid:** `outputs/reports/subj01/recon_grid.png`
- **Generated Images:** `outputs/recon/subj01/auto_with_adapter/`

## Methodology

**Metrics:**
- **CLIPScore:** Cosine similarity between generated and GT image embeddings (Hessel et al. 2021)
- **Retrieval@K:** Proportion of samples where GT appears in top-K retrievals
- **Mean Rank:** Average position of GT in ranked retrieval list
- **MRR:** Mean reciprocal rank (1/rank)

**CLIP Space Consistency:**
Generated with CLIP adapter → evaluated in 1024-D target space (consistent).

## Baseline Comparison

| Method | CLIPScore | R@1 | Notes |
|--------|-----------|-----|-------|
| **NN Retrieval** | 0.90-0.95 | 0.70-0.80 | Strong baseline (existing images) |
| **Diffusion (literature)** | 0.60-0.80 | 0.30-0.60 | Novel generation |
| **This Run** | 0.65 | 0.54 | Current results |

*Note: Lower scores for diffusion-based methods don't necessarily indicate worse quality—they reflect the trade-off between semantic similarity and perceptual novelty.*

---

*Generated by `run_reconstruct_and_eval.py` on 2025-10-25 14:32:10*
```

### Arguments

**Required:**
- `--subject`: NSD subject ID (default: `subj01`)
- `--encoder`: Encoder type (`ridge` or `mlp`)
- `--ckpt`: Path to encoder checkpoint
- `--clip-cache`: Path to CLIP embeddings cache (parquet)
- `--output-dir`: Directory for generated images
- `--report-dir`: Directory for evaluation reports

**Optional:**
- `--limit`: Number of test samples (default: `64`)
- `--device`: Device for inference (`auto`, `cuda`, `cpu`)
- `--steps`: Diffusion steps (default: `50`)
- `--index-root`: Root directory with subject indices
- `--index-file`: Direct path to index parquet file

**Adapter:**
- `--use-adapter`: Enable CLIP adapter
- `--adapter`: Path to adapter checkpoint (required if `--use-adapter`)
- `--model-id`: Diffusion model ID (required if `--use-adapter`)

### Guardrails

**1. SD Cache Check:**
- Always checks if diffusion model is cached
- Never auto-downloads (avoids blocking)
- Prompts user with clear instructions if missing:
  ```
  Model 'stabilityai/stable-diffusion-2-1' does not appear to be cached locally.
  
  To download the model, run:
    make download-sd MODEL=stabilityai/stable-diffusion-2-1
  
  Continue anyway? (y/N):
  ```

**2. Space Consistency:**
- Automatically extracts `target_dim` from adapter metadata
- Passes correct flags to evaluation script
- No-adapter → 512-D eval (default)
- With-adapter → target-D eval (768/1024)

**3. Error Propagation:**
- Validates all checkpoints exist before starting
- Exits immediately if decode fails
- Exits immediately if eval fails
- Returns non-zero exit code on any failure

**4. Limit Consistency:**
- Same `--limit` passed to both decode and eval
- Ensures metrics computed on same test set
- Prevents gallery size mismatches

**5. Filename Matching:**
- Decoder outputs `generated_nsd{ID}.png`
- Evaluator parses `nsd{ID}` automatically
- No manual CSV mapping needed

### Testing

**Syntax Check:**
```bash
python3 -m py_compile scripts/run_reconstruct_and_eval.py
```

**Help Output:**
```bash
python scripts/run_reconstruct_and_eval.py --help
```

**Dry Run (no adapter):**
```bash
make recon-eval LIMIT=4
```

**Dry Run (with adapter):**
```bash
make recon-eval-adapter LIMIT=4
```

### Comparison Workflow

**Compare No-Adapter vs Adapter:**

```bash
# Run 1: No adapter (512-D)
make recon-eval LIMIT=64

# Run 2: With adapter (1024-D)
make recon-eval-adapter LIMIT=64

# Compare summaries
diff outputs/reports/subj01/recon_eval_summary.md \
     outputs/reports/subj01/recon_eval_summary.md

# Or manually review both files
cat outputs/reports/subj01/*/recon_eval_summary.md
```

**Compare Ridge vs MLP:**

```bash
# Ridge baseline (no adapter)
make recon-eval \
    ENCODER=ridge \
    CKPT=checkpoints/ridge/subj01/ridge_k4_rel0.15.pt \
    LIMIT=64

# MLP encoder (no adapter)
make recon-eval \
    ENCODER=mlp \
    CKPT=checkpoints/mlp/subj01/mlp.pt \
    LIMIT=64
```

### Integration with Existing Scripts

**Orchestrator calls these scripts:**
1. `scripts/check_hf_cache.py` - Verify SD model cached
2. `scripts/decode_diffusion.py` - Generate images
3. `scripts/eval_reconstruction.py` - Compute metrics

**All scripts share:**
- Same index loading (`--index-root` or `--index-file`)
- Same subject specification
- Same limit handling
- Same CLIP cache format

**No code duplication:**
- Orchestrator shells out to existing scripts
- Reuses all existing logic
- Only adds workflow coordination + summary generation

### Tips

**Quick Testing:**
- Use `LIMIT=4` for fast iteration
- Use `LIMIT=64` for thesis experiments
- Use `LIMIT=256` for paper-quality results

**Thesis Integration:**
- Copy Markdown summary directly into thesis
- Adjust interpretation text if needed
- Include visualization grid PNG in figures

**Debugging:**
- Check individual step outputs if orchestrator fails
- Run decode and eval separately to isolate issues
- Verify SD cache with `make check-sd`

**Performance:**
- Decode is slowest step (~2-5 sec/image on GPU)
- Evaluation is fast (~1 sec total for 64 samples)
- Summary generation is instant

---

## 5. scripts/compare_evals.py - Aggregate Multiple Evaluations

**Purpose**: Discover multiple evaluation JSONs, compute bootstrap 95% confidence intervals, and generate comprehensive comparison reports (CSV, LaTeX, Markdown, plots).

### Features

**✅ Automatic Discovery:**
- Recursively scans directory for evaluation JSONs
- Configurable glob pattern (default: `recon_eval*.json`)
- Handles multiple runs (no-adapter, adapter, different encoders)

**✅ Bootstrap Confidence Intervals:**
- 95% CIs computed from per-sample metrics
- Nonparametric bootstrap with 1000 resamples (configurable)
- Fixed seed (42) for reproducibility
- Falls back to point estimates if CSV unavailable

**✅ Multi-Format Output:**
- **CSV**: Complete metrics with CI bounds
- **LaTeX**: Thesis-ready table with formatted CIs
- **Markdown**: Comparison summary with interpretation
- **Plots**: Bar charts (CLIPScore, R@1) with error bars

**✅ Intelligent Sorting:**
- Sorted by: adapter (desc) → dimension (desc) → R@1 (desc)
- Easy to identify best-performing configurations

**✅ Automatic Interpretation:**
- Highlights best CLIPScore and R@1 runs
- Computes adapter improvement percentage
- Notes on space consistency

### Usage

**Basic (scan all JSONs in subject directory):**
```bash
make compare-evals

# Custom subject
make compare-evals SUBJECT=subj02
```

**Direct invocation:**
```bash
python scripts/compare_evals.py \
    --report-dir outputs/reports/subj01 \
    --out-csv outputs/reports/subj01/recon_compare.csv \
    --out-tex outputs/reports/subj01/recon_compare.tex \
    --out-md outputs/reports/subj01/recon_compare.md \
    --out-fig outputs/reports/subj01/recon_compare.png
```

**Custom pattern:**
```bash
make compare-evals PATTERN="*eval*.json" BOOTS=2000
```

### Workflow

**Step 1: Discover JSONs**
- Recursively globs `--report-dir` with `--pattern`
- Sorts paths for reproducibility
- Exits if no JSONs found

**Step 2: Load and Parse**
- Loads each JSON with `load_eval_json()`
- Extracts metadata: clip_space, use_adapter, encoder, n_samples
- Extracts metrics: clipscore, R@1/5/10, mean_rank, MRR

**Step 3: Bootstrap CIs**
- Loads per-sample CSV if available
- Computes bootstrap 95% CIs for:
  - CLIPScore (from per-sample cosine similarities)
  - R@1/5/10 (from per-sample binary success indicators)
  - MRR (from per-sample reciprocal ranks)
- Falls back to ±std if CSV missing

**Step 4: Aggregate**
- Creates tidy DataFrame (1 row per run)
- Sorts by adapter → dimension → R@1
- Includes run metadata and all metrics with CIs

**Step 5: Generate Outputs**
- CSV with all columns
- LaTeX table with formatted CIs
- Markdown with interpretation
- Bar plots (2 panels: CLIPScore, R@1)

### Output Formats

#### CSV Example
```csv
run_name,encoder,use_adapter,clip_space,clip_dim,n_samples,clipscore_mean,clipscore_ci_low,clipscore_ci_high,r1,r1_ci_low,r1_ci_high,...
auto_with_adapter,mlp,True,1024-D (target),1024,64,0.654,0.613,0.695,0.543,0.502,0.584,...
auto_no_adapter,mlp,False,512-D (base),512,64,0.612,0.571,0.653,0.487,0.446,0.528,...
```

#### LaTeX Table Example
```latex
\begin{table}[htbp]
\centering
\caption{Reconstruction Evaluation Comparison with 95\% Bootstrap Confidence Intervals}
\label{tab:recon_comparison}
\begin{tabular}{lcccccccc}
\hline
Run & CLIP Space & n & CLIPScore & R@1 & R@5 & R@10 & MRR \\
\hline
auto\_with\_adapter & 1024D (target) & 64 & 0.654 ± 0.041 & 0.543 ± 0.042 & 0.812 ± 0.039 & 0.891 ± 0.031 & 0.612 ± 0.045 \\
auto\_no\_adapter & 512D (base) & 64 & 0.612 ± 0.041 & 0.487 ± 0.041 & 0.765 ± 0.042 & 0.843 ± 0.037 & 0.571 ± 0.043 \\
\hline
\end{tabular}
\end{table}
```

#### Markdown Summary Example
```markdown
# Reconstruction Evaluation Comparison

## Evaluated Runs

- **auto_with_adapter**: 1024-D (target), with adapter, encoder=mlp, n=64
- **auto_no_adapter**: 512-D (base), no adapter, encoder=mlp, n=64

---

## Metrics with 95% Bootstrap Confidence Intervals

| Run | CLIP Space | n | CLIPScore | R@1 | R@5 | R@10 | MRR |
|-----|------------|---|-----------|-----|-----|------|-----|
| auto_with_adapter | 1024-D (target) | 64 | 0.654 ± 0.041 | 0.543 ± 0.042 | 0.812 ± 0.039 | 0.891 ± 0.031 | 0.612 ± 0.045 |
| auto_no_adapter | 512-D (base) | 64 | 0.612 ± 0.041 | 0.487 ± 0.041 | 0.765 ± 0.042 | 0.843 ± 0.037 | 0.571 ± 0.043 |

---

## Interpretation

**Best R@1:** auto_with_adapter (0.543) — 1024-D (target), with adapter. 
**Best CLIPScore:** auto_with_adapter (0.654) — 1024-D (target), with adapter. 
Using the CLIP adapter in target space improved average R@1 by 11.5% (0.487 → 0.543).

---

**Note:** Evaluation CLIP space matches generation space where adapter was used. 
Comparisons across different CLIP dimensions should be interpreted cautiously, 
as they represent different semantic spaces.

**Confidence Intervals:** 95% bootstrap CIs computed from per-sample metrics 
using 1000 resamples with replacement.
```

#### Visualization (PNG)

Two panels stacked vertically:
- **Panel A**: CLIPScore comparison with error bars (95% CI)
- **Panel B**: R@1 comparison with error bars (95% CI)

Each panel includes:
- Bar plot with one bar per run
- Error bars showing 95% CI
- Y-axis grid for readability
- Rotated x-axis labels (run names)

### Bootstrap Methodology

**Algorithm:**
1. Load per-sample CSV (if available)
2. Extract per-sample values (e.g., `clipscore` column)
3. For each of B=1000 bootstrap iterations:
   - Resample n values with replacement
   - Compute mean of resampled values
4. Compute 2.5th and 97.5th percentiles of bootstrap distribution
5. Report as 95% CI: [p2.5, p97.5]

**Metrics:**
- **CLIPScore**: Bootstrap over per-sample cosine similarities
- **R@1/5/10**: Bootstrap over per-sample binary success (0 or 1)
- **MRR**: Bootstrap over per-sample reciprocal ranks (1/rank)

**Reproducibility:**
- Fixed random seed: 42
- Same seed for all runs
- Deterministic results

**Fallback:**
- If CSV missing: Use point estimate ± std (not bootstrap)
- Marked as "NA" in CI columns if no per-sample data

### Arguments

**Required:**
- `--report-dir`: Root directory to scan (e.g., `outputs/reports/subj01`)
- `--out-csv`: Output CSV path
- `--out-tex`: Output LaTeX table path
- `--out-md`: Output Markdown summary path
- `--out-fig`: Output figure path (PNG)

**Optional:**
- `--pattern`: Glob pattern (default: `recon_eval*.json`)
- `--metrics`: Comma-separated metrics (default: `clipscore_mean,R@1,R@5,R@10`)
- `--boots`: Bootstrap resamples (default: 1000)

### Example Workflow

**Generate multiple evaluations:**
```bash
# No adapter (512-D)
make recon-eval LIMIT=64

# With adapter (1024-D)
make recon-eval-adapter LIMIT=64

# Ridge baseline
make recon-eval ENCODER=ridge CKPT=checkpoints/ridge/subj01/ridge.pt LIMIT=64
```

**Compare all:**
```bash
make compare-evals
```

**Check outputs:**
```bash
cat outputs/reports/subj01/recon_compare.md
open outputs/reports/subj01/recon_compare.png
```

### Guardrails

**1. Space Consistency Warning:**
- Markdown explicitly notes evaluation space matches generation
- Warns about cross-dimensional comparisons
- Includes footnote on interpretation

**2. Missing Data Handling:**
- Continues if some JSONs fail to load
- Reports which runs lack per-sample CSVs
- Marks CI as "NA" when bootstrap impossible

**3. Sorting Logic:**
- Adapter runs listed first (typically better)
- Higher dimensions listed first
- Best R@1 within each group

**4. Reproducibility:**
- Fixed random seed (42)
- Deterministic bootstrap
- Sorted JSON discovery

**5. Error Handling:**
- Exits non-zero if no JSONs found
- Logs errors for individual runs
- Continues aggregation despite failures

### Fair Comparison Guidelines

**✅ Valid Comparisons:**
- Same CLIP space (all 512-D or all 1024-D)
- Same test set (same `--limit`)
- Same evaluation protocol

**⚠️ Caution Required:**
- Cross-dimensional (512-D vs 1024-D)
  - Different semantic spaces
  - CLIPScore not directly comparable
  - Note this in interpretation
- Different test set sizes
  - Bootstrap CIs have different power
  - Mention in footnote

**❌ Invalid Comparisons:**
- Mixing generation spaces in evaluation
  - Generated with adapter but evaluated in 512-D
  - This would be caught by orchestrator

### Scientific Context

**Bootstrap Confidence Intervals:**
- Standard method for non-parametric CI estimation
- Does not assume normal distribution
- Robust to outliers
- Widely accepted in machine learning literature

**Metrics Context:**
- **CLIPScore**: Hessel et al. (2021) - semantic similarity
- **Retrieval@K**: Standard IR metric adapted for reconstruction
- **MRR**: Classic information retrieval metric

**Interpretation:**
- Higher CLIPScore = better semantic preservation
- Higher R@1 = better one-to-one matching
- Trade-off between metrics common (diffusion generates novel images)

### Helper Module: `scripts/_report_utils.py`

**Functions:**

1. **`load_eval_json(path) -> dict`**
   - Loads and validates JSON
   - Returns parsed dictionary

2. **`guess_run_name(path) -> str`**
   - Extracts meaningful name from path
   - Heuristics: adapter, encoder, dimension

3. **`bootstrap_ci(values, boots=1000, alpha=0.05) -> (low, high)`**
   - Nonparametric bootstrap
   - Fixed seed (42)
   - Returns CI bounds

4. **`format_mean_ci(mean, low, high) -> str`**
   - Formats as "mean ± half_width"
   - Symmetric CI (conservative)

5. **`format_mean_ci_range(mean, low, high) -> str`**
   - Formats as "mean [low, high]"
   - Asymmetric CI (explicit bounds)

---

## Status

✅ **Implementation Complete**  
✅ **Documentation Complete**  
✅ **Ready for Testing** (requires: `pip install matplotlib seaborn scikit-learn`)

**Next Steps**:

1. Install dependencies: `pip install matplotlib seaborn scikit-learn`
2. Test `report_ablation.py` on existing CSV
3. Test `reconstruct_nn.py` with Ridge checkpoint
4. Run full ablation + reconstruction for paper
5. Test one-click orchestrator: `make recon-eval LIMIT=4`
6. **NEW:** Compare multiple evaluations: `make compare-evals`
```
