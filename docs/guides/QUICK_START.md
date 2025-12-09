# Quick Start Guide: Publication Pipeline

This guide provides a complete workflow for running the fMRI→CLIP→Diffusion reconstruction pipeline from start to finish.

## Prerequisites

```bash
# 1. Activate environment
conda activate fmri2img

# 2. Verify NSD dataset is accessible
ls data/indices/nsd_index/
# Should contain: subject=subj01/, subject=subj02/, subject=subj03/

# 3. Verify stimulus cache exists
ls cache/nsd_hdf5/nsd_stimuli.hdf5
# or
ls cache/nsd_png/ | head
```

## Option 1: Complete Automated Pipeline (Recommended)

```bash
# Run everything end-to-end
make pipeline
```

This single command will:
1. Build 1024-D CLIP cache (~2-3 hours, one-time)
2. Reconstruct test sets for subj01-03 (~4-6 hours per subject)
3. Evaluate with 3 gallery types (~30 min per subject)
4. Generate summary CSV and Markdown report (~1 min)
5. Create publication figures (~2 min)

**Total time: ~15-20 hours** (most time is reconstruction)

## Option 2: Step-by-Step (For Development/Debugging)

### Step 1: Build CLIP Caches (One-Time Setup)

```bash
# Build 1024-D target cache for SD 2.1
make build_target_cache
# Time: ~2-3 hours
# Output: outputs/clip_cache/target_clip_stabilityai_stable-diffusion-2-1.parquet
```

### Step 2: Reconstruct Test Sets

```bash
# Single subject (development)
make reconstruct_subj01
# Time: ~4-6 hours per subject

# All subjects (production)
make reconstruct_all
# Time: ~15-18 hours total
```

### Step 3: Evaluate Reconstructions

```bash
# Quick test (50 samples, matched gallery only)
make eval_quick
# Time: ~2 min

# Single subject with all galleries
make eval_full_subj01
# Time: ~10 min per subject

# All subjects with all galleries (production)
make eval_all_subjects
# Time: ~30 min total
```

### Step 4: Generate Reports

```bash
# Aggregate all evaluation JSONs into summary CSV
make summarize_reports
# Time: <1 min
# Output: outputs/reports/summary_by_subject.csv
#         outputs/reports/SUMMARY.md

# Generate publication figures
make generate_figures
# Time: ~2 min
# Output: outputs/reports/figures/*.png
```

## Option 3: Single Subject Quick Test

For rapid iteration during development:

```bash
# Minimal test with 50 samples
SUBJECT=subj01 LIMIT=50 make eval_quick

# Check outputs
ls outputs/reports/subj01/eval_quick*
# Should see: .csv, .json, .png, __nn.jsonl, __clipscore_hist.png, __rank_hist.png
```

## Output Locations

After running the pipeline, you will find:

```
outputs/
├── clip_cache/
│   └── target_clip_stabilityai_stable-diffusion-2-1.parquet  # 1024-D CLIP cache
│
├── recon/
│   ├── subj01/ridge_diffusion/images/*.png                   # Reconstructed images
│   ├── subj02/ridge_diffusion/images/*.png
│   └── subj03/ridge_diffusion/images/*.png
│
└── reports/
    ├── summary_by_subject.csv                                # Aggregate metrics CSV
    ├── SUMMARY.md                                            # Markdown report
    ├── figures/                                              # Publication figures
    │   ├── clipscore_by_subject.png
    │   ├── clipscore_combined.png
    │   ├── rank_distribution.png
    │   ├── retrieval_comparison.png
    │   └── adapter_ablation.png
    │
    ├── subj01/
    │   ├── eval_matched.{csv,json,__nn.jsonl}                # Matched gallery
    │   ├── eval_test.{csv,json,__nn.jsonl}                   # Test gallery
    │   ├── eval_all.{csv,json,__nn.jsonl}                    # All gallery
    │   ├── eval_matched_grid.png                             # Visualization
    │   ├── eval_matched__clipscore_hist.png                  # Distributions
    │   └── eval_matched__rank_hist.png
    │
    ├── subj02/...
    └── subj03/...
```

## Understanding Evaluation Metrics

### Per-Sample CSV Columns

- `nsdId`: NSD stimulus ID
- `clipscore`: Cosine similarity between generated and GT CLIP embeddings
- `rank`: Position of GT in retrieval ranking (1-based)
- `r@1`, `r@5`, `r@10`: Binary indicators (1 if GT in top-K, 0 otherwise)
- `in_gallery`: Whether GT was present in retrieval gallery
- `nn_nsdId`: NSD ID of top-1 nearest neighbor
- `nn_sim`: Similarity to top-1 neighbor
- `gt_sim`: Similarity to own ground truth (same as clipscore)

### Aggregate JSON Structure

```json
{
  "subject": "subj01",
  "gallery_type": "matched",
  "n_samples": 515,
  "gallery_size": 515,
  "retrieval_eligible": 515,
  
  "clipscore": {
    "mean": 0.524,
    "std": 0.108,
    "min": 0.201,
    "max": 0.812
  },
  
  "retrieval": {
    "R@1": 0.452,
    "R@5": 0.712,
    "R@10": 0.823
  },
  
  "ranking": {
    "mean_rank": 8.3,
    "median_rank": 2.0,
    "mrr": 0.561
  },
  
  "top1_mean_sim": 0.587,
  
  "rank_hist": {
    "1": 233,
    "2-5": 134,
    "6-10": 89,
    "11+": 59
  },
  
  "ablations": {
    "with_adapter": {...},
    "without_adapter": {...}
  }
}
```

### Gallery Types Explained

1. **matched** (Standard Evaluation)
   - Gallery = Only GT of reconstructed images
   - Easiest condition
   - Standard benchmark metric
   - Use this for primary comparisons

2. **test** (Medium Difficulty)
   - Gallery = All test split GT embeddings
   - Includes non-reconstructed test images
   - More realistic retrieval scenario
   - Tests generalization within test distribution

3. **all** (Hardest, Most Realistic)
   - Gallery = All GT embeddings (train+val+test)
   - Largest gallery (~10k+ images)
   - Most realistic evaluation
   - Best measure of true semantic similarity

**Note:** CLIPScore is always computed against matched GT (not affected by gallery choice). Only retrieval metrics (R@K, ranks) depend on gallery type.

## Troubleshooting

### CUDA Out of Memory

Reduce batch sizes:
```bash
# Reconstruction
python scripts/decode_diffusion.py --batch-size 1 ...

# CLIP cache building
python scripts/nsd_build_clip_cache.py --batch-size 16 ...
```

### Missing NSD Stimuli

If you see "Failed to load visualization image" warnings:
```bash
# Option 1: Use S3 (requires AWS credentials)
python scripts/eval_reconstruction.py --image-source s3 ...

# Option 2: Cache stimuli locally
python scripts/cache_nsd_stimuli_hdf5.py --output cache/nsd_hdf5/nsd_stimuli.hdf5
```

### FAISS Not Available

FAISS is optional. If you see "FAISS not available, falling back to numpy":
```bash
# Install FAISS (optional, for large galleries >10k)
pip install faiss-cpu  # or faiss-gpu
```

## Expected Metrics (Ballpark)

Based on typical fMRI reconstruction performance:

### Matched Gallery
- CLIPScore: 0.45 - 0.55
- R@1: 0.35 - 0.50
- R@5: 0.65 - 0.80
- Mean Rank: 5 - 15

### Test Gallery
- R@1: 0.25 - 0.40 (harder)
- R@5: 0.50 - 0.70
- Mean Rank: 10 - 30

### All Gallery
- R@1: 0.15 - 0.30 (hardest)
- R@5: 0.40 - 0.60
- Mean Rank: 20 - 50

**Note:** Actual metrics depend on:
- Subject-specific brain-behavior relationships
- Data quality and preprocessing choices
- Model hyperparameters
- Number of training samples

## Tips for Best Results

1. **Use HDF5 for visualization** (faster than S3)
   ```bash
   python scripts/cache_nsd_stimuli_hdf5.py --output cache/nsd_hdf5/nsd_stimuli.hdf5
   ```

2. **Monitor GPU usage**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Save reconstruction outputs**
   - Reconstructed images are ~1-2MB each
   - Budget ~1-2GB per subject for test set

4. **Run evaluations in parallel** (if you have multiple GPUs)
   ```bash
   CUDA_VISIBLE_DEVICES=0 make eval_full_subj01 &
   CUDA_VISIBLE_DEVICES=1 make eval_full_subj02 &
   CUDA_VISIBLE_DEVICES=2 make eval_full_subj03 &
   ```

## Next Steps

After generating all outputs:

1. **Review Summary**
   ```bash
   cat outputs/reports/SUMMARY.md
   ```

2. **Inspect Figures**
   ```bash
   xdg-open outputs/reports/figures/clipscore_combined.png
   ```

3. **Export for Paper**
   ```bash
   # Copy key metrics
   cp outputs/reports/summary_by_subject.csv paper/tables/
   cp outputs/reports/figures/*.png paper/figures/
   ```

4. **Iterate on Hyperparameters**
   - Adjust preprocessing (k, reliability threshold)
   - Try different Ridge alpha values
   - Experiment with adapter architectures
   - Tune diffusion guidance scale

---

**For detailed documentation, see:**
- Main README: `README.md`
- Ridge baseline: `docs/RIDGE_BASELINE.md`
- Preprocessing: `docs/PREPROCESSING_IMPLEMENTATION.md`
- Diffusion decoder: `docs/DIFFUSION_DECODER.md`
