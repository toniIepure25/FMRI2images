# CLIP Adapter Training Guide

## Problem

The CLIP adapter training requires loading ~9000 NSD images to compute target CLIP embeddings. The images are stored in a large HDF5 file (~39GB) on S3, which can be slow or fail to download.

## Solutions

### Solution 1: Pre-build Target Cache (Recommended for Full Pipeline)

Run the target cache builder separately before the full pipeline:

```bash
# Build target cache with smaller batches (resumable)
python scripts/build_target_clip_cache_robust.py \
    --subject subj01 \
    --index-root data/indices/nsd_index \
    --model-id stabilityai/stable-diffusion-2-1 \
    --output outputs/clip_cache/target_clip_sd21.parquet \
    --batch-size 500 \
    --inference-batch-size 32 \
    --device cuda
```

This script:
- Downloads and caches the HDF5 file incrementally
- Saves progress after each batch
- Can be resumed if interrupted
- Takes ~2-3 hours for 9000 images

### Solution 2: Skip Adapter Training (Quick Results)

The adapter improves quality by ~10-15%, but you can skip it for faster results:

1. Use SD-2.1 directly with zero-padding (512-D → 1024-D)
2. Expected CLIPScore: ~0.60-0.62 (vs 0.68-0.70 with adapter)
3. The production script automatically detects if adapter exists

### Solution 3: Train with Limited Dataset

Train adapter on subset for faster turnaround:

```bash
python scripts/train_clip_adapter.py \
    --clip-cache outputs/clip_cache/subj01_clip512.parquet \
    --out checkpoints/clip_adapter/subj01/adapter.pt \
    --model-id stabilityai/stable-diffusion-2-1 \
    --epochs 30 \
    --batch-size 128 \
    --limit 1000 \
    --device cuda
```

This trains on 1000 samples instead of 9000, taking ~30 minutes.

## Production Script Behavior

The `run_production.sh` script intelligently handles adapter training:

1. **Check existing adapter**: If checkpoint exists, skip training
2. **Check target cache**: If sufficient, train with cache (fast)
3. **Fallback**: Train with limited samples if cache incomplete
4. **Skip gracefully**: Continue without adapter if training fails

## Expected Quality Improvements

| Configuration | CLIPScore | R@1 (test) | Training Time |
|--------------|-----------|------------|---------------|
| Baseline (SD-1.5, no adapter) | 0.556 | 0% | - |
| SD-2.1, no adapter | 0.60-0.62 | 3-5% | - |
| SD-2.1, with adapter | 0.68-0.72 | 8-12% | 2-3 hours |
| SD-2.1, adapter (subset) | 0.64-0.68 | 5-8% | 30 mins |

## Troubleshooting

### HDF5 Download Fails
- Error: "truncated file" or "Unable to synchronously open file"
- Solution: Delete corrupted cache file:
  ```bash
  rm cache/s3_cache/*
  ```
  Then retry with smaller batch size

### Out of Memory
- Reduce `--inference-batch-size` from 32 to 16 or 8
- Reduce training `--batch-size` from 128 to 64

### Slow Progress
- First run will be slow (downloads HDF5)
- Subsequent runs use cached file (~5.2GB local)
- Use `--limit` to test with smaller dataset first
