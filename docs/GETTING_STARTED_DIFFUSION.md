# Getting Started with Diffusion Image Generation

## TL;DR (Three Commands)

```bash
# 1. Download model (ONE TIME, ~5GB, 10-30 min)
make download-sd

# 2. Test encoder pipeline (INSTANT, no diffusion)
python scripts/decode_diffusion.py --encoder ridge --ckpt checkpoints/ridge/subj01/ridge.pkl --test-mode [args]

# 3. Generate images (INSTANT after step 1)
python scripts/decode_diffusion.py --encoder ridge --ckpt checkpoints/ridge/subj01/ridge.pkl [args]
```

---

## Quick Start (3 Steps)

### Step 1: Pre-download the Model (One Time - 20-30 min)

```bash
# Option A: Using make
make download-sd

# Option B: Direct script
python scripts/download_sd_model.py
```

**Wait for completion** - you'll see:

```
✅ DOWNLOAD COMPLETE
Model cached at: ~/.cache/huggingface/hub/...
Total files: 13
Total size: 5.1 GB
```

### Step 2: Test the Pipeline (10 seconds)

```bash
python scripts/decode_diffusion.py \
    --encoder ridge \
    --ckpt checkpoints/ridge/subj01/ridge.pkl \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --preproc-dir outputs/preproc/subj01/test_k4 \
    --test-mode
```

**Expected output:**

```
✅ Mean cosine similarity: 0.7848
✅ Test results saved to outputs/recon/subj01/ridge_diffusion/test_predictions.json
```

### Step 3: Generate Images (Instant after Step 1)

```bash
python scripts/decode_diffusion.py \
    --encoder ridge \
    --ckpt checkpoints/ridge/subj01/ridge.pkl \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --preproc-dir outputs/preproc/subj01/test_k4 \
    --device cuda \
    --dtype float16 \
    --steps 50 \
    --limit 16
```

**Output:** Images at `outputs/recon/subj01/ridge_diffusion/images/*.png`

---

## Why This Works

### The Problem (Without Pre-download)

```
1. Run decode_diffusion.py
2. Script tries to download 5GB model
3. Takes 20-30 minutes
4. Appears to "hang" or block
5. User gets frustrated and cancels
```

### The Solution (With Pre-download)

```
1. Run download_sd_model.py (once)
2. Model downloads to cache
3. All future runs are instant
4. No blocking, no waiting
5. Generate images immediately
```

---

## Files Created

After running the 3 steps, you'll have:

```
~/.cache/huggingface/hub/
└── models--stabilityai--stable-diffusion-2-1/  # Cached model (~5GB)

outputs/recon/subj01/ridge_diffusion/
├── test_predictions.json                        # Test mode results
├── images/
│   ├── nsd_0002_generated.png                  # Generated image
│   ├── nsd_0013_generated.png
│   └── ...
├── grids/
│   ├── nsd_0002_comparison.png                 # Side-by-side comparison
│   └── ...
└── decode_summary.json                          # Metadata
```

---

## Common Questions

### Q: Can I skip Step 1?

**A:** No - the model must be downloaded. But you only do it once!

### Q: What if Step 1 takes too long?

**A:** Run it overnight or in a screen/tmux session:

```bash
screen -S download
python scripts/download_sd_model.py
# Press Ctrl+A then D to detach
# Later: screen -r download to check progress
```

### Q: Can I use a different model?

**A:** Yes! Change the model in Step 1:

```bash
python scripts/download_sd_model.py --model-id runwayml/stable-diffusion-v1-5
```

Then use the same model ID in Step 3:

```bash
python scripts/decode_diffusion.py --model-id runwayml/stable-diffusion-v1-5 [...]
```

### Q: What if I don't have GPU?

**A:** Use CPU mode (slower but works):

```bash
--device cpu --dtype float32
```

Expect ~5 seconds per image instead of <1 second.

### Q: How do I know if the model is cached?

**A:** Check the cache directory:

```bash
ls ~/.cache/huggingface/hub/
# Look for: models--stabilityai--stable-diffusion-2-1
```

---

## Troubleshooting

### "ModuleNotFoundError: diffusers"

```bash
pip install diffusers transformers accelerate
```

### "CUDA out of memory"

```bash
# Use CPU instead
python scripts/decode_diffusion.py --device cpu --dtype float32 [...]
```

### "Download keeps failing"

```bash
# Check disk space
df -h ~

# Check network
ping huggingface.co

# Retry - it resumes from where it stopped
python scripts/download_sd_model.py
```

---

## Summary

**The key to preventing blocking:**

1. **Pre-download once** with `download_sd_model.py` (20-30 min)
2. **Test with** `--test-mode` (10 sec)
3. **Generate images** instantly (model is cached)

**Never wait again!** Once the model is cached, image generation is instant.
