# Preventing Model Download Blocking

## The Problem

When running `decode_diffusion.py` without `--test-mode`, the script downloads the Stable Diffusion model (~5GB, 13 files) from HuggingFace. This can take 20-30 minutes and appears to "block" or hang with a progress bar.

**Why it happens:**

```python
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
# ↑ Downloads ~5GB model if not cached
```

---

## Solutions

### ✅ Solution 1: Pre-download the Model (RECOMMENDED)

**Run once to download, then use forever:**

```bash
# Download the model to cache
python scripts/download_sd_model.py

# Now this command will be instant (no download)
python scripts/decode_diffusion.py [your args here]
```

**What it does:**

- Downloads model to `~/.cache/huggingface/hub/`
- Takes 20-30 minutes (one time only)
- All future runs use the cached model instantly

**Benefits:**

- ✅ Separates download from processing
- ✅ Clear progress indication
- ✅ Can be interrupted and resumed
- ✅ Cached model works for all scripts

---

### ✅ Solution 2: Use Test Mode (NO IMAGES)

**Quick validation without model download:**

```bash
python scripts/decode_diffusion.py \
    --encoder ridge \
    --ckpt checkpoints/ridge/subj01/ridge.pkl \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --preproc-dir outputs/preproc/subj01/test_k4 \
    --test-mode  # ← Skip diffusion, test encoder only
```

**What it does:**

- Tests encoder pipeline
- Predicts CLIP embeddings
- Saves predictions to JSON
- **Skips image generation** (no model download)

**Benefits:**

- ✅ Instant (10 seconds)
- ✅ Validates entire pipeline except diffusion
- ✅ Shows cosine similarity scores

---

### ⚡ Solution 3: Use a Smaller/Cached Model

**If you already have a model downloaded:**

```bash
# Check what models you have cached
ls ~/.cache/huggingface/hub/

# Use a different model that might be cached
python scripts/decode_diffusion.py \
    --model-id runwayml/stable-diffusion-v1-5 \  # Alternative model
    [other args]
```

**Alternative models:**

- `stabilityai/stable-diffusion-2-1` - Default, best quality (~5GB)
- `runwayml/stable-diffusion-v1-5` - Slightly faster (~4GB)
- `stabilityai/stable-diffusion-2-1-base` - Base version (~5GB)

---

### 🔧 Solution 4: Download in Background

**Terminal 1: Download model**

```bash
python scripts/download_sd_model.py
# Let this run in background (20-30 min)
```

**Terminal 2: Continue other work**

```bash
# Run other scripts, tests, etc.
python scripts/report_ablation.py
python scripts/reconstruct_nn.py --test-mode
```

Once download completes, run full diffusion script.

---

## Understanding the Download Process

### What Gets Downloaded?

Stable Diffusion model consists of 13 files (~5GB total):

```
text_encoder/          # CLIP text encoder
vae/                   # VAE for image encoding/decoding
unet/                  # Main denoising network
scheduler/             # Noise scheduler
tokenizer/             # Text tokenizer
model_index.json       # Model configuration
```

### Download Progress

You'll see:

```
Fetching 13 files: 31% |████████| 4/13 [23:36<53:08, 354.24s/it]
```

This is **normal** - large files take time!

### Cache Location

Models are cached at:

```
~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/
```

Once cached, the script checks for the model and loads instantly.

---

## Workflow Recommendations

### First Time Setup

```bash
# 1. Pre-download model (one time, 20-30 min)
python scripts/download_sd_model.py

# 2. Test encoder pipeline (instant)
python scripts/decode_diffusion.py \
    --encoder ridge \
    --ckpt checkpoints/ridge/subj01/ridge.pkl \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --preproc-dir outputs/preproc/subj01/test_k4 \
    --test-mode

# 3. Generate images (instant after model is cached)
python scripts/decode_diffusion.py \
    --encoder ridge \
    --ckpt checkpoints/ridge/subj01/ridge.pkl \
    --clip-cache outputs/clip_cache/clip.parquet \
    --use-preproc \
    --preproc-dir outputs/preproc/subj01/test_k4 \
    --steps 50 \
    --limit 16
```

### Daily Development

```bash
# Model is cached, so this is instant:
python scripts/decode_diffusion.py [args]
```

---

## Troubleshooting

### "The download is stuck at 0%"

- **Not stuck** - large files take time to start
- Wait 2-3 minutes before checking progress
- Check network connection: `ping huggingface.co`

### "It's been downloading for 30+ minutes"

- **Expected** for first download
- Check disk space: `df -h ~` (need ~5GB free)
- Check download speed: slower networks take longer

### "Download failed/interrupted"

- **Resume automatically** - HuggingFace caches partial downloads
- Just re-run the command, it picks up where it left off

### "Out of disk space"

- Clear old models: `rm -rf ~/.cache/huggingface/hub/models--*`
- Or use external disk: set `HF_HOME=/path/to/large/disk`

---

## 🤖 Solution 5: Fail-Fast Mode for CI/Scripts

**For automated environments where downloads should never happen:**

```bash
python scripts/decode_diffusion.py \
    --encoder ridge \
    --ckpt checkpoints/ridge/subj01/ridge.pkl \
    --fail-if-missing-model \  # ← Exit immediately if model not cached
    [other args]
```

**What it does:**

- Checks cache **before** attempting any download
- Exits with code 2 if model is missing
- Prints actionable one-line fix: `make download-sd`

**Use cases:**

- ✅ CI/CD pipelines (prevent accidental long downloads)
- ✅ Batch processing scripts (fail fast if model not ready)
- ✅ Scheduled jobs (don't block, just exit)

**Exit codes:**

- `0` - Success (model cached, images generated)
- `1` - Error (processing failure, bad args, etc.)
- `2` - Model not cached (need to run `make download-sd` first)

**Example CI check:**

```bash
# Check if model is ready before running full pipeline
make check-sd || { echo "Model not ready, run: make download-sd"; exit 1; }

# Run with fail-fast flag
python scripts/decode_diffusion.py --fail-if-missing-model [args]
```

---

## Alternative: Offline Mode

If you have the model on another machine:

```bash
# Machine 1 (with internet): Download model
python scripts/download_sd_model.py

# Copy cache to machine 2 (no internet)
scp -r ~/.cache/huggingface/hub/ user@machine2:~/.cache/huggingface/

# Machine 2: Use cached model offline
export HF_HUB_OFFLINE=1
python scripts/decode_diffusion.py [args]
```

---

## Summary

| Method           | Time           | Pros                 | Cons                        |
| ---------------- | -------------- | -------------------- | --------------------------- |
| **Pre-download** | 20-30 min once | Best for production  | Requires patience           |
| **Test mode**    | 10 sec         | Instant validation   | No images generated         |
| **Background**   | 20-30 min      | Can work meanwhile   | Need multiple terminals     |
| **Cached model** | Instant        | Fast after first run | Still need initial download |

**Recommendation:** Use **pre-download** (`download_sd_model.py`) for the best experience. Run it once, then enjoy instant image generation forever!
