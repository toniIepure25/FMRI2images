# Manual Model Download Guide

This guide shows you how to download Hugging Face models manually for faster download speeds.

## Problem: Slow Downloads

When using `transformers` library, downloads can be very slow (~50 KB/s) due to:
- Single connection downloads
- Hugging Face API throttling
- No resume capability by default

## Solution: Git LFS + Concurrent Downloads

Download speeds can be **10-100x faster** using Git LFS with concurrent transfers.

---

## Step 1: Install Git LFS

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
```

---

## Step 2: Download Model Manually

### Create Models Directory
```bash
cd ~
mkdir -p models
cd models
```

### Clone Model Repository
Replace `MODEL_ID` with the Hugging Face model ID:

```bash
# Example: BLIP-2
export MODEL_ID="Salesforce/blip2-opt-2.7b"

# Clone structure first (fast)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/$MODEL_ID

# Enter directory
cd $(basename $MODEL_ID)

# Configure concurrent downloads (8 parallel connections)
git config lfs.concurrenttransfers 8

# Download large files
git lfs pull
```

**Download Speed Comparison:**
- Default: ~50 KB/s (62 hours for 15GB)
- Git LFS (1 connection): ~1-2 MB/s (~2 hours)
- Git LFS (8 connections): ~4-8 MB/s (~30-60 minutes) ✅

---

## Step 3: Link to Hugging Face Cache

Once downloaded, create a symlink so transformers can find it:

```bash
# Find Hugging Face cache directory
export HF_HOME="${HOME}/.cache/huggingface"
mkdir -p "$HF_HOME/hub"

# Create symlink
cd "$HF_HOME/hub"
ln -s ~/models/blip2-opt-2.7b models--Salesforce--blip2-opt-2.7b
```

**Alternative: Use Local Path**

Or modify your script to load from local path:

```python
model = Blip2ForConditionalGeneration.from_pretrained(
    "~/models/blip2-opt-2.7b",  # Local path
    local_files_only=True
)
```

---

## Step 4: Verify Installation

Test that the model loads:

```bash
cd ~/Desktop/Bachelor\ V2
source .venv/bin/activate

python -c "
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

print('Loading model...')
processor = Blip2Processor.from_pretrained('~/models/blip2-opt-2.7b', local_files_only=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    '~/models/blip2-opt-2.7b',
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map='cuda'
)
print('✅ Model loaded successfully!')
"
```

---

## Common Models for This Project

### BLIP (Fast, Good Quality)
```bash
cd ~/models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Salesforce/blip-image-captioning-large
cd blip-image-captioning-large
git config lfs.concurrenttransfers 8
git lfs pull
```

**Size**: ~1.88 GB  
**Download time**: 5-10 minutes (with 8 connections)  
**Use case**: Good captions, fast inference

### BLIP-2 (Best Quality)
```bash
cd ~/models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Salesforce/blip2-opt-2.7b
cd blip2-opt-2.7b
git config lfs.concurrenttransfers 8
git lfs pull
```

**Size**: ~14.9 GB  
**Download time**: 30-60 minutes (with 8 connections)  
**Use case**: Best captions, slower inference

### CLIP (Required)
```bash
# CLIP models are small and download via open_clip, no manual download needed
```

---

## Using Downloaded Models

### Option 1: Modify build_text_clip_cache.py

Update the model loading to use local path:

```python
# In load_captioning_model() function
if model_name == "blip2":
    model_path = "~/models/blip2-opt-2.7b"  # Local path
    processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map=device
    )
```

### Option 2: Set HF_HOME Environment Variable

Point Hugging Face to your models directory:

```bash
export HF_HOME=~/models
python scripts/build_text_clip_cache.py ...
```

### Option 3: Symlink (Recommended)

Create proper Hugging Face cache structure:

```bash
mkdir -p ~/.cache/huggingface/hub
cd ~/.cache/huggingface/hub
ln -s ~/models/blip2-opt-2.7b models--Salesforce--blip2-opt-2.7b
```

---

## Monitoring Download Progress

While downloading:

```bash
# Watch download progress
watch -n 1 'du -sh ~/models/blip2-opt-2.7b'

# Check network speed
sudo apt-get install -y nethogs
sudo nethogs
```

---

## Troubleshooting

### "Out of disk space"
Check available space:
```bash
df -h ~
```

BLIP-2 requires ~15 GB free space.

### "Permission denied"
```bash
chmod -R u+w ~/models
```

### "Git LFS not found"
```bash
git lfs install
```

### Resume Interrupted Download
Git LFS automatically resumes:
```bash
cd ~/models/blip2-opt-2.7b
git lfs pull
```

---

## Speed Comparison Summary

| Method | Speed | Time (15GB) |
|--------|-------|-------------|
| transformers default | 50 KB/s | 62 hours |
| Git LFS (1 thread) | 1-2 MB/s | 2-4 hours |
| Git LFS (8 threads) | 4-8 MB/s | 30-60 min ✅ |
| Direct download (browser) | Varies | 20-60 min |

---

## Alternative: Download via Browser

1. Go to: https://huggingface.co/Salesforce/blip2-opt-2.7b/tree/main
2. Click "Files and versions"
3. Download these files manually:
   - `model-00001-of-00002.safetensors` (~10 GB)
   - `model-00002-of-00002.safetensors` (~5 GB)
   - `config.json`
   - `preprocessor_config.json`
   - All tokenizer files

4. Place in `~/models/blip2-opt-2.7b/`

**Note**: Browser download may be faster or slower depending on your connection and Hugging Face's CDN.

---

## Current Download Status

The download is currently running in the terminal. To check progress:

```bash
# Terminal ID: 2f96b10a-6284-4e22-9d6e-9c42bbf11634
# Monitor with:
cd ~/models/blip2-opt-2.7b && du -sh .
```

Once complete, the model will be at `~/models/blip2-opt-2.7b/` and ready to use!
