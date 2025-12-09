# Inference Configurations

**Image Generation Settings for fMRI-to-Image Reconstruction**

---

## 🚀 Available Configurations

### **Production Configurations**

#### `production.yaml` ⭐ - Balanced Quality/Speed
```yaml
Purpose: Production deployment
Resolution: 512×512
Diffusion Steps: 50
Speed: ~10s per image
Quality: Good
```

**Features**:
- Balanced quality and speed
- Stable Diffusion 2.1
- Guidance scale: 7.5
- No negative prompt (by default)

**When to use**:
- Production deployments
- Demos and presentations
- Interactive applications
- Batch processing (moderate size)

**Command**:
```bash
python scripts/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint checkpoints/two_stage/best.pt \
    --samples 100
```

**Expected Output**:
- Resolution: 512×512 pixels
- Format: PNG
- Quality: Good detail, minimal artifacts
- Speed: ~10 seconds per image (A100 GPU)

---

#### `production_improved.yaml` - Enhanced Production
```yaml
Purpose: Enhanced production quality
Resolution: 512×512
Diffusion Steps: 75
Speed: ~15s per image
Quality: Better
```

**Features**:
- More diffusion steps (75 vs 50)
- Better detail preservation
- Reduced artifacts
- Slightly slower

**When to use**:
- High-quality demos
- Publication figures
- When speed is less critical
- Enhanced visual quality needed

**Command**:
```bash
python scripts/decode_diffusion.py \
    --config configs/inference/production_improved.yaml \
    --checkpoint checkpoints/two_stage/best.pt
```

**Compared to `production.yaml`**:
| Metric | production.yaml | production_improved.yaml |
|--------|----------------|-------------------------|
| Steps | 50 | 75 (+50%) |
| Speed | ~10s | ~15s (+50%) |
| Quality | Good | Better |
| Artifacts | Few | Fewer |

---

### **Speed-Optimized**

#### `fast_inference.yaml` - Rapid Generation
```yaml
Purpose: Fast batch processing
Resolution: 512×512
Diffusion Steps: 25
Speed: ~5s per image
Quality: Fair
```

**Features**:
- Minimal diffusion steps (25)
- Fast generation
- Lower quality but acceptable
- Efficient for large batches

**When to use**:
- Large-scale batch processing
- Quick previews
- Rapid iteration during development
- When throughput > quality

**Command**:
```bash
python scripts/decode_diffusion.py \
    --config configs/inference/fast_inference.yaml \
    --checkpoint checkpoints/two_stage/best.pt \
    --samples 1000  # Process many samples quickly
```

**Trade-offs**:
- ✅ 2× faster than production
- ✅ Suitable for batch jobs
- ❌ Lower quality (visible artifacts)
- ❌ Less detail preservation

**Optimization tips**:
```bash
# Even faster (minimal quality)
--override "generation.num_inference_steps=15"

# Better quality (still faster than production)
--override "generation.num_inference_steps=35"
```

---

### **Quality-Optimized**

#### `highres_quality.yaml` - Maximum Quality
```yaml
Purpose: Publication-quality images
Resolution: 1024×1024
Diffusion Steps: 200
Speed: ~60s per image
Quality: Excellent
```

**Features**:
- High resolution (1024px)
- Many diffusion steps (200)
- Maximum detail preservation
- Minimal artifacts
- Best visual quality

**When to use**:
- Academic publications
- Conference presentations
- Showcase demonstrations
- Research paper figures
- When quality is paramount

**Command**:
```bash
python scripts/decode_diffusion.py \
    --config configs/inference/highres_quality.yaml \
    --checkpoint checkpoints/two_stage/best.pt \
    --samples 50  # Fewer samples due to time
```

**Features**:
```yaml
generation:
  resolution: 1024
  num_inference_steps: 200
  guidance_scale: 8.0
  eta: 0.0  # Deterministic
  
quality_enhancements:
  use_karras_sigmas: true
  use_attention_slicing: true
  enable_vae_tiling: true
```

**Resource Requirements**:
- GPU Memory: 16GB+ VRAM
- Generation Time: ~60 seconds per image
- Disk Space: ~2MB per image (1024px PNG)

---

## 📊 Performance Comparison

| Configuration | Resolution | Steps | Time | Quality | Use Case |
|--------------|-----------|-------|------|---------|----------|
| `fast_inference` | 512×512 | 25 | ~5s | Fair | Batch processing |
| `production` ⭐ | 512×512 | 50 | ~10s | Good | Deployment, demos |
| `production_improved` | 512×512 | 75 | ~15s | Better | High-quality demos |
| `highres_quality` | 1024×1024 | 200 | ~60s | Excellent | Publications |

*Benchmarks on NVIDIA A100 40GB*

---

## 🎯 Recommended Workflow

### **1. Start with Fast Inference** (Preview)
```bash
python scripts/decode_diffusion.py \
    --config configs/inference/fast_inference.yaml \
    --checkpoint checkpoints/two_stage/best.pt \
    --samples 10
```
Quick preview to verify model and embeddings.

### **2. Generate with Production Config** (Main Results)
```bash
python scripts/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint checkpoints/two_stage/best.pt \
    --samples 500
```
Balanced quality/speed for main experiments.

### **3. High-Quality for Selected Samples** (Publication)
```bash
# Select best samples first, then regenerate
python scripts/decode_diffusion.py \
    --config configs/inference/highres_quality.yaml \
    --checkpoint checkpoints/two_stage/best.pt \
    --samples 20  # Only best samples
```
Maximum quality for paper figures.

---

## 🔧 Common Customizations

### **Adjust Diffusion Steps**
```bash
# More steps = better quality, slower
--override "generation.num_inference_steps=100"

# Fewer steps = faster, lower quality
--override "generation.num_inference_steps=30"
```

### **Change Resolution**
```bash
# Higher resolution (requires more memory)
--override "generation.resolution=768"

# Square aspect ratio
--override "generation.width=512" \
--override "generation.height=512"
```

### **Adjust Guidance Scale**
```bash
# Higher = more faithful to prompt, less creative
--override "generation.guidance_scale=9.0"

# Lower = more creative, less controlled
--override "generation.guidance_scale=5.0"
```

### **Enable Advanced Features**
```bash
# Karras noise schedule (better quality)
--override "generation.use_karras_sigmas=true"

# Attention slicing (reduce memory)
--override "generation.enable_attention_slicing=true"

# VAE tiling (enable high-res)
--override "generation.enable_vae_tiling=true"
```

---

## 📝 Advanced Techniques

### **Best-of-N Sampling**
Generate multiple candidates and select best:

```bash
python scripts/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint checkpoints/two_stage/best.pt \
    --best-of-n 5  # Generate 5, keep best
```

### **Negative Prompts**
Improve quality by specifying what to avoid:

```yaml
generation:
  negative_prompt: "blurry, low quality, distorted, deformed"
```

Or at runtime:
```bash
--override "generation.negative_prompt='blurry, low quality'"
```

### **Seed Control**
Reproducible generation:

```bash
# Fixed seed for reproducibility
--override "generation.seed=42"

# Or let it vary
--override "generation.seed=null"
```

### **Batch Processing**
Process multiple subjects efficiently:

```bash
for subject in subj01 subj02 subj03; do
    python scripts/decode_diffusion.py \
        --config configs/inference/production.yaml \
        --checkpoint checkpoints/two_stage/${subject}_best.pt \
        --samples 100 \
        --output outputs/reconstructions/${subject}/
done
```

---

## 🎨 Quality Enhancement Tips

### **For Publications**
```yaml
# configs/inference/my_publication.yaml
_base_: highres_quality.yaml

generation:
  num_inference_steps: 250  # Extra steps
  guidance_scale: 8.5       # Slightly higher
  eta: 0.0                  # Fully deterministic
  use_karras_sigmas: true   # Better noise schedule
```

### **For Real-Time Demos**
```yaml
# configs/inference/my_demo.yaml
_base_: fast_inference.yaml

generation:
  num_inference_steps: 30   # Slightly better than default
  guidance_scale: 7.0       # Balanced
  enable_attention_slicing: true  # Reduce memory
```

### **For Batch Jobs**
```yaml
# configs/inference/my_batch.yaml
_base_: fast_inference.yaml

generation:
  num_inference_steps: 25
  batch_size: 4             # Process 4 in parallel
  enable_cpu_offload: true  # Reduce GPU memory
```

---

## 🚀 Performance Optimization

### **Reduce Memory Usage**
```bash
# Enable memory-efficient features
--override "generation.enable_attention_slicing=true" \
--override "generation.enable_vae_slicing=true" \
--override "generation.enable_cpu_offload=true"
```

### **Increase Throughput**
```bash
# Batch processing
--override "generation.batch_size=4"

# Reduce precision (faster, minimal quality loss)
--override "generation.precision=fp16"
```

### **Multi-GPU Generation**
```bash
# Distribute across GPUs
python scripts/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --devices 0,1,2,3  # Use 4 GPUs
```

---

## 📊 Quality Metrics

### **Automatic Evaluation**
```bash
python scripts/run_reconstruct_and_eval.py \
    --config configs/inference/production.yaml \
    --checkpoint checkpoints/two_stage/best.pt \
    --evaluate  # Compute metrics
```

**Metrics Computed**:
- SSIM (Structural Similarity)
- LPIPS (Perceptual Similarity)
- FID (Fréchet Inception Distance)
- CLIP Score (Semantic Similarity)

---

## 🐛 Troubleshooting

### **Out of Memory**
```bash
# Reduce resolution
--override "generation.resolution=384"

# Enable memory optimizations
--override "generation.enable_attention_slicing=true" \
--override "generation.enable_vae_slicing=true"

# Reduce batch size
--override "generation.batch_size=1"
```

### **Poor Quality**
```bash
# Increase steps
--override "generation.num_inference_steps=100"

# Adjust guidance
--override "generation.guidance_scale=8.0"

# Try Karras noise schedule
--override "generation.use_karras_sigmas=true"
```

### **Too Slow**
```bash
# Reduce steps
--override "generation.num_inference_steps=30"

# Use FP16 precision
--override "generation.precision=fp16"

# Reduce resolution
--override "generation.resolution=384"
```

### **Artifacts in Images**
```bash
# Increase steps (reduces artifacts)
--override "generation.num_inference_steps=75"

# Use negative prompt
--override "generation.negative_prompt='artifacts, noise, blur'"

# Try different sampler
--override "generation.scheduler=DDIM"
```

---

## 📚 Related Documentation

- **[Main Config README](../README.md)** - Overview
- **[Training Configs](../training/README.md)** - Model training
- **[Diffusion Guide](../../docs/guides/GETTING_STARTED_DIFFUSION.md)** - Diffusion basics
- **[Usage Examples](../../USAGE_EXAMPLES.md)** - Complete commands

---

## 💡 Pro Tips

1. **Always start with `fast_inference.yaml`** for quick validation
2. **Use `production.yaml`** for main experiments (best balance)
3. **Reserve `highres_quality.yaml`** for selected showcase samples
4. **Monitor GPU memory** - enable slicing if running out
5. **Set seeds** for reproducible results in papers

---

**Last Updated**: December 7, 2025  
**Status**: Production-Ready  
**Recommended**: `production.yaml` for most use cases
