# Repository Hardening Implementation Guide

## Overview

This document provides exact patches for hardening the fMRI-to-image reconstruction pipeline with robust I/O, schema enforcement, space safety, and speed optimizations.

## 1. Enhanced Image Loading (COMPLETED ✅)

### File: `src/fmri2img/io/image_loader.py` (NEW)

✅ Created `RobustImageLoader` class with:
- Local HDF5 priority (via `$NSD_HDF5` or default path)
- S3 HDF5 fallback
- COCO HTTP fallback with local caching (`.cache/coco/`)
- Single-warning-per-error pattern (no spam)
- Graceful handling of truncated/partial files

### File: `scripts/build_clip_cache.py` (UPDATED ✅)

✅ Updated to use `RobustImageLoader`:
- Added import: `from fmri2img.io.image_loader import RobustImageLoader`
- Replaced manual HDF5/COCO loading with unified loader
- Enhanced stats reporting with all load methods

## 2. CLIP Cache Schema Enforcement

### Status: COMPLETED ✅ (in previous session)

The schema validation in `build_clip_cache.py` already:
- Ensures `nsd_id` (int) and `embedding` (float32[512]) columns
- Creates aliases from legacy column names (`nsdId` → `nsd_id`, `clip512` → `embedding`)
- Validates and saves with correct schema

## 3. Gallery Builder Robustness

###File: `scripts/eval_reconstruction.py` - Add `--strict-gallery` flag

**Location**: Add to argument parser (around line 1400-1500)

```python
# In the argument parser section, add:
parser.add_argument("--strict-gallery", action="store_true",
                   help="Fail fast if gallery IDs cannot be enumerated (for benchmarking)")
```

**Location**: Update gallery enumeration logic (around line 600-700)

```python
# In the gallery building section, wrap enumeration in try/except:
def build_gallery_safe(gallery_type, index_df, limit, strict=False):
    """Build gallery with safe ID enumeration."""
    try:
        # Existing gallery building logic here
        gallery_ids = enumerate_gallery_ids(gallery_type, index_df, limit)
        return gallery_ids
    except Exception as e:
        if strict:
            logger.error(f"❌ Gallery enumeration failed (--strict-gallery): {e}")
            raise
        else:
            logger.warning(f"⚠️  Gallery enumeration failed, continuing without gallery: {e}")
            logger.warning(f"   Generation will still run, but retrieval metrics unavailable")
            return None
```

## 4. Orchestrator Space Safety

### File: `scripts/run_reconstruct_and_eval.py`

**Location**: Update `load_adapter_metadata()` to use robust loader (around line 111)

```python
def load_adapter_metadata(adapter_path: Path) -> Dict[str, Any]:
    """
    Load adapter metadata to get target dimension using robust loader.
    
    Returns:
        Dictionary with 'target_dim' and other metadata.
    """
    from fmri2img.models.clip_adapter import load_adapter
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    try:
        _, metadata = load_adapter(str(adapter_path), map_location="cpu")
        
        # Get target_dim (prefer target_dim, fallback to out_dim)
        target_dim = metadata.get("target_dim", metadata.get("out_dim"))
        if target_dim is None:
            raise ValueError(f"Adapter metadata missing target_dim: {adapter_path}")
        
        # Ensure it's in metadata
        metadata["target_dim"] = target_dim
        
        return metadata
    except Exception as e:
        raise ValueError(f"Failed to load adapter metadata from {adapter_path}: {e}")
```

**Location**: Update `run_decode()` to auto-detect target_dim (around line 133-200)

```python
def run_decode(
    encoder: str,
    ckpt_path: Path,
    output_dir: Path,
    limit: int,
    steps: int,
    device: str,
    use_adapter: bool,
    adapter_path: Optional[Path],
    model_id: Optional[str],
    clip_target_dim: Optional[int],  # This will be auto-filled from adapter
    subject: str,
    index_root: Optional[Path],
    index_file: Optional[Path],
    preproc_enabled: bool,
    preproc_path: Optional[str],
) -> int:
    """
    Run decode_diffusion.py to generate reconstructions.
    
    Returns:
        Exit code (0 = success).
    """
    print_banner("Step 1/3: Generate Reconstructions")
    
    script_path = Path(__file__).parent / "decode_diffusion.py"
    if not script_path.exists():
        print(f"ERROR: decode_diffusion.py not found at {script_path}")
        return 1
    
    # Auto-detect target_dim from adapter if using adapter
    detected_target_dim = None
    if use_adapter and adapter_path:
        try:
            metadata = load_adapter_metadata(adapter_path)
            detected_target_dim = metadata["target_dim"]
            print(f"   Auto-detected adapter target_dim: {detected_target_dim}D")
            
            # Override clip_target_dim if not explicitly set
            if clip_target_dim is None:
                clip_target_dim = detected_target_dim
            elif clip_target_dim != detected_target_dim:
                print(f"   ⚠️  WARNING: --clip-target-dim={clip_target_dim} but adapter uses {detected_target_dim}D")
                print(f"   Using adapter's dimension: {detected_target_dim}D")
                clip_target_dim = detected_target_dim
        except Exception as e:
            print(f"ERROR: Failed to load adapter metadata: {e}")
            return 1
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--encoder", encoder,
        "--ckpt", str(ckpt_path),
        "--output-dir", str(output_dir),
        "--limit", str(limit),
        "--steps", str(steps),
        "--device", device,
        "--subject", subject,
    ]
    
    # ... rest of command building ...
```

**Location**: Add bold NOTE in final summary (around line 800-900, in main())

```python
def main():
    # ... existing code ...
    
    # After all steps complete, print summary with CLIP space info
    print("=" * 80)
    print("✅ RECONSTRUCTION AND EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Images: {output_dir}")
    print(f"JSON report: {eval_json_path}")
    print(f"CSV report: {eval_csv_path}")
    if grid_path and grid_path.exists():
        print(f"Grid visualization: {grid_path}")
    if comparison_md and comparison_md.exists():
        print(f"Markdown report: {comparison_md}")
    
    # BOLD NOTE about CLIP space
    if args.use_adapter and detected_target_dim:
        print()
        print("🔍 " + "=" * 76)
        print(f"   NOTE: Evaluation performed in {detected_target_dim}D CLIP space")
        print(f"         (matching generation with adapter)")
        print("=" * 80)
    else:
        print()
        print("🔍 " + "=" * 76)
        print(f"   NOTE: Evaluation performed in 512D CLIP space (ViT-B/32)")
        print(f"         (no adapter used)")
        print("=" * 80)
    
    print()
    return 0
```

## 5. Makefile Enhancements

### File: `Makefile`

**Add these targets:**

```makefile
.PHONY: help check-sd download-sd recon-eval recon-eval-adapter

# Show help with common commands
help:
	@echo "Common commands:"
	@echo "  make check-sd              - Check if Stable Diffusion models are cached"
	@echo "  make download-sd           - Download SD models to HuggingFace cache"
	@echo "  make recon-eval            - Generate + evaluate (512-D, no adapter)"
	@echo "  make recon-eval-adapter    - Generate + evaluate (1024-D, with adapter)"
	@echo ""
	@echo "Environment variables:"
	@echo "  NSD_HDF5                   - Path to local nsd_stimuli.hdf5 (speeds up loading)"
	@echo "  SUBJECT                    - Subject ID (default: subj01)"
	@echo "  LIMIT                      - Number of samples (default: 128)"
	@echo "  MODEL                      - Diffusion model (default: stabilityai/stable-diffusion-2-1)"
	@echo "  ADAPTER                    - Path to CLIP adapter checkpoint"
	@echo ""
	@echo "Example:"
	@echo "  export NSD_HDF5=cache/nsd_hdf5/nsd_stimuli.hdf5"
	@echo "  make recon-eval SUBJECT=subj01 LIMIT=32"

# Check if Stable Diffusion models are cached
check-sd:
	@echo "Checking HuggingFace cache for Stable Diffusion models..."
	@python3 -c "from huggingface_hub import scan_cache_dir; \
	cache = scan_cache_dir(); \
	models = [r.repo_id for r in cache.repos]; \
	sd_models = [m for m in models if 'stable-diffusion' in m.lower()]; \
	print(f'Found {len(sd_models)} SD models in cache:'); \
	for m in sd_models: print(f'  ✓ {m}'); \
	exit(0 if sd_models else 1)" || \
	(echo "❌ No Stable Diffusion models found in cache"; \
	 echo "   Run: make download-sd"; \
	 exit 1)

# Download Stable Diffusion models to cache
download-sd:
	@echo "Downloading Stable Diffusion models..."
	python scripts/download_sd_model.py --model-id stabilityai/stable-diffusion-2-1
	python scripts/download_sd_model.py --model-id runwayml/stable-diffusion-v1-5
	@echo "✓ Models downloaded to HuggingFace cache"

# Recon + eval without adapter (512-D)
recon-eval: SUBJECT ?= subj01
recon-eval: LIMIT ?= 128
recon-eval: MODEL ?= stabilityai/stable-diffusion-2-1
recon-eval:
	@echo "Running reconstruction + evaluation (512-D, no adapter)..."
	@echo "  Subject: $(SUBJECT)"
	@echo "  Limit: $(LIMIT)"
	@echo "  Model: $(MODEL)"
	python scripts/run_reconstruct_and_eval.py \
		--encoder mlp \
		--ckpt checkpoints/mlp/$(SUBJECT)/mlp.pt \
		--output-dir outputs/recon/$(SUBJECT)/$(MODEL:stabilityai/%=%) \
		--subject $(SUBJECT) \
		--limit $(LIMIT) \
		--device cuda \
		--model-id $(MODEL)

# Recon + eval with adapter (1024-D)
recon-eval-adapter: SUBJECT ?= subj01
recon-eval-adapter: LIMIT ?= 128
recon-eval-adapter: MODEL ?= stabilityai/stable-diffusion-2-1
recon-eval-adapter: ADAPTER ?= checkpoints/clip_adapter/$(SUBJECT)/adapter.pt
recon-eval-adapter:
	@echo "Running reconstruction + evaluation (with adapter)..."
	@echo "  Subject: $(SUBJECT)"
	@echo "  Limit: $(LIMIT)"
	@echo "  Model: $(MODEL)"
	@echo "  Adapter: $(ADAPTER)"
	@test -f "$(ADAPTER)" || (echo "❌ Adapter not found: $(ADAPTER)"; \
	                          echo "   Train adapter first: python scripts/train_clip_adapter.py ..."; \
	                          exit 1)
	python scripts/run_reconstruct_and_eval.py \
		--encoder mlp \
		--ckpt checkpoints/mlp/$(SUBJECT)/mlp.pt \
		--output-dir outputs/recon/$(SUBJECT)/$(MODEL:stabilityai/%=%)_adapter \
		--subject $(SUBJECT) \
		--limit $(LIMIT) \
		--device cuda \
		--model-id $(MODEL) \
		--use-adapter \
		--adapter $(ADAPTER)
```

## 6. Speed Optimizations

### Environment Variables

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# Speed up NSD image loading
export NSD_HDF5=/path/to/cache/nsd_hdf5/nsd_stimuli.hdf5

# Or use default location
export NSD_HDF5=cache/nsd_hdf5/nsd_stimuli.hdf5
```

### Download NSD HDF5 Locally

```bash
# Create cache directory
mkdir -p cache/nsd_hdf5

# Download from S3 (requires aws-cli and credentials, or use other method)
# Option 1: If you have direct access
cp /path/to/existing/nsd_stimuli.hdf5 cache/nsd_hdf5/

# Option 2: The loader will automatically fall back to S3/COCO
# Just set the env variable and it will use it when available
```

### COCO Cache

The `RobustImageLoader` automatically caches COCO images in `.cache/coco/` for reuse.

## 7. Testing the Changes

### Test Image Loading

```bash
# Test with local HDF5
export NSD_HDF5=cache/nsd_hdf5/nsd_stimuli.hdf5
python scripts/build_clip_cache.py \
    --subject subj01 \
    --limit 100 \
    --device cuda \
    --batch 128

# Check logs for:
# ✓ Local HDF5 found: cache/nsd_hdf5/nsd_stimuli.hdf5
# Image loading sources:
#   - Local HDF5: XX images
#   - S3 HDF5: XX images
#   - COCO (cached): XX images
```

### Test Adapter Auto-Detection

```bash
# Run with adapter (should auto-detect target_dim)
python scripts/run_reconstruct_and_eval.py \
    --encoder mlp \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --subject subj01 \
    --limit 32 \
    --use-adapter \
    --adapter checkpoints/clip_adapter/subj01/adapter.pt \
    --model-id stabilityai/stable-diffusion-2-1

# Check for:
# Auto-detected adapter target_dim: 1024D
# NOTE: Evaluation performed in 1024D CLIP space
```

### Test Makefile

```bash
# Check help
make help

# Check SD cache
make check-sd

# Run basic recon+eval
make recon-eval SUBJECT=subj01 LIMIT=32

# Run with adapter
make recon-eval-adapter SUBJECT=subj01 LIMIT=32
```

## 8. Benefits Summary

✅ **IO Robustness**
- Local HDF5 priority (10-100x faster than S3)
- Graceful degradation on truncated files
- COCO fallback with caching
- Single-warning pattern (no log spam)

✅ **Schema Enforcement**
- Guaranteed `nsd_id` (int) and `embedding` (float32[512])
- Automatic aliasing from legacy columns
- Validation on save

✅ **Space Safety**
- Auto-detection of adapter target_dim
- Bold NOTE in summary about CLIP space
- Consistent eval/generation dimensions

✅ **Speed Wins**
- Local HDF5 preferred (env variable)
- COCO image caching
- Batch CLIP encoding (already present)

✅ **Developer Experience**
- Makefile help text
- Clear error messages
- Concise run summaries
- Easy testing with `make` commands

## Implementation Checklist

- [x] 1. Created `src/fmri2img/io/image_loader.py`
- [x] 2. Updated `scripts/build_clip_cache.py` to use `RobustImageLoader`
- [ ] 3. Add `--strict-gallery` to `scripts/eval_reconstruction.py`
- [ ] 4. Update `scripts/run_reconstruct_and_eval.py` adapter auto-detection
- [ ] 5. Add bold NOTE to orchestrator summary
- [ ] 6. Update `Makefile` with help/check-sd/download-sd targets
- [ ] 7. Test all changes end-to-end

## Next Steps

1. Apply remaining patches to `run_reconstruct_and_eval.py`
2. Add `--strict-gallery` flag to `eval_reconstruction.py`
3. Update Makefile with new targets
4. Test with small limit (32 samples)
5. Verify CLIP space consistency in output
