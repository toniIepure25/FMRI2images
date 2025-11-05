# CLIP Adapter Metadata Enhancement

## Summary

Enhanced CLIP adapter checkpoint format to include standardized metadata for better reproducibility, debugging, and model tracking. Implemented robust backward-compatible loading that handles legacy checkpoints gracefully.

## Changes Made

### TASK A: Training Script Metadata (`scripts/train_clip_adapter.py`)

**Updated checkpoint save hook** to include required metadata fields:

```python
metadata = {
    "subject": args.subject,              # Subject ID (e.g., "subj01")
    "model_id": args.model_id,            # Target model (e.g., "stabilityai/stable-diffusion-2-1")
    "input_dim": 512,                     # Input dimension (ViT-B/32 CLIP)
    "target_dim": target_dim,             # Output dimension (1024 for SD 2.1)
    "created_at": datetime.now().isoformat(),  # ISO timestamp
    "repo_version": repo_version,         # From pyproject.toml
    # Additional training info...
}
```

**Checkpoint format**:
```python
{
    "state_dict": adapter.state_dict(),
    "metadata": {
        # Required fields above
        # + training metrics, hyperparameters
    }
}
```

**Logging output**:
```
✅ Adapter saved to checkpoints/clip_adapter/subj01/adapter.pt
   Saved adapter with metadata: {subject=subj01, model_id=stabilityai/stable-diffusion-2-1, 
   input_dim=512, target_dim=1024, created_at=2025-11-05T..., repo_version=0.1.0}
```

### TASK B: Loader Repair (`src/fmri2img/models/clip_adapter.py`)

**Implemented `load_adapter()` with robust fallback handling**:

1. **Legacy state_dict wrapping**: Automatically wraps raw state_dicts into proper checkpoint format
2. **Missing metadata repair**: Fills missing required fields with sensible defaults:
   - `subject`: "unknown"
   - `model_id`: "stabilityai/stable-diffusion-2-1" (default)
   - `input_dim`: 512
   - `target_dim`: 1024
   - `use_layernorm`: Inferred from state_dict structure

3. **Backward compatibility**: Handles both "metadata" (new) and "meta" (legacy) keys

**Logging output**:
```
Adapter metadata repaired: {subject=unknown, model_id=stabilityai/stable-diffusion-2-1, 
                            input_dim=512, target_dim=1024, use_layernorm=False (inferred)}
Loaded adapter (target_dim=1024) with metadata: {subject=subj01, model_id=..., input_dim=512, target_dim=1024}
```

### TASK C: Call Site Updates

#### `scripts/decode_diffusion.py`

**Enhanced adapter loading with validation**:
- Uses new `load_adapter()` function
- Validates dimension consistency between adapter metadata and CLI args
- Warns if adapter's `model_id` doesn't match `--model-id`
- Clear error messages with guidance if adapter is missing

**Logging output**:
```
Loading CLIP adapter from checkpoints/clip_adapter/subj01/adapter.pt
Loaded adapter (target_dim=1024) with metadata: {subject=subj01, model_id=stabilityai/stable-diffusion-2-1, input_dim=512, target_dim=1024}
✅ CLIP Adapter loaded: 512D → 1024D
   Adapter metadata: model_id=stabilityai/stable-diffusion-2-1, subject=subj01
```

**Warning scenarios**:
```
⚠️  --clip-target-dim=768 but adapter outputs 1024D
   Using adapter's dimension: 1024D

⚠️  Adapter was trained for stabilityai/stable-diffusion-2-1 but using runwayml/stable-diffusion-v1-5
   This may cause dimension mismatches or degraded quality
```

#### `scripts/eval_reconstruction.py`

**Replaced custom `_load_adapter()` with new loader**:
- Returns tuple `(adapter, metadata)` instead of just adapter
- Uses metadata to determine actual output dimensions
- Better error handling with actionable hints

**Logging output**:
```
🔧 Loading adapter: checkpoints/clip_adapter/subj01/adapter.pt
Loaded adapter (target_dim=1024) with metadata: {...}
🔧 Applying adapter: 512D → 1024D
✅ Adapter applied: new shape=(128, 1024)
```

### TASK D: CLI Flags & Defaults

**Dimension resolution logic**:
1. If `--adapter` provided → use `adapter.metadata["target_dim"]`
2. Else → fallback to 512 (ViT-B/32 default)

**No changes needed to CLI**: Existing flags work correctly with new metadata system.

## Testing

**Test suite**: `scripts/test_adapter_metadata.py`

Tests cover:
1. ✅ Save and load with full metadata
2. ✅ Load legacy checkpoint (raw state_dict)
3. ✅ Load non-existent file (proper error)
4. ✅ Load checkpoint with legacy "meta" key

**All tests pass**: 4/4 ✅

## Migration Guide

### For Users

**No action required!** Legacy checkpoints continue to work with auto-repair:
- Missing metadata fields filled with defaults
- Warnings logged for awareness
- `use_layernorm` inferred from model structure

### For Developers

**New checkpoints** automatically include full metadata after training.

**Manual checkpoint creation**:
```python
from fmri2img.models.clip_adapter import CLIPAdapter
from datetime import datetime

adapter = CLIPAdapter(in_dim=512, out_dim=1024)
# ... train adapter ...

metadata = {
    "subject": "subj01",
    "model_id": "stabilityai/stable-diffusion-2-1",
    "input_dim": 512,
    "target_dim": 1024,
    "created_at": datetime.now().isoformat(),
    "repo_version": "0.1.0",
}

adapter.save("path/to/adapter.pt", metadata)
```

**Loading**:
```python
from fmri2img.models.clip_adapter import load_adapter

adapter, metadata = load_adapter("path/to/adapter.pt", map_location="cuda")
print(f"Loaded: {metadata['input_dim']}D → {metadata['target_dim']}D")
print(f"Trained for: {metadata['model_id']}")
```

## Benefits

1. **Reproducibility**: Know exactly which model an adapter was trained for
2. **Debugging**: Track subject, creation date, repo version
3. **Validation**: Automatic dimension consistency checks
4. **Safety**: Clear warnings for model/adapter mismatches
5. **Backward compatibility**: Legacy checkpoints work seamlessly

## Files Modified

- `scripts/train_clip_adapter.py` - Enhanced save hook with metadata
- `src/fmri2img/models/clip_adapter.py` - Robust loader with fallbacks
- `scripts/decode_diffusion.py` - Updated adapter loading and validation
- `scripts/eval_reconstruction.py` - Updated adapter loading
- `scripts/test_adapter_metadata.py` - Comprehensive test suite (new)

## Example Output

### Training (save)
```
✓ CLIP cache build complete!
✅ Adapter saved to checkpoints/clip_adapter/subj01/adapter.pt
   Saved adapter with metadata: {subject=subj01, model_id=stabilityai/stable-diffusion-2-1, 
   input_dim=512, target_dim=1024, created_at=2025-11-05T00:12:27.934379, repo_version=0.1.0}
```

### Inference (load)
```
Loading CLIP adapter from checkpoints/clip_adapter/subj01/adapter.pt
Loaded adapter (target_dim=1024) with metadata: {subject=subj01, model_id=stabilityai/stable-diffusion-2-1, 
                                                  input_dim=512, target_dim=1024}
✅ CLIP Adapter loaded: 512D → 1024D
   Adapter metadata: model_id=stabilityai/stable-diffusion-2-1, subject=subj01
CLIP Adapter: ENABLED (512D → 1024D)
```

### Legacy checkpoint (auto-repair)
```
Adapter metadata repaired: {subject=unknown, model_id=stabilityai/stable-diffusion-2-1, 
                            input_dim=512, target_dim=1024, use_layernorm=False (inferred from state_dict)}
Loaded adapter (target_dim=1024) with metadata: {subject=unknown, model_id=stabilityai/stable-diffusion-2-1, 
                                                  input_dim=512, target_dim=1024}
```

## Notes

- Version reading from `pyproject.toml` uses `tomllib` (Python 3.11+) with regex fallback
- All dimension fields use both new names (`input_dim`/`target_dim`) and legacy names (`in_dim`/`out_dim`) for maximum compatibility
- Checkpoint key changed from "meta" to "metadata" but loader handles both
- `use_layernorm` correctly inferred by checking for "layernorm.weight" in state_dict
