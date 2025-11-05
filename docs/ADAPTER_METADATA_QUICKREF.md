# Adapter Metadata - Quick Reference

## What Changed?

CLIP adapter checkpoints now include standardized metadata for better reproducibility and validation.

## For Users

### Nothing breaks! 🎉
- Old checkpoints work automatically with auto-repair
- New checkpoints include full metadata
- All your existing scripts and workflows continue working

### What You'll See

**When training** (`train_clip_adapter.py`):
```
✅ Adapter saved to checkpoints/clip_adapter/subj01/adapter.pt
   Saved adapter with metadata: {subject=subj01, model_id=stabilityai/stable-diffusion-2-1, 
   input_dim=512, target_dim=1024, created_at=2025-11-05T00:12:27, repo_version=0.1.0}
```

**When loading** (`decode_diffusion.py`, `eval_reconstruction.py`):
```
Loaded adapter (target_dim=1024) with metadata: {subject=subj01, model_id=..., input_dim=512, target_dim=1024}
✅ CLIP Adapter loaded: 512D → 1024D
```

**With legacy checkpoints**:
```
Adapter metadata repaired: {subject=unknown, model_id=stabilityai/stable-diffusion-2-1, input_dim=512, target_dim=1024}
```

**With model mismatches**:
```
⚠️  Adapter was trained for stabilityai/stable-diffusion-2-1 but using runwayml/stable-diffusion-v1-5
   This may cause dimension mismatches or degraded quality
```

## For Developers

### Loading Adapters

**Old way** (don't do this):
```python
checkpoint = torch.load(path)
adapter = CLIPAdapter(...)
adapter.load_state_dict(checkpoint['state_dict'])
```

**New way** (recommended):
```python
from fmri2img.models.clip_adapter import load_adapter

adapter, metadata = load_adapter(path, map_location="cuda")
print(f"Model: {metadata['model_id']}")
print(f"Dims: {metadata['input_dim']}D → {metadata['target_dim']}D")
```

### Saving Adapters

**Automatic** (if using `train_clip_adapter.py`):
- Metadata added automatically
- No changes needed

**Manual**:
```python
from datetime import datetime

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

## Metadata Fields

### Required (auto-filled if missing)
- `subject`: Subject ID (e.g., "subj01", default: "unknown")
- `model_id`: Target diffusion model (default: "stabilityai/stable-diffusion-2-1")
- `input_dim`: Input dimension (default: 512)
- `target_dim`: Output dimension (default: 1024)

### Optional (but recommended)
- `created_at`: ISO timestamp
- `repo_version`: Package version
- `test_cosine`: Test set cosine similarity
- `best_epoch`: Best training epoch
- Training hyperparameters, metrics, etc.

## Testing

### Unit Tests
```bash
python scripts/test_adapter_metadata.py
```

### Integration Test
```bash
python scripts/test_adapter_integration.py
```

## Common Scenarios

### Scenario 1: Using Existing Adapter
**You don't need to do anything!** Just use it as before:
```bash
python scripts/decode_diffusion.py \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --clip-adapter checkpoints/clip_adapter/subj01/adapter.pt \
    --model-id stabilityai/stable-diffusion-2-1
```

### Scenario 2: Training New Adapter
**No changes needed!** Metadata added automatically:
```bash
python scripts/train_clip_adapter.py \
    --subject subj01 \
    --model-id stabilityai/stable-diffusion-2-1 \
    --out checkpoints/clip_adapter/subj01/adapter.pt
```

### Scenario 3: Model Mismatch
If you see a warning, it means:
- Adapter trained for model A
- You're using it with model B
- **May work**, but dimensions might not align perfectly

**Solution**: Train a new adapter for model B, or use model A.

### Scenario 4: Missing Adapter
If you see:
```
❌ Adapter checkpoint not found: checkpoints/clip_adapter/subj01/adapter.pt
   Hint: Train an adapter first with scripts/train_clip_adapter.py
```

**Solution**: Train an adapter:
```bash
python scripts/train_clip_adapter.py \
    --subject subj01 \
    --clip-cache outputs/clip_cache/clip.parquet \
    --model-id stabilityai/stable-diffusion-2-1 \
    --out checkpoints/clip_adapter/subj01/adapter.pt
```

## Benefits

1. **Know what you're using**: See which model an adapter was trained for
2. **Catch mistakes early**: Warnings if dimensions don't match
3. **Better debugging**: Track creation date, subject, version
4. **Reproducibility**: Full training info preserved
5. **Safety**: Auto-repair ensures nothing breaks

## Need Help?

- **Full docs**: `docs/ADAPTER_METADATA.md`
- **Examples**: `scripts/test_adapter_metadata.py`, `scripts/test_adapter_integration.py`
- **Summary**: `docs/ADAPTER_METADATA_SUMMARY.md`

## TL;DR

✅ **Users**: Nothing changes, everything works, you get better logging  
✅ **Developers**: Use `load_adapter()`, get `(adapter, metadata)` back  
✅ **Legacy**: Old checkpoints auto-repair with sensible defaults  
✅ **Safety**: Warnings for model mismatches and dimension issues  
