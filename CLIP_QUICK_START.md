# CLIP Cache Quick Start

## Build CLIP Cache

```bash
# From partitioned index (recommended)
python scripts/build_clip_cache.py \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --cache outputs/clip_cache/clip.parquet \
    --batch 64 --device cuda --limit 256

# From index root with subject filter
python scripts/build_clip_cache.py \
    --index-root data/indices/nsd_index \
    --subject subj01 \
    --cache outputs/clip_cache/clip.parquet \
    --batch 128 --device cuda

# Using Makefile
make build-clip-cache INDEX_FILE=data/indices/nsd_index/subject=subj01/index.parquet LIMIT=100
```

## Resume Support

Resume is automatic - just re-run the same command:
```bash
# Initial run (processes 1000 images, then interrupted)
python scripts/build_clip_cache.py --index-file ... --limit 1000

# Resume (skips already-cached, processes remaining)
python scripts/build_clip_cache.py --index-file ... --limit 1000
```

## Use in Training

```python
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.data.torch_dataset import NSDIterableDataset

# Initialize cache
clip_cache = CLIPCache(cache_path="outputs/clip_cache/clip.parquet")

# Wire into dataset
dataset = NSDIterableDataset(
    index_path_or_root="data/indices/nsd_index",
    subject="subj01",
    clip_cache=clip_cache
)

# Iterate - each batch includes "clip" key with (512,) float32 array
for batch in dataset:
    fmri = batch["fmri"]   # (H,W,D) or (k,) after PCA
    clip = batch["clip"]   # (512,) CLIP embedding (if cached)
    nsd_id = batch["nsdId"]
```

## Image Loading

**Primary Path:** `nsd_stimuli.hdf5` via nsdId (fast, S3-backed)
```python
with hdf5_loader.open(hdf5_path) as f:
    img_arr = f["imgBrick"][nsd_id]  # (H, W, 3)
```

**Fallback:** COCO HTTP if HDF5 fails and cocoId is available
```python
url = layout.coco_http_url(coco_id, coco_split)
response = requests.get(url)
img = Image.open(BytesIO(response.content))
```

## Column Normalization

Handles both naming conventions automatically:
- `nsd_id` → `nsdId`
- `coco_id` → `cocoId`
- `coco_split` → `cocoSplit`

## CLI Flags

### Primary Flags
- `--index-file FILE` - Single parquet index file
- `--index-root DIR` - Partitioned index root (subject=subjXX/)
- `--subject SUBJ` - Subject filter (e.g., 'subj01')
- `--cache FILE` - Output cache path (default: outputs/clip_cache/clip.parquet)
- `--batch N` - Batch size for CLIP inference (default: 128)
- `--device cuda|cpu` - Device for CLIP model (default: cuda)
- `--limit N` - Max items to process (for testing)

### Aliases (Backward Compatible)
- `--batch-size` → `--batch`
- `--max-items` → `--limit`

### Deprecated (Still Work)
- `--index` → `--index-file` (with warning)
- `--use-hdf5` → no-op (with warning)

## Testing

```bash
# Run comprehensive tests
python3 scripts/test_clip_refactoring.py

# Test with small dataset
python3 scripts/build_clip_cache.py \
    --index-file data/indices/test_nsd_index.parquet \
    --cache outputs/clip_cache/test_clip.parquet \
    --batch 4 --device cpu --limit 2

# Verify cache
python3 -c "
from fmri2img.data.clip_cache import CLIPCache
cache = CLIPCache('outputs/clip_cache/test_clip.parquet')
cache.load()
print(cache.stats())
print('Cached IDs:', cache.list_cached_ids())
"
```

## Common Issues

**Q: HDF5 file download fails?**
A: Falls back to COCO HTTP automatically (if cocoId available)

**Q: Column 'nsdId' not found?**
A: Column normalization handles `nsd_id` → `nsdId` automatically

**Q: Cache not resuming?**
A: Make sure you're using the same `--cache` path

**Q: Out of GPU memory?**
A: Reduce `--batch` size or use `--device cpu`

## Key Features

✅ HDF5 primary path (fast S3 access)
✅ COCO HTTP fallback (resilient)
✅ Automatic column normalization
✅ Flexible index loading (file or root)
✅ Resume support (skips cached IDs)
✅ Batch CLIP lookup in dataset (efficient)
✅ Backward compatible CLI
✅ L2 normalized embeddings (ready for cosine similarity)

## Performance

- **Batch Size**: 128 images/batch on GPU (configurable)
- **Storage**: ~2KB per embedding (Snappy compressed)
- **Resume**: Near-instant cached ID lookup
- **Dataset Integration**: Single batch CLIP lookup (not per-sample)

