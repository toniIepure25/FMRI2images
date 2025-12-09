# Gallery Support in run_reconstruct_and_eval.py

## Overview

The `scripts/run_reconstruct_and_eval.py` script now has first-class support for retrieval galleries, allowing you to evaluate reconstructions against different gallery compositions in a single workflow.

## New Features

### 1. Gallery Selection

Control which retrieval gallery to use for evaluation:

```bash
--gallery {matched,test,all}   # Default: matched
```

**Gallery types:**
- `matched`: Only images seen by the subject (standard evaluation)
- `test`: Full test set (including unseen images)
- `all`: Combined training + test set (largest gallery)

### 2. All-Galleries Mode

Evaluate against all three gallery types in sequence:

```bash
--all-galleries
```

When this flag is set, the script will:
1. Run reconstruction once
2. Evaluate against `matched`, `test`, and `all` galleries
3. Generate separate output files for each gallery

### 3. Image Source Control

Specify where to load visualization images from:

```bash
--image-source {auto,s3,png,hdf5}   # Default: auto
```

**Source types:**
- `auto`: Try sources in order (S3 → PNG → HDF5)
- `s3`: Load from AWS S3 (natural-scenes-dataset bucket)
- `png`: Load from local PNG files
- `hdf5`: Load from NSD HDF5 file

### 4. HDF5 Path

Optionally specify the NSD HDF5 file path:

```bash
--nsd-hdf5 /path/to/nsd_stimuli.hdf5
```

## Usage Examples

### Single Gallery Evaluation

Evaluate against the matched gallery (default):

```bash
python scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder mlp \
  --ckpt checkpoints/mlp/subj01/mlp.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-dir outputs/recon/subj01/auto_with_adapter \
  --report-dir outputs/reports/subj01 \
  --use-adapter \
  --model-id stabilityai/stable-diffusion-2-1 \
  --gallery matched \
  --limit 32
```

Evaluate against the test gallery:

```bash
python scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder mlp \
  --ckpt checkpoints/mlp/subj01/mlp.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-dir outputs/recon/subj01/auto_with_adapter \
  --report-dir outputs/reports/subj01 \
  --use-adapter \
  --model-id stabilityai/stable-diffusion-2-1 \
  --gallery test \
  --image-source hdf5 \
  --nsd-hdf5 data/nsd_stimuli.hdf5 \
  --limit 32
```

### All-Galleries Evaluation

Run evaluation against all gallery types:

```bash
python scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder mlp \
  --ckpt checkpoints/mlp/subj01/mlp.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-dir outputs/recon/subj01/auto_with_adapter \
  --report-dir outputs/reports/subj01 \
  --use-adapter \
  --model-id stabilityai/stable-diffusion-2-1 \
  --all-galleries \
  --image-source hdf5 \
  --limit 32
```

This will generate:
- `recon_eval_matched.{csv,json,png}`
- `recon_eval_test.{csv,json,png}`
- `recon_eval_all.{csv,json,png}`

## Output Files

### File Naming Convention

All evaluation outputs now include a gallery suffix:

**Single gallery mode:**
```
outputs/reports/subj01/
├── recon_eval_matched.csv
├── recon_eval_matched.json
├── recon_grid_matched.png
└── recon_eval_summary.md
```

**All-galleries mode:**
```
outputs/reports/subj01/
├── recon_eval_matched.csv
├── recon_eval_matched.json
├── recon_grid_matched.png
├── recon_eval_test.csv
├── recon_eval_test.json
├── recon_grid_test.png
├── recon_eval_all.csv
├── recon_eval_all.json
├── recon_grid_all.png
└── recon_eval_summary.md
```

### JSON Structure

Each evaluation JSON includes a `retrieval_gallery` block with metadata:

```json
{
  "subject": "subj01",
  "clipscore": {...},
  "retrieval": {...},
  "ranking": {...},
  "retrieval_gallery": {
    "type": "test",
    "size": 5000,
    "n_eligible": 982,
    "n_total": 1000
  }
}
```

## Workflow Steps

The script orchestrates three steps:

1. **Reconstruction** (Step 1/3)
   - Runs `decode_diffusion.py` to generate images
   - Outputs to: `{output_dir}/images/`
   - Only runs once, regardless of gallery mode

2. **Evaluation** (Step 2/3)
   - Runs `eval_reconstruction.py` for each gallery
   - Generates CSV, JSON, PNG for each gallery
   - Handles CLIP space consistency automatically

3. **Summary** (Step 3/3)
   - Creates Markdown summary from primary gallery results
   - Uses first gallery for backward compatibility

## Integration with Makefile

The Makefile has been updated to support gallery workflows:

```bash
# Single gallery
make eval-recon SUBJ=subj01 GALLERY=test

# All galleries
make eval-recon-all SUBJ=subj01
```

## Comparison Across Galleries

After running with `--all-galleries`, compare results:

```bash
python scripts/compare_evals.py \
  --report-dir outputs/reports/subj01 \
  --pattern "recon_eval_*.json" \
  --out-csv outputs/reports/subj01/gallery_comparison.csv \
  --out-tex outputs/reports/subj01/gallery_comparison.tex \
  --out-md outputs/reports/subj01/gallery_comparison.md \
  --out-fig outputs/reports/subj01/gallery_comparison.png
```

This will generate a comparison table showing how metrics vary across gallery types.

## Expected Metric Patterns

Different galleries produce different difficulty levels:

| Gallery | Size | Difficulty | Expected R@1 | Expected CLIPScore |
|---------|------|------------|--------------|-------------------|
| matched | ~500 | Easiest | 0.40-0.70 | 0.60-0.80 |
| test | ~1000 | Medium | 0.30-0.60 | 0.60-0.80 |
| all | ~5000 | Hardest | 0.10-0.40 | 0.60-0.80 |

**Note:** CLIPScore measures semantic similarity (remains stable), while retrieval metrics (R@K, rank) measure discriminability (decreases with gallery size).

## Backward Compatibility

The script maintains full backward compatibility:

- Default gallery is `matched` (standard behavior)
- Default image source is `auto` (tries all sources)
- If no gallery arguments provided, behaves exactly as before
- Old output filenames without suffix still work for primary gallery

## Technical Implementation

### Command Construction

The script builds subprocess commands dynamically:

```python
cmd = [
    sys.executable, "scripts/eval_reconstruction.py",
    "--subject", args.subject,
    "--recon-dir", str(recon_dir),
    "--clip-cache", args.clip_cache,
    "--out-csv", str(Path(args.report_dir) / f"recon_eval_{gallery}.csv"),
    "--out-json", str(Path(args.report_dir) / f"recon_eval_{gallery}.json"),
    "--out-fig", str(Path(args.report_dir) / f"recon_grid_{gallery}.png"),
    "--gallery", gallery,
    "--image-source", args.image_source,
]

if args.nsd_hdf5:
    cmd.extend(["--nsd-hdf5", str(args.nsd_hdf5)])

if args.use_adapter:
    cmd.extend(["--use-adapter", "--model-id", args.model_id])
```

### Error Handling

- Creates output directories automatically
- Validates all required paths exist
- Propagates evaluation script errors with proper exit codes
- Logs all commands before execution for transparency

## Testing

Test the implementation:

```bash
# Test argument parsing
python scripts/test_gallery_support.py

# Test single gallery (small limit)
python scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder mlp \
  --ckpt checkpoints/mlp/subj01/mlp.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-dir outputs/recon/subj01/test \
  --report-dir outputs/reports/subj01/test \
  --gallery test \
  --limit 2 \
  --skip-sd-cache-check

# Test all galleries (small limit)
python scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder mlp \
  --ckpt checkpoints/mlp/subj01/mlp.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-dir outputs/recon/subj01/test_all \
  --report-dir outputs/reports/subj01/test_all \
  --all-galleries \
  --limit 2 \
  --skip-sd-cache-check
```

## Benefits

1. **Comprehensive evaluation**: Compare difficulty levels across galleries
2. **Reproducibility**: Deterministic file naming with gallery suffix
3. **Flexibility**: Choose appropriate gallery for your research question
4. **Efficiency**: Single reconstruction, multiple evaluations
5. **Publication-ready**: Structured outputs for thesis/paper comparisons
6. **Automation-friendly**: Easy to integrate into batch workflows

## Files Modified

- `scripts/run_reconstruct_and_eval.py`: Added gallery orchestration logic
- Documentation: `docs/GALLERY_SUPPORT.md` (this file)
- Test: `scripts/test_gallery_support.py`

## Related Scripts

- `scripts/eval_reconstruction.py`: Core evaluation script (supports `--gallery`)
- `scripts/compare_evals.py`: Compare evaluations across galleries
- `scripts/summarize_reports.py`: Aggregate results from multiple runs
