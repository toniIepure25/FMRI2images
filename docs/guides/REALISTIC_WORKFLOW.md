# 🎯 Realistic End-to-End Workflow with Novel Contributions

**Actual commands you need to run from zero to reconstruction with the 3 novel contributions**

---

## ⚠️ Reality Check

The "3 simple commands" version was **oversimplified**. Here's what you **actually** need to do:

---

## 📋 Prerequisites (One-Time Setup)

### 1. Environment Setup

```bash
cd "/home/tonystark/Desktop/Bachelor V2"

# Activate conda environment
conda activate fmri2img

# Install package in development mode
pip install -e .
```

### 2. Verify Installation

```bash
# Run smoke tests
python -m pytest tests/test_losses.py tests/test_soft_reliability.py tests/test_uncertainty.py -v

# Should see: 53 tests passed ✅
```

---

## 🔄 Complete Pipeline (7 Steps)

### Step 1: Build NSD Index (One-Time, ~10 minutes)

**What it does**: Creates Parquet index of all fMRI trials with train/val/test splits

```bash
make index SUBJECTS=subj01
```

**Output**: `data/indices/nsd_index/subject=subj01/index.parquet`

**Verify**:
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/indices/nsd_index/subject=subj01/index.parquet'); print(f'{len(df)} trials')"
# Should print: ~9000-10000 trials (depending on subject)
```

---

### Step 2: Build CLIP Cache (One-Time, ~2-3 hours)

**What it does**: Pre-computes CLIP embeddings for all 73K NSD images

```bash
make build-clip-cache CACHE=outputs/clip_cache/clip.parquet BATCH=256
```

**Output**: `outputs/clip_cache/clip.parquet` (~500MB)

**Monitor progress**:
```bash
# In another terminal
watch -n 10 "ls -lh outputs/clip_cache/clip.parquet"
```

**Verify**:
```bash
python -c "import pandas as pd; df = pd.read_parquet('outputs/clip_cache/clip.parquet'); print(f'{len(df)} images cached')"
# Should print: 73000 images cached
```

---

### Step 3: Preprocessing (NEW - With Soft Reliability)

#### Option A: Baseline (Hard Threshold)

```bash
python -m fmri2img.data.preprocess \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --output-dir outputs/preproc/baseline \
    --reliability-mode hard_threshold \
    --reliability-threshold 0.1 \
    --n-components 3072
```

**Output**: `outputs/preproc/baseline/subj01/`
- `scaler_mean.npy`, `scaler_std.npy`
- `reliability_mask.npy` (binary)
- `pca_components.npy`
- `meta.json`

#### Option B: Novel (Soft Weighting) ⭐

```bash
python -m fmri2img.data.preprocess \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --output-dir outputs/preproc/soft \
    --reliability-mode soft_weight \
    --reliability-threshold 0.1 \
    --reliability-curve sigmoid \
    --reliability-temperature 0.1 \
    --n-components 3072
```

**Output**: `outputs/preproc/soft/subj01/`
- Same as baseline, **plus**:
- `reliability_weights.npy` ⭐ (continuous [0,1])

**Verify soft weights**:
```python
import numpy as np
weights = np.load('outputs/preproc/soft/subj01/reliability_weights.npy')
mask = np.load('outputs/preproc/soft/subj01/reliability_mask.npy')

print(f"Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
print(f"Mask has {mask.sum()} voxels")
print(f"Continuous: {not np.allclose(weights, mask.astype(float))}")
# Should show: range [0, 1], continuous=True
```

---

### Step 4: Training

#### A. Train Baseline MLP (Hard threshold, no InfoNCE)

```bash
python scripts/train_mlp.py \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --clip-cache outputs/clip_cache/clip.parquet \
    --preproc-dir outputs/preproc/baseline \
    --output-dir checkpoints/mlp_baseline/subj01 \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --cosine-weight 1.0 \
    --mse-weight 0.0 \
    --infonce-weight 0.0 \
    --save-every 10
```

**Time**: ~2 hours on RTX 3090

**Output**: 
- `checkpoints/mlp_baseline/subj01/best_model.pt`
- `checkpoints/mlp_baseline/subj01/training_log.json`

#### B. Train with Soft Reliability Only ⭐

```bash
python scripts/train_mlp.py \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --clip-cache outputs/clip_cache/clip.parquet \
    --preproc-dir outputs/preproc/soft \
    --output-dir checkpoints/mlp_soft/subj01 \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --cosine-weight 1.0 \
    --mse-weight 0.0 \
    --infonce-weight 0.0 \
    --save-every 10
```

#### C. Train with InfoNCE Only ⭐

```bash
python scripts/train_mlp.py \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --clip-cache outputs/clip_cache/clip.parquet \
    --preproc-dir outputs/preproc/baseline \
    --output-dir checkpoints/mlp_infonce/subj01 \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --cosine-weight 1.0 \
    --mse-weight 0.0 \
    --infonce-weight 0.3 \
    --temperature 0.07 \
    --save-every 10
```

**Note**: Requires batch-size ≥16 for InfoNCE to work well!

#### D. Train with Both Novel Contributions ⭐⭐

```bash
python scripts/train_mlp.py \
    --subject subj01 \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --clip-cache outputs/clip_cache/clip.parquet \
    --preproc-dir outputs/preproc/soft \
    --output-dir checkpoints/mlp_full_novel/subj01 \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --cosine-weight 1.0 \
    --mse-weight 0.0 \
    --infonce-weight 0.3 \
    --temperature 0.07 \
    --save-every 10
```

**Monitor training**:
```bash
tail -f checkpoints/mlp_full_novel/subj01/train.log

# Look for lines like:
# Epoch 10: train_loss=0.2534 (cosine=0.2189, infonce=0.1149)
```

---

### Step 5: Evaluation

#### A. Standard Metrics (Existing Script)

```bash
python scripts/run_reconstruct_and_eval.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/mlp_full_novel/subj01/best_model.pt \
    --encoder-type mlp \
    --clip-cache outputs/clip_cache/clip.parquet \
    --output-dir outputs/eval/full_novel \
    --split test \
    --limit 500
```

**Output**: 
- `outputs/eval/full_novel/metrics.json`
  - cosine_similarity
  - retrieval_top1, top5, top10, top20, top50
  - mean_rank, median_rank

**Time**: ~5-10 minutes for 500 samples

#### B. Uncertainty Evaluation ⭐ (NEW)

You need to **create this script** since it doesn't exist yet. Here's the code:

```bash
# Create the evaluation script
cat > scripts/eval_uncertainty.py << 'EOF'
#!/usr/bin/env python3
"""Evaluate model with MC dropout uncertainty estimation."""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from fmri2img.models.mlp import MLPEncoder
from fmri2img.data.datasets import NSDDataset
from fmri2img.eval.uncertainty import (
    predict_with_mc_dropout,
    compute_uncertainty_error_correlation,
    plot_calibration_curve
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='subj01')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--index-file', type=str, required=True)
    parser.add_argument('--clip-cache', type=str, required=True)
    parser.add_argument('--preproc-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=20,
                        help='Number of MC dropout samples')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = MLPEncoder(
        input_dim=checkpoint['config']['input_dim'],
        hidden_dims=checkpoint['config']['hidden_dims'],
        output_dim=checkpoint['config']['output_dim'],
        dropout=checkpoint['config'].get('dropout', 0.3)
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load dataset
    print(f"Loading dataset: {args.split}")
    dataset = NSDDataset(
        index_file=args.index_file,
        clip_cache=args.clip_cache,
        preproc_dir=args.preproc_dir,
        subject=args.subject,
        split=args.split
    )
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))

    # Run MC dropout inference
    print(f"Running MC dropout inference (n={args.n_samples} samples)")
    results = []
    
    for i in tqdm(range(len(dataset))):
        fmri, clip_target = dataset[i]
        fmri = fmri.unsqueeze(0).to(args.device)
        clip_target = clip_target.unsqueeze(0).to(args.device)
        
        # MC dropout prediction
        mc_result = predict_with_mc_dropout(
            model, fmri, n_samples=args.n_samples, device=args.device
        )
        
        # Compute error
        pred_mean = torch.from_numpy(mc_result['mean']).to(args.device)
        error = 1.0 - torch.nn.functional.cosine_similarity(
            pred_mean, clip_target, dim=-1
        ).item()
        
        results.append({
            'trial_idx': i,
            'uncertainty': mc_result['uncertainty'],
            'error': error,
            'std_norm': mc_result['std_norm']
        })
    
    # Compute correlation
    uncertainties = np.array([r['uncertainty'] for r in results])
    errors = np.array([r['error'] for r in results])
    
    correlation = compute_uncertainty_error_correlation(
        uncertainties, errors, method='pearson'
    )
    
    print(f"\n{'='*60}")
    print(f"Uncertainty-Error Correlation: {correlation:.4f}")
    print(f"Mean uncertainty: {uncertainties.mean():.4f} ± {uncertainties.std():.4f}")
    print(f"Mean error: {errors.mean():.4f} ± {errors.std():.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'uncertainty_results.csv', index=False)
    
    summary = {
        'correlation_pearson': float(correlation),
        'mean_uncertainty': float(uncertainties.mean()),
        'std_uncertainty': float(uncertainties.std()),
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'n_samples': len(results),
        'mc_dropout_samples': args.n_samples
    }
    
    with open(output_dir / 'uncertainty_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot calibration curve
    print("Generating calibration plot...")
    plot_calibration_curve(
        uncertainties, errors,
        save_path=output_dir / 'calibration_curve.png'
    )
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - uncertainty_results.csv")
    print(f"  - uncertainty_summary.json")
    print(f"  - calibration_curve.png")


if __name__ == '__main__':
    main()
EOF

chmod +x scripts/eval_uncertainty.py
```

**Now run it**:

```bash
python scripts/eval_uncertainty.py \
    --subject subj01 \
    --checkpoint checkpoints/mlp_full_novel/subj01/best_model.pt \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --clip-cache outputs/clip_cache/clip.parquet \
    --preproc-dir outputs/preproc/soft \
    --output-dir outputs/eval/uncertainty \
    --n-samples 20 \
    --split test \
    --limit 500
```

**Output**:
- `outputs/eval/uncertainty/uncertainty_results.csv`
- `outputs/eval/uncertainty/uncertainty_summary.json`
- `outputs/eval/uncertainty/calibration_curve.png` ⭐

**Time**: ~10-20 minutes for 500 samples (20 forward passes each)

---

### Step 6: Image Reconstruction (Optional)

**Using existing script** `scripts/decode_diffusion.py`:

```bash
python scripts/decode_diffusion.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/mlp_full_novel/subj01/best_model.pt \
    --encoder-type mlp \
    --index-file data/indices/nsd_index/subject=subj01/index.parquet \
    --preproc-dir outputs/preproc/soft \
    --output-dir outputs/recon/full_novel \
    --num-samples 16 \
    --num-inference-steps 50 \
    --split test
```

**Time**: ~5-10 minutes for 16 images

**Output**: `outputs/recon/full_novel/*.png`

---

### Step 7: Compare All Ablations

```bash
python scripts/compare_evals.py \
    outputs/eval/baseline/metrics.json \
    outputs/eval/soft/metrics.json \
    outputs/eval/infonce/metrics.json \
    outputs/eval/full_novel/metrics.json \
    --labels "Baseline" "Soft Only" "InfoNCE Only" "Full Novel" \
    --output outputs/reports/ablation_comparison.csv
```

**Output**: CSV table with bootstrap confidence intervals

---

## 📊 Summary Table

| Step | Command | Time | Output | One-Time? |
|------|---------|------|--------|-----------|
| 1. Index | `make index` | 10m | `data/indices/` | ✅ |
| 2. CLIP Cache | `make build-clip-cache` | 2-3h | `outputs/clip_cache/` | ✅ |
| 3. Preprocess | `python -m fmri2img.data.preprocess` | 5-10m | `outputs/preproc/` | Per config |
| 4. Train | `python scripts/train_mlp.py` | 2h | `checkpoints/` | Per config |
| 5. Eval | `python scripts/run_reconstruct_and_eval.py` | 10m | `outputs/eval/` | Per model |
| 5b. Uncertainty | `python scripts/eval_uncertainty.py` | 15m | `outputs/eval/uncertainty/` | Per model |
| 6. Reconstruct | `python scripts/decode_diffusion.py` | 10m | `outputs/recon/` | Optional |
| 7. Compare | `python scripts/compare_evals.py` | 1m | `outputs/reports/` | Final |

**Total time (first run)**: ~8-10 hours
**Total time (subsequent runs with different configs)**: ~2-3 hours

---

## 🎯 Minimal Workflow for Paper Results

If you already have the index and CLIP cache:

```bash
# 1. Preprocess (soft weights)
python -m fmri2img.data.preprocess \
    --subject subj01 \
    --output-dir outputs/preproc/soft \
    --reliability-mode soft_weight

# 2. Train (with InfoNCE)
python scripts/train_mlp.py \
    --subject subj01 \
    --preproc-dir outputs/preproc/soft \
    --output-dir checkpoints/mlp_full_novel/subj01 \
    --infonce-weight 0.3 \
    --epochs 150

# 3. Evaluate standard metrics
python scripts/run_reconstruct_and_eval.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/mlp_full_novel/subj01/best_model.pt \
    --encoder-type mlp \
    --output-dir outputs/eval/full_novel

# 4. Evaluate uncertainty (NEW)
python scripts/eval_uncertainty.py \
    --subject subj01 \
    --checkpoint checkpoints/mlp_full_novel/subj01/best_model.pt \
    --output-dir outputs/eval/uncertainty \
    --n-samples 20
```

**Total: 4 commands, ~3-4 hours**

---

## ⚠️ Common Issues

### 1. "FileNotFoundError: index.parquet not found"

**Fix**: Run Step 1 first
```bash
make index SUBJECTS=subj01
```

### 2. "CLIP cache is empty or missing"

**Fix**: Run Step 2 first
```bash
make build-clip-cache
```

### 3. "ModuleNotFoundError: No module named 'fmri2img'"

**Fix**: Install package
```bash
pip install -e .
```

### 4. "No reliability_weights.npy found"

This is **OK** - the code falls back to the binary mask for backward compatibility.

To generate weights, re-run preprocessing with `--reliability-mode soft_weight`.

### 5. "InfoNCE returns NaN"

**Fixes**:
- Increase batch size (≥16)
- Reduce `--infonce-weight` (try 0.1 instead of 0.3)
- Increase `--temperature` (try 0.10 instead of 0.07)

---

## 🔗 Related Documentation

- **Quick Reference**: `docs/NOVEL_CONTRIBUTIONS_QUICK_REF.md`
- **Full Pipeline Guide**: `docs/guides/NOVEL_CONTRIBUTIONS_PIPELINE.md`
- **Implementation Details**: `docs/NOVEL_CONTRIBUTIONS_IMPLEMENTATION.md`

---

## ✅ Pre-Flight Checklist

Before running experiments:

- [ ] Environment activated (`conda activate fmri2img`)
- [ ] Package installed (`pip install -e .`)
- [ ] Tests passing (`pytest tests/test_*.py`)
- [ ] Index built (`data/indices/nsd_index/`)
- [ ] CLIP cache built (`outputs/clip_cache/clip.parquet`)
- [ ] GPU available (`nvidia-smi`)
- [ ] Disk space available (≥100GB free)

---

## 🚀 Ready to Run!

You now have the **complete, realistic workflow** from zero to paper-ready results.

**Estimated total time**:
- First run (with data prep): 8-10 hours
- Ablation experiments: 2-3 hours per config
- Full 4-way ablation study: ~10-12 hours total
