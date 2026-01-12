#!/usr/bin/env python3
"""
Standalone smoke test for NSD data - works without pip install
Just run: python test_real_data.py
"""
import sys
import os
from pathlib import Path

# Add src to path so we can import without pip install
repo_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(repo_root / "src"))
os.chdir(repo_root)

print(f"[DEBUG] Repo root: {repo_root}")
print(f"[DEBUG] Python path: {sys.path[:3]}")
print(f"[DEBUG] Working dir: {os.getcwd()}")
print()

import logging
import torch
import numpy as np

# Silence nibabel warnings
logging.getLogger("nibabel.global").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("smoke_test")

# Now import after path is set
try:
    from fmri2img.data.torch_dataset import NSDIterableDataset
    from fmri2img.io.nsd_layout import NSDLayout
    log.info("[DEBUG] ✓ Imports successful")
except ImportError as e:
    log.error(f"[DEBUG] ❌ Import failed: {e}")
    log.error(f"[DEBUG] Checking if files exist...")
    torch_ds = repo_root / "src" / "fmri2img" / "data" / "torch_dataset.py"
    log.error(f"[DEBUG] torch_dataset.py exists: {torch_ds.exists()}")
    if torch_ds.exists():
        log.error(f"[DEBUG] File path: {torch_ds}")
    sys.exit(1)

def main():
    log.info("=" * 80)
    log.info("NSD REAL DATA SMOKE TEST")
    log.info("=" * 80)
    log.info("")
    
    # Configuration
    subject = 'subj01'
    sessions = [1]
    limit = 8
    
    log.info(f"Configuration:")
    log.info(f"  Subject: {subject}")
    log.info(f"  Sessions: {sessions}")
    log.info(f"  Limit: {limit} samples")
    log.info("")
    
    # Check data location
    layout = NSDLayout()
    log.info(f"Data root: {layout.nsd_data_root}")
    
    beta_dir = Path(layout.nsd_data_root) / f"nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    if not beta_dir.exists():
        log.error(f"❌ Beta directory not found: {beta_dir}")
        return 1
    
    beta_files = list(beta_dir.glob("betas_session*.nii.gz"))
    log.info(f"✓ Found {len(beta_files)} beta files")
    log.info("")
    
    # Create dataset
    log.info("Creating NSD dataset...")
    try:
        dataset = NSDIterableDataset(
            subject=subject,
            sessions=sessions,
            limit=limit,
            shuffle=False
        )
        log.info("✓ Dataset created successfully")
        log.info("")
    except Exception as e:
        log.error(f"❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Load samples
    log.info(f"Loading {limit} samples...")
    log.info("-" * 80)
    
    samples = []
    try:
        for i, sample in enumerate(dataset, 1):
            fmri_shape = sample['fmri'].shape if isinstance(sample['fmri'], torch.Tensor) else sample['fmri'].shape
            has_image = 'image' in sample and sample['image'] is not None
            
            log.info(f"Sample {i}/{limit}:")
            log.info(f"  ✓ fMRI loaded: shape={fmri_shape}")
            log.info(f"  ✓ Image loaded: {has_image}")
            if 'nsd_id' in sample:
                log.info(f"  - NSD ID: {sample['nsd_id']}")
            if 'coco_id' in sample:
                log.info(f"  - COCO ID: {sample['coco_id']}")
            
            samples.append(sample)
            
            if i >= limit:
                break
                
    except Exception as e:
        log.error(f"❌ Error loading samples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    log.info("-" * 80)
    log.info("")
    
    # Summary
    log.info("=" * 80)
    log.info("✅ SMOKE TEST PASSED!")
    log.info("=" * 80)
    log.info("")
    log.info(f"Summary:")
    log.info(f"  • Loaded {len(samples)}/{limit} samples successfully")
    log.info(f"  • fMRI data shape: {samples[0]['fmri'].shape}")
    log.info(f"  • Images loaded: {sum(1 for s in samples if 'image' in s and s['image'] is not None)}/{len(samples)}")
    log.info("")
    log.info("🎉 Your NSD data is working! You can now run full experiments.")
    log.info("")
    log.info("Next steps:")
    log.info("  1. Run baseline: bash scripts/run_experiment_simple.sh experiments/baseline_subj01.yaml")
    log.info("  2. Run novel: bash scripts/run_experiment_simple.sh experiments/novel_subj01.yaml")
    log.info("")
    
    return 0

if __name__ == '__main__':
    exit(main())
