#!/usr/bin/env python3
"""
Create Mock Preprocessing Files for Testing
============================================

Generates dummy but structurally valid preprocessing files to enable
testing the training pipeline without needing to download 100GB of fMRI data.

Usage:
    python scripts/create_mock_preprocessing.py --subject subj01 --k 512
"""

import argparse
import json
import numpy as np
from pathlib import Path

def create_mock_preprocessing(subject: str, k: int = 512, out_dir: str = "outputs/preproc"):
    """Create mock preprocessing files with correct structure."""
    
    # Create output directory
    subj_dir = Path(out_dir) / subject
    subj_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating mock preprocessing files for {subject}...")
    print(f"Output directory: {subj_dir}")
    
    # Standard NSD voxel dimensions (1.8mm isotropic)
    voxel_shape = (83, 104, 81)  # Typical NSD dimensions
    n_voxels_total = np.prod(voxel_shape)
    
    # Create reliability mask (keep ~50% of voxels, typical for NSD)
    print(f"\nGenerating reliability mask ({voxel_shape})...")
    mask = np.random.rand(*voxel_shape) > 0.5
    n_voxels_kept = mask.sum()
    print(f"  Keeping {n_voxels_kept:,} / {n_voxels_total:,} voxels ({100*n_voxels_kept/n_voxels_total:.1f}%)")
    
    # Save reliability mask
    np.save(subj_dir / "reliability_mask.npy", mask)
    
    # Create scaler (mean and std for each voxel)
    print("\nGenerating scaler parameters...")
    scaler_mean = np.random.randn(*voxel_shape).astype(np.float32) * 100  # Typical BOLD scale
    scaler_std = np.random.rand(*voxel_shape).astype(np.float32) * 50 + 10  # Positive std
    np.save(subj_dir / "scaler_mean.npy", scaler_mean)
    np.save(subj_dir / "scaler_std.npy", scaler_std)
    
    # Create voxel indices (flat indices of kept voxels)
    print("\nGenerating voxel indices...")
    voxel_indices = np.where(mask.ravel())[0]
    np.save(subj_dir / "voxel_indices.npy", voxel_indices)
    
    # Create PCA components
    k_eff = min(k, n_voxels_kept, 24000)  # Can't exceed n_features or n_samples
    if k_eff < k:
        print(f"\n⚠️  Requested k={k}, but capping to k_eff={k_eff} (n_voxels_kept={n_voxels_kept})")
    
    print(f"\nGenerating PCA with {k_eff} components...")
    pca_components = np.random.randn(k_eff, n_voxels_kept).astype(np.float32)
    pca_mean = np.random.randn(n_voxels_kept).astype(np.float32) * 10
    
    # Normalize components (unit norm)
    for i in range(k_eff):
        pca_components[i] /= np.linalg.norm(pca_components[i])
    
    np.save(subj_dir / "pca_components.npy", pca_components)
    np.save(subj_dir / "pca_mean.npy", pca_mean)
    
    # Create metadata
    print("\nGenerating metadata...")
    meta = {
        "subject": subject,
        "roi_mode": None,
        "n_train_samples": 24000,  # Typical NSD train split
        "n_voxels_total": int(n_voxels_total),
        "n_voxels_kept": int(n_voxels_kept),
        "voxel_retention_rate": float(n_voxels_kept / n_voxels_total),
        "reliability_method": "mock",
        "reliability_threshold": 0.0,
        "split_half_seed": None,
        "pca_fitted": True,
        "pca_components": k_eff,
        "explained_variance_ratio": 0.95,  # Mock value
        "note": "Mock preprocessing generated for testing"
    }
    
    with open(subj_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Create reliability metadata
    rel_meta = {
        "method": "mock",
        "reliability_threshold": 0.0,
        "n_repeated_ids": 0,
        "seed": None,
        "mean_r_retained": 0.0,
        "note": "Mock reliability metadata"
    }
    
    with open(subj_dir / "reliability_meta.json", 'w') as f:
        json.dump(rel_meta, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ Mock Preprocessing Files Created!")
    print("="*70)
    print(f"\nArtifacts in: {subj_dir}/")
    for artifact in sorted(subj_dir.glob("*")):
        if artifact.is_file():
            size_mb = artifact.stat().st_size / (1024 * 1024)
            print(f"  ✓ {artifact.name:30s} ({size_mb:6.2f} MB)")
    
    print("\n" + "="*70)
    print("METADATA:")
    print("="*70)
    print(f"  Subject: {meta['subject']}")
    print(f"  Voxels kept: {meta['n_voxels_kept']:,} / {meta['n_voxels_total']:,} ({100*meta['voxel_retention_rate']:.1f}%)")
    print(f"  PCA components: {meta['pca_components']}")
    print(f"  Train samples: {meta['n_train_samples']:,}")
    print(f"  Explained variance: {meta['explained_variance_ratio']:.1%}")
    
    print("\n" + "="*70)
    print("⚠️  WARNING: These are MOCK files for testing only!")
    print("="*70)
    print("These files have the correct structure but contain random data.")
    print("Use them to test the training pipeline, but don't expect meaningful results.")
    print("\nTo use with training:")
    print(f"  python scripts/train_two_stage.py \\")
    print(f"    --config configs/sota_two_stage.yaml \\")
    print(f"    --subject {subject} \\")
    print(f"    --limit 100 \\")  # Use small limit for testing
    print(f"    --output-dir checkpoints/two_stage/test")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Create mock preprocessing files for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--k", type=int, default=512, help="Number of PCA components")
    parser.add_argument("--out-dir", default="outputs/preproc", help="Output directory")
    
    args = parser.parse_args()
    
    create_mock_preprocessing(args.subject, args.k, args.out_dir)


if __name__ == "__main__":
    main()
