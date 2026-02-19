#!/usr/bin/env python3
"""
Quick status check for CLIP cache and data preparation
"""

import os
import sys
from pathlib import Path

def check_file(path, min_size_mb=0):
    """Check if file exists and meets size requirement"""
    p = Path(path)
    if not p.exists():
        return False, "Not found"
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb < min_size_mb:
        return False, f"Too small ({size_mb:.1f} MB)"
    return True, f"OK ({size_mb:.1f} MB)"

def main():
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)
    
    print("=" * 70)
    print("  SOTA Pipeline - Quick Status Check")
    print("=" * 70)
    print()
    
    # Check CLIP cache
    clip_cache = "outputs/clip_cache/clip.parquet"
    print("📊 CLIP Cache Status:")
    if Path(clip_cache).exists():
        try:
            import pandas as pd
            df = pd.read_parquet(clip_cache)
            num_embeddings = len(df)
            expected = 73000
            pct = 100 * num_embeddings / expected
            
            if num_embeddings >= expected * 0.95:
                status = "✅ COMPLETE"
            elif num_embeddings >= 1000:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ INCOMPLETE"
            
            print(f"  {status}")
            print(f"  {num_embeddings:,} / ~{expected:,} embeddings ({pct:.1f}%)")
            
            if num_embeddings < expected * 0.95:
                print()
                print("  → Action: Run CLIP cache builder")
                print(f"    python scripts/build_clip_cache.py \\")
                print(f"      --index-root data/indices/nsd_index \\")
                print(f"      --subject subj01 \\")
                print(f"      --cache {clip_cache} \\")
                print(f"      --batch-size 256")
                print()
                print("  ⏱️  Estimated time: ~2-3 hours on GPU")
                print("  💡 Tip: Use tmux/screen for long-running process")
                print("  💡 Resume: Script automatically skips cached embeddings")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    else:
        print(f"  ❌ NOT FOUND")
        print()
        print("  → Action: Build CLIP cache (REQUIRED)")
        print(f"    python scripts/build_clip_cache.py \\")
        print(f"      --index-root data/indices/nsd_index \\")
        print(f"      --subject subj01 \\")
        print(f"      --cache {clip_cache} \\")
        print(f"      --batch-size 256")
    
    print()
    
    # Check NSD index
    print("📁 NSD Index:")
    index_dir = Path("data/indices/nsd_index")
    if index_dir.exists():
        indices = list(index_dir.glob("subj*.csv"))
        if indices:
            print(f"  ✅ Found {len(indices)} subject indices")
            for idx in sorted(indices):
                size = idx.stat().st_size / 1024
                print(f"     - {idx.name} ({size:.1f} KB)")
        else:
            print(f"  ⚠️  Directory exists but no indices found")
    else:
        print(f"  ❌ NOT FOUND")
        print()
        print("  → Action: Build NSD index")
        print(f"    python scripts/build_full_index.py \\")
        print(f"      --cache-root cache \\")
        print(f"      --subject subj01 \\")
        print(f"      --output data/indices/nsd_index/subj01.csv")
    
    print()
    
    # Check preprocessing
    print("🔧 Preprocessing:")
    preproc_dir = Path("cache/preproc")
    if preproc_dir.exists():
        scalers = list(preproc_dir.glob("*_t1_scaler.pkl"))
        pcas = list(preproc_dir.glob("*_t2_pca_*.npz"))
        
        if scalers or pcas:
            print(f"  ✅ Found {len(scalers)} scalers, {len(pcas)} PCA files")
            for f in sorted(list(scalers) + list(pcas)):
                size = f.stat().st_size / 1024
                print(f"     - {f.name} ({size:.1f} KB)")
        else:
            print(f"  ⚠️  Directory exists but no preprocessing files")
    else:
        print(f"  ❌ NOT FOUND")
        print()
        print("  → Action: Run preprocessing (after index is built)")
        print(f"    # T1 scaler")
        print(f"    python scripts/preprocess_fmri.py \\")
        print(f"      --subject subj01 \\")
        print(f"      --method t1 \\")
        print(f"      --output cache/preproc/subj01_t1_scaler.pkl")
        print()
        print(f"    # T2 PCA")
        print(f"    python scripts/preprocess_fmri.py \\")
        print(f"      --subject subj01 \\")
        print(f"      --method t2 \\")
        print(f"      --pca-dim 512 \\")
        print(f"      --output cache/preproc/subj01_t2_pca_k512.npz")
    
    print()
    
    # Check trained models
    print("🤖 Trained Models:")
    ckpt_dir = Path("checkpoints/two_stage")
    if ckpt_dir.exists():
        models = list(ckpt_dir.glob("*/two_stage_best.pt"))
        if models:
            print(f"  ✅ Found {len(models)} trained models")
            for m in sorted(models):
                size = m.stat().st_size / (1024 * 1024)
                subj = m.parent.name
                print(f"     - {subj}: {size:.1f} MB")
        else:
            print(f"  ⚠️  No trained models found")
    else:
        print(f"  ❌ NOT FOUND")
        print()
        print("  → Action: Train model (after all data is prepared)")
        print(f"    python scripts/train_two_stage.py \\")
        print(f"      --config configs/sota_two_stage.yaml \\")
        print(f"      --subject subj01 \\")
        print(f"      --output-dir checkpoints/two_stage/subj01")
    
    print()
    print("=" * 70)
    print()
    print("📚 Documentation:")
    print("  - SETUP_GUIDE.md (complete setup steps)")
    print("  - USAGE_EXAMPLES.md (ready-to-run commands)")
    print("  - SOTA_QUICK_START.md (detailed guide)")
    print()

if __name__ == "__main__":
    main()
