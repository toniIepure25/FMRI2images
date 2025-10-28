#!/usr/bin/env python3
"""
Repair MLP Checkpoint Metadata
===============================

Patches an existing MLP checkpoint to add/update preprocessing metadata.

Usage:
    python scripts/repair_mlp_meta.py \
        --ckpt checkpoints/mlp/subj01/mlp.pt \
        --subject subj01 \
        --preproc-dir outputs/preproc/subj01/rel=0.100_k=4 \
        --k 4 \
        --thr 0.1
"""

import argparse
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Repair MLP checkpoint metadata")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to MLP checkpoint")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    parser.add_argument("--preproc-dir", type=Path, required=True, help="Preprocessing directory")
    parser.add_argument("--k", type=int, required=True, help="PCA components")
    parser.add_argument("--thr", type=float, required=True, help="Reliability threshold")
    parser.add_argument("--backup", type=bool, default=True, help="Create backup (default: True)")
    
    args = parser.parse_args()
    
    if not args.ckpt.exists():
        print(f"ERROR: Checkpoint not found: {args.ckpt}")
        return 1
    
    if not args.preproc_dir.exists():
        print(f"ERROR: Preprocessing directory not found: {args.preproc_dir}")
        return 1
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    # Ensure meta dict exists
    if "meta" not in ckpt:
        ckpt["meta"] = {}
    
    meta = ckpt["meta"]
    
    # Update input_dim if not set
    if "input_dim" not in meta or meta["input_dim"] is None:
        meta["input_dim"] = args.k
        print(f"✓ Set input_dim: {args.k}")
    else:
        print(f"✓ Existing input_dim: {meta['input_dim']}")
    
    # Set preprocessing metadata
    meta["preproc"] = {
        "used_preproc": True,
        "k": int(args.k),
        "reliability_thr": float(args.thr),
        "path": str(args.preproc_dir),
        "subject": args.subject
    }
    
    print(f"✓ Updated preprocessing metadata:")
    print(f"  - used_preproc: True")
    print(f"  - k: {args.k}")
    print(f"  - reliability_thr: {args.thr}")
    print(f"  - path: {args.preproc_dir}")
    print(f"  - subject: {args.subject}")
    
    # Create backup if requested
    if args.backup:
        backup_path = args.ckpt.with_suffix(".pt.bak")
        shutil.copy2(args.ckpt, backup_path)
        print(f"✓ Backup created: {backup_path}")
    
    # Save updated checkpoint
    torch.save(ckpt, args.ckpt)
    print(f"✓ Checkpoint saved: {args.ckpt}")
    print("\n✅ Metadata repair complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
