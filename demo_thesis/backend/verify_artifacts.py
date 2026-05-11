#!/usr/bin/env python3
"""Verify that all backend artifacts are present and loadable."""
from __future__ import annotations

import os
import sys
from pathlib import Path

def _green(s: str) -> str: return f"\033[92m{s}\033[0m"
def _red(s: str) -> str: return f"\033[91m{s}\033[0m"
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m"

def check_torch():
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "n/a"
        print(f"  PyTorch:  {_green('OK')} v{torch.__version__}  cuda={'yes' if cuda else 'no'}  device={dev}")
        return True
    except ImportError:
        print(f"  PyTorch:  {_red('MISSING')} — install with: pip install torch")
        return False

def check_file(label: str, env_var: str, *fallbacks: str):
    val = os.environ.get(env_var)
    path = None
    if val and Path(val).exists():
        path = Path(val)
    else:
        for fb in fallbacks:
            if Path(fb).exists():
                path = Path(fb)
                break
    if not path:
        tried = [val or f"${env_var} (not set)"] + list(fallbacks)
        print(f"  {label}: {_red('MISSING')}")
        for t in tried:
            print(f"           tried: {t}")
        return None
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  {label}: {_green('OK')} {path} ({size_mb:.1f} MB)")
    return path

def check_npy(label: str, path: Path | None):
    if not path:
        return
    import numpy as np
    arr = np.load(str(path), mmap_mode='r')
    print(f"           shape={arr.shape}  dtype={arr.dtype}")

def check_parquet(label: str, path: Path | None):
    if not path:
        return
    try:
        import pandas as pd
        df = pd.read_parquet(str(path))
        print(f"           rows={len(df)}  columns={list(df.columns)[:8]}")
    except Exception as e:
        print(f"           {_yellow(f'read error: {e}')}")

def check_checkpoint(path: Path | None, has_torch: bool):
    if not path or not has_torch:
        return
    import torch
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    keys = list(ckpt.keys())
    cfg = ckpt.get('model_config', ckpt.get('config', {}).get('model', {}))
    sd = ckpt.get('model_state_dict', {})
    n_params = sum(v.numel() for v in sd.values())
    enc_type = cfg.get('encoder', {}).get('encoder_type', '?')
    out_dim = cfg.get('decoder', {}).get('output_dim', '?')
    print(f"           keys={keys}")
    print(f"           encoder={enc_type}  output_dim={out_dim}  params={n_params:,}")

def main():
    print("=" * 60)
    print("Cortex2Canvas Backend — Artifact Verification")
    print("=" * 60)
    print()

    has_torch = check_torch()
    print()

    subj = os.environ.get("C2C_SUBJECT", "subj01")
    fmri_path = check_file(
        "fMRI features", "C2C_FMRI_FEATURES_PATH",
        f"../local_backend_data/{subj}/fmri_features.npy",
        f"local_backend_data/{subj}/fmri_features.npy",
    )
    check_npy("fMRI features", fmri_path)

    index_path = check_file(
        "Trial index", "C2C_TRIAL_INDEX_PATH",
        f"../local_backend_data/{subj}/index.parquet",
        f"local_backend_data/{subj}/index.parquet",
    )
    check_parquet("Trial index", index_path)

    gallery_path = check_file(
        "CLIP gallery", "C2C_CLIP_GALLERY_PATH",
        "../local_backend_data/clip/clip.parquet",
        "local_backend_data/clip/clip.parquet",
    )
    check_parquet("CLIP gallery", gallery_path)

    ckpt_path = check_file(
        "Checkpoint", "C2C_CHECKPOINT_PATH",
        "../local_backend_data/checkpoints/V62a_model_only.pt",
        "local_backend_data/checkpoints/V62a_model_only.pt",
        "../local_backend_data/checkpoints/model.pt",
    )
    if ckpt_path and has_torch:
        check_checkpoint(ckpt_path, has_torch)

    print()
    print("-" * 60)
    all_data = all([fmri_path, index_path, gallery_path])
    has_model = ckpt_path is not None and has_torch
    if has_model and all_data:
        print(f"Status: {_green('INFERENCE READY')} — model + data loaded")
    elif all_data:
        print(f"Status: {_yellow('DATA READY')} — data loaded, model {'missing' if not ckpt_path else 'needs torch'}")
    else:
        missing = []
        if not fmri_path: missing.append("fmri_features")
        if not index_path: missing.append("trial_index")
        if not gallery_path: missing.append("clip_gallery")
        if not ckpt_path: missing.append("checkpoint")
        if not has_torch: missing.append("torch")
        print(f"Status: {_red('NOT READY')} — missing: {', '.join(missing)}")
    print("-" * 60)

if __name__ == "__main__":
    main()
