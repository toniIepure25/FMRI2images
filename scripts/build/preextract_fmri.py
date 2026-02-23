#!/usr/bin/env python3
"""
Pre-extract ROI-masked fMRI features from NIfTI beta files.

Reads the NSD index parquet, groups trials by session NIfTI file, loads each
file exactly once, applies the ROI mask, and saves all masked feature vectors
as a contiguous float32 numpy array.  This eliminates per-step NIfTI I/O
during training and reduces memory from ~150 GB to ~2 GB.

Output per subject (under cache/preextracted/subject={SUBJECT}/):
    fmri_features.npy  -- (N_trials, N_voxels) float32
    trial_meta.parquet  -- original index row order + nsdId for alignment
    meta.json           -- n_trials, n_voxels, roi_mask_path, timestamp

Usage:
    python scripts/build/preextract_fmri.py --subject subj01
    python scripts/build/preextract_fmri.py --subject subj01 --output-dir cache/preextracted/subject=subj01
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_roi_mask_path(subject: str) -> Path:
    """Search multiple candidate locations for the nsdgeneral ROI mask."""
    nsd_root = os.environ.get("NSD_DATA_ROOT", "data/nsd")
    candidates = [
        Path(nsd_root) / "nsddata" / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz",
        Path(nsd_root) / "nsddata" / "ppdata" / subject / "func1pt8mm" / "nsdgeneral.nii.gz",
        Path(nsd_root) / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz",
        Path(nsd_root) / "ppdata" / subject / "func1pt8mm" / "nsdgeneral.nii.gz",
        Path("data") / "nsd" / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract ROI-masked fMRI features from NIfTI beta files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g. subj01)")
    parser.add_argument(
        "--index-file", type=str, default=None,
        help="Path to index parquet (default: data/indices/nsd_index/subject={SUBJECT}/index.parquet)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: cache/preextracted/subject={SUBJECT})",
    )
    parser.add_argument(
        "--roi-mask", type=str, default=None,
        help="Path to ROI mask NIfTI (default: auto-resolve nsdgeneral.nii.gz)",
    )
    args = parser.parse_args()

    subject = args.subject
    index_file = Path(args.index_file) if args.index_file else Path(f"data/indices/nsd_index/subject={subject}/index.parquet")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"cache/preextracted/subject={subject}")

    if not index_file.exists():
        logger.error("Index not found: %s", index_file)
        logger.error("Run: make index SUBJECT=%s", subject)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Subject: %s", subject)
    logger.info("Index:   %s", index_file)
    logger.info("Output:  %s", output_dir)

    # --- Load index ---
    index_df = pd.read_parquet(index_file)
    n_trials = len(index_df)
    logger.info("Loaded index: %d trials", n_trials)

    # --- Load ROI mask ---
    roi_mask_path = Path(args.roi_mask) if args.roi_mask else resolve_roi_mask_path(subject)
    if not roi_mask_path.exists():
        logger.error("ROI mask not found: %s", roi_mask_path)
        sys.exit(1)

    mask_img = nib.load(str(roi_mask_path))
    roi_mask = mask_img.get_fdata() > 0.5
    n_voxels = int(roi_mask.sum())
    logger.info("ROI mask: %s (%d voxels)", roi_mask_path, n_voxels)

    # --- Pre-allocate output array ---
    features = np.zeros((n_trials, n_voxels), dtype=np.float32)

    # --- Group trials by session file for efficient I/O ---
    beta_col = "beta_path"
    idx_col = "beta_index" if "beta_index" in index_df.columns else "volume_index"

    groups = OrderedDict()
    for row_idx, row in index_df.iterrows():
        bp = row[beta_col]
        if bp not in groups:
            groups[bp] = []
        groups[bp].append((row_idx, int(row[idx_col])))

    n_sessions = len(groups)
    logger.info("Grouped into %d session files", n_sessions)

    # --- Extract features ---
    t0 = time.time()
    rows_done = 0

    for sess_i, (beta_path, trials) in enumerate(groups.items()):
        logger.info(
            "[%d/%d] Loading %s (%d trials)",
            sess_i + 1, n_sessions, Path(beta_path).name, len(trials),
        )
        img = nib.load(beta_path)
        data_4d = img.get_fdata(dtype=np.float32)

        for row_idx, vol_idx in trials:
            vol_3d = data_4d[..., vol_idx]
            features[row_idx] = vol_3d[roi_mask]

        rows_done += len(trials)
        elapsed = time.time() - t0
        rate = rows_done / elapsed if elapsed > 0 else 0
        logger.info(
            "  Progress: %d/%d trials (%.1f%%) — %.0f trials/s",
            rows_done, n_trials, 100 * rows_done / n_trials, rate,
        )

        del data_4d

    elapsed_total = time.time() - t0
    logger.info("Extraction complete: %d trials in %.1f seconds", n_trials, elapsed_total)

    # --- Save features ---
    features_path = output_dir / "fmri_features.npy"
    np.save(features_path, features)
    logger.info("Saved features: %s (%.2f GB)", features_path, features.nbytes / 1e9)

    # --- Save trial metadata (preserves row alignment) ---
    meta_df = index_df[["nsdId"]].copy()
    if "session" in index_df.columns:
        meta_df["session"] = index_df["session"]
    meta_df.to_parquet(output_dir / "trial_meta.parquet", index=True)
    logger.info("Saved trial metadata: %s", output_dir / "trial_meta.parquet")

    # --- Save meta.json ---
    meta = {
        "subject": subject,
        "n_trials": n_trials,
        "n_voxels": n_voxels,
        "roi_mask_path": str(roi_mask_path),
        "index_file": str(index_file),
        "dtype": "float32",
        "shape": list(features.shape),
        "size_gb": round(features.nbytes / 1e9, 3),
        "extraction_time_s": round(elapsed_total, 1),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved meta: %s", output_dir / "meta.json")

    logger.info("Done! Pre-extracted features for %s: %d trials x %d voxels = %.2f GB",
                subject, n_trials, n_voxels, features.nbytes / 1e9)


if __name__ == "__main__":
    main()
