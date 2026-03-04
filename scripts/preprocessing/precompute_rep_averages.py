#!/usr/bin/env python3
"""
Pre-compute repetition-averaged fMRI features (V11).

NSD provides ~3 fMRI repetitions per image.  Averaging repetitions
improves SNR by sqrt(n_reps) ≈ 1.73x.  This script reads the raw
pre-extracted features and trial index, groups trials by nsdId,
averages, and saves the result alongside the original features.

The averaged file is automatically detected by train_unified.py when
``data.average_repetitions: true`` is set in the config.  If the
averaged file exists, it is loaded directly (skipping the per-epoch
averaging inside the dataset constructor).

Usage:
    python scripts/preprocessing/precompute_rep_averages.py --subject subj01
    python scripts/preprocessing/precompute_rep_averages.py --subject subj01 subj02 subj05 subj07
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def precompute_averages(subject: str, cache_root: str | None = None) -> None:
    cache = Path(cache_root or os.environ.get("CACHE_ROOT", "cache"))
    subj_dir = cache / "preextracted" / f"subject={subject}"

    features_path = subj_dir / "fmri_features.npy"
    meta_path = subj_dir / "trial_meta.parquet"
    out_path = subj_dir / "fmri_features_avg.npy"
    out_meta = subj_dir / "trial_meta_avg.parquet"

    if not features_path.exists():
        logger.error("Features not found: %s — run `make preextract SUBJECT=%s` first", features_path, subject)
        sys.exit(1)

    features = np.load(features_path)
    meta = pd.read_parquet(meta_path) if meta_path.exists() else None

    if meta is None or "nsdId" not in meta.columns:
        logger.error("trial_meta.parquet missing or lacks nsdId column")
        sys.exit(1)

    logger.info("Loaded %d trials x %d voxels for %s", *features.shape, subject)

    unique_ids = meta["nsdId"].unique()
    n_unique = len(unique_ids)
    avg_features = np.zeros((n_unique, features.shape[1]), dtype=np.float32)
    new_rows = []

    nsd_vals = meta["nsdId"].values
    for i, nsd_id in enumerate(unique_ids):
        mask = nsd_vals == nsd_id
        avg_features[i] = features[mask].mean(axis=0)
        new_rows.append(meta[mask].iloc[0].to_dict())

    avg_meta = pd.DataFrame(new_rows).reset_index(drop=True)

    np.save(out_path, avg_features)
    avg_meta.to_parquet(out_meta, index=False)

    snr_gain = np.sqrt(len(features) / n_unique)
    logger.info(
        "Saved %d averaged images to %s (SNR ~%.2fx, %.2f GB)",
        n_unique, out_path, snr_gain, avg_features.nbytes / 1e9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute repetition-averaged fMRI features")
    parser.add_argument("--subject", nargs="+", required=True, help="Subject IDs")
    parser.add_argument("--cache-root", default=None, help="Override CACHE_ROOT")
    args = parser.parse_args()

    for subj in args.subject:
        precompute_averages(subj, args.cache_root)


if __name__ == "__main__":
    main()
