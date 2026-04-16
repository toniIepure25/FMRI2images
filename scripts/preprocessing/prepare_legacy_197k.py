#!/usr/bin/env python3
"""Prepare 197376-D legacy predictions and ground truth for N1v28a.

The N1v28a model outputs (N, 257, 768) = (N, 197376) predictions.
Ground truth must match this dimension. This script:
  1. Restores 197376-D predictions from backup if CLS-extracted
  2. Generates 197376-D GT for train from the CLIP token HDF5 cache
  3. Copies 197376-D GT for val/shared1000 from V35's embedded legacy GT
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n1v28a-dir",
        type=str,
        default="experimental_results/N1v28a_dual_head/subj01/metrics",
    )
    parser.add_argument(
        "--v35-dir",
        type=str,
        default="experimental_results/V35_legacy_teacher_distill/subj01/metrics",
    )
    parser.add_argument(
        "--token-cache",
        type=str,
        default="outputs/clip_cache/tokens_ViT-L-14_projected.h5",
    )
    args = parser.parse_args()

    n1 = Path(args.n1v28a_dir)
    v35 = Path(args.v35_dir)

    # ── Step 1: Restore 197376-D predictions from backups ──────────────
    for split in ["train", "val", "shared1000"]:
        backup = n1 / f"{split}_predictions_197k.npy"
        target = n1 / f"{split}_predictions_compact.npy"

        if backup.exists():
            current = np.load(target)
            if current.shape[1] != 197376:
                logger.info("Restoring %s predictions from 197k backup", split)
                shutil.copy2(str(backup), str(target))
            else:
                logger.info("  %s predictions already 197376-D", split)
        else:
            current = np.load(target)
            if current.shape[1] == 197376:
                logger.info("  %s predictions already 197376-D (no backup needed)", split)
            else:
                # Check if non-compact version exists
                fallback = n1 / f"{split}_predictions.npy"
                if fallback.exists():
                    fb = np.load(fallback)
                    if fb.shape[1] == 197376:
                        logger.info("Restoring %s from predictions.npy", split)
                        np.save(str(target), fb)
                    else:
                        logger.warning("  %s: no 197376-D source found!", split)

    # ── Step 2: Generate 197376-D GT for train from HDF5 ──────────────
    train_gt_path = n1 / "train_ground_truth_compact.npy"
    train_nsd_ids = np.load(n1 / "train_nsd_ids.npy")
    train_gt_current = np.load(train_gt_path)

    if train_gt_current.shape[1] == 197376:
        logger.info("Train GT already 197376-D")
    else:
        logger.info("Generating 197376-D train GT from token cache...")
        with h5py.File(args.token_cache, "r") as f:
            cache_nsd_ids = f["nsd_ids"][:]
            cache_tokens = f["tokens"]  # (37000, 257, 768)

            id_to_idx = {int(nid): i for i, nid in enumerate(cache_nsd_ids)}

            train_gt_197k = np.zeros((len(train_nsd_ids), 197376), dtype=np.float32)
            missing = 0
            for i, nid in enumerate(train_nsd_ids):
                nid_int = int(nid)
                # Token cache may use 1-indexed nsd_ids
                idx = id_to_idx.get(nid_int) or id_to_idx.get(nid_int + 1)
                if idx is not None:
                    tokens = cache_tokens[idx]  # (257, 768)
                    flat = tokens.reshape(-1)  # (197376,)
                    train_gt_197k[i] = flat
                else:
                    missing += 1

            if missing > 0:
                logger.warning("  %d/%d train nsd_ids not found in token cache", missing, len(train_nsd_ids))

            train_gt_197k = _l2_normalize(train_gt_197k)
            np.save(str(train_gt_path), train_gt_197k)
            logger.info("  Saved train GT: %s", train_gt_197k.shape)
            # Also save as non-compact name
            np.save(str(n1 / "train_ground_truth.npy"), train_gt_197k)

    # ── Step 3: Copy val/shared1000 GT from V35's legacy GT ───────────
    for split in ["val", "shared1000"]:
        v35_legacy_gt = v35 / f"{split}_ground_truth_legacy.npy"
        n1_gt_compact = n1 / f"{split}_ground_truth_compact.npy"
        n1_gt = n1 / f"{split}_ground_truth.npy"

        if v35_legacy_gt.exists():
            v35_gt = np.load(v35_legacy_gt)
            if v35_gt.shape[1] == 197376:
                # Verify nsd_id alignment
                n1_nsd = np.load(n1 / f"{split}_nsd_ids.npy")
                v35_nsd = np.load(v35 / f"{split}_nsd_ids.npy")

                if np.array_equal(n1_nsd, v35_nsd):
                    np.save(str(n1_gt_compact), v35_gt)
                    logger.info("  %s GT: copied from V35 legacy (same order), shape=%s", split, v35_gt.shape)
                else:
                    v35_id2idx = {int(nid): i for i, nid in enumerate(v35_nsd)}
                    reordered = np.zeros_like(v35_gt)
                    for i, nid in enumerate(n1_nsd):
                        reordered[i] = v35_gt[v35_id2idx[int(nid)]]
                    np.save(str(n1_gt_compact), reordered)
                    logger.info("  %s GT: copied from V35 legacy (reindexed), shape=%s", split, reordered.shape)
            else:
                logger.warning("  %s: V35 legacy GT is not 197376-D: %s", split, v35_gt.shape)
        else:
            # Check if N1v28a already has 197376-D GT (original Mar 19 files)
            if n1_gt.exists():
                current = np.load(n1_gt)
                if current.shape[1] == 197376:
                    np.save(str(n1_gt_compact), current)
                    logger.info("  %s GT: using existing 197376-D from ground_truth.npy", split)
                else:
                    logger.warning("  %s: no 197376-D GT source found!", split)

    # ── Step 4: Final verification ────────────────────────────────────
    print("\n=== Final dimension check ===")
    all_ok = True
    for split in ["train", "val", "shared1000"]:
        p = n1 / f"{split}_predictions_compact.npy"
        g = n1 / f"{split}_ground_truth_compact.npy"

        if p.exists() and g.exists():
            pa = np.load(p, mmap_mode="r")
            ga = np.load(g, mmap_mode="r")
            match = pa.shape[1] == ga.shape[1]
            status = "OK" if match else "MISMATCH"
            print(f"  {split}: pred={pa.shape}, gt={ga.shape} [{status}]")
            if not match:
                all_ok = False
        else:
            print(f"  {split}: MISSING files")
            all_ok = False

    if all_ok:
        print("\nAll dimensions match! Ready for GPU-accelerated cache building.")
    else:
        print("\nWARNING: Some dimensions still mismatch!")


if __name__ == "__main__":
    main()
