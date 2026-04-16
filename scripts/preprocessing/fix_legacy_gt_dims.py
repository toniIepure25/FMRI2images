#!/usr/bin/env python3
"""Fix N1v28a ground truth dimensions for shared1000.

The N1v28a model produces 197376-D predictions (257 tokens x 768).
The predictions have been CLS-extracted to 768-D, but shared1000 GT
is still 197376-D. This copies the 768-D GT from V35's directory.
"""
import numpy as np
from pathlib import Path

n1_base = Path("experimental_results/N1v28a_dual_head/subj01/metrics")
v35_base = Path("experimental_results/V35_legacy_teacher_distill/subj01/metrics")

n1_nsd = np.load(n1_base / "shared1000_nsd_ids.npy")
v35_nsd = np.load(v35_base / "shared1000_nsd_ids.npy")

print(f"N1v28a nsd_ids: {n1_nsd.shape}, range [{n1_nsd.min()}, {n1_nsd.max()}]")
print(f"V35 nsd_ids:    {v35_nsd.shape}, range [{v35_nsd.min()}, {v35_nsd.max()}]")

common = set(n1_nsd.tolist()) & set(v35_nsd.tolist())
print(f"Common IDs: {len(common)} / {len(n1_nsd)}")

if len(common) == len(n1_nsd) and len(common) == len(v35_nsd):
    v35_gt = np.load(v35_base / "shared1000_ground_truth_compact.npy")
    print(f"V35 GT compact: {v35_gt.shape}")

    if np.array_equal(n1_nsd, v35_nsd):
        print("Same order! Direct copy.")
        np.save(str(n1_base / "shared1000_ground_truth_compact.npy"), v35_gt)
    else:
        print("Different order, reindexing...")
        v35_id2idx = {int(nid): i for i, nid in enumerate(v35_nsd)}
        reordered_gt = np.zeros((len(n1_nsd), 768), dtype=np.float32)
        for i, nid in enumerate(n1_nsd):
            reordered_gt[i] = v35_gt[v35_id2idx[int(nid)]]
        np.save(str(n1_base / "shared1000_ground_truth_compact.npy"), reordered_gt)
        print(f"Saved reindexed GT: {reordered_gt.shape}")
    print("Done: shared1000_ground_truth_compact.npy created (768-D)")
else:
    print("WARNING: IDs don't fully match, extracting CLS from 197k GT")
    gt_197k = np.load(n1_base / "shared1000_ground_truth.npy")
    tokens = gt_197k.reshape(gt_197k.shape[0], 257, 768)
    cls_gt = tokens[:, 0, :].copy()
    norms = np.linalg.norm(cls_gt, axis=-1, keepdims=True)
    cls_gt = cls_gt / np.maximum(norms, 1e-8)
    np.save(str(n1_base / "shared1000_ground_truth_compact.npy"), cls_gt.astype(np.float32))
    print(f"Saved CLS-extracted GT: {cls_gt.shape}")

print("\n=== Final dimension check ===")
for split in ["train", "val", "shared1000"]:
    p = n1_base / f"{split}_predictions_compact.npy"
    g_compact = n1_base / f"{split}_ground_truth_compact.npy"
    g_fallback = n1_base / f"{split}_ground_truth.npy"
    g = g_compact if g_compact.exists() else g_fallback

    if p.exists() and g.exists():
        pa, ga = np.load(p), np.load(g)
        match = "OK" if pa.shape[1] == ga.shape[1] else "MISMATCH"
        print(f"  {split}: pred={pa.shape}, gt={ga.shape} [{match}]")
    else:
        print(f"  {split}: MISSING files")
