#!/usr/bin/env python3
"""Generate missing val_nsd_ids.npy for OOF folds from split files + subject index."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

base = Path("/home/jovyan/work/FMRI2images")
subject = "subj01"

index_path = base / f"data/indices/nsd_index/subject={subject}/index.parquet"
idx_df = pd.read_parquet(index_path)
available_nsd_ids = set(idx_df["nsdId"].unique())
print(f"Subject {subject}: {len(available_nsd_ids)} available unique nsd_ids")

manifest_path = base / "experimental_results/V55d_oof_folds/splits/oof_fold_manifest.json"
manifest = json.load(open(manifest_path))
n_folds = len(manifest["folds"])
print(f"Manifest loaded: {n_folds} folds\n")

for fold_info in manifest["folds"]:
    fold_idx = fold_info["fold_index"]
    split_path = base / f"experimental_results/V55d_oof_folds/splits/oof_fold_{fold_idx:02d}.json"
    metrics_dir = base / f"experimental_results/V55d_oof_fold_{fold_idx}/{subject}/metrics"

    split_data = json.load(open(split_path))
    fold_val_ids = set(int(x) for x in split_data["val_nsd_ids"])

    actual_val_ids = sorted(fold_val_ids & available_nsd_ids)

    preds = np.load(metrics_dir / "val_predictions_compact.npy")
    n_preds = preds.shape[0]

    print(f"Fold {fold_idx}: intersection={len(actual_val_ids)}, preds={n_preds}", end="")

    if n_preds == len(actual_val_ids):
        arr = np.array(actual_val_ids, dtype=np.int64)
        np.save(metrics_dir / "val_nsd_ids.npy", arr)
        print(f" -> Saved ({arr.shape})")
    else:
        print(f" !! MISMATCH")

print("\nDone.")
