#!/usr/bin/env python3
"""Batch-level alignment audit for NeuroBridge-OT.

Verifies that fMRI samples are correctly paired with CLIP targets by nsdId,
and that no data leakage or misalignment exists.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fmri2img.data.neurobridge_dataset import NeuroBridgeDataset, neurobridge_collate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def audit_batch_alignment(
    subjects: List[str],
    config_path: str,
    n_batches: int = 20,
    batch_size: int = 32,
    output_dir: str = "experimental_results/neurobridge_ot_debug/batch_alignment_audit",
):
    """Run batch alignment audit."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    errors = []

    # Load CLIP cache
    clip_path = Path(os.environ.get(
        "CLIP_CACHE", "outputs/clip_cache/clip_multilayer.parquet"
    ))
    if not clip_path.exists():
        clip_path = Path("outputs/clip_cache/clip.parquet")
    if not clip_path.exists():
        errors.append(f"CLIP cache not found: {clip_path}")
        _write_errors(output_path, errors)
        return

    clip_df = pd.read_parquet(clip_path)
    embedding_col = config.get("data", {}).get("embedding_column", "layer_18_proj")
    logger.info("CLIP cache: %d entries, columns: %s", len(clip_df), list(clip_df.columns))

    if embedding_col not in clip_df.columns:
        errors.append(
            f"embedding_column='{embedding_col}' NOT in cache columns: {list(clip_df.columns)}"
        )
        _write_errors(output_path, errors)
        return

    # Build reference lookup: nsdId -> embedding
    reference_lookup: Dict[int, np.ndarray] = {}
    for _, row in clip_df.iterrows():
        nsd_id = int(row["nsdId"])
        emb = row[embedding_col]
        if isinstance(emb, np.ndarray):
            reference_lookup[nsd_id] = emb.astype(np.float32)
        elif isinstance(emb, (list, tuple)):
            reference_lookup[nsd_id] = np.array(emb, dtype=np.float32)

    logger.info("Reference lookup: %d nsdIds", len(reference_lookup))

    # Load dataset
    cache_root = Path(os.environ.get("CACHE_ROOT", "cache")) / "preextracted"
    index_root = Path("data/indices/nsd_index")

    dataset = NeuroBridgeDataset(
        subjects=subjects,
        cache_root=cache_root,
        index_root=index_root,
        embeddings_df=clip_df,
        embedding_column=embedding_col,
        exclude_shared1000=config.get("data", {}).get("exclude_shared1000", True),
        split_by_image=config.get("data", {}).get("split_by_image", True),
        val_ratio=config.get("data", {}).get("val_ratio", 0.10),
        seed=config.get("data", {}).get("seed", 42),
    )

    # Audit training split
    train_ds = Subset(dataset, dataset.train_indices)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=neurobridge_collate, num_workers=0,
    )

    sample_rows = []
    target_norms = []
    mismatches = 0
    missing_targets = 0
    shared1000_leaks = 0

    # Get shared1000 nsdIds for leak detection
    shared1000_nsd_ids = set()
    for subj in subjects:
        idx_path = Path("data/indices/nsd_index") / f"subject={subj}" / "index.parquet"
        if idx_path.exists():
            idx_df = pd.read_parquet(idx_path)
            if "shared1000" in idx_df.columns:
                shared1000_nsd_ids.update(
                    idx_df[idx_df["shared1000"] == True]["nsdId"].unique().tolist()
                )

    logger.info("SHARED1000 nsdIds found: %d", len(shared1000_nsd_ids))

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break

        nsd_ids = batch["nsd_id"].numpy()
        clip_targets = batch["clip_target"].numpy()
        subj_names = batch["subject_name"]

        for i in range(len(nsd_ids)):
            nsd_id = int(nsd_ids[i])
            target_vec = clip_targets[i]
            target_norm = float(np.linalg.norm(target_vec))
            subj = subj_names[i]

            # Check SHARED1000 leak
            if nsd_id in shared1000_nsd_ids:
                shared1000_leaks += 1

            # Check target exists in reference
            if nsd_id not in reference_lookup:
                missing_targets += 1
                target_norms.append({"nsd_id": nsd_id, "norm": target_norm, "status": "missing"})
                continue

            # Check alignment: batch target == reference target
            ref_vec = reference_lookup[nsd_id]
            if target_vec.shape != ref_vec.shape:
                mismatches += 1
                errors.append(f"Shape mismatch nsdId={nsd_id}: batch={target_vec.shape}, ref={ref_vec.shape}")
                continue

            cos_sim = float(np.dot(target_vec, ref_vec) / (
                np.linalg.norm(target_vec) * np.linalg.norm(ref_vec) + 1e-8
            ))
            is_aligned = cos_sim > 0.999

            if not is_aligned:
                mismatches += 1
                errors.append(f"Alignment mismatch nsdId={nsd_id}: cosine={cos_sim:.6f}")

            sample_rows.append({
                "batch_idx": batch_idx,
                "sample_idx": i,
                "subject": subj,
                "nsd_id": nsd_id,
                "target_norm": target_norm,
                "ref_norm": float(np.linalg.norm(ref_vec)),
                "cosine_with_ref": cos_sim,
                "aligned": is_aligned,
                "in_shared1000": nsd_id in shared1000_nsd_ids,
            })
            target_norms.append({"nsd_id": nsd_id, "norm": target_norm, "status": "ok"})

    # Write results
    audit_result = {
        "status": "PASS" if (mismatches == 0 and missing_targets == 0 and shared1000_leaks == 0) else "FAIL",
        "n_batches_audited": min(n_batches, batch_idx + 1),
        "n_samples_audited": len(sample_rows),
        "mismatches": mismatches,
        "missing_targets": missing_targets,
        "shared1000_leaks_in_train": shared1000_leaks,
        "embedding_column": embedding_col,
        "clip_cache_path": str(clip_path),
        "clip_cache_entries": len(clip_df),
        "reference_lookup_size": len(reference_lookup),
        "shared1000_nsd_ids_found": len(shared1000_nsd_ids),
        "subjects": subjects,
        "dataset_train_size": len(train_ds),
        "dataset_val_size": len(dataset.val_indices),
    }

    if sample_rows:
        norms = [r["target_norm"] for r in sample_rows]
        audit_result["target_norm_mean"] = float(np.mean(norms))
        audit_result["target_norm_std"] = float(np.std(norms))
        audit_result["target_norm_min"] = float(np.min(norms))
        audit_result["target_norm_max"] = float(np.max(norms))

    with open(output_path / "alignment_audit.json", "w") as f:
        json.dump(audit_result, f, indent=2)

    pd.DataFrame(sample_rows).to_csv(output_path / "sample_rows.csv", index=False)
    pd.DataFrame(target_norms).to_csv(output_path / "target_norms.csv", index=False)
    _write_errors(output_path, errors)

    logger.info("=== BATCH ALIGNMENT AUDIT ===")
    logger.info("Status: %s", audit_result["status"])
    logger.info("Samples audited: %d", len(sample_rows))
    logger.info("Mismatches: %d", mismatches)
    logger.info("Missing targets: %d", missing_targets)
    logger.info("SHARED1000 leaks in train: %d", shared1000_leaks)
    if sample_rows:
        logger.info("Target norm: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                    audit_result["target_norm_mean"], audit_result["target_norm_std"],
                    audit_result["target_norm_min"], audit_result["target_norm_max"])

    return audit_result


def _write_errors(output_path: Path, errors: List[str]):
    with open(output_path / "errors.txt", "w") as f:
        if errors:
            f.write("\n".join(errors))
        else:
            f.write("No errors.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroBridge-OT batch alignment audit")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--subjects", nargs="+",
                        default=["subj01", "subj02", "subj03", "subj04",
                                 "subj05", "subj06", "subj07", "subj08"])
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="experimental_results/neurobridge_ot_debug/batch_alignment_audit")
    args = parser.parse_args()

    audit_batch_alignment(
        subjects=args.subjects,
        config_path=args.config,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
