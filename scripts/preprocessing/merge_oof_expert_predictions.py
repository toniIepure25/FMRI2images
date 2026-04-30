#!/usr/bin/env python3
"""Merge fold-heldout predictions into TRAIN OOF metrics files.

Each fold directory must contain metrics for split prefix `val` produced by a
checkpoint trained on fold-train IDs only. This script re-keys rows by nsd_id,
verifies full non-overlapping coverage of the base train pool, and writes
`train_oof_*` arrays into the target metrics directory.

Supported experts:
- tri/compact: predictions_compact, ground_truth_compact, optional kappas
- tri/rerank:  predictions_rerank,  ground_truth_rerank (optional)
- legacy:      predictions(_compact), ground_truth(_compact)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path) as f:
        return json.load(f)


def _resolve_existing_file(base: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate files exist in {base}: {candidates}")


def _load_fold_arrays(metrics_dir: Path, split_prefix: str) -> dict[str, np.ndarray]:
    nsd_ids = np.load(metrics_dir / f"{split_prefix}_nsd_ids.npy").astype(np.int32)

    compact_pred = np.load(
        _resolve_existing_file(
            metrics_dir,
            [
                f"{split_prefix}_predictions_compact.npy",
                f"{split_prefix}_predictions.npy",
            ],
        )
    ).astype(np.float32)
    compact_gt = np.load(
        _resolve_existing_file(
            metrics_dir,
            [
                f"{split_prefix}_ground_truth_compact.npy",
                f"{split_prefix}_ground_truth.npy",
            ],
        )
    ).astype(np.float32)

    if compact_pred.shape[0] != nsd_ids.shape[0] or compact_gt.shape[0] != nsd_ids.shape[0]:
        raise ValueError(
            f"{metrics_dir}: row mismatch for compact arrays "
            f"(ids={nsd_ids.shape[0]}, pred={compact_pred.shape[0]}, gt={compact_gt.shape[0]})"
        )

    out: dict[str, np.ndarray] = {
        "nsd_ids": nsd_ids,
        "predictions_compact": compact_pred,
        "ground_truth_compact": compact_gt,
    }

    kappas_path = metrics_dir / f"{split_prefix}_kappas.npy"
    if kappas_path.exists():
        kappas = np.load(kappas_path).astype(np.float32)
        if kappas.shape[0] != nsd_ids.shape[0]:
            raise ValueError(f"{metrics_dir}: kappas rows {kappas.shape[0]} != ids rows {nsd_ids.shape[0]}")
        out["kappas"] = kappas

    vmf_pred_path = metrics_dir / f"{split_prefix}_predictions_vmf.npy"
    vmf_kappa_path = metrics_dir / f"{split_prefix}_kappas_vmf.npy"
    if vmf_pred_path.exists():
        vmf_pred = np.load(vmf_pred_path).astype(np.float32)
        if vmf_pred.shape[0] != nsd_ids.shape[0]:
            raise ValueError(f"{metrics_dir}: vmf prediction rows {vmf_pred.shape[0]} != ids rows {nsd_ids.shape[0]}")
        out["predictions_vmf"] = vmf_pred
    if vmf_kappa_path.exists():
        vmf_kappa = np.load(vmf_kappa_path).astype(np.float32)
        if vmf_kappa.shape[0] != nsd_ids.shape[0]:
            raise ValueError(f"{metrics_dir}: vmf kappas rows {vmf_kappa.shape[0]} != ids rows {nsd_ids.shape[0]}")
        out["kappas_vmf"] = vmf_kappa

    rerank_pred_path = metrics_dir / f"{split_prefix}_predictions_rerank.npy"
    rerank_gt_path = metrics_dir / f"{split_prefix}_ground_truth_rerank.npy"
    if rerank_pred_path.exists() and rerank_gt_path.exists():
        rerank_pred = np.load(rerank_pred_path).astype(np.float32)
        rerank_gt = np.load(rerank_gt_path).astype(np.float32)
        if rerank_pred.shape[0] != nsd_ids.shape[0] or rerank_gt.shape[0] != nsd_ids.shape[0]:
            raise ValueError(f"{metrics_dir}: rerank row mismatch with nsd_ids")
        out["predictions_rerank"] = rerank_pred
        out["ground_truth_rerank"] = rerank_gt

    def _optional_component(name_candidates: list[str]) -> np.ndarray | None:
        for name in name_candidates:
            path = metrics_dir / f"{split_prefix}_{name}.npy"
            if path.exists():
                return np.load(path).astype(np.float32)
        return None

    component_mu = _optional_component([
        "predictions_compact_component_mu",
        "predictions_compact_components_mu",
    ])
    component_kappa = _optional_component([
        "predictions_compact_component_kappa",
        "predictions_compact_components_kappa",
    ])
    component_logits = _optional_component([
        "predictions_compact_component_logits",
        "predictions_compact_components_logits",
    ])
    for name, arr in {
        "predictions_compact_component_mu": component_mu,
        "predictions_compact_component_kappa": component_kappa,
        "predictions_compact_component_logits": component_logits,
    }.items():
        if arr is not None:
            if arr.shape[0] != nsd_ids.shape[0]:
                raise ValueError(f"{metrics_dir}: {name} rows {arr.shape[0]} != ids rows {nsd_ids.shape[0]}")
            out[name] = arr

    return out


def _merge_rows_by_nsd_id(
    fold_entries: list[dict[str, Any]],
    expected_train_ids: list[int],
    split_prefix: str,
) -> dict[str, np.ndarray]:
    train_id_set = set(int(x) for x in expected_train_ids)
    row_map: dict[int, dict[str, np.ndarray]] = {}
    provenance: dict[int, dict[str, Any]] = {}

    for entry in fold_entries:
        fold_index = int(entry["fold_index"])
        fold_metrics_dir = Path(entry["metrics_dir"])
        holdout_ids = set(int(x) for x in entry["holdout_nsd_ids"])
        arrays = _load_fold_arrays(fold_metrics_dir, split_prefix)

        for row_idx, nid in enumerate(arrays["nsd_ids"].tolist()):
            nid = int(nid)
            if nid not in train_id_set:
                raise ValueError(
                    f"Fold {fold_index}: nsd_id={nid} is not in expected train pool"
                )
            if nid not in holdout_ids:
                raise ValueError(
                    f"Fold {fold_index}: nsd_id={nid} not in that fold's holdout IDs"
                )
            if nid in row_map:
                prev = provenance[nid]
                raise ValueError(
                    f"Duplicate OOF row for nsd_id={nid}: fold {prev['fold_index']} and fold {fold_index}"
                )

            row_payload: dict[str, np.ndarray] = {
                "predictions_compact": arrays["predictions_compact"][row_idx],
                "ground_truth_compact": arrays["ground_truth_compact"][row_idx],
            }
            if "kappas" in arrays:
                row_payload["kappas"] = np.asarray(arrays["kappas"][row_idx], dtype=np.float32)
            if "predictions_vmf" in arrays:
                row_payload["predictions_vmf"] = arrays["predictions_vmf"][row_idx]
            if "kappas_vmf" in arrays:
                row_payload["kappas_vmf"] = np.asarray(arrays["kappas_vmf"][row_idx], dtype=np.float32)
            if "predictions_rerank" in arrays:
                row_payload["predictions_rerank"] = arrays["predictions_rerank"][row_idx]
            if "ground_truth_rerank" in arrays:
                row_payload["ground_truth_rerank"] = arrays["ground_truth_rerank"][row_idx]
            for key in [
                "predictions_compact_component_mu",
                "predictions_compact_component_kappa",
                "predictions_compact_component_logits",
            ]:
                if key in arrays:
                    row_payload[key] = arrays[key][row_idx]

            row_map[nid] = row_payload
            provenance[nid] = {
                "fold_index": fold_index,
                "metrics_dir": str(fold_metrics_dir),
            }

    missing = sorted(train_id_set.difference(row_map.keys()))
    extra = sorted(set(row_map.keys()).difference(train_id_set))
    if missing:
        logger.warning(
            "Missing OOF predictions for %d / %d train pool nsd_ids "
            "(expected when fold pool spans multiple subjects but training is single-subject). "
            "Proceeding with %d available predictions.",
            len(missing), len(train_id_set), len(row_map),
        )
    if extra:
        raise AssertionError(f"Found {len(extra)} unexpected nsd_ids outside train pool")

    available_ids = sorted(row_map.keys())
    ordered_ids = np.array(available_ids, dtype=np.int32)

    compact_pred = np.stack([row_map[int(nid)]["predictions_compact"] for nid in ordered_ids], axis=0)
    compact_gt = np.stack([row_map[int(nid)]["ground_truth_compact"] for nid in ordered_ids], axis=0)

    merged: dict[str, np.ndarray] = {
        "train_oof_nsd_ids": ordered_ids,
        "train_oof_predictions_compact": compact_pred.astype(np.float32),
        "train_oof_ground_truth_compact": compact_gt.astype(np.float32),
        # Keep legacy aliases for compatibility with helper loaders.
        "train_oof_predictions": compact_pred.astype(np.float32),
        "train_oof_ground_truth": compact_gt.astype(np.float32),
    }

    has_kappa = all("kappas" in row_map[int(nid)] for nid in ordered_ids)
    if has_kappa:
        kappa = np.array([float(row_map[int(nid)]["kappas"]) for nid in ordered_ids], dtype=np.float32)
        merged["train_oof_kappas"] = kappa

    has_vmf_pred = all("predictions_vmf" in row_map[int(nid)] for nid in ordered_ids)
    if has_vmf_pred:
        merged["train_oof_predictions_vmf"] = np.stack(
            [row_map[int(nid)]["predictions_vmf"] for nid in ordered_ids], axis=0
        ).astype(np.float32)
    has_vmf_kappa = all("kappas_vmf" in row_map[int(nid)] for nid in ordered_ids)
    if has_vmf_kappa:
        merged["train_oof_kappas_vmf"] = np.array(
            [float(row_map[int(nid)]["kappas_vmf"]) for nid in ordered_ids], dtype=np.float32
        )

    has_rerank = all(
        "predictions_rerank" in row_map[int(nid)] and "ground_truth_rerank" in row_map[int(nid)]
        for nid in ordered_ids
    )
    if has_rerank:
        merged["train_oof_predictions_rerank"] = np.stack(
            [row_map[int(nid)]["predictions_rerank"] for nid in ordered_ids], axis=0
        ).astype(np.float32)
        merged["train_oof_ground_truth_rerank"] = np.stack(
            [row_map[int(nid)]["ground_truth_rerank"] for nid in ordered_ids], axis=0
        ).astype(np.float32)

    for src_key, out_key in [
        ("predictions_compact_component_mu", "train_oof_predictions_compact_component_mu"),
        ("predictions_compact_component_kappa", "train_oof_predictions_compact_component_kappa"),
        ("predictions_compact_component_logits", "train_oof_predictions_compact_component_logits"),
    ]:
        has_component = all(src_key in row_map[int(nid)] for nid in ordered_ids)
        if has_component:
            merged[out_key] = np.stack(
                [row_map[int(nid)][src_key] for nid in ordered_ids], axis=0
            ).astype(np.float32)

    return merged


def _write_merged_metrics(target_metrics_dir: Path, merged: dict[str, np.ndarray]) -> None:
    target_metrics_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in merged.items():
        np.save(target_metrics_dir / f"{key}.npy", arr)
        logger.info("Saved %s: %s", target_metrics_dir / f"{key}.npy", arr.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge fold-heldout predictions into train_oof arrays")
    parser.add_argument(
        "--fold-manifest",
        type=str,
        required=True,
        help="Path to oof_fold_manifest.json produced by build_oof_split_folds.py",
    )
    parser.add_argument(
        "--fold-results-root",
        type=str,
        required=True,
        help="Root containing per-fold result directories",
    )
    parser.add_argument(
        "--fold-dir-pattern",
        type=str,
        default="fold_{fold_index:02d}",
        help="Pattern under fold-results-root for each fold (default: fold_{fold_index:02d})",
    )
    parser.add_argument(
        "--metrics-subdir",
        type=str,
        default="metrics",
        help="Metrics subdirectory inside each fold dir (default: metrics)",
    )
    parser.add_argument(
        "--split-prefix",
        type=str,
        default="val",
        help="Split prefix to read from each fold metrics (default: val)",
    )
    parser.add_argument(
        "--target-metrics-dir",
        type=str,
        required=True,
        help="Directory where merged train_oof_*.npy files will be written",
    )
    parser.add_argument(
        "--provenance-json",
        type=str,
        default=None,
        help="Optional path to write merge provenance JSON",
    )
    args = parser.parse_args()

    manifest = _load_json(Path(args.fold_manifest))
    if "folds" not in manifest or "train_pool_nsd_ids" not in manifest:
        raise KeyError("Invalid fold manifest: missing 'folds' or 'train_pool_nsd_ids'")

    fold_results_root = Path(args.fold_results_root)
    if not fold_results_root.exists():
        raise FileNotFoundError(
            f"fold-results-root does not exist: {fold_results_root}. "
            "Run fold predictions first or pass the correct --fold-results-root."
        )
    fold_entries: list[dict[str, Any]] = []

    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        fold_dir_rel = args.fold_dir_pattern.format(fold_index=fold_index)
        fold_dir = fold_results_root / fold_dir_rel
        metrics_dir = fold_dir / args.metrics_subdir
        if not metrics_dir.exists():
            nearby = sorted(p.name for p in fold_results_root.iterdir() if p.is_dir())[:20]
            raise FileNotFoundError(
                "Fold metrics dir not found: "
                f"{metrics_dir}. "
                "Expected per-fold outputs like "
                f"{fold_results_root}/fold_00/{args.metrics_subdir}/val_*.npy. "
                "Either generate fold-heldout predictions first, or pass the correct "
                "--fold-results-root/--fold-dir-pattern/--metrics-subdir. "
                f"Nearby dirs under fold-results-root: {nearby}"
            )

        fold_entries.append(
            {
                "fold_index": fold_index,
                "metrics_dir": str(metrics_dir),
                "holdout_nsd_ids": fold.get("holdout_nsd_ids", []),
            }
        )

    merged = _merge_rows_by_nsd_id(
        fold_entries=fold_entries,
        expected_train_ids=[int(x) for x in manifest["train_pool_nsd_ids"]],
        split_prefix=args.split_prefix,
    )

    target_metrics_dir = Path(args.target_metrics_dir)
    _write_merged_metrics(target_metrics_dir, merged)

    # Quick disagreement sanity on merged OOF compact vs legacy-compatible compact
    # (legacy disagreement is measured later when tri/legacy are aligned in cache build).
    n_rows = merged["train_oof_nsd_ids"].shape[0]
    logger.info("Merged OOF rows: %d", n_rows)

    provenance = {
        "oof": True,
        "version": 1,
        "fold_manifest": str(Path(args.fold_manifest)),
        "fold_results_root": str(fold_results_root),
        "fold_dir_pattern": args.fold_dir_pattern,
        "metrics_subdir": args.metrics_subdir,
        "split_prefix": args.split_prefix,
        "target_metrics_dir": str(target_metrics_dir),
        "n_rows": int(n_rows),
        "train_pool_nsd_ids": [int(x) for x in manifest["train_pool_nsd_ids"]],
        "fold_entries": fold_entries,
        "has_kappas": bool("train_oof_kappas" in merged),
        "has_rerank": bool("train_oof_predictions_rerank" in merged),
        "has_compact_components": bool("train_oof_predictions_compact_component_mu" in merged),
    }

    provenance_path = Path(args.provenance_json) if args.provenance_json else (
        target_metrics_dir / "train_oof_merge_provenance.json"
    )
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Saved provenance to %s", provenance_path)


if __name__ == "__main__":
    main()
