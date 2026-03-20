#!/usr/bin/env python3
"""Create deterministic K-fold OOF split files from an existing split.json.

This utility takes a base experiment split (with train_nsd_ids / val_nsd_ids),
partitions the train image IDs into K folds, and writes per-fold split files.
Each fold file is compatible with train_unified.py (via data.split_file):
- train_nsd_ids: all train IDs except this fold
- val_nsd_ids: held-out fold IDs

The fold val_nsd_ids are the OOF holdout queries that must be predicted only
by checkpoints trained on the corresponding fold-train IDs.
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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_base_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Base split.json not found: {path}")
    with open(path) as f:
        payload = json.load(f)

    if "train_nsd_ids" not in payload or "val_nsd_ids" not in payload:
        raise KeyError(
            f"split.json must contain train_nsd_ids and val_nsd_ids; keys={sorted(payload.keys())}"
        )
    return payload


def _assert_disjoint(a: list[int], b: list[int], label: str) -> None:
    overlap = set(a).intersection(b)
    if overlap:
        raise ValueError(f"{label}: overlap detected ({len(overlap)} ids)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build K-fold OOF split files from split.json")
    parser.add_argument("base_split_json", type=str, help="Path to source split.json")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where fold split JSON files and manifest will be written",
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Fold assignment seed (default: 42)")
    parser.add_argument(
        "--prefix",
        type=str,
        default="oof_fold",
        help="Fold filename prefix (default: oof_fold)",
    )
    args = parser.parse_args()

    if args.num_folds < 2:
        raise ValueError("num_folds must be >= 2")

    base_split_path = Path(args.base_split_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _load_base_split(base_split_path)
    train_ids = sorted(int(x) for x in base["train_nsd_ids"])
    val_ids = sorted(int(x) for x in base["val_nsd_ids"])

    _assert_disjoint(train_ids, val_ids, "base_split")

    rng = np.random.default_rng(args.seed)
    shuffled = np.array(train_ids, dtype=np.int32)
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, args.num_folds)

    manifest_folds: list[dict[str, Any]] = []
    all_holdout: list[int] = []

    for fold_idx, holdout_arr in enumerate(folds):
        holdout = sorted(int(x) for x in holdout_arr.tolist())
        holdout_set = set(holdout)
        fold_train = [x for x in train_ids if x not in holdout_set]

        _assert_disjoint(fold_train, holdout, f"fold_{fold_idx:02d}")
        if len(fold_train) + len(holdout) != len(train_ids):
            raise AssertionError(
                f"fold_{fold_idx:02d}: train+holdout count mismatch: "
                f"{len(fold_train)} + {len(holdout)} != {len(train_ids)}"
            )

        fold_payload = {
            "source_split_json": str(base_split_path),
            "oof": True,
            "seed": int(args.seed),
            "num_folds": int(args.num_folds),
            "fold_index": int(fold_idx),
            "train_pool_nsd_ids": train_ids,
            "base_val_nsd_ids": val_ids,
            "train_nsd_ids": fold_train,
            "val_nsd_ids": holdout,
            "holdout_nsd_ids": holdout,
            "n_train_images": len(fold_train),
            "n_val_images": len(holdout),
            "split_by_image": True,
            "exclude_shared1000": bool(base.get("exclude_shared1000", False)),
        }

        fold_name = f"{args.prefix}_{fold_idx:02d}.json"
        fold_path = out_dir / fold_name
        _save_json(fold_path, fold_payload)

        manifest_folds.append(
            {
                "fold_index": int(fold_idx),
                "split_json": str(fold_path),
                "n_train": len(fold_train),
                "n_holdout": len(holdout),
                "holdout_nsd_ids": holdout,
            }
        )
        all_holdout.extend(holdout)

        logger.info(
            "Fold %02d: train=%d holdout=%d -> %s",
            fold_idx,
            len(fold_train),
            len(holdout),
            fold_path,
        )

    holdout_sorted = sorted(all_holdout)
    if holdout_sorted != train_ids:
        raise AssertionError(
            "Fold holdout union does not exactly match base train_nsd_ids "
            f"(union={len(holdout_sorted)}, expected={len(train_ids)})"
        )

    if len(set(all_holdout)) != len(train_ids):
        raise AssertionError("At least one train nsd_id appears in multiple holdout folds")

    manifest = {
        "oof": True,
        "version": 1,
        "seed": int(args.seed),
        "num_folds": int(args.num_folds),
        "source_split_json": str(base_split_path),
        "train_pool_nsd_ids": train_ids,
        "base_val_nsd_ids": val_ids,
        "folds": manifest_folds,
    }
    manifest_path = out_dir / "oof_fold_manifest.json"
    _save_json(manifest_path, manifest)

    logger.info(
        "Saved OOF fold manifest to %s (train_pool=%d, folds=%d)",
        manifest_path,
        len(train_ids),
        args.num_folds,
    )


if __name__ == "__main__":
    main()
