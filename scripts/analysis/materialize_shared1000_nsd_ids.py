#!/usr/bin/env python3
"""Write ``shared1000_nsd_ids.npy`` in the same row order as ``train_unified._evaluate_shared1000``.

The shared1000 prediction arrays are ordered by ``np.unique(nsd_ids)`` (sorted)
over all shared1000 trials for the subject.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def shared1000_unique_nsd_ids(subject: str, index_root: Path) -> np.ndarray:
    """Return sorted unique ``nsdId`` values for shared1000 trials."""
    index_path = index_root / f"subject={subject}" / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    index_df = pd.read_parquet(index_path)
    if "shared1000" not in index_df.columns:
        raise KeyError(f"No 'shared1000' column in {index_path}")
    s1000_mask = index_df["shared1000"].fillna(False).astype(bool).values
    s1000_df = index_df[s1000_mask].reset_index(drop=True)
    nsd_ids = s1000_df["nsdId"].values
    return np.unique(nsd_ids).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="subj01", help="NSD subject id")
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("data/indices/nsd_index"),
        help="Directory containing subject=*/index.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for shared1000_nsd_ids.npy",
    )
    parser.add_argument(
        "--expect-n",
        type=int,
        default=1000,
        help="Expected number of unique images (warn if mismatch)",
    )
    args = parser.parse_args()

    uids = shared1000_unique_nsd_ids(args.subject, args.index_root)
    if len(uids) != args.expect_n:
        print(
            f"WARNING: got {len(uids)} unique shared1000 ids, expected {args.expect_n}",
            file=sys.stderr,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, uids)
    print(f"Saved {args.output} shape={uids.shape} dtype={uids.dtype}")


if __name__ == "__main__":
    main()
