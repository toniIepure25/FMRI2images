#!/usr/bin/env python3
"""
Validate an NSD index parquet/CSV by checking contiguity and beta header bounds.

Usage:
    python scripts/validate_index.py \
        --index data/indices/nsd_index/subject=subj01/index.parquet
"""

from __future__ import annotations
import argparse
import logging
import sys
import pandas as pd

from fmri2img.data.nsd_index_builder import NSDIndexBuilder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def validate_index_file(index_path: str, allow_fallback_index: bool = False) -> None:
    df = pd.read_parquet(index_path) if index_path.endswith(".parquet") else pd.read_csv(index_path)

    if "beta_path" in df.columns and df["beta_path"].nunique() == 1 and not allow_fallback_index:
        raise ValueError(
            "Index maps all trials to a single beta_path. Rebuild a canonical index or pass --allow-fallback-index to override (not for paper runs)."
        )

    builder = NSDIndexBuilder()
    builder.validate_index(df)


def main():
    parser = argparse.ArgumentParser(description="Validate NSD index file")
    parser.add_argument("--index", required=True, help="Path to index parquet/csv")
    parser.add_argument("--allow-fallback-index", action="store_true", help="Permit collapsed single-file indices (not recommended)")

    args = parser.parse_args()

    try:
        validate_index_file(args.index, allow_fallback_index=args.allow_fallback_index)
        logger.info("Index validation passed")
    except Exception as e:
        logger.error(f"Index validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
