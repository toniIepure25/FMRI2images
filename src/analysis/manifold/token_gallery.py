"""Build a flat token-level CLIP gallery aligned to a parquet nsdId column.

Used for manifold analysis when decoder targets live in flattened token
space (e.g. 257×768 = 197376) instead of 768-D CLS embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_flat_token_gallery_from_parquet(
    parquet_path: str | Path,
    token_h5_path: str | Path,
    id_column: str = "nsdId",
    mmap_token_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one flat token embedding per parquet row, in parquet row order.

    Args:
        parquet_path: Parquet with at least *id_column* (e.g. ``clip.parquet``).
        token_h5_path: HDF5 from ``build_token_clip_cache`` (TokenCLIPCache).
        id_column: NSD image id column name.
        mmap_token_cache: If True, memory-map the HDF5 (slower per-row reads).

    Returns:
        ``gallery`` of shape ``(G, T*D)`` float32 L2-normalised per row,
        and ``gallery_ids`` of shape ``(G,)`` int64 (same order as *parquet_path*).
    """
    from fmri2img.data.token_clip_cache import TokenCLIPCache

    parquet_path = Path(parquet_path)
    token_h5_path = Path(token_h5_path)
    df = pd.read_parquet(parquet_path)
    if id_column not in df.columns:
        raise KeyError(f"Column {id_column!r} not in {parquet_path}; have {list(df.columns)}")
    nsd_ids = df[id_column].astype(np.int64).values
    unique_ids = sorted(set(int(x) for x in nsd_ids.tolist()))
    logger.info(
        "Building flat token gallery: %d rows, %d unique nsdIds from %s",
        len(nsd_ids),
        len(unique_ids),
        parquet_path,
    )

    cache = TokenCLIPCache(str(token_h5_path))
    cache.load(mmap=mmap_token_cache, preload_ids=unique_ids)

    rows: list[np.ndarray] = []
    for nid in nsd_ids:
        flat = cache.get_flat(int(nid)).astype(np.float32, copy=False)
        nrm = np.linalg.norm(flat)
        if nrm > 1e-8:
            flat = flat / nrm
        rows.append(flat)
    gallery = np.stack(rows, axis=0)
    logger.info("Token gallery shape=%s dtype=%s", gallery.shape, gallery.dtype)
    return gallery, nsd_ids
