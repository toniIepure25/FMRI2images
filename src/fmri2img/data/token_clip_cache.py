"""
Token-Level CLIP Cache (HDF5)
==============================

Loads and serves token-level CLIP embeddings stored in HDF5 format
by :mod:`scripts.build.build_token_clip_cache`.

Each NSD image is represented as (T, D) where T = 257 tokens
(1 CLS + 256 spatial patches) and D = hidden/projected dim.

Usage:
    cache = TokenCLIPCache("outputs/clip_cache/tokens_ViT-L-14_projected.h5")
    cache.load()
    tokens = cache[42]          # (257, 768) for nsdId=42
    batch  = cache.get_batch([1, 2, 3])  # (3, 257, 768)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class TokenCLIPCache:
    """
    On-disk cache of token-level CLIP embeddings in HDF5 format.

    HDF5 schema (written by build_token_clip_cache.py):
        /tokens   — (N, T, D) float32, L2-normalized per token
        /nsd_ids  — (N,) int32
        attrs: model_name, num_tokens, token_dim, projected, mode, ...

    Args:
        cache_path: Path to the HDF5 cache file.
    """

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self._tokens: Optional[np.ndarray] = None    # (N, T, D)
        self._nsd_ids: Optional[np.ndarray] = None    # (N,)
        self._id_to_idx: Dict[int, int] = {}
        self._meta: Dict[str, Any] = {}
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def num_tokens(self) -> int:
        """Number of tokens per image (e.g. 257 for ViT-L/14)."""
        return self._meta.get("num_tokens", 0)

    @property
    def token_dim(self) -> int:
        """Dimension of each token (e.g. 768 for ViT-L/14 projected)."""
        return self._meta.get("token_dim", 0)

    @property
    def shape(self) -> tuple:
        """(num_images, num_tokens, token_dim)."""
        if self._tokens is not None:
            return self._tokens.shape
        return (0, self.num_tokens, self.token_dim)

    def load(
        self,
        mmap: bool = False,
        preload_ids: Optional[list[int]] = None,
    ) -> "TokenCLIPCache":
        """Load cache from disk.

        Args:
            mmap: If True keep HDF5 file open and read lazily (saves RAM).
                  If False (default), load entire array into RAM for speed.
            preload_ids: If given, load only these nsdIds into RAM (ignores
                         *mmap*). Avoids 29+ GB page cache from mmapping the
                         full file -- only the needed subset (~8 GB for 10K
                         images) is resident.

        Returns:
            self (fluent API)
        """
        if self._is_loaded:
            return self

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Token CLIP cache not found: {self.cache_path}\n"
                "Build it with: python scripts/build/build_token_clip_cache.py"
            )

        logger.info("Loading token CLIP cache from %s", self.cache_path)
        f = h5py.File(self.cache_path, "r")

        for key in f.attrs:
            self._meta[key] = f.attrs[key]

        all_nsd_ids = f["nsd_ids"][:]

        if preload_ids is not None:
            needed = set(int(x) for x in preload_ids)
            disk_indices = [
                i for i, nid in enumerate(all_nsd_ids) if int(nid) in needed
            ]
            disk_indices.sort()
            self._nsd_ids = all_nsd_ids[disk_indices]
            self._id_to_idx = {
                int(nid): j for j, nid in enumerate(self._nsd_ids)
            }
            self._tokens = np.empty(
                (len(disk_indices), f["tokens"].shape[1], f["tokens"].shape[2]),
                dtype=np.float32,
            )
            _chunk = 500
            for start in range(0, len(disk_indices), _chunk):
                batch_idx = disk_indices[start : start + _chunk]
                self._tokens[start : start + len(batch_idx)] = f["tokens"][batch_idx]
            f.close()
            self._is_loaded = True
            _t = self._tokens.shape[1]
            _d = self._tokens.shape[2]
            _gb = len(self._nsd_ids) * _t * _d * 4 / 1e9
            logger.info(
                "TokenCLIPCache loaded (preload_ids): %d/%d images, "
                "%d tokens × %d dim (%.2f GB RAM)",
                len(self._nsd_ids), len(all_nsd_ids), _t, _d, _gb,
            )
            return self

        self._nsd_ids = all_nsd_ids
        self._id_to_idx = {int(nid): i for i, nid in enumerate(self._nsd_ids)}

        if mmap:
            self._tokens = f["tokens"]
            self._h5file = f
        else:
            self._tokens = f["tokens"][:]
            f.close()

        self._is_loaded = True
        _n = len(self._nsd_ids)
        _t = self._meta.get("num_tokens", self._tokens.shape[1])
        _d = self._meta.get("token_dim", self._tokens.shape[2])
        _size_gb = _n * _t * _d * 4 / 1e9
        if mmap:
            logger.info(
                "TokenCLIPCache loaded (mmap/lazy): %d images, %d tokens × %d dim "
                "(%.2f GB on disk, ~0 GB RAM)",
                _n, _t, _d, _size_gb,
            )
        else:
            logger.info(
                "TokenCLIPCache loaded (in-memory): %d images, %d tokens × %d dim "
                "(%.2f GB RAM)",
                _n, _t, _d, _size_gb,
            )
        return self

    def close(self):
        """Close HDF5 file if opened in mmap mode."""
        if hasattr(self, "_h5file"):
            self._h5file.close()

    def __contains__(self, nsd_id: int) -> bool:
        if not self._is_loaded:
            self.load()
        return int(nsd_id) in self._id_to_idx

    def __len__(self) -> int:
        if not self._is_loaded:
            self.load()
        return len(self._nsd_ids)

    def __getitem__(self, nsd_id: int) -> np.ndarray:
        """Get token embeddings for a single nsdId.

        Args:
            nsd_id: NSD stimulus ID.

        Returns:
            (T, D) float32 array of L2-normalized token embeddings.
        """
        if not self._is_loaded:
            self.load()
        idx = self._id_to_idx.get(int(nsd_id))
        if idx is None:
            raise KeyError(f"nsdId={nsd_id} not in token cache ({len(self)} entries)")
        return np.array(self._tokens[idx], dtype=np.float32)

    def get_batch(self, nsd_ids: list[int]) -> np.ndarray:
        """Get token embeddings for multiple nsdIds.

        Args:
            nsd_ids: List of NSD stimulus IDs.

        Returns:
            (B, T, D) float32 array.
        """
        if not self._is_loaded:
            self.load()
        indices = [self._id_to_idx[int(nid)] for nid in nsd_ids]
        return np.array(self._tokens[indices], dtype=np.float32)

    def get_flat(self, nsd_id: int) -> np.ndarray:
        """Get flattened token embedding for a single nsdId.

        Returns:
            (T*D,) float32 array — e.g. (197376,) for 257×768.
        """
        return self[nsd_id].reshape(-1)

    def get_flat_batch(self, nsd_ids: list[int]) -> np.ndarray:
        """Get flattened token embeddings for multiple nsdIds.

        Returns:
            (B, T*D) float32 array.
        """
        tokens = self.get_batch(nsd_ids)
        return tokens.reshape(tokens.shape[0], -1)

    def list_ids(self) -> list[int]:
        """Get all cached nsdIds."""
        if not self._is_loaded:
            self.load()
        return [int(x) for x in self._nsd_ids]

    def stats(self) -> dict:
        """Cache statistics."""
        if not self._is_loaded:
            self.load()
        return {
            "path": str(self.cache_path),
            "num_images": len(self._nsd_ids),
            "num_tokens": self.num_tokens,
            "token_dim": self.token_dim,
            "mode": self._meta.get("mode", "unknown"),
            "model_name": self._meta.get("model_name", "unknown"),
            **self._meta,
        }

    def as_embedding_df(self, flat: bool = True):
        """Convert to a pandas DataFrame compatible with the training pipeline.

        Args:
            flat: If True, flatten tokens to (T*D,) per row (for direct
                  use as embedding column). If False, store as (T, D) arrays.

        Returns:
            DataFrame with columns ['nsdId', 'clip_embedding'].
        """
        import pandas as pd

        if not self._is_loaded:
            self.load()

        rows = []
        for i, nsd_id in enumerate(self._nsd_ids):
            tokens = np.array(self._tokens[i], dtype=np.float32)
            if flat:
                emb = tokens.reshape(-1)
            else:
                emb = tokens
            rows.append({"nsdId": int(nsd_id), "clip_embedding": emb})

        return pd.DataFrame(rows)
