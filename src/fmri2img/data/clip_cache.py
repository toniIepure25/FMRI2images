"""
Professional CLIP Embedding Cache for NSD Dataset
=================================================

On-disk cache of CLIP embeddings with clean schema, resume support, and batch processing.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, Optional, Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

log = logging.getLogger(__name__)


class CLIPCache:
    """
    On-disk cache of CLIP embeddings for nsdId.
    Stored as Parquet with columns:
      - nsdId: int32
      - clip_embedding: fixed-length list[float32] (len=dim)

    Also supports legacy column names (clip512) for backward compatibility.

    Args:
        cache_path: Path to the parquet cache file.
        dim: Embedding dimensionality (512 for ViT-B/32, 768 for ViT-L/14).
    """
    
    LEGACY_COLUMN = "clip512"
    COLUMN = "clip_embedding"
    
    def __init__(self, cache_path: str = "outputs/clip_cache/clip.parquet",
                 dim: int = 768):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._df: Optional[pd.DataFrame] = None
        self._is_loaded: bool = False
        self._meta: Optional[Dict[str, Any]] = None

    @property
    def is_loaded(self) -> bool:
        """Check if cache has been loaded from disk."""
        return self._is_loaded

    def _schema(self) -> pa.schema:
        """PyArrow schema for CLIP cache (dimension-aware)."""
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow required for CLIP cache. Install with: pip install pyarrow")
        return pa.schema([
            pa.field("nsdId", pa.int32()),
            pa.field(self.COLUMN, pa.list_(pa.float32(), list_size=self.dim)),
        ])

    @property
    def meta_path(self) -> Path:
        """Metadata json path stored alongside the parquet cache."""
        return self.cache_path.with_name(f"{self.cache_path.stem}_meta.json")

    def load_metadata(self, raise_on_missing: bool = True) -> Optional[Dict[str, Any]]:
        """Load cache metadata if present."""
        path = self.meta_path
        if not path.exists():
            if raise_on_missing:
                raise FileNotFoundError(
                    f"CLIP cache metadata not found: {path}. Rebuild the cache with metadata "
                    "or create it manually to record model info."
                )
            return None
        import json
        with open(path) as f:
            self._meta = json.load(f)
        return self._meta

    def write_metadata(self, meta: Dict[str, Any], overwrite: bool = True) -> Path:
        """Persist metadata next to the cache parquet."""
        path = self.meta_path
        if path.exists() and not overwrite:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        self._meta = meta
        log.info(f"Wrote CLIP cache metadata to {path}")
        return path

    @property
    def _emb_col(self) -> str:
        """Detect the embedding column name in the loaded DataFrame."""
        if self._df is not None:
            if self.COLUMN in self._df.columns:
                return self.COLUMN
            if self.LEGACY_COLUMN in self._df.columns:
                return self.LEGACY_COLUMN
        return self.COLUMN

    def load(self) -> "CLIPCache":
        """
        Load cache from disk (fluent API).
        
        Returns:
            self (for chaining: CLIPCache(...).load())
        """
        if self._is_loaded:
            return self
            
        if self.cache_path.exists():
            self._df = pd.read_parquet(self.cache_path)
            if self.dim == 0 and self._emb_col in self._df.columns:
                first = self._df[self._emb_col].iloc[0]
                self.dim = len(first) if first is not None else 768
            log.debug("Loaded CLIP cache with %d entries (col=%s)", len(self._df), self._emb_col)
        else:
            self._df = pd.DataFrame(columns=["nsdId", self.COLUMN])
            log.debug("Initialized empty CLIP cache")
        
        self._is_loaded = True
        return self

    def contains(self, nsd_id: int) -> bool:
        """
        Check if nsdId is in cache.
        
        Args:
            nsd_id: NSD stimulus ID
            
        Returns:
            True if nsdId is cached, False otherwise
        """
        if not self._is_loaded:
            self.load()
        return int(nsd_id) in set(self._df["nsdId"].tolist())

    def list_cached_ids(self) -> list[int]:
        """
        Get list of all cached nsdIds.
        
        Returns:
            List of cached nsdIds
        """
        if not self._is_loaded:
            self.load()
        return [] if self._df is None or len(self._df) == 0 else self._df["nsdId"].astype(int).tolist()

    def get(self, nsd_ids: Iterable[int]) -> Dict[int, np.ndarray]:
        """
        Get embeddings for multiple nsdIds (vectorized).
        
        Args:
            nsd_ids: Iterable of nsdIds to retrieve
            
        Returns:
            Dict mapping nsdId to float32 (512,) L2-normalized array
        """
        if not self._is_loaded:
            self.load()
        
        ids = set(int(i) for i in nsd_ids)
        sub = self._df[self._df["nsdId"].isin(list(ids))]
        
        emb_col = self._emb_col
        
        result = {}
        for _, r in sub.iterrows():
            emb = np.array(r[emb_col], dtype=np.float32)
            # Ensure L2 normalization
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            result[int(r.nsdId)] = emb
        
        return result

    def save_rows(self, rows: pd.DataFrame) -> None:
        """
        Save new rows to cache, deduplicating on nsdId.
        
        Args:
            rows: DataFrame with columns ['nsdId', 'clip_embedding'] (or legacy 'clip512').
        """
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow required for CLIP cache. Install with: pip install pyarrow")
        
        if not self._is_loaded:
            self.load()
        
        # Normalize column name to standard
        if self.LEGACY_COLUMN in rows.columns and self.COLUMN not in rows.columns:
            rows = rows.rename(columns={self.LEGACY_COLUMN: self.COLUMN})
        
        # Normalize existing df column name
        if self._df is not None and self.LEGACY_COLUMN in self._df.columns and self.COLUMN not in self._df.columns:
            self._df = self._df.rename(columns={self.LEGACY_COLUMN: self.COLUMN})
        
        df = pd.concat([self._df, rows], ignore_index=True)
        df = df.drop_duplicates(subset=["nsdId"], keep='last').reset_index(drop=True)
        
        df["nsdId"] = df["nsdId"].astype("int32")
        df[self.COLUMN] = df[self.COLUMN].apply(
            lambda v: list(np.asarray(v, dtype=np.float32).reshape(self.dim))
        )
        
        table = pa.Table.from_pandas(df, schema=self._schema(), preserve_index=False)
        pq.write_table(table, self.cache_path, compression="snappy")
        
        self._df = df
        log.info("CLIP cache now has %d items at %s", len(self._df), self.cache_path)

    def stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache_size and path
        """
        if not self._is_loaded:
            self.load()
        n = 0 if self._df is None else len(self._df)
        return {"cache_size": n, "path": str(self.cache_path)}
