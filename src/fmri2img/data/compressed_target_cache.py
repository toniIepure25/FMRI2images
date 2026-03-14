"""Compressed rerank target cache.

Loads PCA-compressed token embeddings for the dedicated rerank head.
Targets are L2-normalized and indexed by NSD stimulus ID.

The cache is produced by ``scripts/preprocessing/build_rerank_cache.py``
which fits PCA on train-split-only images, then projects all images.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CompressedTargetCache:
    """Provides PCA-compressed, L2-normalised rerank targets by NSD ID.

    Parameters
    ----------
    cache_path : str or Path
        Path to ``.npz`` file produced by ``build_rerank_cache.py``.
    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Compressed target cache not found: {self.cache_path}")

        data = np.load(self.cache_path, allow_pickle=True)

        self._targets: np.ndarray = data["targets"]  # (N, rerank_dim), L2-normed
        self._nsd_ids: np.ndarray = data["nsd_ids"]  # (N,)
        self._id_to_idx: Dict[int, int] = {
            int(nid): i for i, nid in enumerate(self._nsd_ids)
        }

        # Parse metadata
        self._metadata: dict = {}
        if "metadata" in data:
            raw = data["metadata"]
            if isinstance(raw, np.ndarray):
                raw = str(raw)
            self._metadata = json.loads(raw)

        self.rerank_dim = int(self._targets.shape[1])
        self.n_images = len(self._nsd_ids)

        # Validate L2 normalization
        norms = np.linalg.norm(self._targets, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            logger.warning(
                "Compressed targets not L2-normalised (norm range %.4f–%.4f). "
                "Re-normalising.",
                norms.min(), norms.max(),
            )
            self._targets = self._targets / np.maximum(
                norms[:, None], 1e-8)

        logger.info(
            "CompressedTargetCache: %d images, rerank_dim=%d, "
            "cumulative_variance=%.4f, cache=%s",
            self.n_images, self.rerank_dim,
            self._metadata.get("cumulative_variance", -1),
            self.cache_path.name,
        )

    def __contains__(self, nsd_id: int) -> bool:
        return int(nsd_id) in self._id_to_idx

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, nsd_id: int) -> np.ndarray:
        """Get L2-normalised compressed target for a single NSD ID.

        Returns
        -------
        target : (rerank_dim,) float32
        """
        idx = self._id_to_idx.get(int(nsd_id))
        if idx is None:
            raise KeyError(
                f"NSD ID {nsd_id} not in compressed target cache "
                f"({self.n_images} images loaded)")
        return np.array(self._targets[idx], dtype=np.float32)

    def get_batch(self, nsd_ids: list[int]) -> np.ndarray:
        """Get compressed targets for multiple NSD IDs.

        Returns
        -------
        targets : (B, rerank_dim) float32, L2-normalised
        """
        indices = [self._id_to_idx[int(nid)] for nid in nsd_ids]
        return np.array(self._targets[indices], dtype=np.float32)

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    @property
    def split_seed(self) -> Optional[int]:
        return self._metadata.get("split_seed")

    @property
    def explained_variance(self) -> Optional[float]:
        return self._metadata.get("cumulative_variance")
