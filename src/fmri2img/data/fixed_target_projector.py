"""Deterministic fixed projector for retrieval targets.

Used when the compact retrieval head dimension differs from the raw CLIP CLS
embedding dimension. For V31 this lifts 768-D CLIP targets into a deterministic
1024-D space while preserving angular geometry as closely as possible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FixedTargetProjector:
    """Deterministic linear target transform with optional L2 renormalisation."""

    input_dim: int
    output_dim: int
    seed: int = 42
    method: str = "orthogonal_lift"
    l2_normalize: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError(
                f"Projector dims must be positive, got {self.input_dim} -> {self.output_dim}"
            )

        self.matrix: np.ndarray | None = None
        if self.output_dim == self.input_dim:
            logger.info(
                "FixedTargetProjector: identity %d -> %d (seed=%d)",
                self.input_dim, self.output_dim, self.seed,
            )
            return

        if self.method != "orthogonal_lift":
            raise ValueError(f"Unsupported projector method: {self.method}")

        rng = np.random.default_rng(self.seed)
        if self.output_dim > self.input_dim:
            basis = rng.standard_normal((self.output_dim, self.input_dim)).astype(np.float32)
            q, _ = np.linalg.qr(basis, mode="reduced")
            self.matrix = q[:, : self.input_dim].astype(np.float32)
        else:
            basis = rng.standard_normal((self.input_dim, self.output_dim)).astype(np.float32)
            q, _ = np.linalg.qr(basis, mode="reduced")
            self.matrix = q[:, : self.output_dim].astype(np.float32)

        logger.info(
            "FixedTargetProjector: %s %d -> %d (seed=%d, l2_normalize=%s)",
            self.method,
            self.input_dim,
            self.output_dim,
            self.seed,
            self.l2_normalize,
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project a single vector or a batch of vectors."""
        arr = np.asarray(x, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]

        if arr.shape[-1] != self.input_dim:
            raise ValueError(
                f"Projector expected input_dim={self.input_dim}, got {arr.shape[-1]}"
            )

        if self.matrix is None:
            out = arr.copy()
        elif self.output_dim > self.input_dim:
            out = arr @ self.matrix.T
        else:
            out = arr @ self.matrix

        if self.l2_normalize:
            norms = np.linalg.norm(out, axis=-1, keepdims=True)
            out = out / np.maximum(norms, 1e-8)

        out = out.astype(np.float32, copy=False)
        return out[0] if single else out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)
