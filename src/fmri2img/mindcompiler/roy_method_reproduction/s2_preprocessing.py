"""S2.0 preprocessing policies (train-only fit; explicit zero-variance accounting).

Two named, frozen policies:

* ``PREPROC_HISTORICAL_D0`` -- train-only centering, no scaling (the historical D0
  smoke behaviour). Preserved unchanged; exists only as engineering-history
  sensitivity (P0).
* ``PREPROC_ZSCORE_TRAIN_ONLY`` -- train-only z-score (center + scale by train std),
  the primary paper-compatible independent reconstruction (P1).

Statistics are fit on TRAIN rows only; validation and test are transformed with the
frozen train statistics -- no validation/test value may influence preprocessing.
Zero-variance columns are handled explicitly (scale set to 1.0, indices and counts
recorded), never divided silently by zero. The policy choice is NOT made using test
performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

PREPROC_HISTORICAL_D0 = "PREPROC_HISTORICAL_D0"          # P0: center-only
PREPROC_ZSCORE_TRAIN_ONLY = "PREPROC_ZSCORE_TRAIN_ONLY"  # P1: z-score
POLICIES = (PREPROC_HISTORICAL_D0, PREPROC_ZSCORE_TRAIN_ONLY)

_ZERO_VAR_EPS = 1e-12


@dataclass(frozen=True)
class FittedScaler:
    """A train-fit scaler carrying explicit zero-variance accounting."""

    policy: str
    mean_: np.ndarray
    scale_: np.ndarray
    with_scaling: bool
    zero_var_idx: np.ndarray
    n_features: int

    @property
    def n_zero_var(self) -> int:
        return int(self.zero_var_idx.size)

    @classmethod
    def fit(cls, X_train: np.ndarray, policy: str) -> "FittedScaler":
        """Fit on TRAIN rows only. Raises on an unresolved policy."""
        if policy not in POLICIES:
            raise ValueError(f"unresolved preprocessing policy {policy!r}; choose {POLICIES}")
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2-D (n_train, n_features)")
        mean = X_train.mean(axis=0)
        with_scaling = policy == PREPROC_ZSCORE_TRAIN_ONLY
        if with_scaling:
            std = X_train.std(axis=0)
            zero_var = np.where(std < _ZERO_VAR_EPS)[0]
            scale = np.where(std < _ZERO_VAR_EPS, 1.0, std)  # never divide by ~0
        else:
            zero_var = np.array([], dtype=int)
            scale = np.ones(X_train.shape[1])
        return cls(policy=policy, mean_=mean, scale_=scale, with_scaling=with_scaling,
                   zero_var_idx=zero_var, n_features=X_train.shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the FROZEN train statistics. Never refits."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features:
            raise ValueError(f"feature dim {X.shape[1]} != fitted {self.n_features}")
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map scaled values back to raw units (Z * scale + mean).

        Used so cross-fit denoised predictions from DIFFERENT per-model scalers all
        land in the same raw-response space, keeping downstream inputs comparable.
        """
        Z = np.asarray(Z, dtype=np.float64)
        if Z.shape[1] != self.n_features:
            raise ValueError(f"feature dim {Z.shape[1]} != fitted {self.n_features}")
        return Z * self.scale_ + self.mean_

    def report(self) -> dict:
        return {
            "policy": self.policy, "with_scaling": self.with_scaling,
            "n_features": self.n_features, "n_zero_var": self.n_zero_var,
            "zero_var_idx": self.zero_var_idx.tolist(),
        }
