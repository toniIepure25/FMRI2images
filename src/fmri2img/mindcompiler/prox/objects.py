"""MINDIR-PROX canonical scientific object model (typed). Replaces anonymous dicts for the platform layer. These
are lightweight dataclasses with schema versions; sealed gate artifacts are NOT migrated (see the migration map)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence, Dict, Any

import numpy as np

SCHEMA_VERSION = "prox-object/1.0.0"


@dataclass
class Subject:
    subject_id: str
    cohort: str                      # A_HISTORICAL_N8_CLOSED / B_INDEPENDENT_O2_16 / C_CONFIRMATION / SYNTHETIC
    schema_version: str = SCHEMA_VERSION


@dataclass
class Representation:
    """A frozen feature representation (voxel-native / PCA / SRM-like / anatomical). Frozen before outcome access."""
    name: str
    family: str
    dim: int
    frozen: bool = True
    schema_version: str = SCHEMA_VERSION


@dataclass
class Subspace:
    """Row-orthonormal basis (k x dim)."""
    basis: np.ndarray
    schema_version: str = SCHEMA_VERSION

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    def orthonormal(self, tol=1e-8) -> bool:
        G = self.basis @ self.basis.T
        return bool(np.max(np.abs(G - np.eye(self.rank))) < tol)


@dataclass
class TargetState:
    """A subject's target internal-state samples for one state (obs x dim)."""
    subject_id: str
    state: str                       # e.g. imagery / recall
    samples: np.ndarray
    schema_version: str = SCHEMA_VERSION


@dataclass
class MetricResult:
    name: str
    value: float
    metric_version: str
    participant_level: bool
    n: Optional[int] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        d = asdict(self); return d


@dataclass
class NullResult:
    name: str
    observed: float
    null_median: float
    p_value: float
    n_null: int
    method: str
    schema_version: str = SCHEMA_VERSION


@dataclass
class InferenceResult:
    family: str
    per_test_p: Dict[str, float]
    holm_reject: Dict[str, bool]
    inferential_unit: str = "participant"
    schema_version: str = SCHEMA_VERSION


@dataclass
class FrozenArtifact:
    path: str
    sha256: str
    protocol_version: str
    schema_version: str = SCHEMA_VERSION


def orthonormalize(rows: np.ndarray, k: Optional[int] = None) -> Subspace:
    X = np.asarray(rows, np.float64)
    Q = np.linalg.qr(X.T)[0]
    k = k or X.shape[0]
    return Subspace(basis=Q[:, :k].T)
