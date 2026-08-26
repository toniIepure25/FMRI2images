"""O2 shared operator (gauge-equivariant scalar ridge) + gauge-invariant baselines & geometry.

Common-space axes are arbitrary up to a shared orthogonal rotation Q, so every estimator and
diagnostic here is orthogonally equivariant/invariant: full-matrix scalar ridge (never per-target
or diagonal), group-imagery-mean and global-scalar-gain baselines, Frobenius/singular-value
geometry, and the rotation-invariant NONTRIVIAL_TRANSFORMATION_INDEX.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

RIDGE_LO, RIDGE_HI, RIDGE_N = 1e-4, 1e4, 25
DELTA_RIDGE_LO, DELTA_RIDGE_HI, DELTA_RIDGE_N = 1e-4, 1e4, 25
_EPS = 1e-12


def ridge_grid() -> np.ndarray:
    return np.logspace(np.log10(RIDGE_LO), np.log10(RIDGE_HI), RIDGE_N)


def fit_shared_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Full-matrix scalar-ridge operator with intercept: Y ~ b + X T. Gauge-equivariant."""
    xbar, ybar = X.mean(0), Y.mean(0)
    Xc, Yc = X - xbar, Y - ybar
    K = Xc.shape[1]
    T = np.linalg.solve(Xc.T @ Xc + lam * np.eye(K), Xc.T @ Yc)
    return T, ybar - xbar @ T


def predict_shared(T: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return b[None, :] + X @ T


def fit_global_gain(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
    """S1 GLOBAL_SHARED_GAIN: Y ~ b + a X (scalar a). Gauge-invariant."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    denom = float((Xc * Xc).sum())
    a = float((Xc * Yc).sum() / denom) if denom > _EPS else 0.0
    return a, Y.mean(0) - a * X.mean(0)


def group_imagery_mean(Y: np.ndarray) -> np.ndarray:
    """S0 GROUP_IMAGERY_MEAN in shared space (training subjects x training identities)."""
    return Y.mean(0)


def pattern_r(pred: np.ndarray, meas: np.ndarray) -> float:
    a = pred - pred.mean(); b = meas - meas.mean()
    d = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / d) if d > _EPS else float("nan")


# ------------------------------------------------------------------ gauge-invariant geometry
def nontrivial_transformation_index(T: np.ndarray) -> float:
    """||T - a*I||_F / ||T||_F with a* = argmin_a ||T - aI||_F = trace(T)/K. Rotation-invariant."""
    K = T.shape[0]
    a_star = float(np.trace(T) / K)
    num = np.linalg.norm(T - a_star * np.eye(K))
    den = np.linalg.norm(T)
    return float(num / den) if den > _EPS else float("nan")


def shared_operator_spectrum(T: np.ndarray) -> dict:
    s = np.linalg.svd(T, compute_uv=False)
    s = s[s > _EPS]
    pr = float((s.sum() ** 2) / (np.square(s).sum())) if s.size else float("nan")
    e90 = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.90) + 1) if s.size else 0
    return {"spectral_norm": float(s.max()) if s.size else float("nan"),
            "frobenius_norm": float(np.linalg.norm(T)),
            "singular_values": s.tolist(),
            "effective_rank_participation": pr, "energy90_rank": e90,
            "condition_number": float(s.max() / s.min()) if s.size and s.min() > _EPS else float("inf"),
            "fraction_contracted": float((s < 1).mean()) if s.size else float("nan"),
            "fraction_expanded": float((s > 1).mean()) if s.size else float("nan"),
            "nontrivial_transformation_index": nontrivial_transformation_index(T),
            "distance_to_best_scalar_identity": float(np.linalg.norm(T - (np.trace(T) / T.shape[0]) * np.eye(T.shape[0])))}


def shared_action_fraction(X: np.ndarray, T_shared: np.ndarray, Delta_s: np.ndarray) -> float:
    """||X T_shared||_F^2 / (||X T_shared||_F^2 + ||X Delta_s||_F^2). Invariant to shared rotation."""
    sh = float(np.linalg.norm(X @ T_shared) ** 2)
    ind = float(np.linalg.norm(X @ Delta_s) ** 2)
    return sh / (sh + ind) if (sh + ind) > _EPS else float("nan")
