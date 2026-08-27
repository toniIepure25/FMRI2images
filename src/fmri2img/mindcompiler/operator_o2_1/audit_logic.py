"""O2.1 frozen pure decision logic (audit only)."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def contrast_retention(delta: np.ndarray, W: np.ndarray) -> float:
    """||delta W W^T||^2 / ||delta||^2 for row-vector delta and orthonormal W (p x K)."""
    P = W @ W.T
    return float((delta @ P @ delta) / (delta @ delta + EPS))


def shared_contrast_relative_error(dyhat: np.ndarray, dy: np.ndarray) -> float:
    return float(np.linalg.norm(dyhat - dy) ** 2 / max(float(np.linalg.norm(dy) ** 2), EPS))


def std_median_diff(median_a: float, median_b: float, pooled_mad: float) -> float:
    return (median_a - median_b) / (pooled_mad if pooled_mad > EPS else EPS)


def smd_label(smd: float) -> str:
    a = abs(smd)
    return "STRONG_DESCRIPTIVE_SEPARATION" if a >= 1.0 else "MODERATE" if a >= 0.5 else "WEAK"


def normalized_shared_gain(g_shared: float, s0: float, eps: float = 0.05) -> float:
    return g_shared / max(1.0 - s0, eps)


def decomposition_confidence(upper_boundary_rate: float, median_delta_cv_increment: float) -> str:
    concern = upper_boundary_rate >= 0.25
    if concern and median_delta_cv_increment <= 0:
        return "DECOMPOSITION_CONFIDENCE_LOW"
    if (concern and median_delta_cv_increment > 0) or (not concern and abs(median_delta_cv_increment) < 0.005):
        return "DECOMPOSITION_CONFIDENCE_MODERATE"
    return "DECOMPOSITION_CONFIDENCE_HIGH"


def o3_ready(a: bool, b: bool, c: bool, d_conf: str, e: bool) -> str:
    d = d_conf in ("DECOMPOSITION_CONFIDENCE_HIGH", "DECOMPOSITION_CONFIDENCE_MODERATE")
    return "O3_READY_FOR_MULTI_STATE_FEASIBILITY" if (a and b and c and d and e) else "O3_NOT_READY"
