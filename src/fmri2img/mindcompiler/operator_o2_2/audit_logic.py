"""O2.2 frozen pure geometry/decision logic (audit only)."""
from __future__ import annotations

import numpy as np

EPS = 1e-12
RY_HIGH, TRUNC_LARGE, REL_MOD, SH_MOD = 0.40, 0.15, 0.20, 0.20


def subspace_projector(B: np.ndarray) -> np.ndarray:
    return B @ B.T


def retention(delta: np.ndarray, B: np.ndarray) -> float:
    """||delta B B^T||^2 / ||delta||^2 (row-vector delta, orthonormal B: voxels x rank)."""
    P = B @ B.T
    return float((delta @ P @ delta) / (delta @ delta + EPS))


def cross_state_overlap(Bv: np.ndarray, Bi: np.ndarray) -> float:
    """||Bv^T Bi||_F^2 / min(rank_v, rank_i) in [0,1]."""
    return float(np.sum((Bv.T @ Bi) ** 2) / min(Bv.shape[1], Bi.shape[1]))


def classify_regime(r_y_visfull: float, r_y_srm: float, srm_truncation_loss: float,
                    outside_reliability: float, outside_loso_sharedness: float) -> str:
    if r_y_visfull >= RY_HIGH and r_y_srm < 0.25 and srm_truncation_loss >= TRUNC_LARGE:
        return "A_SRM_DIMENSION_TRUNCATION_DOMINANT"
    if r_y_visfull < RY_HIGH and outside_reliability >= REL_MOD and outside_loso_sharedness >= SH_MOD:
        return "B_STATE_SPECIFIC_SHARED_GEOMETRY_OUTSIDE_VISUAL_SPAN"
    if r_y_visfull < RY_HIGH and outside_reliability >= REL_MOD and outside_loso_sharedness < SH_MOD:
        return "C_SUBJECT_SPECIFIC_IMAGERY_GEOMETRY"
    if r_y_visfull < RY_HIGH and outside_reliability < REL_MOD:
        return "D_IMAGERY_MEASUREMENT_LIMITED"
    return "MIXED"


def overall_status(roi_regimes: dict) -> str:
    distinct = set(roi_regimes.values())
    if len(distinct) > 1:
        return "CROSS_STATE_TRANSPORT_MULTIREGIME"
    reg = next(iter(distinct))
    return {"A_SRM_DIMENSION_TRUNCATION_DOMINANT": "CROSS_STATE_TRANSPORT_SRM_TRUNCATION_DOMINANT",
            "B_STATE_SPECIFIC_SHARED_GEOMETRY_OUTSIDE_VISUAL_SPAN": "CROSS_STATE_TRANSPORT_SHARED_IMAGERY_SPACE_OUTSIDE_VISION",
            "C_SUBJECT_SPECIFIC_IMAGERY_GEOMETRY": "CROSS_STATE_TRANSPORT_SUBJECT_SPECIFIC_IMAGERY_SPACE",
            "D_IMAGERY_MEASUREMENT_LIMITED": "CROSS_STATE_TRANSPORT_MEASUREMENT_LIMITED"}.get(reg, "CROSS_STATE_TRANSPORT_UNRESOLVED")
