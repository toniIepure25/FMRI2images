"""S2.2B primary H-A spatial-profile scorecard (frozen criteria; descriptive only).

Six prospective ROIs of one participant -- NOT six subjects. No population inference,
no fold p-values. Criteria A-D and the classification are frozen before evaluation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr


def spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def evaluate_criteria(L_rel_img: Dict[str, float], L_perf_RAW: Dict[str, float],
                      n_prospective: int = 6) -> dict:
    """Evaluate the four frozen H-A criteria over EVALUABLE ROIs.

    A: median(L_rel_img) > 0. B: median(L_perf_RAW) > 0.
    C: Spearman(L_rel_img, L_perf_RAW) > 0.
    D: >=4 of 6 (or >=2/3 of evaluable if some non-evaluable) ROIs with both losses > 0.
    """
    rois = sorted(set(L_rel_img) & set(L_perf_RAW))
    rel = [L_rel_img[r] for r in rois]
    perf = [L_perf_RAW[r] for r in rois]
    n_eval = len(rois)
    concordant = sum(1 for r in rois if L_rel_img[r] > 0 and L_perf_RAW[r] > 0)
    d_threshold = 4 if n_eval == n_prospective else int(np.ceil(2 / 3 * n_eval))
    rho = spearman(rel, perf)
    A = bool(np.median(rel) > 0) if rel else False
    B = bool(np.median(perf) > 0) if perf else False
    C = bool(rho > 0) if np.isfinite(rho) else False
    D = bool(concordant >= d_threshold)
    n_passed = sum([A, B, C, D])
    return dict(criterion_A=A, criterion_B=B, criterion_C=C, criterion_D=D,
                n_criteria_passed=n_passed, n_evaluable=n_eval,
                concordant_ROI_count=concordant, d_threshold=d_threshold,
                rho_reliability_vs_RAW_loss=rho,
                median_imagery_reliability_loss=float(np.median(rel)) if rel else float("nan"),
                median_RAW_performance_loss=float(np.median(perf)) if perf else float("nan"))


def classify(scorecard: dict, n_prospective: int = 6) -> str:
    if scorecard["n_evaluable"] < 4:
        return "H_A_SPATIAL_PROFILE_INCONCLUSIVE"
    n = scorecard["n_criteria_passed"]
    if n == 4:
        return "H_A_SPATIAL_PROFILE_SUPPORTED"
    if n in (2, 3):
        return "H_A_SPATIAL_PROFILE_PARTIAL"
    return "H_A_SPATIAL_PROFILE_NOT_SUPPORTED"


def leave_one_roi_rhos(L_rel_img: Dict[str, float], L_perf_RAW: Dict[str, float]) -> dict:
    rois = sorted(set(L_rel_img) & set(L_perf_RAW))
    rhos = []
    for drop in rois:
        keep = [r for r in rois if r != drop]
        rhos.append(spearman([L_rel_img[r] for r in keep], [L_perf_RAW[r] for r in keep]))
    fin = [r for r in rhos if np.isfinite(r)]
    return dict(rhos={rois[i]: rhos[i] for i in range(len(rois))},
                n_positive=int(sum(1 for r in fin if r > 0)),
                min=float(min(fin)) if fin else float("nan"),
                max=float(max(fin)) if fin else float("nan"))
