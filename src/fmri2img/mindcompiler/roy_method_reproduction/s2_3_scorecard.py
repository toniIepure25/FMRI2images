"""S2.3 cross-participant H-A scorecard: participant-level medians + exact permutation.

The PARTICIPANT is the unit of analysis (7 primary: subj02..subj08). For each
participant the seven ROI losses are reduced to one median (equal ROI weighting).
The primary association is an EXACT one-sided permutation Spearman over 7! = 5040
orderings -- not an asymptotic p-value. Criteria A-D and classification are frozen
before outcomes.
"""
from __future__ import annotations

from itertools import permutations
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

PRIMARY_PARTICIPANTS = ("subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08")
DEVELOPMENT_REFERENCE = "subj01"
ALPHA = 0.05


def participant_summary(L_rel_img_by_roi: Dict[str, float],
                        L_perf_raw_by_roi: Dict[str, float]) -> dict:
    """Reduce the seven ROI losses to one participant row (equal-weight median)."""
    rois = sorted(set(L_rel_img_by_roi) & set(L_perf_raw_by_roi))
    rel = np.array([L_rel_img_by_roi[r] for r in rois], float)
    perf = np.array([L_perf_raw_by_roi[r] for r in rois], float)
    concordant = int(np.sum((rel > 0) & (perf > 0)))
    return dict(S_rel=float(np.median(rel)), S_perf=float(np.median(perf)),
                n_evaluable_rois=len(rois),
                S_rel_mean=float(rel.mean()), S_perf_mean=float(perf.mean()),
                min_L_rel=float(rel.min()), max_L_rel=float(rel.max()),
                iqr_L_rel=float(np.percentile(rel, 75) - np.percentile(rel, 25)),
                roi_concordant_signs=concordant)


def _rho(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def exact_permutation_spearman(S_rel: List[float], S_perf: List[float]) -> dict:
    """Exact one-sided (positive) permutation p over all orderings of S_perf."""
    x = list(S_rel)
    y = list(S_perf)
    obs = _rho(x, y)
    perms = list(permutations(range(len(y))))
    null = [_rho(x, [y[i] for i in p]) for p in perms]
    n_ge = int(sum(1 for r in null if np.isfinite(r) and r >= obs - 1e-12))
    return dict(rho_observed=obs, n_permutations=len(perms),
                n_null_ge_observed=n_ge, p_exact_one_sided=float(n_ge / len(perms)),
                alternative="positive", alpha=ALPHA)


def evaluate_criteria(S_rel: Dict[str, float], S_perf: Dict[str, float]) -> dict:
    """Criteria A-D over evaluable participants (frozen)."""
    subs = sorted(set(S_rel) & set(S_perf))
    rel = [S_rel[s] for s in subs]
    perf = [S_perf[s] for s in subs]
    n = len(subs)
    perm = exact_permutation_spearman(rel, perf) if n >= 3 else {
        "rho_observed": float("nan"), "p_exact_one_sided": float("nan"),
        "n_permutations": 0, "n_null_ge_observed": 0}
    concordant = int(sum(1 for s in subs if S_rel[s] > 0 and S_perf[s] > 0))
    A = bool(np.median(rel) > 0) if rel else False
    B = bool(np.median(perf) > 0) if perf else False
    C = bool(np.isfinite(perm["rho_observed"]) and perm["rho_observed"] > 0
             and perm["p_exact_one_sided"] <= ALPHA)
    D = bool(concordant >= 5)
    return dict(criterion_A=A, criterion_B=B, criterion_C=C, criterion_D=D,
                n_criteria_passed=sum([A, B, C, D]), n_evaluable_participants=n,
                concordant_participant_count=concordant,
                median_S_rel=float(np.median(rel)) if rel else float("nan"),
                median_S_perf=float(np.median(perf)) if perf else float("nan"),
                permutation=perm)


def classify(scorecard: dict) -> str:
    if scorecard["n_evaluable_participants"] < 5:
        return "H_A_CROSS_PARTICIPANT_INCONCLUSIVE"
    n = scorecard["n_criteria_passed"]
    if n == 4:
        return "H_A_CROSS_PARTICIPANT_SUPPORTED"
    if n in (2, 3):
        return "H_A_CROSS_PARTICIPANT_PARTIAL"
    return "H_A_CROSS_PARTICIPANT_NOT_SUPPORTED"


def leave_one_participant(S_rel: Dict[str, float], S_perf: Dict[str, float]) -> dict:
    subs = sorted(set(S_rel) & set(S_perf))
    rhos = {}
    for drop in subs:
        keep = [s for s in subs if s != drop]
        rhos[drop] = _rho([S_rel[s] for s in keep], [S_perf[s] for s in keep])
    vals = [r for r in rhos.values() if np.isfinite(r)]
    return dict(rhos=rhos, n_positive=int(sum(1 for r in vals if r > 0)),
                min=float(min(vals)) if vals else float("nan"),
                max=float(max(vals)) if vals else float("nan"),
                median=float(np.median(vals)) if vals else float("nan"))
