"""S2.4H post-hoc descriptive heterogeneity utilities (no rescue, no new stats).

Pure functions over the ALREADY-COMMITTED S2.3 summaries. The frozen S2.3 primary
result is immutable here: this module offers NO path to recompute a primary p-value,
exclude a participant, or change the classification.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction.s2_3_scorecard import _rho as spearman

#: Frozen S2.3 primary result (immutable). S2.4H may verify but never recompute it.
S2_3_PRIMARY_IMMUTABLE = {
    "rho_observed": 0.6785714285714286,
    "n_permutations": 5040,
    "n_null_ge_observed": 276,
    "p_exact_one_sided": 276 / 5040,
    "classification": "H_A_CROSS_PARTICIPANT_PARTIAL",
}


class PrimaryRescueError(RuntimeError):
    """Raised if code attempts to recompute/alter the frozen S2.3 primary."""


def assert_primary_immutable(perm_artifact: dict) -> None:
    """Verify a loaded permutation artifact matches the frozen primary EXACTLY."""
    for k in ("n_permutations", "n_null_ge_observed"):
        if int(perm_artifact.get(k, -1)) != S2_3_PRIMARY_IMMUTABLE[k]:
            raise PrimaryRescueError(f"S2.3 primary altered: {k}")
    if abs(float(perm_artifact.get("p_exact_one_sided", -1)) - S2_3_PRIMARY_IMMUTABLE["p_exact_one_sided"]) > 1e-12:
        raise PrimaryRescueError("S2.3 primary p altered")


def median(x: List[float]) -> float:
    return float(np.median(x)) if len(x) else float("nan")


def mad(x: List[float]) -> float:
    """Median absolute deviation (robust dispersion)."""
    if not len(x):
        return float("nan")
    a = np.asarray(x, float)
    return float(np.median(np.abs(a - np.median(a))))


def sign_counts(rel: List[float], perf: List[float]) -> dict:
    """Per-ROI sign bookkeeping for one participant (positive = B1 worse)."""
    rel = np.asarray(rel, float); perf = np.asarray(perf, float)
    return dict(
        n_rel_positive=int((rel > 0).sum()),
        n_perf_positive=int((perf > 0).sum()),
        n_concordant_positive=int(((rel > 0) & (perf > 0)).sum()),
        n_concordant_negative=int(((rel < 0) & (perf < 0)).sum()),
        n_discordant=int(((rel > 0) != (perf > 0)).sum()),
        n_roi=int(len(rel)))


def ranks(values: Dict[str, float]) -> Dict[str, int]:
    """Ascending competition-free rank (1 = smallest) per key."""
    items = sorted(values, key=lambda k: values[k])
    return {k: i + 1 for i, k in enumerate(items)}


def rank_influence(S_rel: Dict[str, float], S_perf: Dict[str, float]) -> dict:
    rr, rp = ranks(S_rel), ranks(S_perf)
    return {s: dict(rank_S_rel=rr[s], rank_S_perf=rp[s], abs_rank_diff=abs(rr[s] - rp[s]))
            for s in S_rel}
