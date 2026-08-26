"""O1 participant-level inference (exact 2^8 sign-flip + Holm), operator-class logic, 2nd metrics.

Participant is the inferential unit (N=8). No folds/identities/voxels as subjects.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np

COMPETITIVE_TOL = 0.02        # frozen: |score diff| <= this => "performance-competitive"
EXISTENCE_MARGIN = 0.0        # frozen: O4 > O0 by more than this => reusable-operator evidence at the cell
ALPHA = 0.05


def signflip_test(diffs: Sequence[float]) -> dict:
    """Exact one-sided sign-flip test that mean(diffs) > 0 over N participants (2^N enumeration)."""
    d = np.asarray([x for x in diffs if x == x], float)
    n = d.size
    obs = float(d.mean()) if n else float("nan")
    if n == 0:
        return {"n": 0, "observed_mean": float("nan"), "p_one_sided_greater": float("nan")}
    ge = 0; total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        total += 1
        if float((np.asarray(signs) * d).mean()) >= obs - 1e-12:
            ge += 1
    return {"n": int(n), "observed_mean": obs, "n_assignments": total,
            "p_one_sided_greater": ge / total}


def holm(pvals: Dict[str, float], alpha: float = ALPHA) -> Dict[str, dict]:
    """Holm-Bonferroni across keys (e.g. 7 ROIs). Returns per-key adjusted p and reject flag."""
    items = [(k, v) for k, v in pvals.items() if v == v]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)             # enforce monotonic non-decreasing adjusted p
        out[k] = {"raw_p": p, "holm_p": running, "reject": running <= alpha}
    for k, v in pvals.items():
        if v != v:
            out[k] = {"raw_p": float("nan"), "holm_p": float("nan"), "reject": False}
    return out


def classify_operator(scores: Dict[str, float]) -> str:
    """Frozen deterministic operator-class label from one cell's outer-test scores {O0..O4}."""
    o0, o1, o2, o3, o4 = (scores[m] for m in ("O0", "O1", "O2", "O3", "O4"))
    if not (o4 > o0 + EXISTENCE_MARGIN):
        return "NO_REUSABLE_OPERATOR_EVIDENCE"
    if o1 >= o4 - COMPETITIVE_TOL:
        return "GLOBAL_GAIN_SUFFICIENT"
    if o2 >= o4 - COMPETITIVE_TOL:
        return "DIAGONAL_REWEIGHTING_SUFFICIENT"
    if o3 >= o4 - COMPETITIVE_TOL:
        return "LOW_RANK_DEFORMATION_SUFFICIENT"
    return "FULL_CROSS_VOXEL_OPERATOR_NEEDED"


# ------------------------------------------------------------------ secondary metrics
def secondary_metrics(pred: np.ndarray, meas: np.ndarray) -> dict:
    diff = pred - meas
    denom = float((meas - meas.mean()) @ (meas - meas.mean()))
    nmse = float((diff @ diff) / (meas @ meas)) if float(meas @ meas) > 1e-12 else float("nan")
    cos = float(pred @ meas / (np.linalg.norm(pred) * np.linalg.norm(meas) + 1e-12))
    ev = float(1.0 - (diff - diff.mean()) @ (diff - diff.mean()) / denom) if denom > 1e-12 else float("nan")
    norm_ratio = float(np.linalg.norm(pred) / (np.linalg.norm(meas) + 1e-12))
    return {"normalized_mse": nmse, "cosine": cos, "explained_variance": ev, "centroid_norm_ratio": norm_ratio}
