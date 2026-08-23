"""S2.6R Roy-defined dimensionality + visual<->imagery subspace alignment geometry.

Reconstructs the two central Roy scientific quantities that S2.5M deferred:

* dimensionality d_report from a TEST prediction-accuracy vs retained-rank curve
  (Roy Figure 4), using the frozen first-crossing 99%-of-peak rule;
* the visual<->imagery subspace alignment ratio a_g (Roy Figure 5), computed from the
  reduced-rank model OUTPUT-space bases (right singular vectors of the fitted TRAIN
  prediction), never from a proxy (no principal angles / CCA / Procrustes / RSA).

Everything here operates on the SAME per-target-Lambda reduced-rank ridge solutions
frozen in S2.5M; nothing re-optimizes the prediction pipeline.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

DOI = "10.1101/2025.09.02.672180"

# --- dimensionality (Roy Figure 4) ---
D99_RULE = "FIRST_INTEGER_RANK_REACHING_99_PERCENT_OF_TEST_PEAK"
D99_THRESHOLD = 0.99
NONPOSITIVE_PEAK_TAG = "NOT_EVALUABLE_NONPOSITIVE_TEST_PEAK"
NO_FINITE_TAG = "NOT_EVALUABLE_NO_FINITE_TEST"
D99_TIE_NOTE = "D99_EXACT_TIE_INTERPOLATION_NOT_PUBLICLY_SPECIFIED"  # no splines / no interpolation

# --- alignment (Roy Figure 5) ---
ALIGNMENT_VAR_DDOF = 1                     # frozen sample-variance convention (numerator == denominator)
N_ALIGNMENT_NULL = 100                     # Roy Figure-5 null repeats (exact)
ALIGNMENT_NULL_BASIS = "FULL_V_VIS_FITTED_TRAIN_PREDICTION"
ALIGNMENT_NULL_BASIS_SOURCE_NOT_FULLY_IDENTIFIABLE = True
ALIGNMENT_SPACE = "S2_5M_TRAIN_STANDARDIZED_MODEL_SPACE"
_TV_EPS = 1e-12
_ORTHO_ATOL = 1e-8


# ----------------------------------------------------------------------------- dimensionality
def d99_from_curve(test_scores: Sequence[float]) -> Tuple[Optional[int], str]:
    """First integer rank (1-indexed) whose TEST score >= 0.99 * max finite TEST score.

    Returns (d_report, reason). Non-positive or all-NaN peaks are NOT_EVALUABLE; we never
    force a dimensionality from an uninformative curve, and never interpolate between ranks.
    """
    s = np.asarray(test_scores, dtype=np.float64)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return None, NO_FINITE_TAG
    peak = float(finite.max())
    if peak <= 0.0:
        return None, NONPOSITIVE_PEAK_TAG
    thr = D99_THRESHOLD * peak
    for r, val in enumerate(s, start=1):           # smallest integer rank first-crossing
        if np.isfinite(val) and val >= thr:
            return r, "OK"
    return None, NO_FINITE_TAG                      # unreachable given peak is finite & >0


def argmax_rank(test_scores: Sequence[float]) -> Optional[int]:
    """Descriptive argmax rank (1-indexed) of the TEST curve; None if no finite value."""
    s = np.asarray(test_scores, dtype=np.float64)
    if not np.isfinite(s).any():
        return None
    return int(np.nanargmax(np.where(np.isfinite(s), s, -np.inf))) + 1


# ----------------------------------------------------------------------------- bases
def extract_output_basis(X_train: np.ndarray, W_lambda: np.ndarray) -> np.ndarray:
    """Right singular vectors (columns) of the fitted TRAIN prediction X_train @ W_lambda.

    These are the model's estimated OUTPUT-space dimensions; the first r columns are
    exactly the projector basis used by the reduced-rank construction W_RRR = W V_r V_r^T.
    Returns V_full with shape (n_output_voxels, k), k = min(n_train, n_output_voxels).
    """
    fitted = np.asarray(X_train, np.float64) @ np.asarray(W_lambda, np.float64)
    _, _, Vt = np.linalg.svd(fitted, full_matrices=False)
    V = Vt.T
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=_ORTHO_ATOL), "basis not orthonormal"
    return V


def projector(V_d: np.ndarray) -> np.ndarray:
    """Symmetric idempotent projector P = V_d V_d^T (sign-invariant subspace identity)."""
    return V_d @ V_d.T


def projector_hash(V_d: np.ndarray, decimals: int = 9) -> str:
    """Sign-invariant hash of the subspace spanned by V_d (hashes the projector, not V)."""
    P = np.round(projector(V_d), decimals) + 0.0     # +0.0 normalises -0.0
    return hashlib.sha256(P.astype(np.float64).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------- alignment
def subspace_total_variance(X: np.ndarray, V_d: np.ndarray, ddof: int = ALIGNMENT_VAR_DDOF) -> float:
    """Sum over voxels of the test-trial variance of X projected onto span(V_d).

    Uses P = V_d V_d^T; equals sum var(X @ V_d) for orthonormal V_d (verified in tests).
    """
    proj = np.asarray(X, np.float64) @ projector(V_d)
    return float(np.var(proj, axis=0, ddof=ddof).sum())


def alignment_ratio(X_test_vis: np.ndarray, V_vis_full: np.ndarray, V_img_full: np.ndarray,
                    d: int) -> dict:
    """Roy a_g = TV(X_vis on first-d IMAGERY dims) / TV(X_vis on first-d VISUAL dims).

    Same dimension count d (= d_img_fold) in numerator and denominator. Never clipped;
    finite noisy data may exceed 1. Non-evaluable if the visual denominator vanishes.
    """
    Vvis_d = V_vis_full[:, :d]
    Vimg_d = V_img_full[:, :d]
    tv_vis = subspace_total_variance(X_test_vis, Vvis_d)
    tv_img = subspace_total_variance(X_test_vis, Vimg_d)
    if tv_vis <= _TV_EPS:
        return dict(TV_vis=tv_vis, TV_img=tv_img, a_g=float("nan"),
                    evaluable=False, reason="NON_EVALUABLE_ZERO_VISUAL_DENOMINATOR", d=int(d))
    return dict(TV_vis=tv_vis, TV_img=tv_img, a_g=tv_img / tv_vis,
                evaluable=True, reason="OK", d=int(d))


def _null_seed(participant: str, roi: str, fold: str, k: int) -> int:
    key = f"{DOI}|S2.6R|{participant}|{roi}|{fold}|null={k}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def alignment_null(X_test_vis: np.ndarray, V_vis_full: np.ndarray, tv_vis: float, d: int,
                   participant: str, roi: str, fold: str, n: int = N_ALIGNMENT_NULL) -> List[float]:
    """Replace the d imagery dims with d random VISUAL singular vectors (Roy Fig-5 null).

    Draws d distinct columns of V_vis_full without replacement, projects the SAME held-out
    visual activity, divides by the SAME TV_vis. DOI-derived per-iteration seeds; the
    observed alignment value never enters the seed.
    """
    k_cols = V_vis_full.shape[1]
    out: List[float] = []
    for k in range(n):
        rng = np.random.default_rng(_null_seed(participant, roi, fold, k))
        idx = rng.choice(k_cols, size=d, replace=False)
        tv_null = subspace_total_variance(X_test_vis, V_vis_full[:, idx])
        out.append(tv_null / tv_vis if tv_vis > _TV_EPS else float("nan"))
    return out


def null_summary(a_g_obs: float, null: Sequence[float]) -> dict:
    arr = np.asarray(null, np.float64)
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return dict(null_mean=float("nan"), null_median=float("nan"),
                    null_p05=float("nan"), null_p95=float("nan"),
                    empirical_percentile=float("nan"), n_null=int(arr.size))
    pct = float(np.mean(fin <= a_g_obs)) if np.isfinite(a_g_obs) else float("nan")
    return dict(null_mean=float(fin.mean()), null_median=float(np.median(fin)),
                null_p05=float(np.percentile(fin, 5)), null_p95=float(np.percentile(fin, 95)),
                empirical_percentile=pct, n_null=int(arr.size))
