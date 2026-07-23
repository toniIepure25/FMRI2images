"""Reduced-rank ridge regression and selection, per Roy et al. (2025) Methods.

Paper-specified components implemented here (constants in ``contracts.py``):

* ridge over 100 log-spaced values 1e-3..1e5, selected on validation;
* reduced-rank projection of the fitted ridge prediction, ranks 1..r_max;
* rank selection near 99% of peak validation performance;
* per-voxel Pearson correlation with explicit zero-variance handling.

Ambiguous selection choices (rank/ridge tie-breaks) are **explicit policy
arguments with no silent default** -- callers must resolve them from the
ambiguity registry. NumPy/SciPy only; no h5py/nibabel, so this is host-independent
and testable against synthetic ground truth without any real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction.contracts import (
    RANK_SELECTION_FRACTION_OF_PEAK,
    RIDGE_GRID_HI,
    RIDGE_GRID_LO,
    RIDGE_GRID_N,
)

RankTieBreak = Literal["smallest_rank_at_threshold", "argmax_validation"]
RidgeTieBreak = Literal["argmax_validation", "one_se_rule"]


def ridge_grid() -> np.ndarray:
    """The paper's 100 log-spaced ridge values, 1e-3..1e5."""
    return np.logspace(np.log10(RIDGE_GRID_LO), np.log10(RIDGE_GRID_HI), RIDGE_GRID_N)


@dataclass(frozen=True)
class Standardizer:
    """Train-only centering (and optional scaling). Fit on train, apply everywhere.

    Fitting on anything but training rows leaks target-distribution information,
    which is the failure mode the reproduction must avoid.
    """

    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, with_scaling: bool) -> "Standardizer":
        mean = X.mean(axis=0)
        if with_scaling:
            scale = X.std(axis=0)
            scale = np.where(scale < 1e-12, 1.0, scale)
        else:
            scale = np.ones(X.shape[1])
        return cls(mean, scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


def _ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge W minimizing ||Y - X W||^2 + lam ||W||^2 (no intercept).

    Intercept is handled by centering (``Standardizer``), controlled explicitly by
    the caller, never implicitly here.
    """
    n_features = X.shape[1]
    A = X.T @ X + lam * np.eye(n_features)
    return np.linalg.solve(A, X.T @ Y)


def fit_reduced_rank(X: np.ndarray, Y: np.ndarray, lam: float, rank: int) -> np.ndarray:
    """Reduced-rank ridge: project the fitted ridge prediction to ``rank`` dims.

    W_full = ridge(X, Y, lam); the fitted values X W_full are truncated to their
    top-``rank`` right-singular subspace, giving W_rr = W_full P_rank. Deterministic.

    Args:
        X: (n, p) training source. Y: (n, q) training target.
        lam: ridge penalty. rank: retained rank (>=1).

    Returns:
        (p, q) reduced-rank weight matrix.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    W = _ridge_solve(X, Y, lam)
    fitted = X @ W
    # right-singular vectors of the fitted prediction define the target subspace
    _, _, Vt = np.linalg.svd(fitted, full_matrices=False)
    r = min(rank, Vt.shape[0])
    P = Vt[:r].T @ Vt[:r]
    return W @ P


def per_voxel_pearson(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Per-column Pearson r. Zero-variance columns -> NaN (never a favorable 0/1).

    Returns (q,) correlations. NaN marks an undefined correlation and must be
    handled explicitly by the caller, not silently replaced.
    """
    yt = Y_true - Y_true.mean(axis=0)
    yp = Y_pred - Y_pred.mean(axis=0)
    st = np.sqrt((yt ** 2).sum(axis=0))
    sp = np.sqrt((yp ** 2).sum(axis=0))
    denom = st * sp
    num = (yt * yp).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 1e-12, num / denom, np.nan)
    return r


@dataclass(frozen=True)
class Selection:
    lam: float
    rank: int
    val_score: float


def select_hyperparameters(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_va: np.ndarray,
    Y_va: np.ndarray,
    rank_tie_break: RankTieBreak,
    ridge_tie_break: RidgeTieBreak,
    r_max: int | None = None,
) -> Selection:
    """Select (lam, rank) on validation only. Tie-breaks are explicit, never default.

    Validation score is the mean over finite per-voxel Pearson r. Rank selection
    uses the ``smallest_rank_at_threshold`` rule (smallest rank reaching
    ``RANK_SELECTION_FRACTION_OF_PEAK`` of the peak validation score) or
    ``argmax_validation``, per the caller's resolved policy.

    Args:
        X_tr, Y_tr: training source/target. X_va, Y_va: validation source/target.
        rank_tie_break, ridge_tie_break: explicit policies (registry-resolved).
        r_max: cap on rank; defaults to the max supportable by the matrices.

    Returns:
        The selected :class:`Selection`.
    """
    n_tr, p = X_tr.shape
    q = Y_tr.shape[1]
    hard_max = min(p, q, n_tr)
    r_top = hard_max if r_max is None else min(r_max, hard_max)
    grid = ridge_grid()

    # score[lam_i, rank_j]
    scores = np.full((grid.size, r_top), -np.inf)
    for i, lam in enumerate(grid):
        W = _ridge_solve(X_tr, Y_tr, lam)
        fitted = X_tr @ W
        _, _, Vt = np.linalg.svd(fitted, full_matrices=False)
        for j, rank in enumerate(range(1, r_top + 1)):
            r = min(rank, Vt.shape[0])
            P = Vt[:r].T @ Vt[:r]
            pred = X_va @ (W @ P)
            rr = per_voxel_pearson(Y_va, pred)
            scores[i, j] = np.nanmean(rr) if np.isfinite(rr).any() else -np.inf

    peak = scores.max()
    # ridge: pick the lam whose best-rank score is optimal, with explicit tie-break
    lam_best_per = scores.max(axis=1)
    if ridge_tie_break == "argmax_validation":
        i_sel = int(np.argmax(lam_best_per))
    elif ridge_tie_break == "one_se_rule":
        se = np.nanstd(scores[np.isfinite(scores)]) / max(np.sqrt(scores.size), 1)
        ok = np.where(lam_best_per >= peak - se)[0]
        i_sel = int(ok.max()) if ok.size else int(np.argmax(lam_best_per))  # strongest reg within 1SE
    else:
        raise ValueError(f"unresolved ridge_tie_break: {ridge_tie_break!r}")

    row = scores[i_sel]
    if rank_tie_break == "argmax_validation":
        j_sel = int(np.argmax(row))
    elif rank_tie_break == "smallest_rank_at_threshold":
        thr = RANK_SELECTION_FRACTION_OF_PEAK * peak if peak > 0 else peak
        ok = np.where(row >= thr)[0]
        j_sel = int(ok.min()) if ok.size else int(np.argmax(row))
    else:
        raise ValueError(f"unresolved rank_tie_break: {rank_tie_break!r}")

    return Selection(lam=float(grid[i_sel]), rank=int(j_sel + 1), val_score=float(row[j_sel]))
