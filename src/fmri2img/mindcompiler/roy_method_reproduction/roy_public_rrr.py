"""S2.5M public-method-aligned reduced-rank ridge with PER-TARGET lambda (Lambda).

Corrects the S2.5R-identified material gap: the public Methods fit one independent
ridge regression per target voxel with a voxel-specific lambda_i, whereas the earlier
S2 code used a single scalar lambda for the whole target matrix. This module is NEW;
the historical S2.0/S2.4R scalar-lambda path is left untouched and replayable.

Pipeline (on already train-only-scaled arrays):
  1. per-target lambda selection on VALIDATION (per-voxel Pearson r);
  2. assemble W_Lambda (column i uses lambda_i);
  3. reduced rank via SVD of the fitted training prediction: W_RRR = W_Lambda V_r V_r^T;
  4. operational rank r selected on VALIDATION (argmax mean voxelwise r; smallest on tie);
  5. full rank-performance curves recorded (validation + test) for later S2.6R.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction.contracts import (
    RIDGE_GRID_HI, RIDGE_GRID_LO, RIDGE_GRID_N,
)

_EPS = 1e-12
LAMBDA_TIE_RULE = "largest_lambda"          # among exact validation-score ties (frozen)
RANK_TIE_RULE = "smallest_rank"             # among exact validation-score ties (frozen)


def ridge_grid() -> np.ndarray:
    return np.logspace(np.log10(RIDGE_GRID_LO), np.log10(RIDGE_GRID_HI), RIDGE_GRID_N)


def _ridge_W(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ Y)


def _per_voxel_pearson(Yt: np.ndarray, Yp: np.ndarray) -> np.ndarray:
    yt = Yt - Yt.mean(axis=0); yp = Yp - Yp.mean(axis=0)
    st = np.sqrt((yt ** 2).sum(axis=0)); sp = np.sqrt((yp ** 2).sum(axis=0))
    denom = st * sp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > _EPS, (yt * yp).sum(axis=0) / denom, np.nan)


def select_per_target_lambda(Xtr, Ytr, Xva, Yva, grid: Optional[np.ndarray] = None) -> np.ndarray:
    """Choose lambda_i per target voxel on validation. Tie -> largest lambda (frozen)."""
    grid = ridge_grid() if grid is None else grid
    n = Ytr.shape[1]
    scores = np.full((grid.size, n), -np.inf)
    for j, lam in enumerate(grid):
        W = _ridge_W(Xtr, Ytr, lam)
        r = _per_voxel_pearson(Yva, Xva @ W)
        scores[j] = np.where(np.isfinite(r), r, -np.inf)
    # per target: best score; among exact ties pick the LARGEST lambda (highest index)
    best = scores.max(axis=0)
    lam_idx = np.where(scores == best, np.arange(grid.size)[:, None], -1).max(axis=0)
    return grid[lam_idx]


def fit_W_lambda(Xtr, Ytr, lambdas: np.ndarray) -> np.ndarray:
    """Assemble W_Lambda: column i comes from the ridge fit at lambda_i."""
    p, n = Xtr.shape[1], Ytr.shape[1]
    W = np.empty((p, n))
    for lam in np.unique(lambdas):
        cols = np.where(lambdas == lam)[0]
        Wl = _ridge_W(Xtr, Ytr, float(lam))
        W[:, cols] = Wl[:, cols]
    return W


def _rrr(W_lambda: np.ndarray, Xtr: np.ndarray, rank: int) -> np.ndarray:
    fitted = Xtr @ W_lambda
    _, _, Vt = np.linalg.svd(fitted, full_matrices=False)
    r = min(rank, Vt.shape[0])
    P = Vt[:r].T @ Vt[:r]
    return W_lambda @ P


@dataclass
class RRRFit:
    lambdas: np.ndarray
    W_lambda: np.ndarray
    r_model: int
    val_score: float
    rank_max: int


def select_operational_rank(Xtr, Ytr, Xva, Yva, W_lambda, rank_max: int):
    """argmax validation mean voxelwise r over ranks 1..rank_max; smallest rank on tie."""
    best_r, best_score = 1, -np.inf
    for r in range(1, rank_max + 1):
        W = _rrr(W_lambda, Xtr, r)
        rr = _per_voxel_pearson(Yva, Xva @ W)
        score = float(np.nanmean(rr)) if np.isfinite(rr).any() else -np.inf
        if score > best_score + 1e-12:       # strict improvement -> keep smallest rank on tie
            best_score, best_r = score, r
    return best_r, best_score


def rank_curve(Xtr, Ytr, Xva, Yva, Xte, Yte, W_lambda, rank_max: int):
    """Validation + test mean voxelwise r for r = 1..rank_max (for S2.6R; NOT used to tune)."""
    out = []
    for r in range(1, rank_max + 1):
        W = _rrr(W_lambda, Xtr, r)
        v = _per_voxel_pearson(Yva, Xva @ W)
        t = _per_voxel_pearson(Yte, Xte @ W)
        out.append(dict(rank=r,
                        val=float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"),
                        test=float(np.nanmean(t)) if np.isfinite(t).any() else float("nan")))
    return out


def fit_select(Xtr, Ytr, Xva, Yva, rank_max: int) -> RRRFit:
    lambdas = select_per_target_lambda(Xtr, Ytr, Xva, Yva)
    W_lambda = fit_W_lambda(Xtr, Ytr, lambdas)
    r_model, val = select_operational_rank(Xtr, Ytr, Xva, Yva, W_lambda, rank_max)
    return RRRFit(lambdas=lambdas, W_lambda=W_lambda, r_model=r_model, val_score=val, rank_max=rank_max)


def predict(fit: RRRFit, Xtr: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ _rrr(fit.W_lambda, Xtr, fit.r_model)
