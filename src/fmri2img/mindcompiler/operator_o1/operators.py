"""O1 operator ladder O0-O4 + common-vision scaling + reduced-rank ridge + hyperparameter CV.

All operators map vision identity-centroids -> imagery identity-centroids in a COMMON
vision-derived coordinate system (train-only mu_vis/sigma_vis applied to BOTH states, so
state-dependent gain differences are preserved). Fitting sample size = number of training
identities (never inflated by repeats). No nonlinear models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

MODELS = ("O0", "O1", "O2", "O3", "O4")
MODEL_NAMES = {"O0": "IMAGERY_MEAN", "O1": "GLOBAL_GAIN", "O2": "DIAGONAL_AFFINE",
               "O3": "LOW_RANK_RESIDUAL", "O4": "FULL_RRR_OPERATOR"}
RIDGE_LO, RIDGE_HI, RIDGE_N = 1e-4, 1e4, 50
RANK_HARD_MAX = 8
_ZV_EPS = 1e-12
RIDGE_TIE = "larger_ridge"      # frozen hyperparameter tie rule
RANK_TIE = "smaller_rank"


def ridge_grid() -> np.ndarray:
    return np.logspace(np.log10(RIDGE_LO), np.log10(RIDGE_HI), RIDGE_N)


def rank_candidates(n_train_ident: int, eff_source_rank: int, tgt_dim: int) -> List[int]:
    hi = max(1, min(RANK_HARD_MAX, n_train_ident - 1, eff_source_rank, tgt_dim))
    return list(range(1, hi + 1))


# ------------------------------------------------------------------ common vision scaler
@dataclass(frozen=True)
class CommonVisionScaler:
    mu: np.ndarray
    sigma: np.ndarray
    support: np.ndarray            # boolean mask of nonzero-variance vision voxels
    n_zero_var: int

    @classmethod
    def fit(cls, vis_train: np.ndarray) -> "CommonVisionScaler":
        mu = vis_train.mean(0)
        sigma = vis_train.std(0)                       # ddof=0
        support = sigma >= _ZV_EPS
        scale = np.where(support, sigma, 1.0)
        return cls(mu=mu, sigma=scale, support=support, n_zero_var=int((~support).sum()))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale by train vision stats and restrict to supported (nonzero-var) voxels."""
        return ((X - self.mu) / self.sigma)[:, self.support]


# ------------------------------------------------------------------ reduced-rank ridge
def rrr_fit(X: np.ndarray, T: np.ndarray, lam: float, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reduced-rank ridge with intercept: T ~ b + X @ W_r. Returns (W_r, b)."""
    xbar, tbar = X.mean(0), T.mean(0)
    Xc, Tc = X - xbar, T - tbar
    p = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T @ Tc)
    fitted = Xc @ W
    _, _, Vt = np.linalg.svd(fitted, full_matrices=False)
    r = max(1, min(rank, Vt.shape[0]))
    Wr = W @ (Vt[:r].T @ Vt[:r])
    return Wr, tbar - xbar @ Wr


# ------------------------------------------------------------------ operators
@dataclass
class OperatorFit:
    model: str
    params: dict


def fit_operator(model: str, X: np.ndarray, Y: np.ndarray, lam: float = 0.0, rank: int = 1) -> OperatorFit:
    if model == "O0":
        return OperatorFit(model, {"b": Y.mean(0)})
    if model == "O1":
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        denom = float((Xc * Xc).sum())
        a = float((Xc * Yc).sum() / denom) if denom > _ZV_EPS else 0.0
        return OperatorFit(model, {"a": a, "b": Y.mean(0) - a * X.mean(0)})
    if model == "O2":
        xbar, ybar = X.mean(0), Y.mean(0)
        Xc, Yc = X - xbar, Y - ybar
        num = (Xc * Yc).sum(0); den = (Xc * Xc).sum(0) + lam
        d = np.where(den > _ZV_EPS, num / den, 0.0)
        return OperatorFit(model, {"d": d, "b": ybar - d * xbar})
    if model == "O3":
        Wr, b = rrr_fit(X, Y - X, lam, rank)
        return OperatorFit(model, {"delta": Wr, "b": b})
    if model == "O4":
        Wr, b = rrr_fit(X, Y, lam, rank)
        return OperatorFit(model, {"W": Wr, "b": b})
    raise ValueError(f"unknown model {model}")


def predict_operator(fit: OperatorFit, X: np.ndarray) -> np.ndarray:
    p = fit.params
    if fit.model == "O0":
        return np.tile(p["b"], (X.shape[0], 1))
    if fit.model == "O1":
        return p["b"][None, :] + p["a"] * X
    if fit.model == "O2":
        return p["b"][None, :] + p["d"][None, :] * X
    if fit.model == "O3":
        return X + p["b"][None, :] + X @ p["delta"]
    if fit.model == "O4":
        return p["b"][None, :] + X @ p["W"]
    raise ValueError(fit.model)


# ------------------------------------------------------------------ metric
def pattern_r(pred: np.ndarray, meas: np.ndarray) -> float:
    """Pearson correlation across voxels between a predicted and a measured centroid."""
    a = pred - pred.mean(); b = meas - meas.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / denom) if denom > _ZV_EPS else float("nan")


# ------------------------------------------------------------------ hyperparameter CV
def _eff_rank(X: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(X - X.mean(0)))


def _prep_inner(cents_vis, cents_img, inner, scaler_of):
    """Precompute scaled (Xtr, Ytr, Xte, Yte) per inner fold once (reused across all hyperparameters)."""
    out = []
    for f in inner:
        sc = scaler_of(f.train)
        Xtr = sc.transform(np.array([cents_vis[i] for i in f.train]))
        Ytr = sc.transform(np.array([cents_img[i] for i in f.train]))
        Xte = sc.transform(np.array([cents_vis[i] for i in f.test]))
        Yte = sc.transform(np.array([cents_img[i] for i in f.test]))
        out.append((Xtr, Ytr, Xte, Yte))
    return out


def _mean_pattern_r(P, Y):
    return float(np.nanmean([pattern_r(P[i], Y[i]) for i in range(Y.shape[0])]))


def select_hyperparams(model: str, cents_vis: dict, cents_img: dict, inner, scaler_of) -> Tuple[float, int, float]:
    """Inner identity-held-out CV. Returns (lam, rank, val_mean_pattern_r).

    Tie -> larger ridge, then smaller rank. Uses an eigenbasis ridge + one SVD per (fold, lam)
    truncated across ranks, so the full 50xrank grid stays cheap.
    """
    folds = _prep_inner(cents_vis, cents_img, inner, scaler_of)
    if model == "O0":
        return 0.0, 1, float(np.mean([_mean_pattern_r(np.tile(Ytr.mean(0), (Yte.shape[0], 1)), Yte)
                                      for Xtr, Ytr, Xte, Yte in folds]))
    if model == "O1":
        s = []
        for Xtr, Ytr, Xte, Yte in folds:
            f = fit_operator("O1", Xtr, Ytr); s.append(_mean_pattern_r(predict_operator(f, Xte), Yte))
        return 0.0, 1, float(np.mean(s))

    grid = ridge_grid()
    if model == "O2":
        best = (-np.inf, 0.0)
        for lam in grid:
            s = []
            for Xtr, Ytr, Xte, Yte in folds:
                f = fit_operator("O2", Xtr, Ytr, lam); s.append(_mean_pattern_r(predict_operator(f, Xte), Yte))
            sc = float(np.mean(s))
            if sc > best[0] + 1e-12 or (abs(sc - best[0]) <= 1e-12 and lam > best[1]):
                best = (sc, lam)
        return float(best[1]), 1, float(best[0])

    # O3 / O4: precompute eig(Xc^T Xc) and Q^T Xc^T Tc per fold; loop lam cheaply; one SVD per (fold,lam)
    n_tr = len(inner[0].train)
    tgt_dim = folds[0][1].shape[1]
    prep = []
    for Xtr, Ytr, Xte, Yte in folds:
        T = Ytr if model == "O4" else (Ytr - Xtr)
        xbar, tbar = Xtr.mean(0), T.mean(0)
        Xc, Tc = Xtr - xbar, T - tbar
        w, Q = np.linalg.eigh(Xc.T @ Xc)
        A = Q.T @ (Xc.T @ Tc)
        prep.append((Xc, xbar, tbar, w, Q, A, Xte, Yte))
    ranks = rank_candidates(n_tr, min(n_tr - 1, np.linalg.matrix_rank(prep[0][0])), tgt_dim)
    best = (-np.inf, (0.0, 1))
    for lam in grid:
        per_rank = {r: [] for r in ranks}
        for Xc, xbar, tbar, w, Q, A, Xte, Yte in prep:
            W_full = Q @ (A / (w + lam)[:, None])
            _, _, Vt = np.linalg.svd(Xc @ W_full, full_matrices=False)
            for r in ranks:
                rr = min(r, Vt.shape[0])
                Wr = W_full @ (Vt[:rr].T @ Vt[:rr])
                b = tbar - xbar @ Wr
                pred = (b + Xte @ Wr) if model == "O4" else (Xte + b + Xte @ Wr)
                per_rank[r].append(_mean_pattern_r(pred, Yte))
        for r in ranks:
            sc = float(np.mean(per_rank[r]))
            better = sc > best[0] + 1e-12
            tie_lam = abs(sc - best[0]) <= 1e-12 and lam > best[1][0] + 1e-30
            if better or tie_lam:
                best = (sc, (lam, r))
    return float(best[1][0]), int(best[1][1]), float(best[0])
