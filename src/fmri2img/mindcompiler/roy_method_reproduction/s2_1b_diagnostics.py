"""S2.1B exploratory diagnostics (subj01 × V1). Descriptive only -- NO fold p-values.

Pure numpy/scipy on response matrices (trial × voxel). Every function returns compact
summary statistics (floats/lists) so artifacts carry no raw beta arrays. Metric
definitions are frozen in ``s2_1b_frozen_config.json``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.linalg import orthogonal_procrustes, subspace_angles
from scipy.stats import spearmanr

_EPS = 1e-12


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > _EPS else float("nan")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > _EPS else float("nan")


# --- covariance / spectrum (item 8) ------------------------------------------

def covariance_spectrum(X: np.ndarray) -> dict:
    """X: (n_trials, n_vox). Train-safe summary of the voxel covariance spectrum."""
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    ev = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)
    tot = float(ev.sum())
    p = ev / tot if tot > _EPS else np.zeros_like(ev)
    ent = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return dict(
        voxel_var_mean=float(X.var(axis=0).mean()), trial_var_mean=float(X.var(axis=1).mean()),
        cov_trace=float(np.trace(cov)), frobenius=float(np.linalg.norm(cov)),
        eigen_spectrum=[float(x) for x in ev],
        cumulative_explained=[float(x) for x in np.cumsum(p)],
        participation_ratio=float((ev.sum() ** 2) / (np.sum(ev ** 2) + _EPS)),
        effective_rank_entropy=float(np.exp(ent)),
        condition_number=float(ev[0] / ev[-1]) if ev[-1] > _EPS else float("inf"))


# --- repeat reliability (item 10) --------------------------------------------

def repeat_reliability(M: np.ndarray, rows_by_ident: Dict[str, Sequence[int]]) -> dict:
    """Mean pairwise repeat correlation per identity, aggregated."""
    per = {}
    for ident, rows in rows_by_ident.items():
        rows = list(map(int, rows))
        cs = [_corr(M[a], M[b]) for i, a in enumerate(rows) for b in rows[i + 1:]]
        cs = [c for c in cs if np.isfinite(c)]
        per[ident] = float(np.mean(cs)) if cs else float("nan")
    vals = [v for v in per.values() if np.isfinite(v)]
    return dict(mean=float(np.mean(vals)) if vals else float("nan"),
                median=float(np.median(vals)) if vals else float("nan"),
                min=float(np.min(vals)) if vals else float("nan"),
                max=float(np.max(vals)) if vals else float("nan"), per_identity=per)


# --- identity signal vs repeat noise (item 9) --------------------------------

def identity_signal_noise(M: np.ndarray, rows_by_ident: Dict[str, Sequence[int]]) -> dict:
    cents = {ident: M[list(map(int, rows))].mean(axis=0) for ident, rows in rows_by_ident.items()}
    grand = np.mean(list(cents.values()), axis=0)
    within = float(np.mean([np.mean(np.sum((M[list(map(int, rows))] - cents[ident]) ** 2, axis=1))
                            for ident, rows in rows_by_ident.items()]))
    between = float(np.mean([np.sum((cents[ident] - grand) ** 2) for ident in cents]))
    return dict(within_identity_scatter=within, between_identity_separation=between,
                signal_to_noise=float(between / within) if within > _EPS else float("inf"))


# --- paired B0<->B1 trial similarity (item 7) --------------------------------

def paired_trial_similarity(M0: np.ndarray, M1: np.ndarray, rows: Sequence[int]) -> dict:
    rows = list(map(int, rows))
    pear = [_corr(M0[r], M1[r]) for r in rows]
    cos = [_cosine(M0[r], M1[r]) for r in rows]
    nr = [float(np.linalg.norm(M1[r]) / (np.linalg.norm(M0[r]) + _EPS)) for r in rows]
    vr = [float(M1[r].var() / (M0[r].var() + _EPS)) for r in rows]

    def s(x):
        x = [v for v in x if np.isfinite(v)]
        return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                    p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))
    return dict(pearson=s(pear), cosine=s(cos), norm_ratio=s(nr), variance_ratio=s(vr))


# --- centroids / RDM / cross-state (items 16, 18) ----------------------------

def centroids(M: np.ndarray, rows_by_ident: Dict[str, Sequence[int]]) -> (List[str], np.ndarray):
    idents = sorted(rows_by_ident)
    C = np.vstack([M[list(map(int, rows_by_ident[i]))].mean(axis=0) for i in idents])
    return idents, C


def rdm(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    return np.array([[1.0 - _corr(C[i], C[j]) for j in range(n)] for i in range(n)])


def rdm_similarity(C0: np.ndarray, C1: np.ndarray) -> dict:
    r0, r1 = rdm(C0), rdm(C1)
    iu = np.triu_indices(C0.shape[0], k=1)
    rho, _ = spearmanr(r0[iu], r1[iu])
    return dict(rdm_spearman=float(rho))


def cross_state_matching(C_vis: np.ndarray, C_img: np.ndarray) -> dict:
    """similarity(vision centroid i, imagery centroid j); diagonal vs off-diagonal."""
    n = C_vis.shape[0]
    S = np.array([[_corr(C_vis[i], C_img[j]) for j in range(n)] for i in range(n)])
    diag = float(np.nanmean(np.diag(S)))
    off = float(np.nanmean(S[~np.eye(n, dtype=bool)]))
    ranks = []
    for i in range(n):
        order = np.argsort(-S[i])       # imagery identities ranked by similarity to vision i
        ranks.append(int(np.where(order == i)[0][0]) + 1)  # 1 = correct on top
    return dict(diagonal_sim=diag, offdiag_sim=off, separation=diag - off,
                mean_correct_identity_rank=float(np.mean(ranks)),
                top1_accuracy=float(np.mean([r == 1 for r in ranks])))


# --- subspace alignment (item 17) --------------------------------------------

def principal_angles(A: np.ndarray, B: np.ndarray, dim: int) -> dict:
    """Principal angles between the top-`dim` PCA subspaces of A and B (rows=trials)."""
    def basis(X):
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Vt[:dim].T
    ang = subspace_angles(basis(A), basis(B))
    return dict(dim=dim, principal_angles_deg=[float(np.degrees(a)) for a in ang],
                mean_angle_deg=float(np.degrees(np.mean(ang))))


# --- B0->B1 shrinkage vs rotation (item 19) ----------------------------------

def shrinkage_vs_rotation(M0: np.ndarray, M1: np.ndarray, rows: Sequence[int]) -> dict:
    """Train-only characterization of B0->B1: scale-only vs diagonal affine vs rotation."""
    rows = list(map(int, rows))
    X0, X1 = M0[rows], M1[rows]
    # global scale-only: X1 ~ a*X0 + b
    a = float(np.sum((X0 - X0.mean()) * (X1 - X1.mean())) / (np.sum((X0 - X0.mean()) ** 2) + _EPS))
    b = float(X1.mean() - a * X0.mean())
    r2_scale = _r2(X1, a * X0 + b)
    # diagonal voxelwise affine: X1_v ~ a_v*X0_v + b_v
    av = (((X0 - X0.mean(0)) * (X1 - X1.mean(0))).sum(0) /
          (((X0 - X0.mean(0)) ** 2).sum(0) + _EPS))
    bv = X1.mean(0) - av * X0.mean(0)
    r2_diag = _r2(X1, X0 * av + bv)
    # SCALED orthogonal Procrustes on column-centered data (rotation + optimal
    # scale) so "rotation gain" measures geometry change BEYOND simple rescaling.
    Xc0 = X0 - X0.mean(0); Xc1 = X1 - X1.mean(0)
    R, _ = orthogonal_procrustes(Xc0, Xc1)
    P = Xc0 @ R
    s = float(np.sum(P * Xc1) / (np.sum(P * P) + _EPS))
    r2_proc = _r2(Xc1, s * P)
    return dict(scale_only=dict(a=a, b=b, r2=r2_scale),
                diagonal_affine_r2=r2_diag,
                scaled_procrustes_r2=r2_proc,
                rotation_gain_over_scale=float(r2_proc - r2_scale))


def _r2(Y, P) -> float:
    ss_res = float(np.sum((Y - P) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    return float(1.0 - ss_res / (ss_tot + _EPS))


# --- raw -> denoised retention (item 13) -------------------------------------

def raw_to_denoised(M_raw: np.ndarray, denoised: Dict[int, np.ndarray], rows: Sequence[int]) -> dict:
    rows = [int(r) for r in rows if int(r) in denoised]
    pear = [_corr(M_raw[r], denoised[r]) for r in rows]
    cos = [_cosine(M_raw[r], denoised[r]) for r in rows]
    nr = [float(np.linalg.norm(denoised[r]) / (np.linalg.norm(M_raw[r]) + _EPS)) for r in rows]
    var_ret = [float(denoised[r].var() / (M_raw[r].var() + _EPS)) for r in rows]
    fin = lambda x: [v for v in x if np.isfinite(v)]  # noqa: E731
    return dict(pearson_mean=float(np.mean(fin(pear))), cosine_mean=float(np.mean(fin(cos))),
                norm_ratio_mean=float(np.mean(fin(nr))), variance_retained_mean=float(np.mean(fin(var_ret))))
