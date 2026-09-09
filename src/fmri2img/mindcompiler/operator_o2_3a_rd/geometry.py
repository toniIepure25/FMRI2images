"""O2.3A-RD / O2.4R core deterministic geometry (frozen per o2_3a_rd config c1b2ddb0 / o2_4r ad959446).

Fully-specified, committed, testable machinery that resolves the PROVREC 'missing implementation detail'
gaps: a NEW deterministic SRM (SVD init, max_iter=200, tol=1e-7, float64, canonical gauge), deterministic
orthogonal-Procrustes calibration + orientation application, and the exact null RNG
(SHA256 -> first-16-hex -> uint64 -> numpy PCG64). No RNG anywhere except the frozen null construction.
Everything here is deterministic and gauge-certified; the data-loading / cohort-fit driver lives separately.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np

SRM_MAX_ITER = 200
SRM_TOL = 1e-7


# ----------------------------------------------------------------- canonical gauge
def canonical_gauge(Ws: List[np.ndarray], S: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Deterministic gauge convention so re-fits persist identically.

    Order components by descending shared-response energy (row L2 of S; stable tie by index), then fix each
    component's sign so its largest-absolute loading in S is positive. Applied consistently to every W_s and S.
    """
    S = np.asarray(S, dtype=np.float64)
    energy = np.sqrt((S * S).sum(axis=1))
    order = np.lexsort((np.arange(S.shape[0]), -energy))          # -energy primary, index tiebreak (stable)
    S = S[order]
    Ws = [W[:, order] for W in Ws]
    for k in range(S.shape[0]):
        j = int(np.argmax(np.abs(S[k])))                          # largest-abs element of this component
        if S[k, j] < 0:
            S[k] *= -1.0
            for W in Ws:
                W[:, k] *= -1.0
    return Ws, S


# ----------------------------------------------------------------- deterministic SRM
def _init_W(X: np.ndarray, K: int) -> np.ndarray:
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    if U.shape[1] >= K:
        return U[:, :K]
    return np.pad(U, ((0, 0), (0, K - U.shape[1])))


def det_srm_rd(Xs_list: Sequence[np.ndarray], K: int,
               max_iter: int = SRM_MAX_ITER, tol: float = SRM_TOL) -> Tuple[List[np.ndarray], np.ndarray]:
    """Deterministic SRM: X_s (p_s x n) ~ W_s S, W_s^T W_s = I. SVD init, alternating orthogonal-Procrustes
    updates, convergence when max ||S_new - S_old|| <= tol; float64; canonical gauge applied at the end."""
    Xs = [np.asarray(X, dtype=np.float64) for X in Xs_list]
    Ws = [_init_W(X, K) for X in Xs]
    S = np.mean([W.T @ X for W, X in zip(Ws, Xs)], axis=0)
    for _ in range(int(max_iter)):
        for i, X in enumerate(Xs):
            U, _, Vt = np.linalg.svd(X @ S.T, full_matrices=False)
            Ws[i] = U @ Vt
        S_new = np.mean([W.T @ X for W, X in zip(Ws, Xs)], axis=0)
        if float(np.max(np.abs(S_new - S))) <= tol:
            S = S_new
            break
        S = S_new
    return canonical_gauge(Ws, S)


def new_subject_W(X_target: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Vision-only orthogonal alignment of a held-out subject into the trained common space S (K x n_calib)."""
    U, _, Vt = np.linalg.svd(np.asarray(X_target, dtype=np.float64) @ np.asarray(S, dtype=np.float64).T,
                             full_matrices=False)
    return U @ Vt


# ----------------------------------------------------------------- Procrustes + orientation
def orthogonal_procrustes(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Q* = argmin_Q ||X Q - Z||_F, Q^T Q = I. Deterministic SVD; no scaling/translation."""
    U, _, Vt = np.linalg.svd(np.asarray(X, dtype=np.float64).T @ np.asarray(Z, dtype=np.float64),
                             full_matrices=False)
    return U @ Vt


def _orth(B: np.ndarray) -> np.ndarray:
    U, s, _ = np.linalg.svd(np.asarray(B, dtype=np.float64), full_matrices=False)
    return U[:, s > 1e-9]


def native_projector(Q: np.ndarray, U_res: np.ndarray, W_target: np.ndarray) -> np.ndarray:
    """U_target = Q^T U_res -> B_target = W_target U_target -> P_CAL = orth(B) orth(B)^T (R^{p x p})."""
    Qn = _orth(np.asarray(W_target, dtype=np.float64) @ (np.asarray(Q, dtype=np.float64).T @ np.asarray(U_res, dtype=np.float64)))
    return Qn @ Qn.T


def retention(P: np.ndarray, delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    d2 = float(delta @ delta)
    if d2 <= 0:
        return float("nan")
    return float((P @ delta) @ (P @ delta) / d2)


# ----------------------------------------------------------------- frozen null RNG
def seed_uint64(text: str) -> int:
    """SHA256(UTF-8) -> first 16 hex digits -> unsigned 64-bit int (frozen rule)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def null_subspace(seed_text: str, K: int, r: int) -> np.ndarray:
    """Random r-dim subspace basis (K x r): PCG64(seed_uint64) -> G~N(0,1)^{Kxr} -> reduced QR -> canonical signs."""
    rng = np.random.Generator(np.random.PCG64(seed_uint64(seed_text)))
    G = rng.standard_normal((int(K), int(r)))
    Q, _ = np.linalg.qr(G)                                        # reduced QR: (K x r)
    for k in range(Q.shape[1]):                                  # canonical column sign (largest-abs positive)
        j = int(np.argmax(np.abs(Q[:, k])))
        if Q[j, k] < 0:
            Q[:, k] *= -1.0
    return Q
