"""Deterministic Shared Response Model (Chen et al. 2015) + new-subject vision-only mapping.

X_s (p_s x n) ~ W_s S, with W_s^T W_s = I (K x K), S the shared response (K x n). Samples
(columns) are the synchronized stimulus identities. Deterministic (SVD init, no RNG). The
common space is fit on VISION ONLY; the same W_s maps both vision and imagery into shared space.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _init_W(X: np.ndarray, K: int) -> np.ndarray:
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    if U.shape[1] >= K:
        return U[:, :K]
    return np.pad(U, ((0, 0), (0, K - U.shape[1])))          # pad if voxels < K (shouldn't happen given caps)


def det_srm(Xs_list: List[np.ndarray], K: int, n_iter: int = 30) -> Tuple[List[np.ndarray], np.ndarray]:
    """Fit deterministic SRM on synchronized-column data. Returns (W_list, S)."""
    Ws = [_init_W(X, K) for X in Xs_list]
    S = np.mean([W.T @ X for W, X in zip(Ws, Xs_list)], axis=0)
    for _ in range(n_iter):
        for i, X in enumerate(Xs_list):
            U, _, Vt = np.linalg.svd(X @ S.T, full_matrices=False)
            Ws[i] = U @ Vt
        S = np.mean([W.T @ X for W, X in zip(Ws, Xs_list)], axis=0)
    return Ws, S


def new_subject_W(X_target: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Orthogonal map for a held-out subject from its VISION on the calibration identities only.

    X_target: (p_target x n_calib); S: (K x n_calib). Returns W_target (p_target x K).
    """
    U, _, Vt = np.linalg.svd(X_target @ S.T, full_matrices=False)
    return U @ Vt


def project(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Native -> shared: (K x n). W:(p x K), X:(p x n)."""
    return W.T @ X


def reconstruct(W: np.ndarray, Sshared: np.ndarray) -> np.ndarray:
    """Shared -> native: (p x n). W:(p x K), Sshared:(K x n)."""
    return W @ Sshared
