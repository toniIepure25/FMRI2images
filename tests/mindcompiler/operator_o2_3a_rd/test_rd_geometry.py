"""O2.3A-RD/O2.4R generator geometry certification (data-free, synthetic). Certifies the committed
deterministic machinery: SRM rerun-equality + convergence, canonical gauge, gauge invariance of the
native retention metric, frozen null RNG deterministic replay + seeding, orthogonal-Procrustes recovery,
and identity-permutation null behaviour."""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G

rng = np.random.default_rng(7)


def _synth(n=40, K=6):
    S = rng.standard_normal((K, n))
    Ws = [np.linalg.qr(rng.standard_normal((80, K)))[0][:, :K] for _ in range(4)]
    return [W @ S + 0.01 * rng.standard_normal((80, n)) for W in Ws], S


def test_srm_deterministic_rerun_equality():
    Xs, _ = _synth()
    W1, S1 = G.det_srm_rd(Xs, K=6)
    W2, S2 = G.det_srm_rd(Xs, K=6)
    assert np.allclose(S1, S2, atol=0, rtol=0)                    # bitwise-identical rerun
    assert all(np.array_equal(a, b) for a, b in zip(W1, W2))
    assert all(np.allclose(W.T @ W, np.eye(6), atol=1e-9) for W in W1)  # orthonormal columns


def test_canonical_gauge_convention_and_idempotent():
    Xs, _ = _synth()
    Ws, S = G.det_srm_rd(Xs, K=6)
    # each component's largest-abs element of S is positive
    for k in range(S.shape[0]):
        j = int(np.argmax(np.abs(S[k])))
        assert S[k, j] > 0
    # descending shared-response energy
    energy = np.sqrt((S * S).sum(axis=1))
    assert np.all(np.diff(energy) <= 1e-9)
    Ws2, S2 = G.canonical_gauge([W.copy() for W in Ws], S.copy())  # idempotent
    assert np.allclose(S2, S, atol=0) and all(np.array_equal(a, b) for a, b in zip(Ws2, Ws))


def test_native_retention_gauge_invariant():
    K, r, p = 6, 3, 120
    U_res = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]
    W = np.linalg.qr(rng.standard_normal((p, K)))[0][:, :K]
    Qtrue = np.linalg.qr(rng.standard_normal((K, K)))[0]
    a = rng.standard_normal(r)
    delta = W @ (Qtrue.T @ (U_res @ a))
    Q = Qtrue                                                     # perfect calibration
    P1 = G.native_projector(Q, U_res, W); r1 = G.retention(P1, delta)
    Gg = np.linalg.qr(rng.standard_normal((K, K)))[0]             # arbitrary common-space gauge
    P2 = G.native_projector(Gg @ Q @ Gg.T, Gg @ U_res, W @ Gg.T)
    r2 = G.retention(P2, delta)
    assert abs(r1 - r2) < 1e-10 and abs(r1 - 1.0) < 1e-9          # invariant + recovers (delta in subspace)


def test_null_rng_deterministic_and_seed_specific():
    q1 = G.null_subspace("O2.3A-RD|subj01|ventral|0|7", K=8, r=3)
    q2 = G.null_subspace("O2.3A-RD|subj01|ventral|0|7", K=8, r=3)
    q3 = G.null_subspace("O2.3A-RD|subj01|ventral|0|8", K=8, r=3)
    assert np.array_equal(q1, q2)                                 # same seed -> identical
    assert not np.allclose(q1, q3)                                # different iteration -> different
    assert np.allclose(q1.T @ q1, np.eye(3), atol=1e-9)           # orthonormal
    assert G.seed_uint64("O2.3A-RD|subj01|ventral|0|7") == G.seed_uint64("O2.3A-RD|subj01|ventral|0|7")


def test_orthogonal_procrustes_recovers_rotation():
    K = 6
    Qtrue = np.linalg.qr(rng.standard_normal((K, K)))[0]
    X = rng.standard_normal((30, K)); Z = X @ Qtrue
    Qhat = G.orthogonal_procrustes(X, Z)
    assert np.allclose(Qhat, Qtrue, atol=1e-8) and np.allclose(Qhat.T @ Qhat, np.eye(K), atol=1e-9)


def test_identity_permutation_null_changes_fit():
    K = 6
    Qtrue = np.linalg.qr(rng.standard_normal((K, K)))[0]
    X = rng.standard_normal((10, K)); Z = X @ Qtrue
    Qtrue_fit = G.orthogonal_procrustes(X, Z)
    perm = rng.permutation(10)
    Qshuf = G.orthogonal_procrustes(X, Z[perm])
    assert not np.allclose(Qtrue_fit, Qshuf, atol=1e-3)           # destroying correspondence changes Q
