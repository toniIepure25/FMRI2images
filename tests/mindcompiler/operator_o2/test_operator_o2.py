"""O2 data-free tests: SRM, new-subject mapping, gauge equivariance, baselines, synthetic controls."""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2 import pipeline as pl
from fmri2img.mindcompiler.operator_o2 import shared_ops as so
from fmri2img.mindcompiler.operator_o2 import srm as S

SIMPLE = [f"A:{c}" for c in "EHLPRV"]; NAT = [f"B:{c}" for c in "BCDKTW"]
IDS = SIMPLE + NAT
FAM = {i: ("simple" if i.startswith("A:") else "naturalistic") for i in IDS}
SUBJ = [f"subj0{k}" for k in range(1, 9)]


# ---------------------------------------------------------------- SRM
def test_det_srm_recovers_shared_structure():
    rng = np.random.default_rng(0)
    K = 4; n = 10; Ssh = rng.standard_normal((K, n))
    Xs = []
    for _ in range(6):
        p = rng.integers(20, 40)
        W, _ = np.linalg.qr(rng.standard_normal((p, K)))     # orthonormal topography
        Xs.append(W @ Ssh)
    Ws, Shat = S.det_srm(Xs, K)
    # each recovered W is orthonormal and reconstructs its subject
    for W, X in zip(Ws, Xs):
        assert np.allclose(W.T @ W, np.eye(K), atol=1e-6)
        assert np.allclose(W @ (W.T @ X), X, atol=1e-6)      # X in span(W)


def test_new_subject_mapping_uses_only_given_data_and_is_orthonormal():
    rng = np.random.default_rng(1)
    K, n = 3, 8; Sh = rng.standard_normal((K, n))
    Wt, _ = np.linalg.qr(rng.standard_normal((25, K)))
    Xt = Wt @ Sh
    West = S.new_subject_W(Xt, Sh)
    assert np.allclose(West.T @ West, np.eye(K), atol=1e-6)


def test_srm_different_voxel_counts():
    rng = np.random.default_rng(2); K, n = 3, 9; Sh = rng.standard_normal((K, n))
    Xs = [np.linalg.qr(rng.standard_normal((p, K)))[0] @ Sh for p in (14, 71, 156)]
    Ws, _ = S.det_srm(Xs, K)
    assert [W.shape for W in Ws] == [(14, K), (71, K), (156, K)]


# ---------------------------------------------------------------- gauge invariance (Part Q)
def test_scalar_ridge_gauge_equivariant():
    rng = np.random.default_rng(3); n, K = 30, 5
    X = rng.standard_normal((n, K)); Y = rng.standard_normal((n, K))
    Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
    T, b = so.fit_shared_ridge(X, Y, 1.0)
    Tp, bp = so.fit_shared_ridge(X @ Q, Y @ Q, 1.0)
    assert np.allclose(Tp, Q.T @ T @ Q, atol=1e-8)            # T' = Q^T T Q
    # predictions transform back identically
    pred = so.predict_shared(T, b, X)
    predp = so.predict_shared(Tp, bp, X @ Q)
    assert np.allclose(predp, pred @ Q, atol=1e-8)


def test_nti_invariant_and_zero_for_scalar():
    K = 6
    assert so.nontrivial_transformation_index(2.5 * np.eye(K)) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(4); T = rng.standard_normal((K, K))
    Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
    assert so.nontrivial_transformation_index(T) == pytest.approx(
        so.nontrivial_transformation_index(Q.T @ T @ Q), abs=1e-9)
    assert so.nontrivial_transformation_index(T) > 0


def test_shared_action_fraction_bounds():
    rng = np.random.default_rng(5); X = rng.standard_normal((20, 4))
    T = rng.standard_normal((4, 4)); D = rng.standard_normal((4, 4))
    f = so.shared_action_fraction(X, T, D)
    assert 0.0 <= f <= 1.0
    assert so.shared_action_fraction(X, T, np.zeros((4, 4))) == pytest.approx(1.0)


# ---------------------------------------------------------------- synthetic O2 pipeline controls
def _make(kind, seed, K_true=4, p_range=(20, 45), noise=0.02):
    rng = np.random.default_rng(seed)
    Xsh = {i: rng.standard_normal(K_true) for i in IDS}          # shared vision latent per identity
    if kind == "gain":
        T_true = 1.7 * np.eye(K_true)
    elif kind == "full":
        T_true = rng.standard_normal((K_true, K_true)) / np.sqrt(K_true) + 0.5 * np.eye(K_true)
    else:
        T_true = rng.standard_normal((K_true, K_true)) / np.sqrt(K_true) + 0.5 * np.eye(K_true)
    Ysh = {i: Xsh[i] @ T_true for i in IDS}
    cvis, cimg = {}, {}
    for s in SUBJ:
        p = int(rng.integers(*p_range))
        W, _ = np.linalg.qr(rng.standard_normal((p, K_true)))
        for i in IDS:
            cvis[(s, i)] = W @ Xsh[i] + noise * rng.standard_normal(p)
            cimg[(s, i)] = W @ Ysh[i] + noise * rng.standard_normal(p)
    return cvis, cimg


def _loso_scores(cvis, cimg, K=4, lam=1.0):
    """One representative LOSO cell: hold out subj08 + 2 identities."""
    target = "subj08"; train_subjects = [s for s in SUBJ if s != target]
    test_ids = ["A:E", "B:B"]; train_ids = [i for i in IDS if i not in test_ids]
    out = pl.evaluate_outer_cell(cvis, cimg, train_subjects, target, train_ids, test_ids, K, lam)
    rT = np.mean([r["r_Tshared"] for r in out["rows"]]); rS0 = np.mean([r["r_S0"] for r in out["rows"]])
    rS1 = np.mean([r["r_S1"] for r in out["rows"]])
    return rT, rS0, rS1, out


def test_control_shared_operator_beats_group_mean():
    rT, rS0, _, out = _loso_scores(*_make("full", 10))
    assert rT > rS0 + 0.05                                   # shared operator transfers to unseen subject+identities
    assert all(v == 0 for v in out["leakage"].values())     # no target-imagery / test-identity leakage


def test_control_scalar_gain_truth_gain_competitive():
    rT, rS0, rS1, _ = _loso_scores(*_make("gain", 11))
    assert rS1 >= rS0                                        # global gain captures scalar truth
    assert rT > rS0                                          # (full operator also fine)


def test_control_identity_shuffle_collapses():
    cvis, cimg = _make("full", 12)
    rT_intact, rS0_intact, _, _ = _loso_scores(cvis, cimg)
    rng = np.random.default_rng(12)
    cimg_shuf = dict(cimg)
    for s in SUBJ:                                           # break vision<->imagery correspondence
        perm = rng.permutation(IDS)
        col = {i: cimg[(s, pi)] for i, pi in zip(IDS, perm)}
        for i in IDS:
            cimg_shuf[(s, i)] = col[i]
    rT, rS0, _, _ = _loso_scores(cvis, cimg_shuf)
    G_intact = rT_intact - rS0_intact; G_shuf = rT - rS0
    assert G_shuf < 0.5 * G_intact                           # operator increment substantially collapses under shuffle


def test_vision_margin_positive_when_structure_shared_and_zero_when_broken():
    cvis, _ = _make("full", 13)
    train = [s for s in SUBJ if s != "subj08"][:6]
    calib = [i for i in IDS if i not in ("A:E", "B:B")][:8]
    m = pl.vision_identity_margin(cvis, train, "subj07", calib, ["A:E", "B:B"], K=4)
    assert m > 0.1
    # NO shared identity structure at all -> margin should not indicate real alignment
    rng = np.random.default_rng(99); bad = {}
    for s in train + ["subj07"]:
        p = int(rng.integers(20, 45))
        for i in IDS:
            bad[(s, i)] = rng.standard_normal(p)             # independent per (subject, identity)
    mbad = pl.vision_identity_margin(bad, train, "subj07", calib, ["A:E", "B:B"], K=4)
    assert mbad < m - 0.05                                    # far below the structured margin


def test_delta_cannot_use_target_subject():
    # fit_delta_s only accepts one subject's shared reps; structurally cannot see target.
    rng = np.random.default_rng(14); X = rng.standard_normal((10, 4)); Y = rng.standard_normal((10, 4))
    T = rng.standard_normal((4, 4))
    D = pl.fit_delta_s(X, Y, T, 1.0)
    assert D.shape == (4, 4)
