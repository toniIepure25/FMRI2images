"""O1 data-free tests: folds/leakage, operators, metrics, inference, geometry, negative controls."""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o1 import evaluate as ev
from fmri2img.mindcompiler.operator_o1 import folds as fo
from fmri2img.mindcompiler.operator_o1 import geometry as ge
from fmri2img.mindcompiler.operator_o1 import operators as op

IDS = [f"A:{c}" for c in "EHLPRV"] + [f"B:{c}" for c in "BCDKTW"]
FAM = {i: ("simple" if i.startswith("A:") else "naturalistic") for i in IDS}


# --------------------------------------------------------------- folds / leakage
def test_outer_every_identity_tested_once_and_mixed():
    outer = fo.outer_folds(IDS, FAM)
    assert len(outer) == 6
    tested = [i for f in outer for i in f.test]
    assert sorted(tested) == sorted(IDS)
    for f in outer:
        fams = {FAM[i] for i in f.test}
        assert fams == {"simple", "naturalistic"} and len(f.train) == 10


def test_no_test_identity_in_training():
    for f in fo.outer_folds(IDS, FAM):
        assert not (set(f.test) & set(f.train))


def test_inner_isolation_and_coverage():
    outer = fo.outer_folds(IDS, FAM)
    inner_by = [fo.inner_folds(f.train, FAM) for f in outer]
    cert = fo.certify_no_leakage(outer, inner_by)
    assert all(cert[k] for k in ("every_identity_tested_once", "each_outer_fold_1simple_1nat",
                                 "inner_disjoint_from_outer_test", "inner_covers_outer_train_once"))
    for inners in inner_by:
        assert len(inners) == 5
        for inf in inners:
            assert len(inf.test) == 2 and len(inf.train) == 8


# --------------------------------------------------------------- scaler
def test_common_vision_scaler_same_stats_both_states_and_zero_var():
    rng = np.random.default_rng(0)
    vis = rng.standard_normal((10, 6)); vis[:, 3] = 5.0            # zero-variance vision dim
    sc = op.CommonVisionScaler.fit(vis)
    assert sc.n_zero_var == 1 and sc.support.sum() == 5
    img = rng.standard_normal((10, 6))
    zv = sc.transform(vis); zi = sc.transform(img)
    assert zv.shape[1] == 5 and zi.shape[1] == 5                   # same support applied to both
    assert np.allclose(zv.mean(0), 0, atol=1e-9)                  # vision centered by its own stats


# --------------------------------------------------------------- operators
def _fit_pred(model, X, Y, lam=0.0, rank=2):
    return op.predict_operator(op.fit_operator(model, X, Y, lam, rank), X)


def test_O0_ignores_vision():
    rng = np.random.default_rng(1); X = rng.standard_normal((8, 5)); Y = rng.standard_normal((8, 5))
    p = _fit_pred("O0", X, Y)
    assert np.allclose(p, Y.mean(0)[None, :])


def test_O1_recovers_global_gain():
    rng = np.random.default_rng(2); X = rng.standard_normal((10, 6)); Y = 2.3 * X
    fit = op.fit_operator("O1", X, Y)
    assert fit.params["a"] == pytest.approx(2.3, abs=1e-6)


def test_O2_recovers_diagonal():
    rng = np.random.default_rng(3); X = rng.standard_normal((12, 5)); d = np.array([1., -2., 0.5, 3., -1.])
    Y = X * d
    fit = op.fit_operator("O2", X, Y, lam=0.0)
    assert np.allclose(fit.params["d"], d, atol=1e-6)


def test_O4_rrr_rank_truncation():
    rng = np.random.default_rng(4); X = rng.standard_normal((10, 8))
    W_true = rng.standard_normal((8, 8)); Y = X @ W_true
    Wr, b = op.rrr_fit(X, Y, lam=1e-3, rank=3)
    assert np.linalg.matrix_rank(Wr) <= 3


def test_O3_identity_plus_residual_shape():
    rng = np.random.default_rng(5); X = rng.standard_normal((10, 6)); Y = X + 0.1 * rng.standard_normal((10, 6))
    p = _fit_pred("O3", X, Y, lam=1.0, rank=2)
    assert p.shape == Y.shape


def test_pattern_r_perfect_and_zero():
    v = np.array([1., 2., 3., 4.])
    assert op.pattern_r(v, 2 * v + 5) == pytest.approx(1.0)
    assert abs(op.pattern_r(v, np.array([1., -1., 1., -1.]))) < 1.0


# --------------------------------------------------------------- inference
def test_signflip_exact_all_positive():
    out = ev.signflip_test([0.1, 0.2, 0.05, 0.3, 0.15, 0.25, 0.08, 0.12])
    assert out["n_assignments"] == 256 and out["p_one_sided_greater"] == pytest.approx(1 / 256)


def test_signflip_symmetric_zero_mean():
    out = ev.signflip_test([0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.3, -0.3])
    assert out["observed_mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["p_one_sided_greater"] > 0.4


def test_holm_monotone_and_reject():
    out = ev.holm({"a": 0.001, "b": 0.02, "c": 0.5, "d": 0.6, "e": 0.7, "f": 0.8, "g": 0.9})
    assert out["a"]["reject"] and out["a"]["holm_p"] <= out["b"]["holm_p"] <= out["c"]["holm_p"]


def test_classify_thresholds():
    assert ev.classify_operator({"O0": .1, "O1": .1, "O2": .1, "O3": .1, "O4": .1}) == "NO_REUSABLE_OPERATOR_EVIDENCE"
    assert ev.classify_operator({"O0": .1, "O1": .3, "O2": .3, "O3": .3, "O4": .31}) == "GLOBAL_GAIN_SUFFICIENT"
    assert ev.classify_operator({"O0": .1, "O1": .2, "O2": .35, "O3": .35, "O4": .36}) == "DIAGONAL_REWEIGHTING_SUFFICIENT"
    assert ev.classify_operator({"O0": .1, "O1": .2, "O2": .25, "O3": .40, "O4": .41}) == "LOW_RANK_DEFORMATION_SUFFICIENT"
    assert ev.classify_operator({"O0": .1, "O1": .2, "O2": .25, "O3": .3, "O4": .5}) == "FULL_CROSS_VOXEL_OPERATOR_NEEDED"


# --------------------------------------------------------------- geometry
def test_supported_basis_orthonormal():
    rng = np.random.default_rng(6); V = rng.standard_normal((12, 20))
    B = ge.supported_basis(V)
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-8) and B.shape[1] <= 11


def test_geometry_identity_operator():
    rng = np.random.default_rng(7); V = rng.standard_normal((12, 15))
    B = ge.supported_basis(V)
    g = ge.operator_geometry(np.eye(15), B)         # W = I -> A = I
    assert g["identity_deviation"] == pytest.approx(0.0, abs=1e-9)
    assert g["out_of_visual_span_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_geometry_within_span_operator_has_no_out_of_span():
    rng = np.random.default_rng(8); V = rng.standard_normal((12, 16)); B = ge.supported_basis(V)
    # operator that maps everything inside span(B): A = B M B^T  => W = A^T
    M = rng.standard_normal((B.shape[1], B.shape[1]))
    A = B @ M @ B.T
    g = ge.operator_geometry(A.T, B)
    assert g["out_of_visual_span_fraction"] == pytest.approx(0.0, abs=1e-8)


def test_polar_rotation_orthonormal_and_sign_invariant():
    rng = np.random.default_rng(9); Ab = rng.standard_normal((5, 5))
    pol = ge.polar_decomposition(Ab)
    assert 0.0 <= pol["fraction_contracted"] <= 1.0
    Ab2 = Ab.copy(); Ab2[:, 1] *= -1
    # stretch spectrum (singular values) invariant to a column sign flip
    assert ge.polar_decomposition(Ab2)["stretch_singular_values"]["max"] == pytest.approx(
        pol["stretch_singular_values"]["max"], abs=1e-9)


def test_stability_identical_operators_high():
    rng = np.random.default_rng(10); V = rng.standard_normal((12, 14)); B = ge.supported_basis(V)
    W = rng.standard_normal((14, 14))
    out = ge.operator_stability([W, W.copy(), W.copy()], B)
    assert out["label"] == "OPERATOR_STABILITY_HIGH" and out["mean_pairwise_action_cosine"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------- negative controls (Part AB)
def _synthetic_cells(kind, seed, p=25, noise=0.05):
    rng = np.random.default_rng(seed)
    base = {i: rng.standard_normal(p) for i in IDS}          # vision centroids
    vis, img = {}, {}
    if kind == "D":
        W = rng.standard_normal((p, p)) / np.sqrt(p)
    if kind == "C":
        U = rng.standard_normal((p, 3)); delta = U @ U.T / p
    for i in IDS:
        x = base[i]; vis[i] = x
        if kind == "A":       y = 1.8 * x
        elif kind == "B":     y = x * rng.standard_normal(p) if i == IDS[0] else x * _DIAG
        elif kind == "C":     y = x + x @ delta
        elif kind == "D":     y = x @ W
        elif kind == "E":     y = base[rng.permutation(IDS.__len__())[0]] if False else 2.0 * x
        img[i] = y + noise * rng.standard_normal(p)
    return vis, img


_DIAG = np.random.default_rng(999).standard_normal(25)


def _outer_scores(vis, img):
    outer = fo.outer_folds(IDS, FAM)
    scores = {m: [] for m in op.MODELS}
    for of in outer:
        inner = fo.inner_folds(of.train, FAM)
        scaler_of = lambda ids: op.CommonVisionScaler.fit(np.array([vis[i] for i in ids]))
        for m in op.MODELS:
            lam, rank, _ = op.select_hyperparams(m, vis, img, inner, scaler_of)
            sc = scaler_of(of.train)
            Xtr = sc.transform(np.array([vis[i] for i in of.train]))
            Ytr = sc.transform(np.array([img[i] for i in of.train]))
            fit = op.fit_operator(m, Xtr, Ytr, lam, rank)
            rs = [op.pattern_r(op.predict_operator(fit, sc.transform(vis[t][None, :]))[0],
                               sc.transform(img[t][None, :])[0]) for t in of.test]
            scores[m].append(float(np.mean(rs)))
    return {m: float(np.mean(v)) for m, v in scores.items()}


def test_control_A_global_gain_competitive():
    s = _outer_scores(*_synthetic_cells("A", 11))
    assert s["O1"] >= s["O0"] and s["O4"] > s["O0"]              # gain structure is learnable


def test_control_D_full_beats_diagonal():
    s = _outer_scores(*_synthetic_cells("D", 13))
    assert s["O4"] > s["O2"] + 0.02                              # cross-voxel mixing needed


def test_control_E_shuffle_collapses():
    vis, img = _synthetic_cells("A", 14)
    rng = np.random.default_rng(14)
    perm = rng.permutation(IDS)
    img_shuf = {IDS[k]: img[perm[k]] for k in range(len(IDS))}   # break identity correspondence
    s = _outer_scores(vis, img_shuf)
    assert s["O4"] < 0.3                                         # held-out generalization collapses
