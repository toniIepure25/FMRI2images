"""O2.11 Subject-Specific Target-State Geometry Predictability certification (data-free/synthetic). Verifies the
perception kernels, the 127 active-set convex barycentric simplex solver (planted-recovery + simplex equivalence
+ hull-projection + uniform-dominance), the weighted-subspace predictor, the geometry-permutation null
(derangement), all ROI/program status branches, and config immutability. No target imagery is used in the
prediction (Stage-A) math."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_11 import prediction as P

rng = np.random.default_rng(11)


# ---- A. perception kernel: symmetric PSD, unit Frobenius, centered, basis/space invariant ----
def test_kernel_properties():
    X = rng.standard_normal((512, 40))
    K = P._kernel(X)
    assert K.shape == (512, 512)
    assert np.allclose(K, K.T, atol=1e-10)
    assert abs(np.linalg.norm(K) - 1.0) < 1e-10
    assert np.all(np.linalg.eigvalsh(K) > -1e-9)                      # PSD
    # row-centering: H K H == K (already centered)
    assert np.allclose(P._H @ K @ P._H, K, atol=1e-9)


def test_kernel_invariant_to_voxel_count_scaling():
    # kernel depends on the 512x512 anchor Gram; an orthogonal reparam of voxel space leaves it unchanged
    X = rng.standard_normal((512, 30)); R = np.linalg.qr(rng.standard_normal((30, 30)))[0]
    assert np.allclose(P._kernel(X), P._kernel(X @ R), atol=1e-9)


def test_kernel_degenerate_returns_none():
    assert P._kernel(np.zeros((512, 5))) is None


# ---- B. barycentric solver: planted convex combination recovered exactly ----
def _rand_kernels(n):
    return [P._kernel(rng.standard_normal((512, 20))) for _ in range(n)]


def test_bary_planted_recovery():
    Kd = _rand_kernels(7)
    wtrue = np.zeros(7); wtrue[1] = 0.6; wtrue[4] = 0.4
    Kt = 0.6 * Kd[1] + 0.4 * Kd[4]
    w, eb, eu, act = P.bary_weights(Kt, Kd)
    assert abs(w.sum() - 1.0) < 1e-9 and np.all(w >= -1e-12)
    assert eb < 1e-6                                                  # exact reconstruction
    assert np.allclose(w, wtrue, atol=1e-6)


def test_bary_single_donor_identity():
    Kd = _rand_kernels(7)
    w, eb, eu, act = P.bary_weights(Kd[3], Kd)
    assert eb < 1e-8 and w[3] > 1 - 1e-6 and act == (3,)


def test_bary_simplex_equivalence_bruteforce():
    # small-n exhaustive: enumerator optimum matches a dense simplex grid search lower bound
    Kd = _rand_kernels(4)
    Kt = P._kernel(rng.standard_normal((512, 20)))
    w, eb, eu, act = P.bary_weights(Kt, Kd)
    c = float(np.sum(Kt * Kt))
    b = np.array([float(np.sum(Kt * K)) for K in Kd])
    M = np.array([[float(np.sum(Ki * Kj)) for Kj in Kd] for Ki in Kd])
    best = np.inf
    grid = np.linspace(0, 1, 41)
    for a in grid:
        for bb in grid:
            for cc in grid:
                if a + bb + cc > 1 + 1e-9:
                    continue
                wv = np.array([a, bb, cc, 1 - a - bb - cc])
                if wv[3] < -1e-9:
                    continue
                best = min(best, float(np.sqrt(max(c - 2 * wv @ b + wv @ M @ wv, 0))))
    assert eb <= best + 1e-6                                          # enumerator no worse than grid
    assert abs(w.sum() - 1.0) < 1e-9 and np.all(w >= -1e-12)


def test_bary_hull_projection_when_outside():
    # target far outside donor hull still yields a valid simplex point, err>0, uniform >= bary
    Kd = _rand_kernels(7)
    Kt = P._kernel(rng.standard_normal((512, 20)))
    w, eb, eu, act = P.bary_weights(Kt, Kd)
    assert abs(w.sum() - 1.0) < 1e-9 and np.all(w >= -1e-12)
    assert eu >= eb - 1e-9                                            # barycentric never worse than uniform


# ---- C. weighted subspace predictor ----
def test_weighted_subspace_single_active():
    Qs = [np.linalg.qr(rng.standard_normal((60, 3)))[0][:, :3] for _ in range(5)]
    w = np.zeros(5); w[2] = 1.0
    Q = P._weighted_subspace(Qs, w, 3)
    Pt = Qs[2] @ Qs[2].T; Pp = Q @ Q.T
    assert np.trace(Pt @ Pp) / 3 > 1 - 1e-8


def test_weighted_subspace_identical_donors():
    Q0 = np.linalg.qr(rng.standard_normal((60, 3)))[0][:, :3]
    Qs = [Q0, Q0, Q0]; w = np.array([0.2, 0.3, 0.5])
    Q = P._weighted_subspace(Qs, w, 3)
    assert np.trace((Q0 @ Q0.T) @ (Q @ Q.T)) / 3 > 1 - 1e-8


# ---- D. geometry-permutation null: derangement (no fixed point) ----
def test_derangement_no_fixed_point():
    import fmri2img.mindcompiler.operator_o2_3a_rd.geometry as G
    r = np.random.Generator(np.random.PCG64(G.seed_uint64("O2.11|subj01|ventral|0|IN|7")))
    for _ in range(50):
        p = P._derange(r, 7)
        assert not np.any(p == np.arange(7))


def test_null_permutation_changes_prediction():
    Qs = [np.linalg.qr(rng.standard_normal((60, 3)))[0][:, :3] for _ in range(7)]
    w = rng.random(7); w /= w.sum()
    base = P._weighted_subspace(Qs, w, 3)
    perm = P._derange(np.random.default_rng(0), 7)
    permd = P._weighted_subspace([Qs[j] for j in perm], w, 3)
    assert np.trace((base @ base.T) @ (permd @ permd.T)) / 3 < 0.999  # correspondence destroyed


# ---- E. classification branches ----
def _part(vals):
    return {s: {roi: {k: vals[roi][k] for k in vals[roi]} for roi in vals} for s in P.ALL}


def _cell(sim_in, sim_out, gr_in, gr_out, e_in, e_out, tr_in, tr_out, tot, cf):
    return {"SIM_IN": sim_in, "SIM_OUT": sim_out, "GR_IN": gr_in, "GR_OUT": gr_out, "E_IN": e_in, "E_OUT": e_out,
            "TR_IN": tr_in, "TR_OUT": tr_out, "TOTAL": tot, "CFRAC": cf}


def test_status_predicts_composite():
    v = _cell(0.8, 0.8, 0.7, 0.7, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7)
    rs, prog, infer = P.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert rs["ventral"] == "TARGET_PERCEPTION_PREDICTS_COMPOSITE_GEOMETRY"
    assert prog == "ZERO_TARGET_IMAGERY_COMPOSITE_GEOMETRY_PREDICTED_FROM_TARGET_PERCEPTION"


def test_status_components_but_composite_insufficient():
    v = _cell(0.8, 0.8, 0.7, 0.7, 0.1, 0.1, 0.7, 0.7, 0.2, 0.2)
    rs, prog, infer = P.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert rs["ventral"] == "PERCEPTION_PREDICTS_COMPONENTS_BUT_COMPOSITE_INSUFFICIENT"
    assert prog == "PERCEPTION_PREDICTS_GEOMETRY_BUT_OPERATIONALLY_INSUFFICIENT"


def test_status_not_predictive():
    v = _cell(0.2, 0.2, 0.1, 0.1, -0.02, -0.02, 0.1, 0.1, 0.1, 0.1)
    rs, prog, infer = P.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert rs["ventral"] == "TARGET_PERCEPTION_GEOMETRY_NOT_PREDICTIVE"
    assert prog == "SUBJECT_SPECIFIC_GEOMETRY_NOT_PREDICTABLE_FROM_TESTED_PERCEPTION_PHENOTYPE"


def test_status_within_only_multiregime():
    v = _cell(0.8, 0.8, 0.7, 0.7, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7)                  # ventral: both predictable
    o = _cell(0.8, 0.2, 0.7, 0.1, 0.1, -0.02, 0.7, 0.1, 0.3, 0.3)               # lateral: within only
    rs, prog, infer = P.classify(_part({"ventral": v, "lateral": o}), ["ventral", "lateral"])
    assert rs["lateral"] == "WITHIN_PERCEPTION_PREDICTABLE_OUTSIDE_NOT"
    assert prog == "PERCEPTION_CONDITIONED_GEOMETRY_PREDICTION_MULTIREGIME"


def test_predictable_requires_both_geometry_and_native():
    # strong native effect + Holm reject but geometry recovery < 0.5 -> NOT predictable (both required)
    v = _cell(0.4, 0.4, 0.2, 0.2, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3)
    rs, prog, infer = P.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert infer["ventral|IN"]["holm_reject"] is True
    assert infer["ventral|IN"]["predictable"] is False


# ---- F. config immutability + no model search ----
def test_config_and_no_model_search():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_11/o2_11_frozen_config.json"))
    assert cfg["immutable_history"]["O2_10"] == "COMPOSITE_TARGET_STATE_GEOMETRY_SUBJECT_SPECIFIC_DOMINANT"
    assert cfg["immutable_history"]["O3"] == "O3_NOT_READY" and cfg["O3"] == "O3_NOT_READY"
    assert cfg["inference"]["primary_family"].startswith("2 components")
    assert "config_sha256" in cfg
    src = Path(P.__file__).read_text()
    for bad in ("Ridge", "CCA(", "PLSRegression", "torch", "sklearn", "RandomForest", "MLP"):
        assert bad not in src
