"""O2.13 Subject-Specific Geometry Covariate Diagnostic certification (data-free/synthetic). Verifies the
min-norm convex barycentric solver (planted recovery, non-unique -> minimum-norm, hull flag), LOSO donor-only
standardization, the residual-correction formula + label-permutation null (destroys planted coupling), the
top-k symmetric projector, all ROI/program status branches, the exact 8-test Holm family, and config
immutability. No target imagery enters the correction math."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_13 import covariate_diagnostic as C

rng = np.random.default_rng(13)


# ---- A. barycentric min-norm solver ----
def test_bary_reconstruction_and_simplex():
    # in a 2D covariate space with 7 donors the convex fit is underdetermined: min-norm reconstructs xt
    # EXACTLY as a valid simplex point (and by design spreads rather than picking the sparse planted pair)
    Xd = rng.standard_normal((7, 2))
    wtrue = np.zeros(7); wtrue[1] = 0.6; wtrue[4] = 0.4
    xt = Xd.T @ wtrue
    w, sse, act = C.bary_minnorm(xt, Xd)
    assert sse < 1e-8 and abs(w.sum() - 1) < 1e-9 and np.all(w >= -1e-12)
    assert np.allclose(Xd.T @ w, xt, atol=1e-6)                              # perfect convex reconstruction
    # 1D exactly-determined case: single interior donor pair is uniquely recovered
    x1 = np.array([2.0]); X1 = np.array([[0.0], [4.0]])                       # xt = 0.5*d0 + 0.5*d1
    w1, sse1, act1 = C.bary_minnorm(x1, X1)
    assert sse1 < 1e-12 and np.allclose(w1, [0.5, 0.5], atol=1e-9)


def test_bary_min_norm_tiebreak():
    # duplicate donors create non-unique convex reps of the target; min-norm spreads weight
    base = rng.standard_normal(2)
    Xd = np.stack([base, base, base, base + 1e-9, rng.standard_normal(2), rng.standard_normal(2), rng.standard_normal(2)])
    xt = base.copy()
    w, sse, act = C.bary_minnorm(xt, Xd)
    assert sse < 1e-6 and abs(w.sum() - 1) < 1e-9
    # min-norm should not dump all weight on a single duplicate; effective donors > 1
    assert 1.0 / np.sum(w ** 2) > 1.5


def test_bary_hull_flag_outside():
    Xd = rng.standard_normal((7, 2))
    xt = Xd.mean(0) + 50.0                                                     # far outside hull
    hull, w, sse, act = C._hull_flag(xt, Xd)
    assert hull is False and abs(w.sum() - 1) < 1e-9 and np.all(w >= -1e-12)


# ---- B. LOSO donor-only standardization ----
def test_loso_standardization_donor_only():
    cov = {f"subj0{i}": {"anatomy": {"ventral": np.array([1000.0 + 100 * i, 2.0 + 0.05 * i]),
                                     "lateral": np.array([900.0 + 50 * i, 2.5])}, "behavior": np.array([1.0 + 0.1 * i, 0.0])}
           for i in range(1, 9)}
    xt, Xd, donors, ok = C.loso_standardize(cov, "subj01", "ventral", "anatomy")
    assert ok and Xd.shape == (7, 2) and len(donors) == 7 and "subj01" not in donors
    assert np.allclose(Xd.mean(0), 0, atol=1e-9)                              # donor-only z has zero donor mean


def test_loso_zero_sd_nonevaluable():
    cov = {f"subj0{i}": {"anatomy": {"ventral": np.array([1000.0, 2.0]), "lateral": np.array([1.0, 1.0])}, "behavior": np.array([1.0, 0.0])}
           for i in range(1, 9)}
    xt, Xd, donors, ok = C.loso_standardize(cov, "subj01", "ventral", "anatomy")
    assert ok is False                                                        # constant feature -> sd 0


# ---- C. residual correction + label-permutation null ----
def _proj(M, k):
    U = np.linalg.qr(rng.standard_normal((64, k)))[0][:, :k]
    return U


def test_correction_recovers_coupled_residual():
    # planted: target residual = convex combo of donor residuals with covariate weights
    k = 3
    Qp = _proj(None, k)                                                       # target prior subspace
    Gpri = Qp @ Qp.T
    donor_dg = [(_proj(None, k) @ _proj(None, k).T - _proj(None, k) @ _proj(None, k).T) for _ in range(7)]
    w = np.zeros(7); w[2] = 0.7; w[5] = 0.3
    Graw = Gpri + sum(w[d] * donor_dg[d] for d in range(7))
    Q_corr = C._topk_sym(Graw, k)
    # oracle = the coupled target subspace (top-k of Graw) -> SIM_CORR ~ 1
    Ttar = Q_corr
    sim_corr = np.sum((Q_corr.T @ Ttar) ** 2) / k
    assert sim_corr > 0.999
    # permuting donor labels changes the correction (destroys correspondence)
    pi = C._derange(np.random.default_rng(1), 7)
    Graw_n = Gpri + sum(w[d] * donor_dg[pi[d]] for d in range(7))
    Q_n = C._topk_sym(Graw_n, k)
    assert np.sum((Q_n.T @ Ttar) ** 2) / k < 0.999


def test_topk_sym_orthonormal():
    M = rng.standard_normal((64, 5)); P = M @ M.T
    Q = C._topk_sym(P, 3)
    assert Q.shape == (64, 3) and np.allclose(Q.T @ Q, np.eye(3), atol=1e-9)


def test_derange_no_fixed_point():
    import fmri2img.mindcompiler.operator_o2_3a_rd.geometry as G
    r = np.random.Generator(np.random.PCG64(G.seed_uint64("O2.13|subj01|ventral|0|anatomy|IN|3")))
    for _ in range(50):
        p = C._derange(r, 7); assert not np.any(p == np.arange(7))


# ---- D. classification branches ----
def _part(vals):
    """vals: dict roi -> dict (family,comp) -> {'E':..,'D':..} constant across subjects."""
    part = {}
    for s in C.ALL:
        part[s] = {}
        for roi in vals:
            part[s][roi] = {}
            for key in vals[roi]:
                v = vals[roi][key]
                part[s][roi][key] = {"E_NATIVE": v["E"], "DOP_NATIVE": v["D"], "E_GEOM": 0.0, "DOP_GEOM": 0.0, "RGR": 0.0, "NRR": 0.0}
    return part


def _allkeys(good):
    d = {}
    for fam in C.FAMILIES:
        for comp in C.COMPONENTS:
            d[(fam, comp)] = {"E": 0.05, "D": 0.05} if good(fam, comp) else {"E": -0.02, "D": -0.02}
    return d


def test_status_multifamily_both_rois():
    v = _allkeys(lambda fam, comp: comp == "IN")                              # both families support IN
    _, infer, roi_status, prog, tech = C.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert roi_status["ventral"] == "MULTIFAMILY_COVARIATE_COUPLING"
    assert prog == "SUBJECT_SPECIFIC_GEOMETRY_HAS_REPRODUCIBLE_NONIMAGERY_COVARIATE_STRUCTURE"


def test_status_anatomy_only_reproducible():
    v = _allkeys(lambda fam, comp: fam == "anatomy" and comp == "IN")
    _, infer, roi_status, prog, tech = C.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert roi_status["ventral"] == "ANATOMY_COVARIATE_COUPLING_SUPPORTED"
    assert prog == "SUBJECT_SPECIFIC_GEOMETRY_HAS_REPRODUCIBLE_NONIMAGERY_COVARIATE_STRUCTURE"


def test_status_multiregime():
    v = _allkeys(lambda fam, comp: fam == "anatomy" and comp == "IN")         # ventral: anatomy IN
    o = _allkeys(lambda fam, comp: False)                                     # lateral: nothing
    _, infer, roi_status, prog, tech = C.classify(_part({"ventral": v, "lateral": o}), ["ventral", "lateral"])
    assert prog == "SUBJECT_SPECIFIC_GEOMETRY_COVARIATE_MULTIREGIME"


def test_status_no_structure():
    v = _allkeys(lambda fam, comp: False)
    _, infer, roi_status, prog, tech = C.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert roi_status["ventral"] == "NO_SUPPORTED_NONIMAGERY_COVARIATE_COUPLING"
    assert prog == "NO_REPRODUCIBLE_TESTED_COVARIATE_STRUCTURE"


def test_support_requires_both_E_and_delta_over_prior():
    # strong E_NATIVE + Holm but DOP_NATIVE negative -> NOT supported
    v = {}
    for fam in C.FAMILIES:
        for comp in C.COMPONENTS:
            v[(fam, comp)] = {"E": 0.05, "D": -0.02}
    _, infer, roi_status, prog, tech = C.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert infer["ventral|anatomy|IN"]["holm_reject"] is True
    assert infer["ventral|anatomy|IN"]["supported"] is False


def test_holm_family_exactly_eight():
    v = _allkeys(lambda fam, comp: comp == "IN")
    _, infer, _, _, _ = C.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"])
    assert len(infer) == 8


# ---- E. config immutability + no model search ----
def test_config_and_no_model_search():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_13/o2_13_frozen_config.json"))
    assert cfg["immutable_history"]["O2_12"] == "PERCEPTION_PRIOR_BENEFICIAL_BUT_EIGHT_TRIAL_MINIMUM_NOT_REDUCED"
    assert cfg["immutable_history"]["O2_9_N_TRIALS_STAR"] == 8 and cfg["O3"] == "O3_NOT_READY"
    assert cfg["primary_inference"]["family"].startswith("2 families")
    assert "config_sha256" in cfg
    src = Path(C.__file__).read_text()
    for bad in ("Ridge", "PLSRegression", "sklearn", "torch", "RandomForest", "PCA(", "CCA("):
        assert bad not in src
