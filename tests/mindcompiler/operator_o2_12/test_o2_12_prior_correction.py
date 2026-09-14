"""O2.12 Minimal Imagery Calibration with Perception-Only Prior Control -- certification (data-free/synthetic).
Verifies: hybrid Gram-optimized energy/subspace == dense C_HYB eigen-projection; the one-pseudo-observation
trace identity; control == exact O2.9 topk_energy_gram; perfect-prior helps at low resource; useless-orthogonal
prior yields no false gain; all ROI/program status branches; the exactly-8 primary Holm family; config
immutability + no model search. No held-out data enters any fitted subspace."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_12 import prior_correction as P
from fmri2img.mindcompiler.operator_o2_9 import composite_calibration as O9

rng = np.random.default_rng(12)


# ---- A. hybrid Gram-optimized == dense C_HYB eigen-projection (within & outside) ----
def _dense_hybrid(Zrows, Uprior, e_over, k, g):
    """Dense reference: C = Z^T Z + prior; top-k eigvectors; energy=||U^T g||^2, and return U for sim checks."""
    Kd = Zrows.shape[1]
    C = Zrows.T @ Zrows + e_over * (Uprior @ Uprior.T)
    lam, U = np.linalg.eigh(C)
    idx = np.argsort(lam)[::-1][:k]
    Uk = U[:, idx]
    return float(np.sum((Uk.T @ g) ** 2)), Uk


def test_hybrid_gram_equals_dense_within():
    K, M, r = 9, 4, 3
    Z = rng.standard_normal((M, K)); Uprior = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]
    g = rng.standard_normal(K)
    e_IN = float(np.mean([Z[i] @ Z[i] for i in range(M)]))
    # gram path
    Gin = np.zeros((M + r, M + r)); Gin[:M, :M] = Z @ Z.T; Gin[:M, M:] = Z @ Uprior
    Gin[M:, :M] = (Z @ Uprior).T; Gin[M:, M:] = np.eye(r)
    lam = np.concatenate([np.ones(M), np.full(r, e_IN / r)])
    nu, W, ls = P._hyb_topk(Gin, lam, r)
    e_gram = P._hyb_energy(nu, W, ls, np.concatenate([Z @ g, Uprior.T @ g]), r)
    e_dense, _ = _dense_hybrid(Z, Uprior, e_IN / r, r, g)
    assert abs(e_gram - e_dense) < 1e-9


def test_hybrid_gram_sim_equals_dense():
    K, M, r = 9, 4, 3
    Z = rng.standard_normal((M, K)); Uprior = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]
    Ref = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]
    e_IN = float(np.mean([Z[i] @ Z[i] for i in range(M)]))
    Gin = np.zeros((M + r, M + r)); Gin[:M, :M] = Z @ Z.T; Gin[:M, M:] = Z @ Uprior
    Gin[M:, :M] = (Z @ Uprior).T; Gin[M:, M:] = np.eye(r)
    lam = np.concatenate([np.ones(M), np.full(r, e_IN / r)])
    nu, W, ls = P._hyb_topk(Gin, lam, r)
    Mrhs = np.vstack([Z @ Ref, Uprior.T @ Ref])
    sim_gram = P._hyb_sim(nu, W, ls, Mrhs, r, r)
    _, Uk = _dense_hybrid(Z, Uprior, e_IN / r, r, np.zeros(K))
    sim_dense = float(np.sum((Uk.T @ Ref) ** 2))
    assert abs(sim_gram - sim_dense) < 1e-9


# ---- B. one-pseudo-observation trace identity ----
def test_one_pseudo_observation_trace():
    K, r = 8, 3
    Uprior = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]
    z = rng.standard_normal((5, K)); e_IN = float(np.mean([z[i] @ z[i] for i in range(5)]))
    C_prior = (e_IN / r) * (Uprior @ Uprior.T)
    assert abs(np.trace(C_prior) - e_IN) < 1e-12                              # trace == one avg observation energy


# ---- C. control path is literally O2.9 ----
def test_control_is_o2_9_topk():
    M = 4
    Z = rng.standard_normal((M, 6)); Zg = Z @ Z.T; g = rng.standard_normal(6); h = Z @ g
    e, rank = O9.topk_energy_gram(Zg, h, 2, max(6, M))
    # direct: ||U_2^T g||^2 with U_2 top-2 left sv of Z^T
    U, s, _ = np.linalg.svd(Z.T, full_matrices=False)
    e_direct = float(np.sum((U[:, :2].T @ g) ** 2))
    assert abs(e - e_direct) < 1e-10


# ---- D. perfect within prior recovers oracle direction the data alone (rank-limited) cannot ----
def test_perfect_prior_supplies_missing_rank():
    K, r = 6, 3
    Uor = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]                  # oracle within subspace
    # M=1 calibration observation: data spans only 1 direction; control q=min(r,1)=1
    z1 = 2.0 * Uor[:, 0] + 0.1 * rng.standard_normal(K)
    Z = z1[None, :]; M = 1
    g = Uor @ rng.standard_normal(r)                                          # test lies fully in oracle subspace
    # control: top-1 of Z
    Uc, sc, _ = np.linalg.svd(Z.T, full_matrices=False); Ec = float(np.sum((Uc[:, :1].T @ g) ** 2))
    # hybrid with perfect prior U_PRIOR = Uor supplies full rank r
    e_IN = float(z1 @ z1)
    Gin = np.zeros((M + r, M + r)); Gin[:M, :M] = Z @ Z.T; Gin[:M, M:] = Z @ Uor
    Gin[M:, :M] = (Z @ Uor).T; Gin[M:, M:] = np.eye(r)
    lam = np.concatenate([np.ones(M), np.full(r, e_IN / r)])
    nu, W, ls = P._hyb_topk(Gin, lam, r)
    Eh = P._hyb_energy(nu, W, ls, np.concatenate([Z @ g, Uor.T @ g]), r)
    Eg = float(g @ g)
    assert Eh > Ec + 1e-6 and Eh / Eg > 0.99                                  # hybrid recovers full test energy


# ---- E. useless orthogonal prior: hybrid ~ control as data grows (no false boost) ----
def test_useless_prior_no_false_gain_at_high_resource():
    K, r, M = 8, 3, 8
    Uor = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]
    Z = (rng.standard_normal((M, r)) @ Uor.T) + 0.01 * rng.standard_normal((M, K))  # data spans oracle subspace
    Uprior = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]              # random (useless) prior
    g = Uor @ rng.standard_normal(r)
    Uc, sc, _ = np.linalg.svd(Z.T, full_matrices=False); Ec = float(np.sum((Uc[:, :r].T @ g) ** 2))
    e_IN = float(np.mean([Z[i] @ Z[i] for i in range(M)]))
    Gin = np.zeros((M + r, M + r)); Gin[:M, :M] = Z @ Z.T; Gin[:M, M:] = Z @ Uprior
    Gin[M:, :M] = (Z @ Uprior).T; Gin[M:, M:] = np.eye(r)
    lam = np.concatenate([np.ones(M), np.full(r, e_IN / r)])
    nu, W, ls = P._hyb_topk(Gin, lam, r)
    Eh = P._hyb_energy(nu, W, ls, np.concatenate([Z @ g, Uprior.T @ g]), r)
    assert abs(Eh - Ec) < 0.05 * max(Ec, 1e-9)                                # weak 1-pseudo-obs prior barely perturbs


# ---- F/G. classification branches ----
def _part_builder(vals):
    """vals: dict roi -> dict (M,T) -> dict of per-participant constants."""
    per = {}
    for s in P.ALL:
        per[s] = {}
        for roi in vals:
            per[s][roi] = {"R0": 0.0, "R_native": 1.0, "r_best": 3, "cells": {}}
            for c in P.ALL_CELLS:
                d = {"evaluable": True, "N_TRIALS": c[0] * c[1], "q": min(c[0], 3)}
                v = vals[roi].get(c)
                if v is not None:
                    d["R_CONTROL"] = v["ctrl"]; d["R_HYBRID"] = v["hyb"]
                per[s][roi]["cells"][c] = d
    return per


def _cellvals(ctrl, hyb):
    return {"ctrl": ctrl, "hyb": hyb}


def test_status_reduces_below_eight():
    # every primary cell: hybrid >> control (gain) AND hybrid recovery >=0.5 -> prior-enabled in both ROIs
    v = {c: _cellvals(0.2, 0.7) for c in P.PRIMARY_CELLS}
    v.update({c: _cellvals(0.9, 0.95) for c in P.REFERENCE_CELLS})
    per = _part_builder({"ventral": v, "lateral": v})
    R111 = {(s, roi): 1.0 for s in P.ALL for roi in ("ventral", "lateral")}
    part, infer, roi_status, common_pe, common_hs, n_star, minimal_set, prog, tech = P.classify(per, R111, ["ventral", "lateral"])
    assert prog == "PERCEPTION_PRIOR_REDUCES_COMPOSITE_CALIBRATION_BELOW_EIGHT_TRIALS"
    assert roi_status["ventral"] == "PERCEPTION_PRIOR_REDUCES_IMAGERY_CALIBRATION"
    assert n_star == 2 and {tuple(x) for x in minimal_set} == {(2, 1)}


def test_status_sufficient_without_gain():
    # hybrid reaches recovery >=0.5 but hybrid==control (no gain) -> sufficient-without-gain
    v = {c: _cellvals(0.7, 0.7) for c in P.PRIMARY_CELLS}
    v.update({c: _cellvals(0.9, 0.9) for c in P.REFERENCE_CELLS})
    per = _part_builder({"ventral": v, "lateral": v})
    R111 = {(s, roi): 1.0 for s in P.ALL for roi in ("ventral", "lateral")}
    _, infer, roi_status, common_pe, common_hs, n_star, _, prog, _ = P.classify(per, R111, ["ventral", "lateral"])
    assert prog == "SUBEIGHT_HYBRID_RECOVERY_WITHOUT_PRIOR_UTILITY_ESTABLISHED"
    assert roi_status["ventral"] == "PERCEPTION_PRIOR_OPERATIONALLY_SUFFICIENT_WITHOUT_GAIN_EVIDENCE"


def test_status_beneficial_not_reduced():
    # supported gain but hybrid recovery <0.5 -> improves but not to threshold; no common operational
    v = {c: _cellvals(0.1, 0.3) for c in P.PRIMARY_CELLS}
    v.update({c: _cellvals(0.9, 0.95) for c in P.REFERENCE_CELLS})
    per = _part_builder({"ventral": v, "lateral": v})
    R111 = {(s, roi): 1.0 for s in P.ALL for roi in ("ventral", "lateral")}
    _, infer, roi_status, common_pe, common_hs, n_star, _, prog, _ = P.classify(per, R111, ["ventral", "lateral"])
    assert prog == "PERCEPTION_PRIOR_BENEFICIAL_BUT_EIGHT_TRIAL_MINIMUM_NOT_REDUCED"
    assert roi_status["ventral"] == "PERCEPTION_PRIOR_IMPROVES_BUT_NOT_TO_OPERATIONAL_THRESHOLD"


def test_status_no_benefit():
    v = {c: _cellvals(0.3, 0.3) for c in P.PRIMARY_CELLS}                      # no gain, no sufficiency
    v.update({c: _cellvals(0.9, 0.9) for c in P.REFERENCE_CELLS})
    per = _part_builder({"ventral": v, "lateral": v})
    R111 = {(s, roi): 1.0 for s in P.ALL for roi in ("ventral", "lateral")}
    _, _, roi_status, _, _, _, _, prog, _ = P.classify(per, R111, ["ventral", "lateral"])
    assert prog == "PERCEPTION_PRIOR_DOES_NOT_REDUCE_TARGET_IMAGERY_BURDEN"
    assert roi_status["ventral"] == "PERCEPTION_PRIOR_NO_SUPPORTED_BENEFIT"


def test_technical_inconclusive_when_cell_missing():
    v = {c: _cellvals(0.2, 0.7) for c in P.PRIMARY_CELLS}
    v.update({c: _cellvals(0.9, 0.95) for c in P.REFERENCE_CELLS})
    per = _part_builder({"ventral": v, "lateral": v})
    per["subj03"]["ventral"]["cells"][(2, 1)] = {"evaluable": False, "N_TRIALS": 2, "q": 2}  # drop one
    R111 = {(s, roi): 1.0 for s in P.ALL for roi in ("ventral", "lateral")}
    _, _, roi_status, _, _, _, _, prog, tech = P.classify(per, R111, ["ventral", "lateral"])
    assert tech is True and prog == "O2_12_PRIOR_CORRECTION_INCONCLUSIVE"
    assert roi_status["ventral"] == "PERCEPTION_PRIOR_CALIBRATION_INCONCLUSIVE"


# ---- H. primary Holm family is exactly 8 ----
def test_holm_family_exactly_eight():
    assert len(P.PRIMARY_CELLS) == 4
    v = {c: _cellvals(0.2, 0.7) for c in P.PRIMARY_CELLS}
    v.update({c: _cellvals(0.9, 0.95) for c in P.REFERENCE_CELLS})
    per = _part_builder({"ventral": v, "lateral": v})
    R111 = {(s, roi): 1.0 for s in P.ALL for roi in ("ventral", "lateral")}
    _, infer, _, _, _, _, _, _, _ = P.classify(per, R111, ["ventral", "lateral"])
    tests = [k for k in infer if "|" in k and "__cells__" not in k]
    assert len(tests) == 8


# ---- I. config immutability + no model search ----
def test_config_and_no_model_search():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_12/o2_12_frozen_config.json"))
    assert cfg["immutable_history"]["O2_11"] == "SUBJECT_SPECIFIC_GEOMETRY_NOT_PREDICTABLE_FROM_TESTED_PERCEPTION_PHENOTYPE"
    assert cfg["immutable_history"]["O2_9_N_TRIALS_STAR"] == 8 and cfg["O3"] == "O3_NOT_READY"
    assert cfg["primary_inference"]["family"].startswith("4 sub-eight")
    assert "config_sha256" in cfg
    src = Path(P.__file__).read_text()
    for bad in ("Ridge", "lambda_grid", "Procrustes", "CCA(", "PLSRegression", "torch", "sklearn", "RandomForest"):
        assert bad not in src
