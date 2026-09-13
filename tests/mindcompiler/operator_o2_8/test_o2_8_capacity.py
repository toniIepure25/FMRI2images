"""O2.8 Residual-Capacity Decomposition certification (data-free/synthetic). Verifies exact 3-factor Shapley
(efficiency + additivity), support-oracle/outside-basis properties, the subject-level pipeline efficiency, and
the full classification/program-status logic incl. the synthetic capacity controls (A/B/C/multicomponent/
interaction/beyond-tested). No competing model is fit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
from fmri2img.mindcompiler.operator_o2_8 import capacity_decomposition as CD

rng = np.random.default_rng(8)


# --- Shapley math ------------------------------------------------------------
def test_shapley_efficiency_and_additive():
    # additive R: R_abc = R000 + a*qa + b*qb + c*qc -> phi_i == q_i
    R000, qa, qb, qc = 0.1, 0.3, 0.2, 0.15
    R = {(a, b, c): R000 + a * qa + b * qb + c * qc for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    phi = CD.shapley(R)
    assert abs(phi["A"] - qa) < 1e-12 and abs(phi["B"] - qb) < 1e-12 and abs(phi["C"] - qc) < 1e-12
    assert abs(sum(phi.values()) - (R[(1, 1, 1)] - R[(0, 0, 0)])) < 1e-12


def test_shapley_efficiency_with_interactions():
    R = {(a, b, c): 0.1 + 0.2 * a + 0.1 * b + 0.05 * c + 0.3 * a * b + 0.1 * a * b * c
         for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    phi = CD.shapley(R)
    assert abs(sum(phi.values()) - (R[(1, 1, 1)] - R[(0, 0, 0)])) < 1e-12   # efficiency holds with interactions


def test_support_oracle_and_outside_basis():
    W = np.linalg.qr(rng.standard_normal((60, 8)))[0][:, :8]
    Z = rng.standard_normal((10, 8))
    Us = CD.support_oracle_basis(Z, 3); assert Us.shape == (8, 3) and np.allclose(Us.T @ Us, np.eye(3), atol=1e-9)
    def _oc(): x = rng.standard_normal(60); return x - W @ (W.T @ x)
    dout = np.stack([_oc() for _ in range(5)]).T
    B = CD.outside_basis(dout, 2)
    assert B.shape == (60, 2) and float(abs(W.T @ B).max()) < 1e-10
    assert CD.outside_basis(dout[:, :1], 2) is None                          # rank<D -> None


# --- subject-level pipeline efficiency (real geometry) -----------------------
def _synth(V=50, K=8, r=1, seedn=0):
    rr = np.random.default_rng(200 + seedn)
    W = np.linalg.qr(rr.standard_normal((V, K)))[0][:, :K]
    cells, en = [], []
    for f in range(CD.N_FOLDS):
        U_res = CD._canon(np.linalg.qr(rr.standard_normal((K, r)))[0][:, :r])
        Z = rr.standard_normal((10, K))
        Dn = rr.standard_normal((10, V))
        dt = [rr.standard_normal(V) for _ in range(2)]
        P0 = G.native_projector(np.eye(K), U_res, W)
        cells.append({"W_target": W, "U_res": U_res, "Z": Z, "deltas_test": dt,
                      "R_CAL0": float(np.mean([G.retention(P0, d) for d in dt])), "K": K, "r_best": r,
                      "simple_ids": np.array([0, 1, 2, 3, 4]), "nat_ids": np.array([5, 6, 7, 8, 9]),
                      "train_ids": np.array([f"id{i}" for i in range(10)], dtype=object),
                      "test_ids": np.array(["t0", "t1"], dtype=object)})
        en.append({"delta_native": Dn})
    rd = {"per_target": {"ventral": {"subjX": [{"R_ORACLE": 0.9}] * 6}}}
    return cells, en, rd


def test_subject_roi_shapley_efficiency():
    c, e, rd = _synth(seedn=1)
    cert = {"m0_dev": 0, "sym": 0, "idem": 0, "cross": 0, "rank_ok": True}
    o = CD.subject_roi("subjX", "ventral", c, e, rd, G, cert, {"max": 0})
    assert o["evaluable"] and abs(o["shapley_sum"] - (o["R111"] - o["R000"])) < 1e-12
    assert cert["m0_dev"] <= 1e-10 and cert["sym"] <= 1e-10 and cert["idem"] <= 1e-8 and cert["cross"] <= 1e-9 and cert["rank_ok"]


# --- classification / program status -----------------------------------------
def _mk(phiA, phiB, phiC, fc):
    per = {}
    for i, s in enumerate(CD.ALL):
        d = {"evaluable": True, "phi": {"A": phiA[i], "B": phiB[i], "C": phiC[i]},
             "FULL_CLOSURE": fc[i], "R000": 0.1, "R111": 0.5, "G_REMAIN": 0.6, "R_native": 0.7,
             "shapley_sum": phiA[i] + phiB[i] + phiC[i], "R": {}, "c1_replay": []}
        per[s] = {"ventral": dict(d), "lateral": dict(d)}
    return per


def test_A_only_outside_identity_limit():
    rs, prog = CD.aggregate(_mk([0.2] * 8, [-0.05] * 8, [-0.05] * 8, [0.7] * 8), G)
    assert rs["ventral"]["status"] == "OUTSIDE_IDENTITY_DIVERSITY_LIMIT"
    assert prog == "RESIDUAL_CAPACITY_OUTSIDE_IDENTITY_DOMINANT"


def test_B_only_second_dimension_limit():
    rs, prog = CD.aggregate(_mk([-0.05] * 8, [0.2] * 8, [-0.05] * 8, [0.7] * 8), G)
    assert prog == "RESIDUAL_CAPACITY_SECOND_OUTSIDE_DIMENSION_DOMINANT"


def test_C_only_within_support_limit():
    rs, prog = CD.aggregate(_mk([-0.05] * 8, [-0.05] * 8, [0.2] * 8, [0.7] * 8), G)
    assert prog == "RESIDUAL_CAPACITY_WITHIN_SUPPORT_ORIENTATION_DOMINANT"


def test_multicomponent():
    rs, prog = CD.aggregate(_mk([0.2] * 8, [0.2] * 8, [-0.05] * 8, [0.7] * 8), G)
    assert rs["ventral"]["status"] == "MULTICOMPONENT_RESIDUAL_CAPACITY_LIMIT"
    assert prog == "RESIDUAL_CAPACITY_MULTICOMPONENT"


def test_interaction_dominant():
    # none individually supported (mixed signs -> not Holm) but FC>=0.5
    mixed = [0.3, -0.2, 0.1, -0.1, 0.2, -0.3, 0.1, -0.05]
    rs, prog = CD.aggregate(_mk(mixed, mixed, mixed, [0.7] * 8), G)
    assert rs["ventral"]["status"] == "CAPACITY_INTERACTION_DOMINANT"


def test_beyond_tested_factors():
    rs, prog = CD.aggregate(_mk([0.2] * 8, [0.2] * 8, [0.2] * 8, [0.2] * 8), G)   # FC<0.5
    assert rs["ventral"]["status"] == "RESIDUAL_CAPACITY_BEYOND_TESTED_FACTORS"
    assert prog == "RESIDUAL_CAPACITY_BEYOND_TESTED_FACTORS"


def test_inconclusive_low_evaluable():
    per = _mk([0.2] * 8, [-0.05] * 8, [-0.05] * 8, [0.7] * 8)
    for s in CD.ALL[:2]:
        per[s]["ventral"]["evaluable"] = False
    rs, prog = CD.aggregate(per, G)
    assert rs["ventral"]["status"] == "RESIDUAL_CAPACITY_DECOMPOSITION_INCONCLUSIVE"


def test_config_and_no_model_search():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_8/o2_8_frozen_config.json"))
    assert cfg["immutable_history"]["O2_7A"] == "REDUCED_TRIAL_OUTSIDE_AXIS_CALIBRATION_NOT_ESTABLISHED"
    assert cfg["inference"]["primary_family"].startswith("3 factors") and cfg["O3"] == "O3_NOT_READY"
    src = Path(CD.__file__).read_text()
    for bad in ("orthogonal_procrustes", "sklearn", "Ridge", "CCA(", "torch"):
        assert bad not in src
