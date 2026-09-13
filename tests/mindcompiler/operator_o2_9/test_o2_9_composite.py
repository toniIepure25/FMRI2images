"""O2.9 Minimal Composite Target-State Calibration certification (data-free/synthetic). Verifies Gram==direct
equivalence, the within (q=min) / outside-D2 estimators, rank guards (no subset dropping), success/common/
minimal/Pareto frontier logic, and synthetic resource controls. No Procrustes, no model search."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
from fmri2img.mindcompiler.operator_o2_9 import composite_calibration as CC

rng = np.random.default_rng(9)


# --- Gram == direct equivalence (control H) ----------------------------------
def test_gram_equals_direct_within_and_outside():
    V, K, M = 60, 8, 4
    W = np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]
    Zc = rng.standard_normal((M, K)); delta = rng.standard_normal(V); g = W.T @ delta
    # within, q=3
    q = 3
    Zg = Zc @ Zc.T; zcg = Zc @ g
    e_gram, _ = CC.topk_energy_gram(Zg, zcg, q, max(K, M))
    U_in = CC.within_basis_direct(Zc, q); B_in = W @ U_in
    e_direct = float((B_in.T @ delta) @ (B_in.T @ delta))
    assert abs(e_gram - e_direct) < 1e-10
    # outside D2
    dout = np.stack([(lambda x: x - W @ (W.T @ x))(rng.standard_normal(V)) for _ in range(M)]).T
    Og = dout.T @ dout; oh = dout.T @ delta
    eo_gram, _ = CC.topk_energy_gram(Og, oh, 2, V)
    B_out = CC.outside_basis_direct(dout, 2)
    eo_direct = float((B_out.T @ delta) @ (B_out.T @ delta))
    assert abs(eo_gram - eo_direct) < 1e-10


def test_rank_guard_returns_none():
    # rank-1 outside matrix cannot yield D2
    V = 40; W = np.linalg.qr(rng.standard_normal((V, 8)))[0][:, :8]
    x = rng.standard_normal(V); x = x - W @ (W.T @ x)
    dout = np.stack([x, 2 * x]).T                                            # rank 1
    Og = dout.T @ dout; oh = dout.T @ rng.standard_normal(V)
    e, rank = CC.topk_energy_gram(Og, oh, 2, V)
    assert e is None and rank < 2


def test_num_rank_rule():
    sv = np.array([1.0, 1e-3, 1e-20])
    assert CC._num_rank(sv, 10) == 2                                          # third below eps*10*1


# --- synthetic subject_roi (small, fast) -------------------------------------
def _synth(V=60, K=8, r_best=3, out_dirs=2, snr=20.0, seedn=0):
    rr = np.random.default_rng(300 + seedn)
    W = np.linalg.qr(rr.standard_normal((V, K)))[0][:, :K]
    # planted shared outside basis (out_dirs directions) + within target coords
    Bt = CC.outside_basis_direct(np.stack([(lambda x: x - W @ (W.T @ x))(rr.standard_normal(V)) for _ in range(out_dirs)]).T, out_dirs)
    Ein = rr.standard_normal((K, r_best))                                    # within target coords generator
    cells, tn, en = [], [], []
    for f in range(CC.N_FOLDS):
        U_res = CC._canon(np.linalg.qr(rr.standard_normal((K, r_best)))[0][:, :r_best])
        dt = []
        for _ in range(2):
            wc = Ein @ rr.standard_normal(r_best); oc = Bt @ rr.standard_normal(out_dirs)
            dt.append(W @ wc + oc)                                           # test = within + outside (recoverable)
        P0 = G.native_projector(np.eye(K), U_res, W)
        cells.append({"W_target": W, "U_res": U_res, "deltas_test": dt, "K": K, "r_best": r_best,
                      "R_CAL0": float(np.mean([G.retention(P0, d) for d in dt])),
                      "simple_ids": np.array([0, 1, 2, 3, 4]), "nat_ids": np.array([5, 6, 7, 8, 9]),
                      "train_ids": np.array([f"id{i}" for i in range(10)], dtype=object),
                      "test_ids": np.array(["t0", "t1"], dtype=object)})
        Dtr = np.zeros((10, 8, V)); dn = np.zeros((10, V))
        for i in range(10):
            wc = Ein @ rr.standard_normal(r_best); oc = Bt @ rr.standard_normal(out_dirs)
            dn[i] = W @ wc + oc
            for t in range(8):
                Dtr[i, t] = dn[i] + (1.0 / snr) * rr.standard_normal(V)
        tn.append({"trial_native": Dtr}); en.append({"delta_native": dn})
    rd = {"per_target": {"ventral": {"subjX": [{"R_ORACLE": 1.0}] * 6}}}
    return cells, tn, en, rd


def test_subject_roi_replay_and_q(monkeypatch):
    monkeypatch.setattr(CC, "M_GRID", [2, 10]); monkeypatch.setattr(CC, "T_GRID", [1, 8])
    c, t, e, rd = _synth(seedn=1)
    o = CC.subject_roi("subjX", "ventral", c, t, e, rd, {"max": 0}, {}, want_sealed_replay=True)
    assert o["cells"][(2, 1)]["q"] == 2 and o["cells"][(10, 1)]["q"] == 3   # q=min(r_best=3,M)
    assert o["cells"][(2, 1)]["evaluable"] and o["cells"][(10, 8)]["evaluable"]
    assert 0.0 <= o["cells"][(10, 8)]["TOTAL_RECOVERY"] <= 1.0


# --- frontier logic (finalize) -----------------------------------------------
def _mk_group(suff_cells):
    """Build per_subj so that given (M,T) cells are 'sufficient' in both ROIs."""
    per = {}
    for s in CC.ALL:
        d = {}
        for M in CC.M_GRID:
            for T in CC.T_GRID:
                good = (M, T) in suff_cells
                d[(M, T)] = {"evaluable": True, "R_COMP": (0.8 if good else 0.2), "N_TRIALS": M * T, "q": min(M, 3),
                             "TOTAL_RECOVERY": (0.8 if good else 0.2)}
        per[s] = {"ventral": {"cells": {k: dict(v) for k, v in d.items()}, "R0": 0.1},
                  "lateral": {"cells": {k: dict(v) for k, v in d.items()}, "R0": 0.1}}
    return per


def test_common_minimal_pareto():
    per = _mk_group({(4, 2), (8, 1), (10, 8)})                               # 8, 8, 80 trials
    R111 = {(s, roi): 1.0 for s in CC.ALL for roi in ("ventral", "lateral")}
    group, roi_status, common, n_star, minimal, pareto, prog = CC.finalize(per, R111, ["ventral", "lateral"])
    assert set(common) == {(4, 2), (8, 1), (10, 8)}
    assert n_star == 8 and set(minimal) == {(4, 2), (8, 1)}                   # both 8-trial ties reported
    assert set(pareto) == {(4, 2), (8, 1)}                                   # (10,8) dominated
    assert prog == "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED"
    assert all(roi_status[r] == "COMPOSITE_OPERATIONAL_CALIBRATION_ESTABLISHED" for r in ("ventral", "lateral"))


def test_not_established_at_full_resource():
    per = _mk_group(set())                                                   # nothing sufficient
    R111 = {(s, roi): 1.0 for s in CC.ALL for roi in ("ventral", "lateral")}
    group, roi_status, common, n_star, minimal, pareto, prog = CC.finalize(per, R111, ["ventral", "lateral"])
    assert common == [] and prog == "COMPOSITE_OPERATIONAL_RECOVERY_NOT_ESTABLISHED"
    assert all(roi_status[r] == "COMPOSITE_OPERATIONAL_RECOVERY_NOT_ESTABLISHED_AT_FULL_RESOURCE" for r in ("ventral", "lateral"))


def test_multiregime():
    per = _mk_group({(2, 1)})
    for s in CC.ALL:                                                         # make lateral (2,1) NOT sufficient, but (4,4) sufficient
        per[s]["lateral"]["cells"][(2, 1)]["TOTAL_RECOVERY"] = 0.2; per[s]["lateral"]["cells"][(2, 1)]["R_COMP"] = 0.2
        per[s]["lateral"]["cells"][(4, 4)]["TOTAL_RECOVERY"] = 0.8; per[s]["lateral"]["cells"][(4, 4)]["R_COMP"] = 0.8
    R111 = {(s, roi): 1.0 for s in CC.ALL for roi in ("ventral", "lateral")}
    group, roi_status, common, n_star, minimal, pareto, prog = CC.finalize(per, R111, ["ventral", "lateral"])
    assert common == [] and prog == "COMPOSITE_TARGET_STATE_CALIBRATION_MULTIREGIME"


def test_config_and_no_procrustes():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_9/o2_9_frozen_config.json"))
    assert cfg["frozen_geometry"]["D_outside"] == 2 and cfg["frozen_geometry"]["no_procrustes"] is True
    assert cfg["immutable_history"]["O2_8"] == "RESIDUAL_CAPACITY_MULTICOMPONENT" and cfg["O3"] == "O3_NOT_READY"
    src = Path(CC.__file__).read_text()
    for bad in ("orthogonal_procrustes", "sklearn", "Ridge", "CCA(", "torch"):
        assert bad not in src
