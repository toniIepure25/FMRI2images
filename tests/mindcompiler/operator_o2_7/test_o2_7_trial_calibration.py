"""O2.7 Minimal-Trial Target-State Calibration certification (data-free/synthetic; integrity-corrected).
Verifies the reduced-repeat estimator, matched null, T=8 replay certification (numerical + structural),
trial-state semantic verification, exact cell-count evaluability, T_STAR + multiregime status logic, the
6-test Holm family excluding T=8, and the synthetic controls. No competing model is fit."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
from fmri2img.mindcompiler.operator_o2_7 import trial_calibration as TC

rng = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def _fast_null(monkeypatch):
    monkeypatch.setattr(TC, "N_NULL", 6)


def _W(V=40, K=8):
    return np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]


# --- helpers: axis / null / pairs / subset counts ----------------------------
def test_pairs_and_subset_counts():
    assert len(TC.balanced_pairs([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])) == 25
    assert [len(list(itertools.combinations(range(8), T))) for T in (1, 2, 4, 8)] == [8, 28, 70, 1]
    assert TC.EXPECTED_SUBSETS == {1: 8, 2: 28, 4: 70, 8: 1}


def test_outside_axis_and_null():
    W = _W(); V = W.shape[0]
    b = TC._outside_axis(rng.standard_normal(V), rng.standard_normal(V), W)
    assert abs(np.linalg.norm(b) - 1) < 1e-9 and float(abs(W.T @ b).max()) < 1e-10
    a1 = TC._null_axis("O2.7|s|ventral|0|3|2|5|7", W, V, G)
    a2 = TC._null_axis("O2.7|s|ventral|0|3|2|5|7", W, V, G)
    a3 = TC._null_axis("O2.7|s|ventral|0|3|2|5|8", W, V, G)
    assert np.array_equal(a1, a2) and not np.allclose(a1, a3) and float(abs(W.T @ a1).max()) < 1e-9


# --- synthetic cohort --------------------------------------------------------
def _synth(V=50, K=8, snr=1.0, identity_specific=False, no_outside=False, seedn=0, n_simple=2, n_nat=2):
    r = np.random.default_rng(100 + seedn)
    W = np.linalg.qr(r.standard_normal((V, K)))[0][:, :K]
    bt = r.standard_normal(V); bt = bt - W @ (W.T @ bt); bt /= np.linalg.norm(bt)
    cells, tn = [], []
    for f in range(TC.N_FOLDS):
        U_res = TC._canon1(np.linalg.qr(r.standard_normal((K, 1)))[0][:, :1].ravel()).reshape(K, 1)
        X = r.standard_normal((10, K)); Z = r.standard_normal((10, K))
        Dtrial = np.zeros((10, 8, V)); coeff = r.standard_normal(10)
        for i in range(10):
            insup = W @ r.standard_normal(K)
            if no_outside:
                oi = np.zeros(V)
            elif identity_specific:
                bi = r.standard_normal(V); bi = bi - W @ (W.T @ bi); bi /= np.linalg.norm(bi); oi = 1.5 * bi
            else:
                oi = coeff[i] * bt
            for t in range(8):
                Dtrial[i, t] = insup + oi + (0.15 / max(snr, 1e-6)) * r.standard_normal(V)
        test = []
        for _ in range(2):
            insup = W @ r.standard_normal(K)
            op = 0.0 if no_outside else (r.standard_normal() * bt)
            if identity_specific:
                op = r.standard_normal(V); op = op - W @ (W.T @ op)
            test.append(insup + op)
        cells.append({"W_target": W, "U_res": U_res, "X": X, "Z": Z, "deltas_test": test,
                      "simple_ids": np.array(list(range(n_simple))), "nat_ids": np.array(list(range(5, 5 + n_nat))),
                      "R_CAL0": 0.1, "K": K, "train_ids": np.array([f"id{i}" for i in range(10)], dtype=object),
                      "test_ids": np.array(["t0", "t1"], dtype=object)})
        tn.append({"trial_native": Dtrial})
    rd = {"per_target": {"ventral": {"subjX": [{"R_ORACLE": 0.8}] * 6}}}
    return cells, tn, rd


# A (high-SNR) — even T=1 recovers the axis
def test_A_high_snr_T1_recovers():
    c, t, rd = _synth(snr=50, seedn=1)
    o = TC.subject_roi("subjX", "ventral", c, t, rd, G, {"max": 0}, [1, 2, 4], need_null=True)
    assert o["T"][1]["evaluable"] and o["T"][1]["AXIS_FIDELITY"] > 0.9


# B (noisy) — axis fidelity increases with T
def test_B_noisy_fidelity_increases():
    c, t, rd = _synth(snr=0.6, seedn=2)
    o = TC.subject_roi("subjX", "ventral", c, t, rd, G, {"max": 0}, [1, 2, 4], need_null=True)
    assert o["T"][4]["AXIS_FIDELITY"] >= o["T"][1]["AXIS_FIDELITY"] - 1e-9


# C (no outside) — augmentation ~ null
def test_C_no_outside_no_gain():
    c, t, rd = _synth(no_outside=True, seedn=3)
    o = TC.subject_roi("subjX", "ventral", c, t, rd, G, {"max": 0}, [1, 2, 4], need_null=True)
    assert o["T"][4]["E_TRIAL"] < 0.05


# D (identity-specific) — weak generalization
def test_D_identity_specific_weak():
    c, t, rd = _synth(identity_specific=True, snr=5, seedn=4)
    o = TC.subject_roi("subjX", "ventral", c, t, rd, G, {"max": 0}, [1, 2, 4], need_null=True)
    assert o["T"][4]["TOTAL_RECOVERY"] < 0.5


# K — exact expected cell counts (evaluable requires all pairs x folds x subsets)
def test_K_cell_counts_exact():
    c, t, rd = _synth(seedn=5)  # 2 simple x 2 nat = 4 pairs
    o = TC.subject_roi("subjX", "ventral", c, t, rd, G, {"max": 0}, [1, 2, 4], need_null=True)
    assert o["cell_counts"][1] == 6 * 4 * 8 and o["cell_counts"][2] == 6 * 4 * 28 and o["cell_counts"][4] == 6 * 4 * 70
    assert all(o["T"][T]["evaluable"] for T in (1, 2, 4))


# --- T8 certification (FIX 5) ------------------------------------------------
def _ref_csv(tmp, base=0.1, aug=0.5, tot=0.6, drop=None, dupe=False):
    rows = ["subject,roi,M,d,R_BASE,R_AUG,DELTA_OUT,R_NULL_mean,E_AUG,R_COND,OUT_BASIS_RECOVERY,TOTAL_RECOVERY,TRAIN_retention"]
    for roi in ("ventral", "lateral"):
        for s in TC.ALL:
            if drop == (s, roi):
                continue
            rows.append("%s,%s,2,1,%g,%g,0,0,0,0,0,%g,0" % (s, roi, base, aug, tot))
            if dupe and (s, roi) == ("subj01", "ventral"):
                rows.append("%s,%s,2,1,%g,%g,0,0,0,0,0,%g,0" % (s, roi, base, aug, tot))
    p = tmp / "aug.csv"; p.write_text("\n".join(rows) + "\n"); return p


def _per8(base=0.1, aug=0.5, tot=0.6):
    return {s: {roi: {"cell_counts": {8: 150}, "T": {8: {"R_BASE": base, "R_AUG": aug, "TOTAL_RECOVERY": tot}}}}
            for s in TC.ALL for roi in ("ventral", "lateral")} if False else \
        {s: {roi: {"cell_counts": {8: 150}, "T": {8: {"R_BASE": base, "R_AUG": aug, "TOTAL_RECOVERY": tot}}}
             for roi in ("ventral", "lateral")} for s in TC.ALL}


def test_E_t8_compares_all_three_metrics(tmp_path):
    csvp = _ref_csv(tmp_path)
    assert TC.certify_t8(_per8(), str(csvp), None, ["ventral", "lateral"])["ok"] is True
    # differ in TOTAL_RECOVERY beyond tol -> fail
    bad = _per8(); bad["subj01"]["ventral"]["T"][8]["TOTAL_RECOVERY"] = 0.6 + 1e-3
    assert TC.certify_t8(bad, str(csvp), None, ["ventral", "lateral"])["ok"] is False


def test_C_missing_ref_row_fails(tmp_path):
    csvp = _ref_csv(tmp_path, drop=("subj03", "lateral"))
    r = TC.certify_t8(_per8(), str(csvp), None, ["ventral", "lateral"])
    assert r["ok"] is False and r["status"] == "O2_7_FULL_REPEAT_REPLAY_FAILURE"


def test_D_duplicate_ref_row_fails(tmp_path):
    csvp = _ref_csv(tmp_path, dupe=True)
    assert TC.certify_t8(_per8(), str(csvp), None, ["ventral", "lateral"])["ok"] is False


def test_F_t8_requires_150_cells(tmp_path):
    csvp = _ref_csv(tmp_path); bad = _per8(); bad["subj02"]["ventral"]["cell_counts"][8] = 149
    assert TC.certify_t8(bad, str(csvp), None, ["ventral", "lateral"])["ok"] is False


# --- trial-state semantic verification (FIX 4: H, I, J) ----------------------
def _mk_z(V=40, train=None, vh="VH", nrep=8, dup_beta=False):
    train = train or [f"id{i}" for i in range(10)]
    prov = {idn: {"repeat": list(range(nrep)), "beta_index0": ([0] * 8 if dup_beta else list(range(k * 8, k * 8 + 8)))}
            for k, idn in enumerate(train)}
    return {"trial_native": np.zeros((10, 8, V)), "train_ids": np.array(train, dtype=object),
            "fam": np.array(["simple"] * 10, dtype=object), "voxel_hash": vh,
            "provenance": np.array(json.dumps(prov), dtype=object)}


def test_H_train_order_mismatch_fails():
    ids = [f"id{i}" for i in range(10)]
    cell = {"train_ids": np.array(list(reversed(ids)), dtype=object)}
    assert TC.verify_trial_file(_mk_z(train=ids), cell, "VH", 40) == "O2_7_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE"


def test_I_repeat_and_beta_dup_fails():
    ids = [f"id{i}" for i in range(10)]; cell = {"train_ids": np.array(ids, dtype=object)}
    assert TC.verify_trial_file(_mk_z(train=ids, nrep=7), cell, "VH", 40) == "O2_7_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE"
    assert TC.verify_trial_file(_mk_z(train=ids, dup_beta=True), cell, "VH", 40) == "O2_7_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE"


def test_J_wrong_voxel_hash_fails():
    ids = [f"id{i}" for i in range(10)]; cell = {"train_ids": np.array(ids, dtype=object)}
    assert TC.verify_trial_file(_mk_z(train=ids, vh="WRONG"), cell, "VH", 40) == "O2_7_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE"
    assert TC.verify_trial_file(_mk_z(train=ids), cell, "VH", 40) is None       # clean passes


# --- status logic (FIX 8: L, M, N) ------------------------------------------
def _mk_per(Ev, Etot, evaluable=True):
    per = {}
    for s in TC.ALL:
        d = {T: {"evaluable": evaluable, "E_TRIAL": Ev[T], "TOTAL_RECOVERY": Etot[T], "AXIS_FIDELITY": 0.9,
                 "TRAIN_retention": 0.5, "R_AUG": Etot[T]} for T in TC.T_PRIMARY}
        per[s] = {"ventral": {"T": d, "R0": 0.1, "R_native": 0.8},
                  "lateral": {"T": dict(d), "R0": 0.1, "R_native": 0.8}}
    return per


def test_L_one_finite_one_not_multiregime():
    good = {1: 0.2, 2: 0.2, 4: 0.2}; goodtot = {1: 0.7, 2: 0.7, 4: 0.7}
    bad = {1: -0.1, 2: -0.1, 4: -0.1}; badtot = {1: 0.1, 2: 0.1, 4: 0.1}
    per = _mk_per(good, goodtot)
    for s in TC.ALL:  # make lateral not established
        per[s]["lateral"]["T"] = {T: {"evaluable": True, "E_TRIAL": bad[T], "TOTAL_RECOVERY": badtot[T],
                                      "AXIS_FIDELITY": 0.9, "TRAIN_retention": 0.5, "R_AUG": badtot[T]} for T in TC.T_PRIMARY}
    rs, prog = TC.aggregate(per, G)
    assert prog == "TARGET_STATE_TRIAL_CALIBRATION_MULTIREGIME"


def test_M_both_not_established():
    bad = {1: -0.1, 2: -0.1, 4: -0.1}; badtot = {1: 0.1, 2: 0.1, 4: 0.1}
    rs, prog = TC.aggregate(_mk_per(bad, badtot), G)
    assert prog == "REDUCED_TRIAL_TARGET_STATE_CALIBRATION_NOT_ESTABLISHED"


def test_both_T1_two_total_trials():
    rs, prog = TC.aggregate(_mk_per({1: 0.2, 2: 0.2, 4: 0.2}, {1: 0.7, 2: 0.7, 4: 0.7}), G)
    assert prog == "MINIMAL_TARGET_STATE_CALIBRATION_TWO_TOTAL_TRIALS"
    assert rs["ventral"]["T_STAR"] == 1


def test_N_t8_never_in_holm_family():
    per = _mk_per({1: 0.2, 2: 0.2, 4: 0.2}, {1: 0.7, 2: 0.7, 4: 0.7})
    rs, prog = TC.aggregate(per, G)
    for roi in ("ventral", "lateral"):
        assert 8 not in rs[roi]["per_T"]                                        # T8 not in the inference family


# --- immutable seals + no model search ---------------------------------------
def test_immutable_seals_and_o3():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_7/o2_7_frozen_config.json"))
    assert cfg["immutable_history"]["O2_6"] == "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"
    assert cfg["fixed_resource_point"]["M"] == 2 and cfg["fixed_resource_point"]["D"] == 1 and cfg["O3"] == "O3_NOT_READY"


def test_no_model_search():
    src = Path(TC.__file__).read_text()
    for bad in ("sklearn", "Ridge", "CCA(", "PLS", "torch", "RandomForest"):
        assert bad not in src
