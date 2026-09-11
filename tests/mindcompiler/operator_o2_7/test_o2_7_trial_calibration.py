"""O2.7 Minimal-Trial Target-State Calibration certification (data-free/synthetic). Verifies the reduced-repeat
estimator plumbing (outside axis, matched null, repeat-subset enumeration, T=8 centroid replay), axis-fidelity
behavior, T_STAR/status logic + 6-test Holm family, participant-first aggregation, the synthetic controls
(high-SNR / noisy / pure-random / identity-specific / T8-replay / leakage), and immutable seals / O3."""
from __future__ import annotations

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
    # synthetic controls don't need 100 null draws; keep the frontier fast (real run uses N_NULL=100)
    monkeypatch.setattr(TC, "N_NULL", 6)


def _W(V=40, K=8):
    return np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]


# item 5,6 — 25 balanced pairs; repeat-subset counts exactly 8/28/70/1
def test_pairs_and_subset_counts():
    assert len(TC.balanced_pairs([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])) == 25
    assert [len(list(itertools.combinations(range(8), T))) for T in (1, 2, 4, 8)] == [8, 28, 70, 1]
    assert TC.EXPECTED_SUBSETS == {1: 8, 2: 28, 4: 70, 8: 1}


# item 11 — outside axis: unit, orthogonal to W, deterministic sign
def test_outside_axis_properties():
    W = _W(); V = W.shape[0]
    d1 = rng.standard_normal(V); d2 = rng.standard_normal(V)
    b = TC._outside_axis(d1, d2, W)
    assert abs(np.linalg.norm(b) - 1.0) < 1e-9 and float(abs(W.T @ b).max()) < 1e-10
    assert np.array_equal(b, TC._outside_axis(d1, d2, W))                     # deterministic
    j = int(np.argmax(np.abs(b))); assert b[j] > 0                            # canonical sign


# item 11 — matched null outside W, unit, deterministic by seed
def test_null_axis_outside_and_deterministic():
    W = _W(); V = W.shape[0]
    a = TC._null_axis("O2.7|s|ventral|0|3|2|5|7", W, V, G)
    b = TC._null_axis("O2.7|s|ventral|0|3|2|5|7", W, V, G)
    c = TC._null_axis("O2.7|s|ventral|0|3|2|5|8", W, V, G)
    assert np.array_equal(a, b) and not np.allclose(a, c)
    assert abs(np.linalg.norm(a) - 1) < 1e-9 and float(abs(W.T @ a).max()) < 1e-9


# --- synthetic cohort for end-to-end subject_roi ------------------------------
def _synth_subject(V=50, K=8, snr=1.0, identity_specific=False, no_outside=False, seedn=0):
    r = np.random.default_rng(100 + seedn)
    W = np.linalg.qr(r.standard_normal((V, K)))[0][:, :K]
    b_true = r.standard_normal(V); b_true = b_true - W @ (W.T @ b_true); b_true /= np.linalg.norm(b_true)
    cells, tn = [], []
    for f in range(TC.N_FOLDS):
        U_res = TC._canon1(np.linalg.qr(r.standard_normal((K, 1)))[0][:, :1].ravel()).reshape(K, 1)
        X = r.standard_normal((10, K)); Z = r.standard_normal((10, K))
        # identity outside coefficients (shared axis) + per-identity in-support
        coeff = r.standard_normal(10)
        Dtrial = np.zeros((10, 8, V))
        for i in range(10):
            insup = W @ r.standard_normal(K)
            if no_outside:
                out_i = np.zeros(V)
            elif identity_specific:
                bi = r.standard_normal(V); bi = bi - W @ (W.T @ bi); bi /= np.linalg.norm(bi)
                out_i = 1.5 * bi                                              # each identity a different axis
            else:
                out_i = coeff[i] * b_true
            for t in range(8):
                Dtrial[i, t] = insup + out_i + (1.0 / max(snr, 1e-6)) * (r.standard_normal(V) * 0.15)
        # held-out test residuals: share b_true (generalizable) unless identity_specific/no_outside
        test = []
        for _ in range(2):
            insup = W @ r.standard_normal(K)
            outp = (0.0 if no_outside else (r.standard_normal(V) if identity_specific else r.standard_normal() * b_true))
            if identity_specific:
                outp = outp - W @ (W.T @ outp)
            test.append(insup + outp)
        cells.append({"W_target": W, "U_res": U_res, "X": X, "Z": Z, "deltas_test": test,
                      "simple_ids": np.array([0, 1]), "nat_ids": np.array([5, 6]),   # 4 pairs (fast synthetic)
                      "R_CAL0": 0.1, "K": K, "train_ids": np.array([f"id{i}" for i in range(10)], dtype=object),
                      "test_ids": np.array(["t0", "t1"], dtype=object)})
        tn.append({"trial_native": Dtrial})
    rd = {"per_target": {"ventral": {"subjX": [{"R_ORACLE": 0.8} for _ in range(6)]}}}
    return cells, tn, rd


# item 3,4 (E control) — T=8 mean reproduces the centroid axis; centroid replay
def test_E_T8_centroid_replay():
    cells, tn, rd = _synth_subject(snr=5.0)
    Dtrial = tn[0]["trial_native"]; W = cells[0]["W_target"]
    d1_8 = Dtrial[0].mean(0); d2_8 = Dtrial[5].mean(0)
    b8 = TC._outside_axis(d1_8, d2_8, W)
    # full-subset S=all8 mean equals the centroid used for b8
    S = list(range(8))
    assert np.allclose(Dtrial[0][S].mean(0), d1_8, atol=1e-12)
    assert np.array_equal(TC._outside_axis(Dtrial[0][S].mean(0), Dtrial[5][S].mean(0), W), b8)


# A — high-SNR one-axis: even T=1 recovers the planted axis well
def test_A_high_snr_one_axis_T1_recovers():
    cells, tn, rd = _synth_subject(snr=50.0, seedn=1)
    out = TC.subject_roi("subjX", "ventral", cells, tn, rd, G, {"max": 0.0})
    assert out["T"][1]["AXIS_FIDELITY"] > 0.9                                 # T=1 already high-fidelity


# B — noisy one-axis: axis fidelity improves with T
def test_B_noisy_axis_fidelity_increases_with_T():
    cells, tn, rd = _synth_subject(snr=0.6, seedn=2)
    out = TC.subject_roi("subjX", "ventral", cells, tn, rd, G, {"max": 0.0})
    assert out["T"][4]["AXIS_FIDELITY"] >= out["T"][1]["AXIS_FIDELITY"] - 1e-9
    assert out["T"][8]["AXIS_FIDELITY"] >= out["T"][4]["AXIS_FIDELITY"] - 1e-9


# C — pure random outside (no shared axis): augmentation does not beat null much
def test_C_no_outside_no_systematic_gain():
    cells, tn, rd = _synth_subject(no_outside=True, seedn=3)
    out = TC.subject_roi("subjX", "ventral", cells, tn, rd, G, {"max": 0.0})
    assert out["T"][8]["E_TRIAL"] < 0.05                                      # ~ null (no outside structure)


# D — identity-specific outside: held-out generalization weak despite fit
def test_D_identity_specific_weak_generalization():
    cells, tn, rd = _synth_subject(identity_specific=True, snr=5.0, seedn=4)
    out = TC.subject_roi("subjX", "ventral", cells, tn, rd, G, {"max": 0.0})
    assert out["T"][8]["TOTAL_RECOVERY"] < 0.5                                # does not generalize to held-out


# item 12 (F leakage) — pairs index only training identities (0..9); test ids separate
def test_F_no_testid_leakage():
    for (a, b) in TC.balanced_pairs([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]):
        assert 0 <= a <= 4 and 5 <= b <= 9                                    # never a test identity


# item 14,15 — 6-test Holm family + T_STAR logic + program status
def _mk_per_subj(E_by_T, tot_by_T):
    per = {}
    for s in TC.ALL:
        d = {T: {"E_TRIAL": E_by_T[T], "TOTAL_RECOVERY": tot_by_T[T], "AXIS_FIDELITY": 0.9,
                 "FULL_RESOURCE_FRACTION": 0.9, "TRAIN_retention": 0.5, "R_AUG": tot_by_T[T]} for T in TC.T_ALL}
        per[s] = {"ventral": {"T": d, "R0": 0.1, "R_native": 0.8, "R8": 0.5},
                  "lateral": {"T": d, "R0": 0.1, "R_native": 0.8, "R8": 0.5}}
    return per


def test_T_STAR_and_program_status():
    per = _mk_per_subj({1: 0.2, 2: 0.2, 4: 0.2, 8: 0.2}, {1: 0.7, 2: 0.7, 4: 0.7, 8: 0.7})
    roi_status, prog = TC.aggregate(per, G)
    assert roi_status["ventral"]["T_STAR"] == 1                               # T=1 meets all
    assert prog == "MINIMAL_TARGET_STATE_CALIBRATION_TWO_TOTAL_TRIALS"
    # nothing works -> not established
    per2 = _mk_per_subj({1: -0.1, 2: -0.1, 4: -0.1, 8: 0.2}, {1: 0.1, 2: 0.1, 4: 0.1, 8: 0.7})
    rs2, prog2 = TC.aggregate(per2, G)
    assert rs2["ventral"]["T_STAR"] == "BELOW_FULL_RESOURCE_NOT_ESTABLISHED"
    assert prog2 == "REDUCED_TRIAL_TARGET_STATE_CALIBRATION_NOT_ESTABLISHED"


# item 4 — T8 exact O2.6 replay comparison logic
def test_t8_replay_certification(tmp_path):
    csvp = tmp_path / "aug.csv"
    csvp.write_text("subject,roi,M,d,R_BASE,R_AUG,DELTA_OUT,R_NULL_mean,E_AUG,R_COND,OUT_BASIS_RECOVERY,TOTAL_RECOVERY,TRAIN_retention\n"
                    + "\n".join("%s,ventral,2,1,0.1,0.5,0,0,0,0,0,0,0" % s for s in TC.ALL) + "\n")
    per = {s: {"ventral": {"T": {8: {"R_AUG": 0.5}}}} for s in TC.ALL}
    assert TC.certify_t8_replay(per, str(csvp), ["ventral"])["ok"] is True
    per2 = {s: {"ventral": {"T": {8: {"R_AUG": 0.5 + 1e-6}}}} for s in TC.ALL}
    assert TC.certify_t8_replay(per2, str(csvp), ["ventral"])["ok"] is False


# item 16,17 — immutable seals + O3 lock
def test_immutable_seals_and_o3():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_7/o2_7_frozen_config.json"))
    assert cfg["immutable_history"]["O2_6"] == "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"
    assert cfg["fixed_resource_point"]["M"] == 2 and cfg["fixed_resource_point"]["D"] == 1
    assert cfg["O3"] == "O3_NOT_READY"


# no model search — driver imports no ML fitters
def test_no_model_search():
    src = Path(TC.__file__).read_text()
    for bad in ("sklearn", "Ridge", "CCA(", "PLS", "torch", "RandomForest"):
        assert bad not in src
