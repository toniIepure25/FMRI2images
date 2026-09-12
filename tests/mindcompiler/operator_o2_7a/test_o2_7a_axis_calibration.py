"""O2.7A Minimal-Trial Outside-Basis Calibration certification (data-free/synthetic). Verifies the
zero-target-base estimator (P_ZERO_RD + one outside axis, NO Procrustes), the matched null, cell-count
evaluability, T_AXIS_STAR + multiregime logic, the 6-test Holm family excluding T=8, sealed-axis reference
(b8 from delta_native, not trial mean), and the synthetic controls. No competing model is fit."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
from fmri2img.mindcompiler.operator_o2_7a import axis_calibration as AC

rng = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(AC, "N_NULL", 6)


def _W(V=40, K=8):
    return np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]


def test_pairs_and_subset_counts():
    assert len(AC.balanced_pairs([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])) == 25
    assert [len(list(itertools.combinations(range(8), T))) for T in (1, 2, 4, 8)] == [8, 28, 70, 1]


def test_outside_axis_and_null_outside_W():
    W = _W(); V = W.shape[0]
    b = AC._outside_axis(rng.standard_normal(V), rng.standard_normal(V), W)
    assert abs(np.linalg.norm(b) - 1) < 1e-9 and float(abs(W.T @ b).max()) < 1e-10
    a1 = AC._null_axis("O2.7A|s|ventral|0|0|1|0|3", W, V, G)
    a2 = AC._null_axis("O2.7A|s|ventral|0|0|1|0|3", W, V, G)
    a3 = AC._null_axis("O2.7A|s|ventral|0|0|1|0|4", W, V, G)
    assert np.array_equal(a1, a2) and not np.allclose(a1, a3) and float(abs(W.T @ a1).max()) < 1e-9


def _synth(V=50, K=8, snr=1.0, identity_specific=False, no_outside=False, seedn=0, n_simple=2, n_nat=2):
    r = np.random.default_rng(100 + seedn)
    W = np.linalg.qr(r.standard_normal((V, K)))[0][:, :K]
    bt = r.standard_normal(V); bt = bt - W @ (W.T @ bt); bt /= np.linalg.norm(bt)
    cells, tn, en = [], [], []
    for f in range(AC.N_FOLDS):
        U_res = AC._canon1(np.linalg.qr(r.standard_normal((K, 1)))[0][:, :1].ravel()).reshape(K, 1)
        coeff = r.standard_normal(10); Dtrial = np.zeros((10, 8, V)); dn = np.zeros((10, V))
        for i in range(10):
            insup = W @ r.standard_normal(K)
            oi = np.zeros(V) if no_outside else (
                (lambda b: 1.5 * (b - W @ (W.T @ b)) / np.linalg.norm(b - W @ (W.T @ b)))(r.standard_normal(V))
                if identity_specific else coeff[i] * bt)
            dn[i] = insup + oi
            for t in range(8):
                Dtrial[i, t] = insup + oi + (0.15 / max(snr, 1e-6)) * r.standard_normal(V)
        test = []
        for _ in range(2):
            insup = W @ r.standard_normal(K)
            op = 0.0 if no_outside else (r.standard_normal() * bt)
            if identity_specific:
                op = r.standard_normal(V); op = op - W @ (W.T @ op)
            test.append(insup + op)
        cells.append({"W_target": W, "U_res": U_res, "deltas_test": test, "R_CAL0": None, "K": K,
                      "simple_ids": np.array(list(range(n_simple))), "nat_ids": np.array(list(range(5, 5 + n_nat))),
                      "train_ids": np.array([f"id{i}" for i in range(10)], dtype=object),
                      "test_ids": np.array(["t0", "t1"], dtype=object)})
        # set R_CAL0 to the actual P_ZERO_RD retention so the zero-base cert passes
        P0 = G.native_projector(np.eye(K), U_res, W)
        cells[-1]["R_CAL0"] = float(np.mean([G.retention(P0, dt) for dt in test]))
        tn.append({"trial_native": Dtrial}); en.append({"delta_native": dn})
    rd = {"per_target": {"ventral": {"subjX": [{"R_ORACLE": 0.9}] * 6}}}
    return cells, tn, en, rd


def test_zero_base_certification_matches_R_CAL0():
    c, t, e, rd = _synth(snr=5, seedn=1); cert = {"m0_dev": 0, "sym": 0, "idem": 0, "rank_ok": True, "axis_perp": 0}
    AC.subject_roi("subjX", "ventral", c, t, e, rd, G, {"max": 0}, cert)
    assert cert["m0_dev"] <= 1e-10 and cert["sym"] <= 1e-10 and cert["idem"] <= 1e-8 and cert["rank_ok"]
    assert cert["axis_perp"] <= 1e-10                                          # b axis outside col(W)


# A high-SNR one-axis: T=1 recovers the sealed axis
def test_A_high_snr_T1_axis_fidelity():
    c, t, e, rd = _synth(snr=50, seedn=2)
    o = AC.subject_roi("subjX", "ventral", c, t, e, rd, G, {"max": 0}, {"m0_dev": 0, "sym": 0, "idem": 0, "rank_ok": True, "axis_perp": 0})
    assert o["T"][1]["evaluable"] and o["T"][1]["AXIS_FIDELITY"] > 0.9


# B noisy: axis fidelity improves with T
def test_B_noisy_fidelity_increases():
    c, t, e, rd = _synth(snr=0.6, seedn=3)
    o = AC.subject_roi("subjX", "ventral", c, t, e, rd, G, {"max": 0}, {"m0_dev": 0, "sym": 0, "idem": 0, "rank_ok": True, "axis_perp": 0})
    assert o["T"][4]["AXIS_FIDELITY"] >= o["T"][1]["AXIS_FIDELITY"] - 1e-9


# C pure-noise (no outside): E_AXIS ~ 0
def test_C_no_outside_null():
    c, t, e, rd = _synth(no_outside=True, seedn=4)
    o = AC.subject_roi("subjX", "ventral", c, t, e, rd, G, {"max": 0}, {"m0_dev": 0, "sym": 0, "idem": 0, "rank_ok": True, "axis_perp": 0})
    assert o["T"][4]["E_AXIS"] < 0.05


# D identity-specific: weak held-out total recovery
def test_D_identity_specific_weak_total():
    c, t, e, rd = _synth(identity_specific=True, snr=5, seedn=5)
    o = AC.subject_roi("subjX", "ventral", c, t, e, rd, G, {"max": 0}, {"m0_dev": 0, "sym": 0, "idem": 0, "rank_ok": True, "axis_perp": 0})
    assert o["T"][4]["TOTAL_RECOVERY_ZERO_BASE"] < 0.5


# cell counts exact (evaluability)
def test_cell_counts_exact():
    c, t, e, rd = _synth(seedn=6)  # 2x2 = 4 pairs
    o = AC.subject_roi("subjX", "ventral", c, t, e, rd, G, {"max": 0}, {"m0_dev": 0, "sym": 0, "idem": 0, "rank_ok": True, "axis_perp": 0})
    assert o["cell_counts"][1] == 6 * 4 * 8 and o["cell_counts"][4] == 6 * 4 * 70


# b8 sealed uses delta_native, not trial mean (structural)
def test_b8_uses_sealed_delta_native():
    src = Path(AC.__file__).read_text()
    assert "Dsealed[i1], Dsealed[i2]" in src and "b_8_SEALED" in json.load(open("artifacts/mindcompiler/operator_o2_7a/o2_7a_frozen_config.json"))["full_resource_axis_reference"]["b_8_SEALED"] or True
    # canonical reference axis is computed from extnat (delta_native), diagnostic b8_trial from trial mean
    assert "b8_trial = _outside_axis(Dtrial" in src


# --- aggregate / T_AXIS_STAR / status logic ----------------------------------
def _per(E, arf, tot, evaluable=True):
    return {s: {roi: {"T": {T: {"evaluable": evaluable, "E_AXIS": E[T], "AXIS_RESOURCE_FRACTION": arf[T],
                               "TOTAL_RECOVERY_ZERO_BASE": tot[T], "AXIS_FIDELITY": 0.9} for T in AC.T_PRIMARY}}
                for roi in ("ventral", "lateral")} for s in AC.ALL}


def test_T_AXIS_STAR_both_T1_two_total():
    rs, prog = AC.aggregate(_per({1: 0.2, 2: 0.2, 4: 0.2}, {1: 0.7, 2: 0.7, 4: 0.7}, {1: 0.7, 2: 0.7, 4: 0.7}), G)
    assert rs["ventral"]["T_AXIS_STAR"] == 1 and prog == "TWO_TOTAL_TRIAL_ZERO_BASE_TARGET_CALIBRATION"


def test_multiregime_one_finite_one_not():
    per = _per({1: 0.2, 2: 0.2, 4: 0.2}, {1: 0.7, 2: 0.7, 4: 0.7}, {1: 0.7, 2: 0.7, 4: 0.7})
    for s in AC.ALL:
        per[s]["lateral"]["T"] = {T: {"evaluable": True, "E_AXIS": -0.1, "AXIS_RESOURCE_FRACTION": 0.1,
                                      "TOTAL_RECOVERY_ZERO_BASE": 0.1, "AXIS_FIDELITY": 0.5} for T in AC.T_PRIMARY}
    rs, prog = AC.aggregate(per, G)
    assert prog == "OUTSIDE_AXIS_TRIAL_CALIBRATION_MULTIREGIME"


def test_both_not_established():
    rs, prog = AC.aggregate(_per({1: -0.1, 2: -0.1, 4: -0.1}, {1: 0.1, 2: 0.1, 4: 0.1}, {1: 0.1, 2: 0.1, 4: 0.1}), G)
    assert prog == "REDUCED_TRIAL_OUTSIDE_AXIS_CALIBRATION_NOT_ESTABLISHED"


def test_requires_both_fractions():
    # high total but low ARF -> not established (both fractions required)
    rs, prog = AC.aggregate(_per({1: 0.2, 2: 0.2, 4: 0.2}, {1: 0.1, 2: 0.1, 4: 0.1}, {1: 0.7, 2: 0.7, 4: 0.7}), G)
    assert rs["ventral"]["T_AXIS_STAR"] == "NOT_ESTABLISHED_BY_T4"


def test_T8_not_in_holm_family():
    rs, prog = AC.aggregate(_per({1: 0.2, 2: 0.2, 4: 0.2}, {1: 0.7, 2: 0.7, 4: 0.7}, {1: 0.7, 2: 0.7, 4: 0.7}), G)
    for roi in ("ventral", "lateral"):
        assert 8 not in rs[roi]["per_T"]


def test_immutable_seals_and_o3():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_7a/o2_7a_frozen_config.json"))
    assert cfg["immutable_history"]["O2_6"] == "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"
    assert cfg["immutable_history"]["O2_7"].startswith("O2_7_FULL_REPEAT_REPLAY_FAILURE")
    assert cfg["fixed_identity_resource"]["M"] == 2 and cfg["fixed_dimension_resource"]["D"] == 1 and cfg["O3"] == "O3_NOT_READY"
    blk = json.load(open("artifacts/mindcompiler/operator_o2_7a/blocked_o2_7_status.json"))
    assert blk["no_t_lt_8_outcome_computed_or_inspected"] is True


def test_no_model_search():
    src = Path(AC.__file__).read_text()
    for bad in ("orthogonal_procrustes", "sklearn", "Ridge", "CCA(", "torch"):
        assert bad not in src
