"""O2.4R frontier-driver certification (data-free, synthetic per-cell state). Certifies balanced-subset
enumeration, the calibration frontier (true recovers > identity-correspondence null in an identifiable
regime), deterministic null replay, exact sign-flip p, Holm, and the M_STAR rule -- all on the committed
geometry, with no NSD data."""
from __future__ import annotations

import numpy as np

from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G

rng = np.random.default_rng(11)
K, r, p = 6, 3, 100


def _cell(Qtrue):
    U_res = np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r]
    W = np.linalg.qr(rng.standard_normal((p, K)))[0][:, :K]
    A = rng.standard_normal((12, r))                      # 10 train (0-4 simple, 5-9 nat) + 2 test (10,11)
    X = A @ U_res.T
    Z = X @ Qtrue
    deltas = [W @ (Qtrue.T @ (U_res @ A[i])) for i in (10, 11)]
    P0 = G.native_projector(np.eye(K), U_res, W)
    R0 = float(np.mean([G.retention(P0, d) for d in deltas]))
    return {"X": X, "Z": Z, "U_res": U_res, "W_target": W, "deltas_test": deltas,
            "simple_ids": [0, 1, 2, 3, 4], "nat_ids": [5, 6, 7, 8, 9], "R_CAL0": R0}


def test_balanced_subset_counts():
    s, n = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    assert len(F.balanced_subsets(s, n, 0)) == 1
    assert len(F.balanced_subsets(s, n, 2)) == 25
    assert len(F.balanced_subsets(s, n, 4)) == 100
    assert len(F.balanced_subsets(s, n, 8)) == 25
    assert len(F.balanced_subsets(s, n, 10)) == 1


def test_frontier_true_beats_null_in_identifiable_regime():
    Qtrue = np.linalg.qr(rng.standard_normal((K, K)))[0]
    cell = _cell(Qtrue)
    fr = F.cell_frontier(cell, "subj01", "ventral", 0, n_null=10, budgets=[0, 2, 10])
    # calibration recovers: true retention rises above the M=0 baseline and above the null
    assert fr[2]["r_cal_true"] > fr[0]["r_cal_true"]
    assert fr[10]["r_cal_true"] > fr[2]["r_cal_true"] - 1e-9
    assert fr[2]["r_cal_true"] > fr[2]["r_cal_null_mean"]        # identity correspondence matters
    assert fr[10]["r_cal_true"] > 0.95                           # near-perfect recovery at M=10


def test_null_permutation_deterministic():
    p1 = F._null_perm("O2.4R|subj01|ventral|0|2|3|7", 2)
    p2 = F._null_perm("O2.4R|subj01|ventral|0|2|3|7", 2)
    assert np.array_equal(p1, p2)


def test_signflip_exact_and_holm():
    assert abs(F.signflip_p_onesided([1.0] * 8) - 1.0 / 256.0) < 1e-12   # all-positive -> only all-+ config
    assert F.signflip_p_onesided([-1.0] * 8) == 1.0                       # all-negative -> every config >= obs
    rej = F.holm({"a": 0.001, "b": 0.9, "c": 0.02}, alpha=0.05)
    assert rej["a"] is True and rej["b"] is False


def test_participant_E_and_mstar():
    Qtrue = np.linalg.qr(rng.standard_normal((K, K)))[0]
    cfs = [F.cell_frontier(_cell(Qtrue), "subjX", "ventral", f, n_null=6, budgets=[0, 10]) for f in range(2)]
    E10, dz10 = F.participant_E(cfs, 10)                         # M=10 well-determined -> robust effect
    assert E10 > 0 and dz10 > 0                                  # positive calibration effect + improvement
    per_M = {2: {"median_E": 0.0, "frac_pos": 0.5, "holm_p": 0.3, "median_delta_zero": 0.0, "median_oracle_recovery": 0.2},
             4: {"median_E": 0.1, "frac_pos": 7 / 8, "holm_p": 0.01, "median_delta_zero": 0.1, "median_oracle_recovery": 0.6}}
    assert F.m_star(per_M) == 4
    assert F.m_star({2: {"median_E": -1, "frac_pos": 0, "holm_p": 1, "median_delta_zero": -1, "median_oracle_recovery": 0}}) == "NOT_REACHED"
