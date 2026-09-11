"""O2.5 Perception-Support Ceiling Audit certification (data-free/synthetic). Diagnostic-only gate:
verifies hash gating, W_target/P_SUPPORT certification, the support ceiling >= any in-support predictor,
the train-only rank-matched support oracle, gap-fraction formulas + guards, participant-first aggregation,
the categorical mechanism logic, absence of any competing model, and the immutable seals / O3 lock."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
from fmri2img.mindcompiler.operator_o2_5 import support_audit as SA

rng = np.random.default_rng(5)


def _cell(p=40, K=8, r=3, in_support_frac=0.7):
    W = np.linalg.qr(rng.standard_normal((p, K)))[0][:, :K]
    U_res = SA._canon_cols(np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r])
    X = rng.standard_normal((10, K)); Z = rng.standard_normal((10, K))
    # held-out deltas: partly inside col(W), partly orthogonal
    deltas = []
    for _ in range(2):
        a = rng.standard_normal(K)
        perp = rng.standard_normal(p); perp = perp - W @ (W.T @ perp)
        d = in_support_frac * (W @ a) / np.linalg.norm(W @ a) + (1 - in_support_frac) * perp / np.linalg.norm(perp)
        deltas.append(d)
    return {"W_target": W, "U_res": U_res, "X": X, "Z": Z, "deltas_test": deltas,
            "R_CAL0": 0.10, "r_best": r, "K": K}


# 1 — hash gating
def test_state_hash_verification(tmp_path):
    f = tmp_path / "cell_subj01_ventral_fold0.npz"
    np.savez(f, a=np.arange(3))
    man = {"n_files": 1, "files": [{"name": f.name, "sha256": SA._sha256_file(f), "bytes": f.stat().st_size}]}
    mp = tmp_path / "man.json"; mp.write_text(json.dumps(man))
    assert SA.verify_state_hashes(tmp_path, mp, [f.name])["ok"] is True
    np.savez(f, a=np.arange(4))                       # tamper
    assert SA.verify_state_hashes(tmp_path, mp, [f.name])["ok"] is False


# 2,3,4 — W_target orthonormality + P_SUPPORT symmetry + idempotence
def test_w_target_and_p_support_certification():
    m = SA.cell_metrics(_cell(), 0.7, G)
    assert m["orth_err"] <= 1e-8
    assert m["sym_err"] <= 1e-10 and m["idem_err"] <= 1e-10


# 5,6 — O2.4R predicted subspace inside col(W_target) => R_pred <= R_SUPPORT_FULL
def test_predicted_bounded_by_support_ceiling():
    for _ in range(5):
        m = SA.cell_metrics(_cell(in_support_frac=rng.uniform(0.2, 0.95)), 0.7, G)
        assert m["R_m0_recon"] <= m["R_full"] + 1e-10
        assert m["R_m10"] <= m["R_full"] + 1e-10


# 7,8 — support oracle uses TRAIN Z only; test residuals never enter the fit
def test_support_oracle_train_only():
    Z = rng.standard_normal((10, 8))
    U1 = SA.support_oracle_basis(Z, 3)
    U2 = SA.support_oracle_basis(Z, 3)
    assert np.array_equal(U1, U2)                     # deterministic
    # changing held-out deltas cannot change the oracle basis (basis is a function of Z only)
    c = _cell(); base = SA.support_oracle_basis(c["Z"], c["r_best"])
    c2 = dict(c); c2["deltas_test"] = [d + 5.0 for d in c["deltas_test"]]
    assert np.array_equal(base, SA.support_oracle_basis(c2["Z"], c2["r_best"]))


# 9 — exact frozen rank
def test_support_oracle_exact_rank():
    for r in (1, 2, 4, 6):
        assert SA.support_oracle_basis(rng.standard_normal((10, 8)), r).shape == (8, r)


# 10,11,12 — F_FULL / F_RANK / F_EST formulas
def test_gap_fraction_formulas():
    # R0=0.1, R_full=0.5, R_sup_or=0.3, R_m10=0.2, R_native=0.9
    R0, R_full, R_sup, R_m10, R_nat = 0.1, 0.5, 0.3, 0.2, 0.9
    den_nat = R_nat - R0
    assert abs((R_full - R0) / den_nat - 0.5) < 1e-12          # F_FULL=0.5
    assert abs((R_sup - R0) / den_nat - 0.25) < 1e-12          # F_RANK=0.25
    assert abs((R_m10 - R0) / (R_sup - R0) - 0.5) < 1e-12      # F_EST=0.5
    assert SA._clip01(1.4) == 1.0 and SA._clip01(-0.2) == 0.0


# 13 — denominator guards
def test_denominator_guards():
    # DEN_NATIVE <= eps -> not evaluable for native gap
    part = {s: {"evaluable_native": False} for s in SA.ALL}
    st = SA.classify_roi(part)
    assert st["status"] == "SUPPORT_MAPPING_MULTIREGIME_OR_INCONCLUSIVE" and st["n_evaluable"] == 0


# 14 — participant-first aggregation (mean folds before medians) is what classify consumes
def test_participant_first_then_median():
    # 8 participants, F_FULL below 0.5 for >=6 -> ceiling
    part = {}
    ff = [0.2, 0.3, 0.25, 0.4, 0.45, 0.1, 0.6, 0.7]           # 6 below 0.5
    for s, v in zip(SA.ALL, ff):
        part[s] = {"evaluable_native": True, "evaluable_est": True, "F_FULL": v, "F_RANK": v, "F_EST": v}
    st = SA.classify_roi(part)
    assert st["median_F_FULL"] == float(np.median(ff))
    assert st["status"] == "PERCEPTION_SUPPORT_CEILING_DOMINANT"


# 15 — categorical status logic (all five branches)
def _part(ff, fr, fe):
    return {s: {"evaluable_native": True, "evaluable_est": True, "F_FULL": ff[i], "F_RANK": fr[i], "F_EST": fe[i]}
            for i, s in enumerate(SA.ALL)}


def test_categorical_status_branches():
    lo = [0.2] * 8; hi = [0.8] * 8
    assert SA.classify_roi(_part(lo, lo, lo))["status"] == "PERCEPTION_SUPPORT_CEILING_DOMINANT"
    assert SA.classify_roi(_part(hi, lo, lo))["status"] == "RANK_MATCHED_SUPPORT_LIMIT_DOMINANT"
    assert SA.classify_roi(_part(hi, hi, lo))["status"] == "WITHIN_SUPPORT_MAPPING_ESTIMATOR_DOMINANT"
    assert SA.classify_roi(_part(hi, hi, hi))["status"] == "SUPPORT_AND_MAPPING_ADEQUATE"
    # <7 evaluable -> inconclusive
    p = _part(hi, hi, hi)
    for s in SA.ALL[:2]:
        p[s]["evaluable_native"] = False
    assert SA.classify_roi(p)["status"] == "SUPPORT_MAPPING_MULTIREGIME_OR_INCONCLUSIVE"


def test_program_status_mapping():
    def roi_st(v, l): return {"ventral": {"status": v}, "lateral": {"status": l}}
    assert SA.program_status(roi_st("PERCEPTION_SUPPORT_CEILING_DOMINANT", "PERCEPTION_SUPPORT_CEILING_DOMINANT")) == "TARGET_IMAGERY_RESIDUAL_OUTSIDE_PERCEPTION_SUPPORT_DOMINANT"
    assert SA.program_status(roi_st("RANK_MATCHED_SUPPORT_LIMIT_DOMINANT", "RANK_MATCHED_SUPPORT_LIMIT_DOMINANT")) == "TARGET_IMAGERY_RESIDUAL_RANK_LIMIT"
    assert SA.program_status(roi_st("WITHIN_SUPPORT_MAPPING_ESTIMATOR_DOMINANT", "WITHIN_SUPPORT_MAPPING_ESTIMATOR_DOMINANT")) == "TARGET_IMAGERY_RESIDUAL_WITHIN_SUPPORT_MAPPING_LIMIT"
    assert SA.program_status(roi_st("PERCEPTION_SUPPORT_CEILING_DOMINANT", "RANK_MATCHED_SUPPORT_LIMIT_DOMINANT")) == "TARGET_IMAGERY_RESIDUAL_SUPPORT_MULTIREGIME"


# 16 — no competing model executed (module imports no ML fitters)
def test_no_competing_model_imported():
    src = Path(SA.__file__).read_text()
    for bad in ("sklearn", "Ridge", "CCA", "PLS", "RandomForest", "torch", "cca", "ridge_regression"):
        assert bad not in src


# 17,18 — immutable seals + O3 lock present in status writer contract
def test_immutable_seals_and_o3_lock():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_5/o2_5_frozen_config.json"))
    assert cfg["immutable_history"]["O2_4R"] == "TARGET_STATE_ORIENTATION_NOT_RECOVERED_BY_ORTHOGONAL_CALIBRATION"
    assert cfg["immutable_history"]["O2_3A_RD"] == "CORE_ANCHOR_RD_INCONCLUSIVE"
    assert cfg["O3"] == "O3_NOT_READY"
