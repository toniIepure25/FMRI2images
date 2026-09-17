"""O2.16 replication protocol pre-outcome technical certification (data-free/synthetic). Verifies the 1-vs-1
quality monitor (repeat-label symmetry, identical A/B -> Q=1, orthogonal -> low Q, basis rotation/sign
invariance, projector validity, computed WITHOUT held-out data), the M4T2 operational replication criterion
(75% rule generalizing the historical 6/8), the exactly-4-test classification, and config immutability. No
independent cohort is required for these frozen-formula tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_16 import replication as RP

rng = np.random.default_rng(16)


def _acq(noise):
    V, K = 40, 6
    W = np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]
    base = rng.standard_normal((4, V))                                         # 4 identities
    dA = base + noise * rng.standard_normal((4, V)); dB = base + noise * rng.standard_normal((4, V))
    return W, dA, dB, base


# ---- 1. repeat-label symmetry ----
def test_symmetry_A_B():
    W, dA, dB, _ = _acq(0.3)
    qin1, qout1 = RP.q_mon(W, dA, dB, 3); qin2, qout2 = RP.q_mon(W, dB, dA, 3)
    assert abs(qin1 - qin2) < 1e-10 and abs(qout1 - qout2) < 1e-10


# ---- 2. identical A/B -> Q=1 ----
def test_identical_repeats_Q_one():
    W, dA, _, _ = _acq(0.3)
    qin, qout = RP.q_mon(W, dA, dA.copy(), 3)
    assert abs(qin - 1.0) < 1e-9 and abs(qout - 1.0) < 1e-9


# ---- 3. orthogonal-ish -> low Q ----
def test_noisy_repeats_low_Q():
    W, dA, dB, _ = _acq(6.0)                                                   # heavy independent noise
    qin, qout = RP.q_mon(W, dA, dB, 3)
    assert qin < 0.95 and qout < 0.95


# ---- 4. basis rotation / sign invariance of the projector similarity ----
def test_proj_sim_invariance():
    U = np.linalg.qr(rng.standard_normal((30, 3)))[0][:, :3]
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    assert abs(RP._proj_sim(U, U @ R, 3) - 1.0) < 1e-9 and abs(RP._proj_sim(U, -U, 3) - 1.0) < 1e-9


# ---- 5/6. projector basis validity (orthonormal columns) ----
def test_basis_orthonormal():
    W, dA, _, _ = _acq(0.3)
    z = dA @ W; U = RP._within_basis(z, 3); B = RP._outside_basis(np.stack([dA[i] - W @ (W.T @ dA[i]) for i in range(4)]).T, 2)
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-9) and np.allclose(B.T @ B, np.eye(2), atol=1e-9)


# ---- 7. Q computed without held-out data (signature takes only calibration repeats) ----
def test_q_uses_only_calibration():
    import inspect
    params = list(inspect.signature(RP.q_mon).parameters)
    assert params == ["W", "dA", "dB", "r_best"]                              # no held-out / test argument


# ---- 8. M4T2 operational criterion (75% rule generalizes historical 6/8) ----
def test_m4t2_operational_75pct():
    # N=8: need ceil(0.75*8)=6
    tots = [0.6] * 6 + [0.4] * 2; fcfs = [0.6] * 6 + [0.4] * 2
    assert RP.m4t2_operational_pass(tots, fcfs) is True
    tots2 = [0.6] * 5 + [0.4] * 3
    assert RP.m4t2_operational_pass(tots2, [0.6] * 8) is False                # only 5/8
    # N=10: need ceil(7.5)=8
    assert RP._ceil75(10) == 8 and RP._ceil75(8) == 6 and RP._ceil75(12) == 9


def test_quality_monitor_pass():
    assert RP.quality_monitor_pass([0.2] * 8, True) is True
    assert RP.quality_monitor_pass([0.2] * 8, False) is False                 # Holm not rejected
    assert RP.quality_monitor_pass([0.2] * 5 + [-0.1] * 3, True) is False     # only 5/8 positive


# ---- classification branches (exactly 4-test-derived) ----
def test_classify_full_pass():
    roi = {"ventral": RP.classify_roi(True, True, True), "lateral": RP.classify_roi(True, True, True)}
    assert roi["ventral"] == "M4T2_AND_QUALITY_MONITOR_REPLICATED"
    assert RP.classify_program(roi) == "INDEPENDENT_M4T2_AND_CALIBRATION_QUALITY_REPLICATION_PASS"


def test_classify_m4t2_pass_monitor_fail():
    roi = {"ventral": RP.classify_roi(True, False, False), "lateral": RP.classify_roi(True, False, False)}
    assert RP.classify_program(roi) == "INDEPENDENT_M4T2_REPLICATION_PASS_MONITOR_FAIL"


def test_classify_monitor_pass_m4t2_fail():
    roi = {"ventral": RP.classify_roi(False, True, True), "lateral": RP.classify_roi(False, True, True)}
    assert roi["ventral"] == "QUALITY_MONITOR_REPLICATED_M4T2_NOT_OPERATIONALLY_SUFFICIENT"
    assert RP.classify_program(roi) == "INDEPENDENT_FAILURE_MODE_REPLICATION_PASS_M4T2_FAIL"


def test_classify_not_confirmed():
    roi = {"ventral": RP.classify_roi(False, False, False), "lateral": RP.classify_roi(False, False, False)}
    assert RP.classify_program(roi) == "INDEPENDENT_REPLICATION_NOT_CONFIRMED"


# ---- config immutability + no threshold search ----
def test_config_and_no_search():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_16/o2_16_frozen_config.json"))
    assert cfg["immutable_history"]["O2_15"] == "REPEAT_GEOMETRY_INSTABILITY_SUPPORTED_AS_PRIMARY_FAILURE_MODE"
    assert cfg["immutable_history"]["O2_9_N_TRIALS_STAR"] == 8 and cfg["O3"] == "O3_NOT_READY"
    assert cfg["quality_inference"]["family"].startswith("2 components")
    assert cfg["quality_monitor"]["Q_MON_IN"].startswith("trace(P_IN_A P_IN_B)") and "config_sha256" in cfg
    src = Path(RP.__file__).read_text()
    for bad in ("threshold_search", "roc", "best_repeat", "argmax(", "Ridge", "Procrustes", "sklearn"):
        assert bad not in src.lower().replace("_no_", "")
