"""O2.1 data-free audit tests: frozen logic + committed-artifact guards + O2 immutability."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2_1 import audit_logic as al

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/mindcompiler/operator_o2_1"; O2 = ROOT / "artifacts/mindcompiler/operator_o2"


def _j(n):
    return json.loads((A / n).read_text())


# ---------------- frozen logic
def test_contrast_retention_bounds_and_full():
    rng = np.random.default_rng(0); W, _ = np.linalg.qr(rng.standard_normal((10, 3)))
    d_in = (W @ rng.standard_normal(3))                     # lies entirely in span(W)
    assert al.contrast_retention(d_in, W) == pytest.approx(1.0, abs=1e-9)
    d_out = rng.standard_normal(10)
    assert 0.0 <= al.contrast_retention(d_out, W) <= 1.0


def test_shared_contrast_relative_error():
    dy = np.array([1.0, 2.0]); assert al.shared_contrast_relative_error(dy, dy) == pytest.approx(0.0)
    assert al.shared_contrast_relative_error(np.zeros(2), dy) == pytest.approx(1.0)


def test_smd_labels():
    assert al.smd_label(al.std_median_diff(0.55, 0.10, 0.1)) == "STRONG_DESCRIPTIVE_SEPARATION"
    assert al.smd_label(0.6) == "MODERATE" and al.smd_label(0.2) == "WEAK"


def test_decomposition_confidence_logic():
    assert al.decomposition_confidence(0.30, -0.01) == "DECOMPOSITION_CONFIDENCE_LOW"
    assert al.decomposition_confidence(0.30, 0.01) == "DECOMPOSITION_CONFIDENCE_MODERATE"
    assert al.decomposition_confidence(0.10, 0.0001) == "DECOMPOSITION_CONFIDENCE_MODERATE"
    assert al.decomposition_confidence(0.10, 0.02) == "DECOMPOSITION_CONFIDENCE_HIGH"


def test_o3_ready_requires_all_and_confidence():
    assert al.o3_ready(True, True, True, "DECOMPOSITION_CONFIDENCE_MODERATE", True) == "O3_READY_FOR_MULTI_STATE_FEASIBILITY"
    assert al.o3_ready(True, True, False, "DECOMPOSITION_CONFIDENCE_MODERATE", True) == "O3_NOT_READY"
    assert al.o3_ready(True, True, True, "DECOMPOSITION_CONFIDENCE_LOW", True) == "O3_NOT_READY"


def test_normalized_shared_gain_floor():
    assert al.normalized_shared_gain(0.01, 0.99) == pytest.approx(0.01 / 0.05)


# ---------------- artifact guards
def test_cross_state_gap_recomputes_and_present():
    with open(A / "cross_state_projection_retention.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 144
    for r in rows[:20]:
        assert float(r["cross_state_retention_gap"]) == pytest.approx(
            float(r["vision_contrast_retention"]) - float(r["imagery_contrast_retention"]), abs=1e-9)


def test_oracle_not_used_as_predictor():
    # oracle uses target imagery (capacity diagnostic); it must not appear as a G_shared source
    cols = list(csv.DictReader(open(A / "common_space_oracle.csv")).fieldnames)
    assert "oracle_pattern_r" in cols and "r_Tshared" not in cols


def test_low_K_not_the_cause():
    lk = _j("low_K_audit.json")
    assert lk["status"] == "LOW_K_GLOBAL_BOTTLENECK_NOT_SUPPORTED"       # ventral transfers at smallest K


def test_overall_status_and_o3_not_ready():
    c = _j("o2_1_conclusion.json")
    assert c["overall_bottleneck_status"] == "O2_TRANSFER_LIMIT_CROSS_STATE_SPACE_DOMINANT"
    assert c["immutable"] == "SHARED_OPERATOR_PARTIAL (unchanged)"
    o3 = _j("o3_readiness.json")
    assert o3["status"] == "O3_NOT_READY" and o3["criteria"]["C_shared_contrast_present"] is False


def test_o2_status_immutable_not_changed():
    g = _j("o2_1_gate_status.json")
    assert g["immutable_O2_status"] == "SHARED_OPERATOR_PARTIAL"
    # committed O2 scientific status untouched
    assert json.loads((O2 / "scientific_status.json").read_text())["scientific_status"] == "SHARED_OPERATOR_PARTIAL"


def test_frozen_srm_replay_exact():
    assert _j("execution_provenance.json")["frozen_srm_replay_exact"] is True
    assert _j("o2_1_gate_status.json")["frozen_srm_replay_max_err"] < 1e-10


def test_no_refit_no_alt_common_space():
    p = _j("execution_provenance.json")
    assert p["no_refit"] and p["no_alternative_common_space"] and p["no_K_change"] and p["no_status_change"]


def test_delta_boundary_concern_recorded():
    d = _j("delta_regularization_audit.json")
    assert d["DELTA_UPPER_BOUNDARY_RATE_overall"] >= 0.25 and d["status"] == "DELTA_REGULARIZATION_BOUNDARY_CONCERN"


def test_no_operator_fitting_in_audit_logic():
    src = (ROOT / "src/fmri2img/mindcompiler/operator_o2_1/audit_logic.py").read_text()
    for bad in ("det_srm", "fit_shared_ridge", "new_subject_W", "evaluate_outer_cell"):
        assert bad not in src
