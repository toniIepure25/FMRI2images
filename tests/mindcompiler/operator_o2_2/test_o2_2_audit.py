"""O2.2 data-free tests: frozen geometry logic + committed-artifact guards + immutability."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2_2 import audit_logic as al

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/mindcompiler/operator_o2_2"
O2 = ROOT / "artifacts/mindcompiler/operator_o2"; O21 = ROOT / "artifacts/mindcompiler/operator_o2_1"


def _j(n):
    return json.loads((A / n).read_text())


# ---------------- frozen geometry logic
def test_retention_full_and_bounds():
    rng = np.random.default_rng(0); B, _ = np.linalg.qr(rng.standard_normal((12, 4)))
    d_in = B @ rng.standard_normal(4)
    assert al.retention(d_in, B) == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= al.retention(rng.standard_normal(12), B) <= 1.0


def test_cross_state_overlap_bounds_and_identical():
    rng = np.random.default_rng(1); B, _ = np.linalg.qr(rng.standard_normal((12, 3)))
    assert al.cross_state_overlap(B, B) == pytest.approx(1.0, abs=1e-9)
    B2, _ = np.linalg.qr(rng.standard_normal((12, 3)))
    assert 0.0 <= al.cross_state_overlap(B, B2) <= 1.0


def test_regime_classification():
    assert al.classify_regime(0.55, 0.10, 0.30, 0.6, 0.7) == "A_SRM_DIMENSION_TRUNCATION_DOMINANT"
    assert al.classify_regime(0.25, 0.10, 0.10, 0.6, 0.7) == "B_STATE_SPECIFIC_SHARED_GEOMETRY_OUTSIDE_VISUAL_SPAN"
    assert al.classify_regime(0.25, 0.10, 0.10, 0.6, 0.10) == "C_SUBJECT_SPECIFIC_IMAGERY_GEOMETRY"
    assert al.classify_regime(0.25, 0.10, 0.10, 0.10, 0.7) == "D_IMAGERY_MEASUREMENT_LIMITED"


def test_overall_status_multiregime_and_single():
    assert al.overall_status({"ventral": "B_STATE_SPECIFIC_SHARED_GEOMETRY_OUTSIDE_VISUAL_SPAN",
                              "lateral": "B_STATE_SPECIFIC_SHARED_GEOMETRY_OUTSIDE_VISUAL_SPAN",
                              "parietal": "A_SRM_DIMENSION_TRUNCATION_DOMINANT"}) == "CROSS_STATE_TRANSPORT_MULTIREGIME"
    assert al.overall_status({"a": "B_STATE_SPECIFIC_SHARED_GEOMETRY_OUTSIDE_VISUAL_SPAN"}) == "CROSS_STATE_TRANSPORT_SHARED_IMAGERY_SPACE_OUTSIDE_VISION"


# ---------------- artifact guards
def test_srm_truncation_loss_recomputes():
    with open(A / "nested_subspace_retention.csv", newline="") as f:
        for r in list(csv.DictReader(f))[:20]:
            assert float(r["SRM_truncation_loss"]) == pytest.approx(float(r["R_Y_VISFULL"]) - float(r["R_Y_SRM"]), abs=1e-9)
            assert float(r["cross_state_span_loss"]) == pytest.approx(1 - float(r["R_Y_VISFULL"]), abs=1e-9)


def test_R_Y_SRM_reproduces_O2_1():
    # frozen O2 SRM object replays exactly: ventral imagery retention ~0.096 as in O2.1
    import statistics
    with open(A / "nested_subspace_retention.csv", newline="") as f:
        vent = [float(r["R_Y_SRM"]) for r in csv.DictReader(f) if r["ROI"] == "ventral"]
    assert statistics.median(vent) == pytest.approx(0.096, abs=0.01)


def test_o1_perp_exceeds_parallel_recorded():
    a = _j("O1_discarded_component_audit.json")["per_ROI"]
    assert all(a[r]["perp_exceeds_parallel"] for r in ("ventral", "lateral", "parietal"))


def test_identifiability_distinction():
    idn = _j("target_state_identifiability.json")
    assert idn["status"] == "TARGET_STATE_ORIENTATION_NOT_IDENTIFIABLE_FROM_CURRENT_VISION_ONLY_DATA"
    assert "shared geometry" in idn["distinction"].lower() and "identifiable" in idn["distinction"].lower()


def test_overall_and_future_gate():
    c = _j("o2_2_conclusion.json")
    assert c["overall_status"] in {"CROSS_STATE_TRANSPORT_MULTIREGIME", "CROSS_STATE_TRANSPORT_SHARED_IMAGERY_SPACE_OUTSIDE_VISION",
                                   "CROSS_STATE_TRANSPORT_SUBJECT_SPECIFIC_IMAGERY_SPACE", "CROSS_STATE_TRANSPORT_SRM_TRUNCATION_DOMINANT",
                                   "CROSS_STATE_TRANSPORT_MEASUREMENT_LIMITED", "CROSS_STATE_TRANSPORT_UNRESOLVED"}
    assert _j("future_gate_decision.json")["future_gate_decision"] in {"O2_3_RICHER_VISION_SPACE_FEASIBLE",
        "O2_3_STATE_AWARE_SHARED_SPACE_FEASIBLE", "O2_3_SUBJECT_SPECIFIC_STATE_MAPPING_REQUIRED", "O2_3_MEASUREMENT_BOTTLENECK", "O2_3_NO_CLEAR_REMEDY"}


def test_o2_o3_immutable_not_changed():
    g = _j("o2_2_gate_status.json")
    assert g["immutable_O2"] == "SHARED_OPERATOR_PARTIAL" and g["immutable_O3"] == "O3_NOT_READY"
    assert json.loads((O2 / "scientific_status.json").read_text())["scientific_status"] == "SHARED_OPERATOR_PARTIAL"
    assert json.loads((O21 / "o2_1_gate_status.json").read_text())["o3_readiness"] == "O3_NOT_READY"


def test_frozen_o2_srm_replay_exact_and_no_refit():
    p = _j("execution_provenance.json")
    assert p["frozen_o2_srm_replay_exact"] is True and p["no_refit"] and p["no_alt_common_space"] and p["no_status_change"]


def test_no_operator_fitting_in_audit_logic():
    src = (ROOT / "src/fmri2img/mindcompiler/operator_o2_2/audit_logic.py").read_text()
    for bad in ("det_srm", "fit_shared_ridge", "new_subject_W", "fit_operator", "select_hyperparams"):
        assert bad not in src


def test_rdm_sharedness_participant_unit():
    d = _j("residual_RDM_summary.json")
    for r in d["detail"]:
        assert len(r["values"]) == 8                        # one LOSO value per participant, not 28 pairs
