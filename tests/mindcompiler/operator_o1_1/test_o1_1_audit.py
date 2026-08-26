"""O1.1 data-free audit tests: frozen logic + committed-artifact schema/immutability guards."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmri2img.mindcompiler.operator_o1_1 import audit_logic as al

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/mindcompiler/operator_o1_1"
O1 = ROOT / "artifacts/mindcompiler/operator_o1"


def _j(name):
    return json.loads((A / name).read_text())


# ---------------------------------------------------------------- frozen logic
def test_G_is_o4_minus_o0_not_raw_o4():
    assert al.operator_gain(0.80, 0.71) == pytest.approx(0.09)
    # a high raw O4 with an equally high O0 is NOT evidence
    assert al.operator_gain(0.78, 0.78) == pytest.approx(0.0)


def test_headroom_and_normalized_gain():
    assert al.headroom(0.75) == pytest.approx(0.25)
    assert al.normalized_gain(0.80, 0.75) == pytest.approx(0.05 / 0.25)
    # epsilon floor prevents blow-up when O0 ~ 1
    assert al.normalized_gain(0.99, 0.99, eps=0.05) == pytest.approx(0.0)
    assert al.normalized_gain(1.0, 0.999) == pytest.approx(0.001 / 0.05)


def test_std_median_diff():
    assert al.std_median_diff(0.74, 0.53, 0.067) == pytest.approx((0.74 - 0.53) / 0.067, rel=1e-6)


def test_fm4_status_thresholds():
    assert al.fm4_beta_status(0.6, 0.1) == "BETA_OPERATOR_DEPENDENCE_TRACKS_RELIABILITY"
    assert al.fm4_beta_status(0.3, 0.1) == "BETA_OPERATOR_DEPENDENCE_PARTLY_TRACKS_RELIABILITY"
    assert al.fm4_beta_status(0.05, 0.1) == "BETA_OPERATOR_DEPENDENCE_NOT_EXPLAINED_BY_RELIABILITY"
    assert al.fm4_beta_status(float("nan"), float("nan")) == "BETA_OPERATOR_DEPENDENCE_INCONCLUSIVE"


def test_fm5_status():
    assert al.fm5_stability_status(3.13, 0.6) == "INSTABILITY_DOMINANT"
    assert al.fm5_stability_status(3.13, 0.38) == "INSTABILITY_CONTRIBUTES"
    assert al.fm5_stability_status(0.1, 0.1) == "STABILITY_NOT_EXPLANATORY"


def test_participant_regime():
    assert al.participant_regime(5) == "BROAD_OPERATOR_EVIDENCE"
    assert al.participant_regime(2) == "REGION_SPECIFIC_OPERATOR_EVIDENCE"
    assert al.participant_regime(0) == "WEAK_OPERATOR_EVIDENCE"


def test_o2_ready_requires_all():
    assert al.o2_ready(True, True, True, True, True) == "O2_READY_FOR_SHARED_OPERATOR_TEST"
    assert al.o2_ready(True, False, True, True, True) == "O2_NOT_READY"


# ---------------------------------------------------------------- artifact schema / guards
def test_G_recomputed_matches_committed():
    import csv
    with open(A / "participant_ROI_heterogeneity.csv", newline="") as f:
        for r in csv.DictReader(f):
            assert float(r["G"]) == pytest.approx(float(r["O4"]) - float(r["O0"]), abs=1e-9)
            assert float(r["headroom"]) == pytest.approx(1 - float(r["O0"]), abs=1e-9)


def test_no_56_row_pooling_reductions_present():
    import csv
    with open(A / "ROI_heterogeneity_summary.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 7            # per-ROI (L1), not pooled
    with open(A / "participant_heterogeneity_summary.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 8            # per-participant (L2), not pooled


def test_all_participants_and_rois_present_no_exclusion():
    import csv
    with open(A / "participant_ROI_heterogeneity.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len({r["participant"] for r in rows}) == 8 and len({r["ROI"] for r in rows}) == 7


def test_v2_hv4_case_labels_and_decision_table():
    for name in ("V2_case_audit.json", "hV4_case_audit.json"):
        c = _j(name)
        assert c["case_classification"] in {"BASELINE_LIMITED", "MEASUREMENT_LIMITED", "ESTIMATION_INSTABILITY",
                                            "VISION_DEPENDENT_SIGNAL_WEAK", "STRUCTURAL_OPERATOR_DIFFERENCE",
                                            "MULTIFACTORIAL", "UNRESOLVED"}
    table = _j("failure_mode_decision_table.json")
    assert set(table.keys()) == {"V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal"}


def test_overall_status_and_immutable_o1():
    c = _j("heterogeneity_conclusion.json")
    assert c["overall_status"] in {"OPERATOR_HETEROGENEITY_MEASUREMENT_DOMINANT", "OPERATOR_HETEROGENEITY_REGION_STRUCTURE_DOMINANT",
                                   "OPERATOR_HETEROGENEITY_STABILITY_DOMINANT", "OPERATOR_HETEROGENEITY_MULTIFACTORIAL",
                                   "OPERATOR_HETEROGENEITY_UNRESOLVED"}
    assert c["does_not_modify"] == "STIMULUS_INVARIANT_OPERATOR_PARTIAL"
    gate = _j("o1_1_gate_status.json")
    assert gate["O1_status_immutable"] == "STIMULUS_INVARIANT_OPERATOR_PARTIAL"


def test_o2_readiness_logic_matches_criteria():
    o2 = _j("o2_readiness.json")
    crit = o2["criteria"]
    allpass = all(crit[k]["pass"] for k in crit)
    assert (o2["status"] == "O2_READY_FOR_SHARED_OPERATOR_TEST") == allpass
    assert o2["critical_O2_test"] == "LEAVE_ONE_SUBJECT_OUT + HELD_OUT_STIMULUS_IDENTITIES"


def test_cross_family_increment_not_raw():
    x = _j("cross_family_failure_audit.json")
    assert "increment" in x["status"].lower()
    assert x["interpretation"].startswith("positive absolute O4 is NOT")


def test_no_operator_fitting_entry_point_in_o1_1():
    # the audit package must not import or expose operator fitting
    src = (ROOT / "src/fmri2img/mindcompiler/operator_o1_1/audit_logic.py").read_text()
    for bad in ("rrr_fit", "fit_operator", "predict_operator", "run_fold", "select_hyperparams"):
        assert bad not in src


def test_o1_source_artifacts_untouched_present():
    for f in ("participant_ROI_model_summary.csv", "operator_stability.json", "final_operator_summary.csv"):
        assert (O1 / f).exists()
