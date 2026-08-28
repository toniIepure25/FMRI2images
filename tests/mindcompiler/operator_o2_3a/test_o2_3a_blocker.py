"""O2.3A data-free tests: data-availability blocker integrity + immutability + no-B1/C3R guards."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/mindcompiler/operator_o2_3a"
O2 = ROOT / "artifacts/mindcompiler/operator_o2"; O22 = ROOT / "artifacts/mindcompiler/operator_o2_2"


def _j(n):
    return json.loads((A / n).read_text())


def test_execution_status_is_data_unavailable():
    s = _j("scientific_status.json")
    assert s["execution_status"] == "O2_3A_CORE_DATA_UNAVAILABLE"
    assert s["orientation_status"] == "CORE_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE"
    assert "not a scientific failure" in s["reason"].lower() or "not a scientific" in s["reason"].lower()


def test_o2_o3_immutable_not_changed():
    s = _j("scientific_status.json")
    assert s["immutable"]["O2"] == "SHARED_OPERATOR_PARTIAL" and s["immutable"]["O3"] == "O3_NOT_READY"
    # committed O2/O2.2 statuses untouched
    assert json.loads((O2 / "scientific_status.json").read_text())["scientific_status"] == "SHARED_OPERATOR_PARTIAL"
    assert json.loads((O22 / "o2_2_gate_status.json").read_text())["immutable_O3"] == "O3_NOT_READY"


def test_inventory_records_b0_missing_and_b1_prohibited():
    inv = _j("core_data_inventory.json")
    assert "UNAVAILABLE" in inv["verdict"]
    assert inv["available"]["local_machine"]["subj02_08"].startswith("ABSENT")
    # pod b3 present but flagged wrong-version / not admissible / not touched
    pod = inv["available"]["pod_work"]["nsd_betas_download"]
    assert "b3" in pod and ("WRONG" in pod or "NOT admissible" in pod) and "NOT touched" in pod


def test_no_b1_and_no_fitting_and_c3r_untouched():
    p = _j("execution_provenance.json")
    assert p["no_b1_used"] is True and p["no_fitting_performed"] is True
    assert p["no_target_imagery_touched"] is True and p["c3r_data_untouched"] is True
    assert p["no_o2_o3_status_change"] is True and p["halted_at"].startswith("PART C")


def test_methodology_frozen_for_future():
    cfg = _j("o2_3a_frozen_config.json")
    assert "config_sha256" in cfg and cfg["beta"].startswith("B0 lineage only")
    assert cfg["primary_rois"] == ["ventral", "lateral"]
    assert "betas_fithrf_GLMdenoise_RR (b3 / B1)" in _j("core_data_contract.json")["prohibited_beta"]
    assert _j("execution_provenance.json")["methodology_frozen_for_future_execution"] is True


def test_next_gate_records_blocker_not_failure():
    n = _j("next_gate_decision.json")
    assert n["blocker"] == "O2_3A_CORE_DATA_UNAVAILABLE" and n["not_a_scientific_failure"] is True
    assert any("ACQUIRE" in o["option"] for o in n["admissible_next_options"])
    assert any("b3" in x for x in n["not_admissible"]) and "O3_NOT_READY" in n["O3"]


def test_no_result_artifacts_fabricated():
    # blocker path must NOT contain fabricated Phase-B/C result artifacts
    for forbidden in ("target_residual_orientation_results.csv", "anchor_common_space_validation.csv",
                      "core_imagery_retention.csv", "residual_null_summary.json"):
        assert not (A / forbidden).exists(), f"fabricated result artifact present: {forbidden}"
