"""O2.3C data-free gate tests: frozen-config integrity, blocker verdict, immutability, no-fit leakage.

These assert the FROZEN methodology and the data-availability blocker record only. No NSD data is
touched (the gate halted at the availability precondition), so the suite is fully offline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/mindcompiler/operator_o2_3c"
O2 = ROOT / "artifacts/mindcompiler/operator_o2"
O23A = ROOT / "artifacts/mindcompiler/operator_o2_3a"


def _j(base: Path, n: str):
    return json.loads((base / n).read_text())


def test_frozen_sha_matches_content():
    cfg = _j(A, "o2_3c_frozen_config.json")
    stored = cfg.pop("frozen_methodology_sha")
    recomputed = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    assert recomputed == stored == "528f23eb"


def test_frozen_config_core_parameters():
    cfg = _j(A, "o2_3c_frozen_config.json")
    assert cfg["anchor_family"] == "RESTING_STATE_FUNCTIONAL_CONNECTIVITY"
    assert cfg["common_space_method"] == "DETERMINISTIC_CONNECTIVITY_SRM"
    assert cfg["one_anchor_family_only"] is True and cfg["no_method_leaderboard"] is True
    assert cfg["reference_space"] == "native_func1pt8mm" and cfg["no_resampling"] is True
    assert cfg["target_parcellation"]["atlas"] == "FreeSurfer_DesikanKilliany_aparc"
    assert cfg["target_parcellation"]["n_cortical_parcels"] == 68
    assert cfg["qc"]["FD_censor_threshold_mm"] == 0.25
    assert cfg["qc"]["min_usable_rest_minutes_per_subject"] == 30
    assert cfg["temporal_cleaning"]["global_signal_regression"] is False   # NO GSR
    assert cfg["fitting_exclusions"]["no_target_imagery_in_fitting"] is True
    assert cfg["fitting_exclusions"]["no_target_perception_in_fitting"] is True
    assert cfg["residual_under_test"]["source_gate"] == "O2.2"
    assert cfg["residual_under_test"]["imported_unchanged"] is True


def test_blocker_verdict_is_product_incompatible():
    s = _j(A, "scientific_status.json")
    assert s["execution_status"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"
    assert s["orientation_status"] == "CONNECTIVITY_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE"
    assert s["orientation_test_performed"] is False
    assert s["blocker_is_technical_not_scientific"] is True


def test_inventory_supports_verdict_not_scarcity():
    inv = _j(A, "rest_data_inventory.json")
    f = inv["findings"]
    # no prepared native resting timeseries anywhere
    assert inv["findings"]["prepared_native_func1pt8mm_rest_timeseries_runs"] == 0
    assert all(v["native_func1pt8mm_rest_timeseries"] == 0 for v in inv["per_subject"].values())
    # restingbetas are single-volume, not timeseries
    assert f["restingbetas_product"]["usable_as_timeseries_for_connectivity"] is False
    assert f["restingbetas_product"]["nifti_ndim_checked"] == 3
    # raw exists AND meets 30 min -> blocker is NOT scarcity
    assert f["raw_bids_task_rest"]["present"] is True
    assert all(v["meets_30min_in_raw"] for v in inv["per_subject"].values())
    assert inv["gate_evaluation"]["raw_only_and_preprocessing_forbidden_partF6"] is True
    assert inv["verdict"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"


def test_dk_parcellation_not_native_volume():
    f = _j(A, "rest_data_inventory.json")["findings"]["desikan_killiany_aparc"]
    assert f["native_func1pt8mm_volume_present"] is False
    assert f["surface_and_anatomical_present"] is True


def test_leakage_vacuous_no_fit():
    c = _j(A, "o2_3c_contracts.json")["leakage_contract"]
    assert c["no_target_imagery_in_any_fit"] is True
    assert c["no_target_perception_in_any_fit"] is True
    assert c["status"] == "VACUOUS_NO_FIT_PERFORMED"
    p = _j(A, "execution_provenance.json")
    assert p["no_fitting_performed"] is True and p["o2_2_residual_modified"] is False
    assert p["no_credentials_used"] is True and p["no_b1_b3_files_used"] is True


def test_carried_immutable_statuses():
    s = _j(A, "scientific_status.json")["immutable"]
    assert s["TRACK_R"] == "INDEPENDENT_METHOD_REPRODUCTION_PARTIAL"
    assert s["O1"] == "STIMULUS_INVARIANT_OPERATOR_PARTIAL"
    assert s["O2"] == "SHARED_OPERATOR_PARTIAL"
    assert s["O2.3A_execution"] == "O2_3A_CORE_ANCHOR_FEASIBILITY_PASS"
    assert s["O2.3A_orientation"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
    assert s["O3"] == "O3_NOT_READY"


def test_prior_gate_status_files_unmodified():
    # O2 and O2.3A ground-truth statuses must remain exactly as the earlier gates left them
    assert _j(O2, "scientific_status.json")["scientific_status"] == "SHARED_OPERATOR_PARTIAL"
    a = _j(O23A, "scientific_status.json")
    assert a["execution_status"] == "O2_3A_CORE_ANCHOR_FEASIBILITY_PASS"
    assert a["orientation_status"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
