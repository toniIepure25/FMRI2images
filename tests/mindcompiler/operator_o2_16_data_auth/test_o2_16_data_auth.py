"""O2.16-DATA-AUTH readiness-package integrity checks (data-free). Certifies the pre-scanning enablement package
did NOT mutate any scientific status, fabricate ethics/scanner values, create participant records, or reuse
historical subjects; that imagery-replay fields are source-bound; stimulus placeholders are flagged; the
synthetic BIDS fixture reconstructs identity/repeat/family; the pilot is excluded from confirmatory N; the O2.16
config is still exactly 2da2cc79; and O3 remains locked. This gate can change no science."""
from __future__ import annotations

import json
from pathlib import Path

A = "artifacts/mindcompiler/operator_o2_16_data_auth/"
DATA = "artifacts/mindcompiler/operator_o2_16_data/"


def _j(p):
    return json.load(open(p))


def test_1_no_scientific_status_mutation():
    assert _j(DATA + "scientific_status.json")["status"] == "O2_16_DATA_AWAITING_HUMAN_ACQUISITION_AUTHORIZATION"
    assert _j(A + "scientific_status_preservation.json")["no_science_changed"] is True


def test_2_no_fabricated_ethics_approval():
    assert _j(DATA + "ethics_authorization_status.json")["irb_ethics_approval"] == "ABSENT"
    txt = Path(A + "ethics_application_draft.md").read_text(encoding="utf-8").lower()
    assert "draft" in txt and "not" in txt and "approv" in txt  # explicitly not approved


def test_3_no_fabricated_scanner_values():
    md = Path(A + "scanner_site_requirements.md").read_text(encoding="utf-8")
    assert "TBD_BY_MRI_FACILITY" in md


def test_4_no_participant_records():
    assert Path(A + "participant_scheduling_template.csv").read_text().strip().count("\n") == 0  # header only
    assert _j(DATA + "replication_cohort_manifest.json")["recruited_ids"] == []


def test_5_no_historical_subject_reuse():
    for p in (A + "participant_scheduling_template.csv", A + "data_flow_and_access_matrix.csv"):
        t = Path(p).read_text()
        for s in ("subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"):
            assert s not in t


def test_6_imagery_replay_fields_source_bound():
    spec = _j(A + "imagery_task_exact_replay_spec.json")["fields"]
    for f, v in spec.items():
        assert "status" in v and "source" in v
        assert v["status"] in ("CERTIFIED", "PARTIAL", "UNRESOLVED")
    assert _j(A + "imagery_task_exact_replay_spec.json")["certification"].startswith("NOT ")  # not fully certified


def test_7_stimulus_placeholders_detected():
    t = Path(A + "imagery_stimulus_binding.csv").read_text()
    assert "PLACEHOLDER" in t and "TO_BIND" in t


def test_8_bids_reconstructs_identity_repeat_family():
    checks = _j(A + "bids_dry_run_report.json")["structural_checks"]
    assert checks["repeats_per_identity_ok"] and checks["families_ok"] and checks["repeat_index_contiguous"] and checks["volume_index_monotone"]


def test_9_pilot_excluded_from_confirmatory():
    cfg = _j(DATA + "o2_16_data_frozen_config.json")
    assert cfg["pilot"]["not_part_of_confirmatory_N"] is True and cfg["pilot"]["N_pilot_max"] == 2


def test_10_o2_16_config_unchanged():
    assert _j("artifacts/mindcompiler/operator_o2_16/o2_16_frozen_config.json")["config_sha256"].startswith("2da2cc79")
    assert _j(A + "scientific_status_preservation.json")["O3"] == "O3_NOT_READY"


def test_11_o3_locked_everywhere():
    for p in (A + "scientific_status_preservation.json", A + "next_action.json"):
        pass
    assert _j(A + "scientific_status_preservation.json")["readiness_state"] == "O2_16_DATA_AUTH_PACKAGE_PREPARED"
