"""O2.16-AUTH-PREP certification (documentation/governance; ZERO inference). The 20 required checks. No approval
is fabricated; the confirmatory gate defaults closed; scientific state is unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DOCS = Path("docs/research/mindcompiler/o2_16_authorization")
ART = Path("artifacts/mindcompiler/o2_16_auth_prep")


def _gate():
    return json.load(open(ART / "CONFIRMATORY_START_AUTH.json"))


# 1. no approval defaults true
def test_01_no_approval_defaults_true():
    g = _gate()
    for k in ("PI_APPROVED", "ETHICS_ACTIVE", "SITE_APPROVED", "CONSENT_APPROVED", "STIMULUS_PERMISSIONS_COMPLETE",
              "SITE_CONFIG_FROZEN", "REAL_SITE_BIDS_VALIDATED", "ENGINEERING_PILOT_COMPLETE",
              "MRI_SAFETY_WORKFLOW_ACTIVE", "PARTICIPANT_INDEPENDENCE_CHECK_ACTIVE", "STAGE_A_STAGE_B_FIREWALL_ACTIVE"):
        assert g[k] is False


# 2. no fabricated ethics id
def test_02_no_fabricated_ethics_id():
    g = _gate()
    assert g["ethics_protocol_id"] == "TBD" and g["approval_institution"] == "TBD"
    eth = (DOCS / "ETHICS_APPLICATION_DRAFT.md").read_text()
    assert "AWAITING_ETHICS_APPROVAL" in eth


# 3. no participant identities in research schema
def test_03_no_participant_identities():
    dp = json.load(open(ART / "data_protection_certification.json"))
    for bad in ("git", "analysis artifacts", "BIDS dataset", "public repo"):
        assert bad in dp["identity_key_never_in"]


# 4. confirmatory gate defaults closed
def test_04_gate_closed():
    g = _gate()
    assert g["confirmatory_may_begin"] is False and g["gate_state"] == "CONFIRMATORY_GATE_CLOSED"


# 5. all human-facing docs versioned
def test_05_docs_versioned():
    md = list(DOCS.glob("*.md"))
    assert len(md) >= 21
    for f in md:
        t = f.read_text()
        assert "VERSION:" in t and "APPROVAL_STATUS:" in t and "DOC_ID:" in t


# 6. participant sheet contains voluntary withdrawal
def test_06_withdrawal_in_info_sheet():
    t = (DOCS / "PARTICIPANT_INFORMATION_SHEET.md").read_text().lower()
    assert "voluntary" in t and "withdraw" in t


# 7. participant sheet states no guaranteed benefit
def test_07_no_guaranteed_benefit():
    t = (DOCS / "PARTICIPANT_INFORMATION_SHEET.md").read_text().lower()
    assert "no guaranteed" in t and "benefit" in t


# 8. consent remains draft
def test_08_consent_draft():
    t = (DOCS / "INFORMED_CONSENT_DRAFT.md").read_text()
    assert "DRAFT_FOR_HUMAN_REVIEW" in t and "REQUIRES_INSTITUTIONAL_REVIEW" in t


# 9. compensation remains TBD
def test_09_compensation_tbd():
    t = (DOCS / "COMPENSATION_AND_PARTICIPANT_BURDEN.md").read_text()
    assert "TBD_BY_PI_AND_SITE" in t


# 10. raw MRI treated as potentially identifiable
def test_10_raw_mri_identifiable():
    dp = json.load(open(ART / "data_protection_certification.json"))
    assert dp["raw_mri_potentially_identifiable"] is True and dp["defacing_policy_documented"] is True


# 11. historical N=8 clearly development-only
def test_11_historical_development_only():
    t = (DOCS / "STUDY_PROTOCOL.md").read_text().lower()
    assert "development cohort" in t and "closed" in t


# 12-14. O2.16 / O2.16-SEC / M0 unchanged
def _self(pth):
    d = json.load(open(pth)); c2 = {k: v for k, v in d.items() if k != "config_sha256"}
    return hashlib.sha256(json.dumps(c2, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == d["config_sha256"]


def test_12_o2_16_unchanged():
    assert _self("artifacts/mindcompiler/operator_o2_16/o2_16_frozen_config.json")


def test_13_o2_16_sec_unchanged():
    assert _self("artifacts/mindcompiler/o2_16_secondary_prereg/secondary_prereg_frozen_config.json")


def test_14_m0_unchanged():
    assert _self("artifacts/mindcompiler/moonshot_triad/moonshot_triad_frozen_config.json")


# 15. no moonshot described as validated
def test_15_no_moonshot_validated():
    prereg = (DOCS / "PREREGISTRATION_EXPORT_DRAFT.md").read_text().lower()
    assert "future hypotheses" in prereg
    for f in DOCS.glob("*.md"):
        t = f.read_text().lower()
        assert "moonshot validated" not in t and "moonshots are established" not in t


# 16. stimulus permissions incomplete blocks confirmatory
def test_16_stimulus_blocks_confirmatory():
    s = json.load(open(ART / "stimulus_permission_status.json"))
    assert s["confirmatory_blocked"] is True and _gate()["STIMULUS_PERMISSIONS_COMPLETE"] is False


# 17. imagery set incomplete blocks confirmatory
def test_17_imagery_set_incomplete():
    s = json.load(open(ART / "stimulus_permission_status.json"))
    assert s["imagery_set"] == "IMAGERY_STIMULUS_SET_NOT_FINALIZED"


# 18. site config incomplete blocks confirmatory
def test_18_site_config_incomplete():
    assert _gate()["SITE_CONFIG_FROZEN"] is False
    sd = json.load(open(ART / "site_decision_register.json"))
    assert sd["all"] == "TBD_SITE_OPERATOR"


# 19. ethics incomplete blocks confirmatory
def test_19_ethics_incomplete():
    assert _gate()["ETHICS_ACTIVE"] is False


# 20. engineering pilot cannot certify science
def test_20_pilot_not_science():
    t = (DOCS / "ENGINEERING_PILOT_AUTH_CHECKLIST.md").read_text().lower()
    assert "non_confirmatory" in t and "not" in t and "hypothesis testing" in t
    assert "neural effect size is never a go/no-go" in t
