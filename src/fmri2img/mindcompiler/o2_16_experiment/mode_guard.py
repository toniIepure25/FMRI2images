"""Runtime-mode guards. CONFIRMATORY fails CLOSED: it starts only if an authorization manifest explicitly
grants every required approval, the site is certified, stimuli are real, timing is complete, and the
participant is independent of the historical cohort. No --force override exists."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import config as C


class ConfirmatoryAbort(RuntimeError):
    pass


@dataclass
class AuthorizationManifest:
    ethics_active: bool = False
    participant_consent_complete: bool = False
    mri_safety_complete: bool = False
    site_authorization_complete: bool = False
    stimulus_authorization_complete: bool = False
    participant_independent: bool = False
    ethics_protocol_id: Optional[str] = None
    approval_institution: Optional[str] = None
    approval_valid_from: Optional[str] = None
    approval_valid_to: Optional[str] = None
    consent_version: Optional[str] = None
    data_processing_basis: Optional[str] = None


REQUIRED_BOOL = ("ethics_active", "participant_consent_complete", "mri_safety_complete",
                 "site_authorization_complete", "stimulus_authorization_complete", "participant_independent")
REQUIRED_ETHICS_VALUES = ("ethics_protocol_id", "approval_institution", "approval_valid_from",
                          "approval_valid_to", "consent_version", "data_processing_basis")


def ethics_guard(auth: AuthorizationManifest):
    """Returns (ok, missing). CONFIRMATORY requires real (non-null, non-TBD) ethics values."""
    missing = [k for k in REQUIRED_ETHICS_VALUES if C.is_tbd(getattr(auth, k))]
    return len(missing) == 0, missing


def confirmatory_precheck(mode: C.Mode, auth: AuthorizationManifest, cfg: C.ExperimentConfig,
                          manifest_report: dict, participant_id: str):
    """Fail-closed gate. Raises ConfirmatoryAbort on any unmet requirement. Returns a report on pass."""
    if mode != C.Mode.CONFIRMATORY:
        return {"mode": mode.value, "confirmatory": False, "checked": False}
    reasons = []
    for k in REQUIRED_BOOL:
        if getattr(auth, k) is not True:
            reasons.append("authorization_false:%s" % k)
    eok, emiss = ethics_guard(auth)
    if not eok:
        reasons += ["ethics_value_missing:%s" % m for m in emiss]
    if C.is_tbd(cfg.site_id):
        reasons.append("site_id_TBD")
    if C.is_tbd(cfg.trigger_event) or cfg.trigger_event == "KEYBOARD_TRIGGER":
        reasons.append("scanner_trigger_uncertified")
    if C.is_tbd(cfg.bids_root):
        reasons.append("bids_root_TBD")
    tmiss = cfg.timing.confirmatory_missing()
    if tmiss:
        reasons.append("timing_incomplete:%s" % ",".join(tmiss))
    if participant_id in C.HISTORICAL_N8:
        reasons.append("participant_overlaps_historical:%s" % participant_id)
    if not manifest_report.get("ok", False):
        reasons.append("stimulus_manifest_not_ok")
    if manifest_report.get("n_placeholder", 0) > 0:
        reasons.append("stimulus_placeholder_survives")
    if reasons:
        raise ConfirmatoryAbort("CONFIRMATORY fail-closed: " + "; ".join(reasons))
    return {"mode": "CONFIRMATORY", "confirmatory": True, "checked": True, "pass": True}


def participant_independence_ok(participant_id: str) -> bool:
    return participant_id not in C.HISTORICAL_N8


def watermark(mode: C.Mode):
    if mode == C.Mode.ENGINEERING_PILOT:
        return "NON_CONFIRMATORY_ENGINEERING_PILOT"
    if mode == C.Mode.SIMULATION:
        return "SIMULATION_NO_BOLD"
    return "CONFIRMATORY"
