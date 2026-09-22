"""Session state machine:
PRECHECK -> PARTICIPANT_LOAD -> STIMULUS_HASH_CHECK -> DEVICE_CHECK -> SCANNER_CHECK -> RUN_READY ->
WAIT_TRIGGER -> RUN_ACTIVE -> RUN_QC -> RUN_SEALED -> SESSION_COMPLETE. Any critical failure -> SESSION_ABORTED_SAFE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from . import config as C
from . import mode_guard as MG
from . import stimuli as S
from . import participant_registry as PR

STATES = ["PRECHECK", "PARTICIPANT_LOAD", "STIMULUS_HASH_CHECK", "DEVICE_CHECK", "SCANNER_CHECK", "RUN_READY",
          "WAIT_TRIGGER", "RUN_ACTIVE", "RUN_QC", "RUN_SEALED", "SESSION_COMPLETE"]
ABORT = "SESSION_ABORTED_SAFE"


@dataclass
class SessionResult:
    reached: List[str] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""
    final_state: str = ""


def precheck_session(mode: C.Mode, participant_id, auth, cfg, manifest, device_ok=True, scanner_ok=True):
    """Advance the state machine through the pre-run checks; abort-safe on any critical failure."""
    res = SessionResult()

    def step(name):
        res.reached.append(name); res.final_state = name

    def abort(reason):
        res.aborted = True; res.abort_reason = reason; res.final_state = ABORT
        res.reached.append(ABORT)
        return res

    step("PRECHECK")
    step("PARTICIPANT_LOAD")
    if not PR.valid_for_mode(participant_id, mode):
        return abort("participant_id_invalid_for_mode:%s" % participant_id)
    if mode == C.Mode.CONFIRMATORY and not MG.participant_independence_ok(participant_id):
        return abort("participant_overlaps_historical")
    step("STIMULUS_HASH_CHECK")
    mrep = manifest
    if not mrep.get("ok") and mode == C.Mode.CONFIRMATORY:
        return abort("stimulus_manifest_not_ok")
    step("DEVICE_CHECK")
    if not device_ok:
        return abort("device_check_failed")
    step("SCANNER_CHECK")
    if not scanner_ok:
        return abort("scanner_check_failed")
    if mode == C.Mode.CONFIRMATORY:
        try:
            MG.confirmatory_precheck(mode, auth, cfg, mrep, participant_id)
        except MG.ConfirmatoryAbort as e:
            return abort(str(e))
    step("RUN_READY")
    return res
