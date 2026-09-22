"""O2.16-ENG frozen design constants, runtime modes, timing schema, and immutable-hash bindings.

This module BINDS the already-frozen O2.16 / O2.16-DATA / O2.16-SEC design; it never redefines it. All
acquisition counts and structure come from the sealed configs (O2.16-DATA acquisition_protocol.json:
perception 512x3=1536, imagery 12x8=96, N=12/min 8; randomization SHA256(participant)->uint64->PCG64 with
global seed 'O2.16-DATA|frozen|randomization|v1'). No scientific inference happens here."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

PROTOCOL_VERSION = "O2.16-EXP-0.1.0"

# ---- FROZEN design (bound from sealed O2.16-DATA; DO NOT edit to change science) ----
PERCEPTION_ANCHORS = 512
PERCEPTION_PRESENTATIONS = 3
PERCEPTION_TRIALS = PERCEPTION_ANCHORS * PERCEPTION_PRESENTATIONS          # 1536
IMAGERY_IDENTITIES = 12
IMAGERY_SIMPLE = 6
IMAGERY_NATURALISTIC = 6
IMAGERY_REPEATS = 8
IMAGERY_TRIALS = IMAGERY_IDENTITIES * IMAGERY_REPEATS                      # 96
OUTER_FOLDS = 6
M4T2_M = 4
M4T2_T = 2
N_PLANNED = 12
N_CONFIRMATORY_MIN = 8
RANDOMIZATION_GLOBAL_SEED = "O2.16-DATA|frozen|randomization|v1"

# ---- Immutable upstream hashes (engineering must never change these) ----
IMMUTABLE_HASHES = {
    "o2_16_primary_config_sha256": "2da2cc791054406b42eb1896b9b5f123b5b6bacb8386e6fb0f1e19649cf18907",
    "o2_16_data_config_sha256": "804d5b21fc0c8656f9e0ef62ac4b7f412b114b7e0d3de2e328e972d75dec6444",
    "o2_16_sec_config_sha256": "3770816869ac0c8cd1cddcc5d4e4263578fcb08b327bee1031ac6d0a8aa16bfc",
}

# ---- Historical development cohort (confirmatory participants must NOT overlap) ----
HISTORICAL_N8 = tuple("subj0%d" % i for i in range(1, 9))

TASKS = ("perception", "imagery")


class Mode(str, Enum):
    SIMULATION = "SIMULATION"
    ENGINEERING_PILOT = "ENGINEERING_PILOT"
    CONFIRMATORY = "CONFIRMATORY"


PILOT_MAX_PARTICIPANTS = 2


@dataclass
class Timing:
    """Trial-timing model. All fields config-driven (no hidden constants). Confirmatory requires every field
    non-null and site-certified. Defaults are the frozen NSD-Imagery-derived TARGETS for SIMULATION only."""
    TR_s: Optional[float] = None
    perception_image_s: Optional[float] = 3.0          # frozen O2.16-DATA perception.timing.image_s
    perception_gap_s: Optional[float] = 1.0            # frozen O2.16-DATA perception.timing.gap_s
    imagery_trial_s: Optional[float] = 4.0             # ~4 s frozen from NSD-Imagery provenance
    cue_duration_s: Optional[float] = 1.0
    imagery_duration_s: Optional[float] = 1.7
    response_window_s: Optional[float] = 1.3           # ~1.3 s 2AFC decision component
    iti_s: Optional[float] = 1.0
    dummy_volumes: Optional[int] = None                # TBD_BY_MRI_FACILITY
    initial_fixation_s: Optional[float] = None
    end_fixation_s: Optional[float] = None
    inter_run_rest_s: Optional[float] = None

    def confirmatory_missing(self):
        return [k for k, v in asdict(self).items() if v is None]


@dataclass
class ExperimentConfig:
    mode: Mode = Mode.SIMULATION
    protocol_version: str = PROTOCOL_VERSION
    timing: Timing = field(default_factory=Timing)
    site_id: str = "TBD_BY_MRI_FACILITY"
    trigger_event: str = "KEYBOARD_TRIGGER"            # confirmatory must be site-certified, not assumed "5"
    bids_root: str = "TBD_BY_MRI_FACILITY"
    frame_tolerance_s: float = 0.005
    tr_tolerance_s: float = 0.05

    def to_dict(self):
        d = asdict(self); d["mode"] = self.mode.value; return d

    def protocol_hash(self) -> str:
        import json
        payload = {"protocol_version": self.protocol_version, "timing": asdict(self.timing),
                   "perception_trials": PERCEPTION_TRIALS, "imagery_trials": IMAGERY_TRIALS,
                   "outer_folds": OUTER_FOLDS, "m4t2": [M4T2_M, M4T2_T], "trigger_event": self.trigger_event}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


TBD = "TBD_BY_MRI_FACILITY"


def is_tbd(v) -> bool:
    return v is None or (isinstance(v, str) and ("TBD" in v.upper()))
