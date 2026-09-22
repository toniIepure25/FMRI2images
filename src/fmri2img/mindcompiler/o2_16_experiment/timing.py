"""Clock + frame-timing policy. Experimental timing uses a high-resolution MONOTONIC clock; wall-clock UTC is
metadata only and never used to derive trial durations. Frame records capture requested vs actual onset where a
real presentation backend provides flip timestamps (simulation supplies synthetic ones)."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


def monotonic() -> float:
    return time.perf_counter()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FrameRecord:
    requested_onset_s: float
    actual_flip_s: float
    actual_offset_s: float
    frame_duration_s: float
    dropped_frames: int
    deviation_s: float
    exceeds_tolerance: bool

    def to_dict(self):
        return asdict(self)


def make_frame_record(requested_onset, actual_flip, actual_offset, dropped_frames, tolerance_s):
    dur = actual_offset - actual_flip
    dev = abs(actual_flip - requested_onset)
    return FrameRecord(requested_onset, actual_flip, actual_offset, dur, int(dropped_frames), dev,
                       bool(dev > tolerance_s))
