"""Participant response-input abstraction. Task code never embeds device-specific logic; it calls this
interface. SimulatedResponseDevice generates deterministic synthetic responses (with configurable missing/
multiple) for SIMULATION/CI. Real adapters (MRI button box, keyboard) are site-configured."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class Response:
    key: Optional[str]
    response_time_s: Optional[float]
    correct: Optional[bool]
    missing: bool
    multiple: bool


class ResponseDevice:
    def poll(self, trial_key: str, correct_key: Optional[str], window_s: float) -> Response:
        raise NotImplementedError


class SimulatedResponseDevice(ResponseDevice):
    def __init__(self, participant_id, session_id, miss_rate=0.0, multi_rate=0.0, keys=("1", "2")):
        self.participant_id = participant_id
        self.session_id = session_id
        self.miss_rate = miss_rate
        self.multi_rate = multi_rate
        self.keys = keys

    def _u(self, trial_key):
        d = hashlib.sha256(("%s|%s|%s" % (self.participant_id, self.session_id, trial_key)).encode()).hexdigest()
        return int(d[:8], 16) / 0xFFFFFFFF

    def poll(self, trial_key, correct_key, window_s):
        u = self._u(trial_key)
        if u < self.miss_rate:
            return Response(None, None, None, missing=True, multiple=False)
        multiple = ((u * 1000) % 1.0) < self.multi_rate
        key = self.keys[int(u * len(self.keys)) % len(self.keys)]
        rt = 0.15 + (u * max(window_s - 0.2, 0.1))
        correct = None if correct_key is None else (key == correct_key)
        return Response(key, round(rt, 4), correct, missing=False, multiple=multiple)
