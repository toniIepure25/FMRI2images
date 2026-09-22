"""Stimulus-presentation abstraction. Task code depends only on this interface, never on a specific framework.

- SimulationPresenter: no external dependency; produces synthetic flip timestamps. CI-safe and used for all
  automated tests and SIMULATION mode.
- PsychoPyPresenter: a documented, lazily-imported adapter STUB for real-site visual presentation. It is NOT
  exercised in CI and must NOT be treated as tested; a real site integrates and validates flip timing on the
  actual display before any confirmatory use. We do not silently substitute browser timing for scanner
  presentation.
"""
from __future__ import annotations

from . import timing as T


class Presenter:
    def show(self, stimulus_ref, requested_onset_s, duration_s):
        """Return (actual_flip_s, actual_offset_s, dropped_frames)."""
        raise NotImplementedError

    def close(self):
        pass


class SimulationPresenter(Presenter):
    """Synthetic, deterministic 'flip' timing: honours requested onset exactly minus a tiny fixed model latency."""

    MODEL_LATENCY_S = 0.001

    def __init__(self, refresh_hz=60.0):
        self.refresh_hz = refresh_hz

    def show(self, stimulus_ref, requested_onset_s, duration_s):
        flip = requested_onset_s + self.MODEL_LATENCY_S
        offset = flip + duration_s
        return flip, offset, 0


class PsychoPyPresenter(Presenter):
    """STUB adapter. Requires PsychoPy at the real site; not imported or tested in CI. Real flip-timing
    validation is a site-integration step (see the operator guide)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PsychoPyPresenter is a real-site stub. Install and pin PsychoPy at the MRI site, then validate "
            "visual flip timing, keyboard/button response, and scanner trigger on the actual display before "
            "any confirmatory use. CI and SIMULATION use SimulationPresenter.")
