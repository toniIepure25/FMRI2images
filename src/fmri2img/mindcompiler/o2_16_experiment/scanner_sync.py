"""Scanner-trigger abstraction. The real scanner input adapter is site-configured; here we provide a base
interface plus a deterministic SimulatedTrigger for SIMULATION/CI. Confirmatory mode must NOT assume trigger
key '5' unless the site config certifies it. Records a BIDS-compatible scansync log with anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import timing as T


@dataclass
class TriggerEvent:
    trigger_count: int
    monotonic_s: float
    wall_utc: str
    est_volume_index: int
    inter_trigger_interval_s: Optional[float]
    anomaly: Optional[str]


class ScannerSync:
    """Base trigger interface. Subclasses implement wait_for_scanner() and _next_pulse()."""

    def __init__(self, tr_s: Optional[float], tr_tolerance_s: float = 0.05):
        self.tr_s = tr_s
        self.tr_tolerance_s = tr_tolerance_s
        self.events: List[TriggerEvent] = []
        self._count = 0
        self._last_m: Optional[float] = None

    # --- to be overridden ---
    def wait_for_scanner(self):
        raise NotImplementedError

    def _pulse_time(self) -> float:
        raise NotImplementedError

    # --- shared ---
    def register_trigger(self):
        m = self._pulse_time()
        iti = None if self._last_m is None else (m - self._last_m)
        anomaly = None
        if self.tr_s is not None and iti is not None:
            if abs(iti - self.tr_s) > self.tr_tolerance_s:
                anomaly = "TR_DEVIATION"
            if iti < self.tr_s * 0.5:
                anomaly = "DUPLICATE_TRIGGER"
        self._count += 1
        ev = TriggerEvent(self._count, m, T.utc_now(), self._count - 1, iti, anomaly)
        self.events.append(ev)
        self._last_m = m
        return ev

    def count_volume(self) -> int:
        return self._count

    def detect_missing_trigger(self, expected_count: int):
        return max(0, expected_count - self._count)

    def detect_duplicate_trigger(self):
        return [e.trigger_count for e in self.events if e.anomaly == "DUPLICATE_TRIGGER"]

    def compare_tr(self):
        if self.tr_s is None:
            return {"tr_known": False}
        itis = [e.inter_trigger_interval_s for e in self.events if e.inter_trigger_interval_s is not None]
        if not itis:
            return {"tr_known": True, "n_intervals": 0}
        import statistics
        return {"tr_known": True, "n_intervals": len(itis), "median_iti_s": statistics.median(itis),
                "max_abs_dev_s": max(abs(x - self.tr_s) for x in itis),
                "within_tolerance": all(abs(x - self.tr_s) <= self.tr_tolerance_s for x in itis)}

    def scansync_rows(self):
        return [{"trigger_count": e.trigger_count, "monotonic_s": round(e.monotonic_s, 6), "wall_utc": e.wall_utc,
                 "est_volume_index": e.est_volume_index,
                 "inter_trigger_interval_s": (round(e.inter_trigger_interval_s, 6) if e.inter_trigger_interval_s is not None else "n/a"),
                 "anomaly": e.anomaly or "none"} for e in self.events]


class SimulatedTrigger(ScannerSync):
    """Deterministic synthetic triggers at TR spacing, with optional injected faults for failure-mode testing."""

    def __init__(self, tr_s, n_volumes, tr_tolerance_s=0.05, jitter_s=0.0, drop_indices=(), duplicate_indices=()):
        super().__init__(tr_s or 1.6, tr_tolerance_s)
        self.n_volumes = n_volumes
        self.jitter_s = jitter_s
        self.drop_indices = set(drop_indices)
        self.duplicate_indices = set(duplicate_indices)
        self._t = 0.0
        self._schedule = self._build()
        self._i = 0

    def _build(self):
        pulses = []
        t = 0.0
        for v in range(self.n_volumes):
            if v in self.drop_indices:
                t += self.tr_s
                continue
            j = (self.jitter_s if (v % 2 == 0) else -self.jitter_s)
            pulses.append(t + j)
            if v in self.duplicate_indices:
                pulses.append(t + j + self.tr_s * 0.1)   # spurious extra pulse -> DUPLICATE_TRIGGER
            t += self.tr_s
        return pulses

    def wait_for_scanner(self):
        return True

    def _pulse_time(self):
        p = self._schedule[self._i]
        self._i += 1
        return p

    def has_next(self):
        return self._i < len(self._schedule)

    def emit_all(self):
        while self.has_next():
            self.register_trigger()
        return self.events
