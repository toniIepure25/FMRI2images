"""Hardware-validation HARNESS. It does NOT validate real hardware now; it DEFINES exactly how site staff will
test the trigger->clock, flip->visual, and response->event chains, and provides a result recorder. Every real
result stays TBD until run at the site on real hardware."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ValidationResult:
    test: str
    expected: str
    observed: str = "TBD_SITE_HARDWARE"
    latency_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    missed_events: Optional[int] = None
    duplicates: Optional[int] = None
    status: str = "NOT_YET_RUN"           # PASS / WARN / FAIL only after real hardware run


def trigger_to_clock_test():
    return ValidationResult(test="scanner_trigger->task_clock",
                            expected="each scanner pulse registered once; ITI ~ TR within site-defined tolerance")


def flip_to_visual_test():
    return ValidationResult(test="task_flip->visual_presentation",
                            expected="requested onset matches measured flip within site-defined frame tolerance")


def response_to_event_test():
    return ValidationResult(test="response_button->recorded_event",
                            expected="each button press recorded once with correct mapping and plausible RT")


def define_harness():
    tests = [trigger_to_clock_test(), flip_to_visual_test(), response_to_event_test()]
    return {"harness": "site hardware validation (definition only; not executed here)",
            "acceptance_threshold": "SITE_DEFINED_ACCEPTANCE_THRESHOLD (MRI physicist approval required)",
            "tests": [asdict(t) for t in tests], "requires_real_hardware": True,
            "no_participant_needed": True}


def record_result(test, observed, latency_ms=None, jitter_ms=None, missed=None, duplicates=None, status="NOT_YET_RUN"):
    return asdict(ValidationResult(test=test, expected="see define_harness", observed=observed, latency_ms=latency_ms,
                                   jitter_ms=jitter_ms, missed_events=missed, duplicates=duplicates, status=status))
