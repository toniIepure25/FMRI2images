"""Configurable, facility-agnostic scanner-trigger and response adapters. These read a site config (trigger
key/value, debounce, TR, dummy count; button_1..4 mappings) and never hardcode a specific facility's protocol.
SimulatedTrigger/SimulatedResponseDevice remain the CI backends; these adapters are the shape a real site fills
in. No undocumented vendor-specific protocols are implemented."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from . import config as C


@dataclass
class TriggerAdapterConfig:
    trigger_type: str = "TBD_SITE_OPERATOR"     # e.g. KEYBOARD_KEY / SERIAL_EVENT (site-defined)
    trigger_value: str = "TBD_SITE_OPERATOR"    # e.g. the actual key/code -- do NOT assume "5"
    debounce_ms: Optional[int] = None
    expected_TR_s: Optional[float] = None
    initial_dummy_count: Optional[int] = None

    def confirmatory_ready(self):
        missing = [k for k, v in asdict(self).items() if C.is_tbd(v)]
        return len(missing) == 0, missing


@dataclass
class ResponseAdapterConfig:
    device: str = "TBD_SITE_OPERATOR"           # e.g. Current Designs 4-Button Fiber Optic (per site page)
    button_1: str = "TBD_SITE_OPERATOR"
    button_2: str = "TBD_SITE_OPERATOR"
    button_3: str = "TBD_SITE_OPERATOR"
    button_4: str = "TBD_SITE_OPERATOR"

    def mapping(self):
        return {i + 1: getattr(self, "button_%d" % (i + 1)) for i in range(4)}

    def confirmatory_ready(self):
        missing = [k for k, v in asdict(self).items() if C.is_tbd(v)]
        return len(missing) == 0, missing


class KeyboardEventTrigger:
    """Real-input trigger that treats a configured key/event as a scanner pulse (PsychoPy/keyboard-backed at the
    site). Not a vendor protocol; a thin, config-driven event source. CI uses SimulatedTrigger instead."""

    def __init__(self, cfg: TriggerAdapterConfig):
        self.cfg = cfg
        if C.is_tbd(cfg.trigger_value):
            self.ready = False
        else:
            self.ready = True

    def describe(self):
        ok, missing = self.cfg.confirmatory_ready()
        return {"trigger_type": self.cfg.trigger_type, "trigger_value": self.cfg.trigger_value,
                "debounce_ms": self.cfg.debounce_ms, "expected_TR_s": self.cfg.expected_TR_s,
                "initial_dummy_count": self.cfg.initial_dummy_count, "confirmatory_ready": ok, "missing": missing,
                "no_assumed_key_5": True}


class ButtonBoxResponseAdapter:
    def __init__(self, cfg: ResponseAdapterConfig):
        self.cfg = cfg

    def describe(self):
        ok, missing = self.cfg.confirmatory_ready()
        return {"device": self.cfg.device, "mapping": self.cfg.mapping(), "confirmatory_ready": ok,
                "missing": missing}
