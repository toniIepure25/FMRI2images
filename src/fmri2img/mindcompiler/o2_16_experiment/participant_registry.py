"""Pseudonymous participant IDs. Identity data (name/email/health) is NEVER stored in the research dataset;
this registry only mints/validates pseudonyms. Any real identity mapping lives outside the research dataset per
future site policy (not implemented here)."""
from __future__ import annotations

import re

from . import config as C

FORMATS = {
    C.Mode.SIMULATION: re.compile(r"^sub-SYN\d{3}$"),
    C.Mode.ENGINEERING_PILOT: re.compile(r"^sub-PILOT\d{3}$"),
    C.Mode.CONFIRMATORY: re.compile(r"^sub-[A-Z0-9]{4,}$"),   # site-authorized pseudonymous format
}

_FORBIDDEN_IN_RESEARCH = ("name", "full_name", "email", "address", "phone", "dob", "health", "screening")


def make_id(mode: C.Mode, n: int) -> str:
    if mode == C.Mode.SIMULATION:
        return "sub-SYN%03d" % n
    if mode == C.Mode.ENGINEERING_PILOT:
        return "sub-PILOT%03d" % n
    return "sub-%s" % ("SITE%03d" % n)  # placeholder confirmatory pseudonym shape (site overrides)


def valid_for_mode(participant_id: str, mode: C.Mode) -> bool:
    return bool(FORMATS[mode].match(participant_id))


def assert_no_identity_fields(record: dict):
    """Guard: research records must never carry identity fields."""
    bad = [k for k in record if any(f in k.lower() for f in _FORBIDDEN_IN_RESEARCH)]
    if bad:
        raise ValueError("identity fields forbidden in research record: %s" % bad)
    return True
