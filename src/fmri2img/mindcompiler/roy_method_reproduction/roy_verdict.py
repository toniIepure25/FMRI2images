"""S2.7R frozen final-verdict decision logic (pure functions; NO modelling).

Encodes the S2.7R rubric (Part H/I/J/K) as a deterministic decision over already-assigned
claim/domain statuses. This is an AUDIT helper: it fits nothing, reads no data, and only maps
frozen statuses -> one of the four verdicts. Kept importable and unit-tested so the committed
`reproduction_verdict.json` is reproducible from its inputs.
"""
from __future__ import annotations

from typing import Dict

SUPPORTED = "INDEPENDENT_METHOD_REPRODUCTION_SUPPORTED"
PARTIAL = "INDEPENDENT_METHOD_REPRODUCTION_PARTIAL"
NOT_SUPPORTED = "INDEPENDENT_METHOD_REPRODUCTION_NOT_SUPPORTED"
INSUFFICIENT = "PUBLIC_METHODS_INSUFFICIENT_FOR_STRONG_VERDICT"

_CONCORDANT = {"CONCORDANT", "DIRECTIONALLY_CONCORDANT", "NUMERICALLY_CLOSE"}
_DISCORDANT = {"DIRECTIONALLY_DISCORDANT", "DISCORDANT"}
_METHOD_SUFFICIENT = {"METHOD_SUFFICIENT_FOR_INDEPENDENT_REPRODUCTION_ASSESSMENT",
                      "METHOD_PARTIALLY_IDENTIFIABLE_BUT_ASSESSMENT_POSSIBLE"}


def is_concordant(status: str) -> bool:
    return status in _CONCORDANT


def is_discordant(status: str) -> bool:
    return status in _DISCORDANT


def method_sufficient(method_status: str) -> bool:
    return method_status in _METHOD_SUFFICIENT


def decide_verdict(method_status: str, prediction_status: str, dimensionality_domain: str,
                   d_c1_status: str, alignment_domain: str, central_statuses: Dict[str, str]) -> str:
    """Map frozen statuses to exactly one verdict per the S2.7R rubric.

    Order: insufficient-method -> SUPPORTED (all conditions) -> NOT_SUPPORTED (broad failure) -> PARTIAL.
    """
    if not method_sufficient(method_status):
        return INSUFFICIENT

    pred_ok = prediction_status in _CONCORDANT or "DIRECTIONALLY_CONCORDANT" in prediction_status
    pred_fails = prediction_status in _DISCORDANT

    supported = (pred_ok
                 and not is_discordant(d_c1_status)
                 and is_concordant(dimensionality_domain)
                 and is_concordant(alignment_domain)
                 and not any(is_discordant(v) for v in central_statuses.values()))
    if supported:
        return SUPPORTED

    broad_failure = pred_fails or (is_discordant(dimensionality_domain) and is_discordant(alignment_domain))
    if broad_failure:
        return NOT_SUPPORTED

    return PARTIAL
