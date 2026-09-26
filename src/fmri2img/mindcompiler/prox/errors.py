"""Actionable, user-facing exceptions. Each replaces an opaque failure with a message telling the user what to
fix. Used across the platform (geometry, metrics, governance, benchmark)."""
from __future__ import annotations


class MindirError(Exception):
    """Base class for all MINDIR user-facing errors."""


class ShapeError(MindirError):
    def __init__(self, expected, got, what="array"):
        super().__init__("%s shape mismatch: expected %s, got %s. Check your input dimensions." % (what, expected, got))


class RankDeficiencyError(MindirError):
    def __init__(self, rank, needed):
        super().__init__("rank-deficient input: numerical rank %d < required %d. Provide more independent "
                         "observations or reduce the requested subspace dimension." % (rank, needed))


class MissingIdentityError(MindirError):
    def __init__(self, missing):
        super().__init__("missing identity/identities %s. Every identity in the frozen design must be present." % missing)


class DuplicateParticipantError(MindirError):
    def __init__(self, pid):
        super().__init__("duplicate participant id %r. Participant ids must be unique in the inferential set." % pid)


class SchemaMismatchError(MindirError):
    def __init__(self, expected, got):
        super().__init__("schema version mismatch: expected %s, got %s. Migrate the artifact or use a matching "
                         "platform version (no silent reinterpretation)." % (expected, got))


class HashMismatchError(MindirError):
    def __init__(self, name, expected, got):
        super().__init__("hash mismatch for %s: expected %s, got %s. Inputs/config changed; reproduction is not "
                         "exact." % (name, expected[:12], got[:12]))


class LeakageViolation(MindirError):
    def __init__(self, detail):
        super().__init__("leakage violation: %s. Predictor-side code must not see held-out outcomes." % detail)


class ProtectedOutcomeAccess(MindirError):
    def __init__(self, path):
        super().__init__("protected Stage-B outcome access blocked: %s. Obtain a human STAGE_B_RELEASE token." % path)


class HistoricalN8Access(MindirError):
    def __init__(self, hits):
        super().__init__("historical N=8 access blocked: %s. Cohort A is permanently closed for new discovery." % hits)


class UnsupportedMetric(MindirError):
    def __init__(self, name, known):
        super().__init__("unsupported metric %r. Known metrics: %s." % (name, sorted(known)))
