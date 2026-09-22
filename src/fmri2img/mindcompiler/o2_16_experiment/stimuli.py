"""Stimulus manifest system: perception anchor binding + imagery identity manifest, SHA256 verification, and
placeholder guard. In SIMULATION/pilot, synthetic placeholder stimuli are allowed and flagged. In CONFIRMATORY,
every stimulus must have a real file whose SHA matches and placeholder=false, else abort (fail closed)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from . import config as C


@dataclass
class StimulusEntry:
    logical_id: str
    family: str
    source: str
    file_path: Optional[str]
    sha256: Optional[str]
    width: Optional[int]
    height: Optional[int]
    media_type: str
    permission_status: str            # e.g. "AUTHORIZED", "TBD", "PLACEHOLDER"
    acquisition_included: bool
    placeholder: bool


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def synthetic_perception_manifest():
    """512 placeholder anchors for SIMULATION (no real image files)."""
    return [StimulusEntry(
        logical_id="anchor-%04d" % a, family="perception_anchor", source="O2.3A-RD_canonical_512(placeholder)",
        file_path=None, sha256=None, width=None, height=None, media_type="image/placeholder",
        permission_status="PLACEHOLDER", acquisition_included=True, placeholder=True) for a in range(C.PERCEPTION_ANCHORS)]


def synthetic_imagery_manifest():
    """12 placeholder identities (6 simple + 6 naturalistic) for SIMULATION."""
    out = []
    for i in range(C.IMAGERY_IDENTITIES):
        fam = "simple" if i < C.IMAGERY_SIMPLE else "naturalistic"
        out.append(StimulusEntry(
            logical_id="identity-%02d" % i, family=fam, source="NSD-Imagery_family(placeholder)",
            file_path=None, sha256=None, width=None, height=None, media_type="cue/text+image(placeholder)",
            permission_status="PLACEHOLDER", acquisition_included=True, placeholder=True))
    return out


def imagery_identity_records(manifest):
    """The 12-identity manifest with cue text (frozen mapping; never outcome-dependent)."""
    recs = []
    for e in manifest:
        recs.append({"identity_id": e.logical_id, "family": e.family, "display_name": e.logical_id,
                     "cue_text": "Imagine: %s" % e.logical_id, "stimulus_reference": e.file_path,
                     "sha256": e.sha256, "permission_status": e.permission_status, "placeholder": e.placeholder})
    return recs


def verify_manifest(manifest, mode: C.Mode):
    """Verify SHAs + placeholder policy. Returns (ok, report). CONFIRMATORY fails closed on any issue."""
    issues = []
    for e in manifest:
        if e.file_path is not None:
            p = Path(e.file_path)
            if not p.exists():
                issues.append("missing_file:%s" % e.logical_id)
            elif e.sha256 and sha256_file(p) != e.sha256:
                issues.append("sha_mismatch:%s" % e.logical_id)
    placeholders = [e.logical_id for e in manifest if e.placeholder]
    if mode == C.Mode.CONFIRMATORY:
        if placeholders:
            issues.append("confirmatory_placeholder_present:%d" % len(placeholders))
        for e in manifest:
            if e.acquisition_included and (e.file_path is None or e.sha256 is None):
                issues.append("confirmatory_missing_stimulus_binding:%s" % e.logical_id)
            if e.acquisition_included and e.permission_status != "AUTHORIZED":
                issues.append("confirmatory_permission_not_authorized:%s" % e.logical_id)
    ok = len(issues) == 0
    return ok, {"ok": ok, "n_stimuli": len(manifest), "n_placeholder": len(placeholders), "issues": issues,
                "mode": mode.value}


def perception_anchor_binding(manifest):
    return {"n_anchors": len(manifest), "must_not_substitute_subset": True, "no_outcome_based_substitution": True,
            "anchors": [{"canonical_identity": e.logical_id, "source": e.source, "file_sha": e.sha256,
                         "family": e.family, "manifest_order": i, "placeholder": e.placeholder}
                        for i, e in enumerate(manifest)]}


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
