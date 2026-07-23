"""Every 'verified' manifest reference must resolve to an existing file.

Guards the provenance defect this session repaired: acquisition_manifest.json
pointed at a metadata manifest path that had moved. A dangling 'verified'
reference is a provenance lie and must fail CI.
"""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = [
    ROOT / "artifacts/mindcompiler/roy_s1/metadata_manifest.json",
    ROOT / "artifacts/mindcompiler/roy_s1/acquisition_manifest.json",
]


def _iter_verified_paths(obj):
    """Yield local file paths that are asserted 'verified' anywhere in a manifest."""
    if isinstance(obj, dict):
        status = str(obj.get("status", "")).lower()
        dest = obj.get("dest") or obj.get("manifest")
        if status == "verified" and isinstance(dest, str):
            # strip trailing parenthetical annotations, e.g. "... (per-file sha256)"
            yield dest.split(" (")[0].strip()
        for v in obj.values():
            yield from _iter_verified_paths(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_verified_paths(v)


@pytest.mark.parametrize("mpath", MANIFESTS, ids=lambda p: p.name)
def test_manifest_exists_and_is_json(mpath):
    assert mpath.exists(), f"manifest missing: {mpath}"
    json.loads(mpath.read_text())


def test_all_verified_references_resolve():
    missing = []
    for mpath in MANIFESTS:
        if not mpath.exists():
            continue
        for ref in _iter_verified_paths(json.loads(mpath.read_text())):
            p = (ROOT / ref)
            # raw NSD data lives under data/nsd/ (gitignored) but is present locally
            # after download; a 'verified' entry must point at a real file.
            if not p.exists():
                missing.append((mpath.name, ref))
    assert not missing, f"dangling 'verified' manifest references: {missing}"
