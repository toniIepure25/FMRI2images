"""Stage-A predictor-side export + Stage-A/Stage-B access firewall.

Stage A prepares predictor-side artifacts (from acquired-and-preprocessed neuroimaging; here it operates only
on SYNTHETIC schema-matching fixtures) WITHOUT reading any Stage-B held-out outcome. Stage B stays locked until
a human-created STAGE_B_RELEASE.json exists. No CLI/debug path may display protected Stage-B outcomes during
Stage A. This module never runs scientific predictions on the historical N=8."""
from __future__ import annotations

import json
from pathlib import Path

STAGE_A_DIR = "analysis/stage_a"
STAGE_B_DIR = "analysis/stage_b_protected"
RELEASE_FILE = "STAGE_B_RELEASE.json"


class StageBAccessError(RuntimeError):
    pass


def _is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve()); return True
    except Exception:
        return False


def assert_not_stage_b(path):
    """Firewall: refuse any read that resolves inside the protected Stage-B tree."""
    if _is_within(path, STAGE_B_DIR):
        raise StageBAccessError("Stage-A code may not read protected Stage-B path: %s" % path)
    return True


def stage_a_read(path):
    assert_not_stage_b(path)
    return Path(path).read_text()


def export_stage_a(fixture: dict, out_dir=STAGE_A_DIR):
    """Prepare predictor-side artifacts from a synthetic schema-matching fixture. No outcomes, no N=8 data."""
    assert fixture.get("cohort") != "historical_N8", "Stage-A export must not touch historical N=8"
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    art = {"stage": "A", "predictor_side_only": True, "no_heldout_outcome": True,
           "participants": fixture.get("participants", []), "n_runs": fixture.get("n_runs"),
           "synthetic_fixture": True, "source": "SYNTHETIC_SCHEMA_MATCHING_FIXTURE"}
    (d / "stage_a_predictors.json").write_text(json.dumps(art, indent=2))
    return art


def stage_b_available(base="."):
    return (Path(base) / RELEASE_FILE).exists()


def require_stage_b_release(base="."):
    """Stage-B commands abort unless a human release token exists with the required fields."""
    p = Path(base) / RELEASE_FILE
    if not p.exists():
        raise StageBAccessError("Stage B locked: %s not present (human release required)" % RELEASE_FILE)
    tok = json.loads(p.read_text())
    required = ["cohort_certification", "stage_a_commit_sha", "stage_a_hash_manifest", "server_verification_status",
                "human_approver", "timestamp"]
    missing = [k for k in required if k not in tok or tok[k] in (None, "", "TBD")]
    if missing:
        raise StageBAccessError("Stage-B release token incomplete: %s" % missing)
    return tok


def release_token_template():
    return {"cohort_certification": "TBD", "stage_a_commit_sha": "TBD", "stage_a_hash_manifest": "TBD",
            "server_verification_status": "TBD", "human_approver": "TBD", "timestamp": "TBD",
            "note": "Human-created only. Stage B stays locked until every field is real."}
