"""Replay engine. From only {participant, session, protocol_version, seed derivation, n_runs} the expected
trial sequence is reconstructed EXACTLY. Also parses written BIDS events back to a trial manifest for a
round-trip identity/repeat/family/run/order check. Timing tolerance is reported separately (never used for the
exact-sequence check)."""
from __future__ import annotations

import csv
from pathlib import Path

from . import randomization as R


def expected_perception(participant_id, session_id, n_runs=6, protocol_version=None):
    from . import config as C
    seq, _ = R.perception_sequence(participant_id, session_id, n_runs, protocol_version or C.PROTOCOL_VERSION)
    return seq


def expected_imagery(participant_id, session_id, n_runs=6, protocol_version=None):
    from . import config as C
    seq, _ = R.imagery_sequence(participant_id, session_id, n_runs, protocol_version or C.PROTOCOL_VERSION)
    return seq


def reconstruct(participant_id, session_id, task, n_runs=6, protocol_version=None):
    return expected_perception(participant_id, session_id, n_runs, protocol_version) if task == "perception" \
        else expected_imagery(participant_id, session_id, n_runs, protocol_version)


def sequence_matches(expected, actual):
    """Exact match on the identity-defining key + order, per (run, trial_index)."""
    if len(expected) != len(actual):
        return False, "length %d != %d" % (len(expected), len(actual))
    for e, a in zip(expected, actual):
        for k in ("run", "trial_index"):
            if e[k] != a[k]:
                return False, "order mismatch at %s" % k
        if e["task"] == "perception":
            if e["anchor_id"] != a["anchor_id"] or e["presentation"] != a["presentation"]:
                return False, "perception key mismatch"
        else:
            if e["identity_id"] != a["identity_id"] or e["repeat_index"] != a["repeat_index"] or e["family"] != a["family"]:
                return False, "imagery key mismatch"
    return True, "exact"


def parse_events_tsv(path):
    """Parse a written BIDS events.tsv back into trial dicts (identity/family/repeat/run/order)."""
    rows = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            task = r["trial_type"]
            d = {"task": task, "run": int(r["run"]), "trial_index": int(r["trial_index"])}
            if task == "perception":
                d["anchor_id"] = int(r["stimulus_id"].split("-")[1]); d["presentation"] = int(r["presentation"])
            else:
                d["identity_id"] = r["identity_id"]; d["family"] = r["family"]; d["repeat_index"] = int(r["repeat_index"])
            rows.append(d)
    return rows


def roundtrip_from_bids(participant_id, session_id, task, bids_root, n_runs=6, protocol_version=None):
    """events.tsv -> parsed manifest, compared to reconstructed expected sequence."""
    func = Path(bids_root) / ("sub-%s" % participant_id.replace("sub-", "")) / ("ses-%s" % session_id) / "func"
    tsvs = sorted(func.glob("*_task-%s_*_events.tsv" % task))
    actual = []
    for t in tsvs:
        actual += parse_events_tsv(t)
    exp = reconstruct(participant_id, session_id, task, n_runs, protocol_version)
    # normalize imagery identity id form (expected uses int; parsed uses 'identity-##')
    if task == "imagery":
        exp = [dict(e, identity_id="identity-%02d" % e["identity_id"]) for e in exp]
    else:
        exp = [dict(e) for e in exp]
    return sequence_matches(exp, actual)
