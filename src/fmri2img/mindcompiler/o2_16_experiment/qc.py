"""Acquisition QC. Reports operational quality only. It NEVER makes scientific accept/reject decisions based on
outcomes (no 'exclude if model performs poorly'); scientific participant-exclusion logic is frozen elsewhere."""
from __future__ import annotations

from collections import Counter

from . import config as C


def run_qc(records, expected_trials, scanner=None, expected_triggers=None):
    n = len(records)
    ids = [r["stimulus_id"] for r in records]
    dup = [k for k, v in Counter(ids).items() if v > 1 and records[0]["trial_type"] == "imagery"]
    missing_resp = sum(1 for r in records if r.get("response_missing"))
    multi_resp = sum(1 for r in records if r.get("response_multiple"))
    frame_dev = sum(1 for r in records if r.get("frame_exceeds_tol"))
    sha_mismatch = sum(1 for r in records if r.get("stimulus_sha") == "MISMATCH")
    missing_trig = scanner.detect_missing_trigger(expected_triggers) if (scanner and expected_triggers) else 0
    dup_trig = len(scanner.detect_duplicate_trigger()) if scanner else 0
    checks = {
        "expected_trials": expected_trials, "observed_trials": n,
        "missing_trials": max(0, expected_trials - n), "extra_trials": max(0, n - expected_trials),
        "duplicate_ids": dup, "response_missing": missing_resp, "response_multiple": multi_resp,
        "timing_frame_deviations": frame_dev, "missing_triggers": missing_trig, "duplicate_triggers": dup_trig,
        "stimulus_hash_mismatch": sha_mismatch,
    }
    fail = (n != expected_trials) or sha_mismatch > 0 or (expected_triggers and missing_trig > 0) or dup_trig > 0
    warn = frame_dev > 0 or missing_resp > 0 or multi_resp > 0
    status = "FAIL" if fail else ("WARN" if warn else "PASS")
    checks["status"] = status
    checks["note"] = "operational QC only; no outcome-based scientific exclusion"
    return checks
