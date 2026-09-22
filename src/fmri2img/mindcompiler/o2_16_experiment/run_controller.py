"""Run lifecycle: freeze+hash a run manifest BEFORE the run, execute the task, then append completion status
WITHOUT rewriting the frozen trial order. Crash handling persists state; partial and restarted runs are never
merged invisibly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config as C
from . import perception_task as PT
from . import imagery_task as IT
from . import qc as QC


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def freeze_run_manifest(participant_id, session_id, task, run_number, run_trials, cfg, software_commit,
                        stimulus_manifest_sha, expected_triggers=None):
    manifest = {
        "participant": participant_id, "session": session_id, "task": task, "run": run_number,
        "protocol_version": cfg.protocol_version, "protocol_sha": cfg.protocol_hash(),
        "config_sha": _hash(cfg.to_dict()), "stimulus_manifest_sha": stimulus_manifest_sha,
        "trial_sequence": run_trials, "n_trials": len(run_trials),
        "expected_trigger_count": expected_triggers, "software_commit": software_commit,
        "frozen": True,
    }
    manifest["run_manifest_sha"] = _hash({k: v for k, v in manifest.items() if k != "run_manifest_sha"})
    return manifest


def execute_run(manifest, presenter, response_device, scanner, cfg, identity_map=None, stimulus_sha_of=None,
                crash_after_trial=None):
    task = manifest["task"]; trials = manifest["trial_sequence"]
    completed = []
    try:
        if crash_after_trial is not None:
            partial = trials[:crash_after_trial]
            recs = _dispatch(task, manifest, partial, presenter, response_device, scanner, cfg, identity_map, stimulus_sha_of)
            raise RuntimeError("INJECTED_CRASH after trial %d" % crash_after_trial)
        recs = _dispatch(task, manifest, trials, presenter, response_device, scanner, cfg, identity_map, stimulus_sha_of)
        completed = recs
        status = "RUN_SEALED"
        error = None
    except Exception as e:               # crash recovery: persist, do NOT silently restart
        status = "RUN_CRASHED"
        error = str(e)
        completed = locals().get("recs", [])
    qc = QC.run_qc(completed, manifest["n_trials"], scanner, manifest.get("expected_trigger_count"))
    completion = {
        "run_manifest_sha": manifest["run_manifest_sha"], "status": status,
        "last_completed_trial": (completed[-1]["trial_index"] if completed else -1),
        "n_completed": len(completed), "error": error, "qc_status": qc["status"],
        "frozen_trial_order_preserved": True, "recovery_policy": "restart per explicit run-level protocol; no partial+restart merge",
    }
    return completed, qc, completion


def _dispatch(task, manifest, trials, presenter, response_device, scanner, cfg, identity_map, stimulus_sha_of):
    if task == "perception":
        return PT.run_perception_run(manifest["participant"], manifest["session"], manifest["run"], trials,
                                     presenter, response_device, scanner, cfg, stimulus_sha_of)
    return IT.run_imagery_run(manifest["participant"], manifest["session"], manifest["run"], trials,
                              presenter, response_device, scanner, cfg, identity_map, stimulus_sha_of)


def write_run_manifest(manifest, completion, out_dir):
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    p = d / ("run_manifest_%s_%s_%s_run%02d.json" % (manifest["participant"], manifest["session"], manifest["task"], manifest["run"]))
    p.write_text(json.dumps({"manifest": manifest, "completion": completion}, indent=2, default=str))
    return str(p)
