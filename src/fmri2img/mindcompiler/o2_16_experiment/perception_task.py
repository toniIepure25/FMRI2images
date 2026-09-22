"""Perception-task runner. Presents each anchor at its planned onset via an injected presenter, records timing/
response/trigger QC per trial. Computes NO scientific outcome during presentation."""
from __future__ import annotations

from . import config as C
from . import randomization as R
from . import timing as T


def run_perception_run(participant_id, session_id, run_number, run_trials, presenter, response_device,
                       scanner, cfg: C.ExperimentConfig, stimulus_sha_of=None):
    """run_trials: list of trial dicts (from randomization). Returns list of per-trial records."""
    image_s = cfg.timing.perception_image_s or 3.0
    gap_s = cfg.timing.perception_gap_s or 1.0
    trial_len = image_s + gap_s
    records = []
    t0 = T.monotonic()
    for tr in run_trials:
        planned = tr["trial_index"] * trial_len
        flip, offset, dropped = presenter.show("anchor-%04d" % tr["anchor_id"], planned, image_s)
        frame = T.make_frame_record(planned, flip, offset, dropped, cfg.frame_tolerance_s)
        resp = response_device.poll("perc|%d|%d" % (run_number, tr["trial_index"]), None, image_s)
        vol = scanner.count_volume() if scanner is not None else C.PILOT_MAX_PARTICIPANTS * 0  # est only
        sha = stimulus_sha_of("anchor-%04d" % tr["anchor_id"]) if stimulus_sha_of else None
        records.append({
            "participant": participant_id, "session": session_id, "run": run_number,
            "trial_index": tr["trial_index"], "trial_type": "perception", "identity_id": "n/a",
            "family": "perception_anchor", "repeat_index": "n/a", "stimulus_id": "anchor-%04d" % tr["anchor_id"],
            "anchor_id": tr["anchor_id"], "presentation": tr["presentation"], "stimulus_sha": sha,
            "planned_onset": planned, "onset": flip, "duration": image_s, "scanner_volume_index": vol,
            "response": resp.key, "response_time": resp.response_time_s, "response_missing": resp.missing,
            "response_multiple": resp.multiple, "response_correct": resp.correct,
            "frame_deviation_s": frame.deviation_s, "frame_exceeds_tol": frame.exceeds_tolerance,
            "trigger_anomaly": "none",
        })
    return records
