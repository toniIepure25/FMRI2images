"""Imagery-task runner. Cues each identity from the FROZEN identity mapping, runs the imagery period and the
frozen/authorized decision component, records timing/response. Never alters identity based on prior responses;
no adaptive difficulty; no scientific outcome computed during presentation."""
from __future__ import annotations

from . import config as C
from . import timing as T


def run_imagery_run(participant_id, session_id, run_number, run_trials, presenter, response_device,
                    scanner, cfg: C.ExperimentConfig, identity_map=None, stimulus_sha_of=None):
    cue_s = cfg.timing.cue_duration_s or 1.0
    img_s = cfg.timing.imagery_duration_s or 1.7
    resp_s = cfg.timing.response_window_s or 1.3
    iti_s = cfg.timing.iti_s or 1.0
    trial_len = cue_s + img_s + resp_s + iti_s
    records = []
    for tr in run_trials:
        idn = tr["identity_id"]
        stim_id = "identity-%02d" % idn
        planned = tr["trial_index"] * trial_len
        # cue
        cflip, coff, cdrop = presenter.show("cue:%s" % stim_id, planned, cue_s)
        # imagery period
        iflip, ioff, idrop = presenter.show("imagery:%s" % stim_id, planned + cue_s, img_s)
        frame = T.make_frame_record(planned, cflip, ioff, cdrop + idrop, cfg.frame_tolerance_s)
        resp = response_device.poll("img|%d|%d" % (run_number, tr["trial_index"]), None, resp_s)
        vol = scanner.count_volume() if scanner is not None else 0
        sha = stimulus_sha_of(stim_id) if stimulus_sha_of else None
        records.append({
            "participant": participant_id, "session": session_id, "run": run_number,
            "trial_index": tr["trial_index"], "trial_type": "imagery", "identity_id": stim_id,
            "family": tr["family"], "repeat_index": tr["repeat_index"], "stimulus_id": stim_id,
            "presentation": "n/a", "stimulus_sha": sha, "planned_onset": planned, "onset": cflip,
            "duration": cue_s + img_s + resp_s, "scanner_volume_index": vol,
            "response": resp.key, "response_time": resp.response_time_s, "response_missing": resp.missing,
            "response_multiple": resp.multiple, "response_correct": resp.correct,
            "frame_deviation_s": frame.deviation_s, "frame_exceeds_tol": frame.exceeds_tolerance,
            "trigger_anomaly": "none",
        })
    return records
