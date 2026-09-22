"""BIDS events schema for the perception and imagery tasks (columns + JSON sidecar descriptions)."""
from __future__ import annotations

EVENT_COLUMNS = ["onset", "duration", "trial_type", "identity_id", "family", "repeat_index",
                 "stimulus_id", "response", "response_time"]

# extra namespaced columns (allowed by BIDS; kept out of the required-minimum set)
EXTRA_COLUMNS = ["run", "trial_index", "presentation", "stimulus_sha", "scanner_volume_index",
                 "planned_onset", "frame_deviation_s", "frame_exceeds_tol", "trigger_anomaly",
                 "response_missing", "response_multiple", "response_correct"]

EVENTS_JSON = {
    "onset": {"Description": "Trial onset relative to first retained trigger (monotonic clock).", "Units": "s"},
    "duration": {"Description": "Trial duration.", "Units": "s"},
    "trial_type": {"Description": "Task/trial category.", "Levels": {"perception": "perception anchor viewing",
                                                                     "imagery": "cued mental imagery"}},
    "identity_id": {"Description": "Logical imagery identity id (imagery task) or n/a (perception)."},
    "family": {"Description": "Imagery family.", "Levels": {"simple": "simple identity", "naturalistic": "naturalistic identity",
                                                            "perception_anchor": "perception anchor"}},
    "repeat_index": {"Description": "0-based repeat index of an imagery identity (n/a for perception)."},
    "stimulus_id": {"Description": "Logical stimulus id (anchor-#### or identity-##)."},
    "response": {"Description": "Recorded response key/button, or n/a."},
    "response_time": {"Description": "Response time from trial onset.", "Units": "s"},
    "run": {"Description": "Run number (1-based)."},
    "presentation": {"Description": "0-based presentation index of a perception anchor (n/a for imagery)."},
    "stimulus_sha": {"Description": "SHA256 of the presented stimulus file (n/a for placeholder)."},
    "scanner_volume_index": {"Description": "Estimated scanner volume index at trial onset."},
    "planned_onset": {"Description": "Planned onset before flip.", "Units": "s"},
    "frame_deviation_s": {"Description": "abs(actual flip - planned onset).", "Units": "s"},
    "trigger_anomaly": {"Description": "Trigger anomaly flag at/around trial, or none."},
}
NA = "n/a"
