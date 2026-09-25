"""MINDIR-PROX governance: reproducibility contract, scientific immutability, and the historical-N8 firewall.
No sealed scientific status is changed here; this layer only verifies and records provenance."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

IMMUTABLE_HASHES = {
    "o2_16_primary_config_sha256": "2da2cc791054406b42eb1896b9b5f123b5b6bacb8386e6fb0f1e19649cf18907",
    "o2_16_sec_config_sha256": "3770816869ac0c8cd1cddcc5d4e4263578fcb08b327bee1031ac6d0a8aa16bfc",
    "o2_16_data_config_sha256": "804d5b21fc0c8656f9e0ef62ac4b7f412b114b7e0d3de2e328e972d75dec6444",
}
# historical-N8 firewall tokens (assembled from fragments so this file does not trip its own guard)
_FORBIDDEN = tuple(a + b for a, b in (("operator_o2_9", "/results"), ("x4_geometric_drift", "/results"),
                                      ("trial_", "native"), ("delta_", "native"), ("load_", "historical")))


class HistoricalN8Firewall(RuntimeError):
    pass


def assert_no_historical_access(source_text: str):
    hits = [t for t in _FORBIDDEN if t in source_text]
    if hits:
        raise HistoricalN8Firewall("historical N=8 access forbidden in platform code: %s" % hits)
    return True


def _self_consistent(path) -> bool:
    try:
        d = json.load(open(path)); c2 = {k: v for k, v in d.items() if k != "config_sha256"}
        calc = hashlib.sha256(json.dumps(c2, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return calc == d["config_sha256"]
    except Exception:
        return False


def scientific_immutability(repo="."):
    r = Path(repo)
    detail = {
        "o2_16_primary": _self_consistent(r / "artifacts/mindcompiler/operator_o2_16/o2_16_frozen_config.json"),
        "o2_16_sec": _self_consistent(r / "artifacts/mindcompiler/o2_16_secondary_prereg/secondary_prereg_frozen_config.json"),
        "m0": _self_consistent(r / "artifacts/mindcompiler/moonshot_triad/moonshot_triad_frozen_config.json"),
    }
    return {"all_match": all(detail.values()), "detail": detail,
            "O2_16": "O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE",
            "O2_16_DATA": "O2_16_DATA_AWAITING_HUMAN_ACQUISITION_AUTHORIZATION",
            "O2_16_SEC": "O2_16_SECONDARY_ENDPOINTS_PREREGISTERED_AWAITING_INDEPENDENT_COHORT",
            "M0": "MINDIR_MOONSHOT_TRIAD_PREREGISTERED_AWAITING_NEW_DATA", "O3": "O3_NOT_READY",
            "historical_n8": "PERMANENTLY_CLOSED", "no_x5": True, "no_new_inference": True}


@dataclass
class ReproRecord:
    artifact: str
    data_hashes: dict
    code_commit: str
    config_hash: str
    dependency_env: dict
    random_seed: Optional[int]
    participant_set: list
    metric_version: str
    timestamp: str
    frozen: bool

    def to_dict(self):
        return asdict(self)


def dependency_env():
    def ver(m):
        try:
            return __import__(m).__version__
        except Exception:
            return "absent"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": ver("numpy"), "pandas": ver("pandas")}


def semantic_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def repro_record(artifact, data_hashes, code_commit, config_hash, seed, participant_set, metric_version, timestamp, frozen=True):
    return ReproRecord(artifact, data_hashes, code_commit, config_hash, dependency_env(), seed,
                       list(participant_set), metric_version, timestamp, frozen).to_dict()
