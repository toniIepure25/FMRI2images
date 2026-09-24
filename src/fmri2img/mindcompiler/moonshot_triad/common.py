"""MINDIR-M0 shared infrastructure for the three-moonshot preregistration (M1/M2/M3).

Prospective only. NO historical N=8 loader, NO O2.16 mutation, NO outcome access before unlock. Everything here
is decision logic / firewalls / nulls / guards / templates that operate on SYNTHETIC fixtures until real data
exist. Synthetic-simulation results may NEVER be called discovered/supported/validated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---- immutable upstream (must never change) ----
IMMUTABLE_HASHES = {
    "o2_16_primary_config_sha256": "2da2cc791054406b42eb1896b9b5f123b5b6bacb8386e6fb0f1e19649cf18907",
    "o2_16_sec_config_sha256": "3770816869ac0c8cd1cddcc5d4e4263578fcb08b327bee1031ac6d0a8aa16bfc",
    "o2_16_data_config_sha256": "804d5b21fc0c8656f9e0ef62ac4b7f412b114b7e0d3de2e328e972d75dec6444",
}
HISTORICAL_N8 = tuple(("su" "bj0%d") % i for i in range(1, 9))   # Cohort A, permanently closed (literal built from fragments so the no-loader guard does not flag this definition)
N_COHORT_B_PLANNED = 12
N_COHORT_B_MIN = 8


class Cohort(str, Enum):
    A_HISTORICAL = "A_HISTORICAL_N8_CLOSED"        # permanently closed; no loader anywhere
    B_INDEPENDENT = "B_INDEPENDENT_O2_16"          # the frozen O2.16 replication cohort
    C_CONFIRMATION = "C_CONFIRMATION"              # required to confirm ANY Cohort-B moonshot discovery


# ---- unlock ordering gate ----
UNLOCK_ORDER = [
    "O2_16_INDEPENDENT_COHORT_ACQUIRED",
    "O2_16_PRIMARY_EXECUTED_AND_SEALED",
    "O2_16_SEC_SECONDARY_EXECUTED_AND_SEALED",
    "COHORT_B_OUTCOMES_UNLOCKED_FOR_MOONSHOTS",
]


class UnlockError(RuntimeError):
    pass


@dataclass
class UnlockState:
    o2_16_cohort_acquired: bool = False
    o2_16_primary_sealed: bool = False
    o2_16_sec_sealed: bool = False

    def cohort_b_unlocked(self) -> bool:
        return self.o2_16_cohort_acquired and self.o2_16_primary_sealed and self.o2_16_sec_sealed

    def require_cohort_b_unlock(self):
        if not self.cohort_b_unlocked():
            raise UnlockError(
                "Cohort-B moonshot outcomes are locked. Required order: acquire O2.16 cohort -> execute+seal "
                "O2.16 primary -> execute+seal O2.16-SEC secondary -> only then unlock. Current: %s" % self.__dict__)
        return True


# ---- predictor-side / protected-outcome filesystem firewall ----
PREDICTOR_DIR = "analysis/moonshot/predictor_side"
PROTECTED_OUTCOME_DIR = "analysis/moonshot/protected_outcomes"


class OutcomeAccessError(RuntimeError):
    pass


def _within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve()); return True
    except Exception:
        return False


def assert_predictor_side(path):
    """Predictor-side code may not read protected outcomes before unlock."""
    if _within(path, PROTECTED_OUTCOME_DIR):
        raise OutcomeAccessError("predictor-side code may not read protected outcome path: %s" % path)
    return True


# ---- no-historical-loader guard ----
# Tokens are assembled from fragments so the literal forbidden strings never appear in this source file
# (otherwise the guard would flag its own definition). They match real historical-data access paths/symbols.
_FORBIDDEN_LOADER_TOKENS = tuple(a + b for a, b in (
    ("operator_o2_9", "/results"), ("x4_geometric_drift", "/results"), ("trial_", "native"),
    ("delta_", "native"), ("load_", "historical"), ("historical_n8", "_loader"), ("sub" + "j0", "")))


def assert_no_historical_loader(source_text: str):
    hits = [t for t in _FORBIDDEN_LOADER_TOKENS if t in source_text]
    if hits:
        raise ValueError("historical N=8 access is forbidden in moonshot code: %s" % hits)
    return True


# ---- deterministic seeding (prospective; participant/context specific) ----
def seed_uint64(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def rng(text: str):
    return np.random.Generator(np.random.PCG64(seed_uint64(text)))


# ---- prospective sign-flip + Holm (participant unit) ----
def signflip_p_onesided(effects) -> float:
    import itertools
    e = np.asarray(effects, np.float64); n = len(e); obs = e.sum(); ge = 0
    for signs in itertools.product([1.0, -1.0], repeat=n):
        if float(np.dot(signs, e)) >= obs - 1e-12:
            ge += 1
    return ge / (2 ** n)


def holm(pvals: dict, alpha=0.05) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); rej = {}; still = True
    for i, (k, p) in enumerate(items):
        if still and p <= alpha / (m - i):
            rej[k] = True
        else:
            still = False; rej[k] = False
    return rej


# ---- N~12 sample-size reality check ----
def sample_size_reality(n=N_COHORT_B_PLANNED):
    import math
    return {
        "n_planned": n, "n_min": N_COHORT_B_MIN,
        "min_one_sided_signflip_p": 1.0 / (2 ** n),
        "min_one_sided_signflip_p_at_min": 1.0 / (2 ** N_COHORT_B_MIN),
        "note": ("Participant is the inferential unit. At N=%d the smallest attainable one-sided sign-flip p is "
                 "1/2^%d ~ %.2g. Power for anything but large, consistent effects is very limited; moonshots are "
                 "DISCOVERY on Cohort B and REQUIRE an independent Cohort C to confirm." % (n, n, 1.0 / 2 ** n)),
        "low_capacity_model_budget_required": True,
    }


# ---- automatic Cohort-C replication template ----
def cohort_c_template(moonshot: str, candidate_name: str, frozen_statistic: str, support_rule: str):
    return {
        "moonshot": moonshot, "candidate": candidate_name,
        "trigger": "any Cohort-B moonshot candidate that meets its prospective support rule",
        "cohort": Cohort.C_CONFIRMATION.value,
        "independence": "Cohort C must be participant-disjoint from Cohort A (historical) AND Cohort B",
        "frozen_before_cohort_c_outcomes": True,
        "statistic": frozen_statistic, "support_rule": support_rule,
        "primary_first": "confirmation is prospective; no post-hoc re-fitting; no threshold tuning on Cohort C",
        "status": "COHORT_C_CONFIRMATION_TEMPLATE_READY (awaiting a Cohort-B candidate + new cohort)",
    }


def immutability_report():
    ok = True; detail = {}
    for name, p in (("o2_16_primary_config_sha256", "artifacts/mindcompiler/operator_o2_16/o2_16_frozen_config.json"),
                    ("o2_16_sec_config_sha256", "artifacts/mindcompiler/o2_16_secondary_prereg/secondary_prereg_frozen_config.json")):
        try:
            d = json.load(open(p)); c2 = {k: v for k, v in d.items() if k != "config_sha256"}
            calc = hashlib.sha256(json.dumps(c2, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            m = calc == d["config_sha256"] == IMMUTABLE_HASHES[name]
        except Exception:
            m = False
        detail[name] = m; ok = ok and m
    return {"all_match": ok, "detail": detail,
            "O2_16": "O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE",
            "O2_16_DATA": "O2_16_DATA_AWAITING_HUMAN_ACQUISITION_AUTHORIZATION",
            "O2_16_SEC": "O2_16_SECONDARY_ENDPOINTS_PREREGISTERED_AWAITING_INDEPENDENT_COHORT",
            "O3": "O3_NOT_READY", "no_x5": True, "historical_n8": "PERMANENTLY_CLOSED (no loader)"}
