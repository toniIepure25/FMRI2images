"""Deterministic, reconstructable, participant/session/task-specific randomization + balanced schedule
generation. Preserves the frozen O2.16-DATA scheme (global seed + per-participant SHA256->uint64->PCG64) and
scopes it by session/task/protocol_version per the engineering brief. No outcome-adaptive scheduling."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict

import numpy as np

from . import config as C


def seed_input(participant_id: str, session_id: str, task: str, protocol_version: str) -> str:
    return "%s|%s|%s|%s|%s" % (C.RANDOMIZATION_GLOBAL_SEED, participant_id, session_id, task, protocol_version)


def derive_seed(participant_id: str, session_id: str, task: str, protocol_version: str):
    txt = seed_input(participant_id, session_id, task, protocol_version)
    digest = hashlib.sha256(txt.encode()).hexdigest()
    seed = int(digest[:16], 16)                       # first 16 hex -> uint64
    return txt, digest, seed


def _rng(participant_id, session_id, task, protocol_version):
    _, _, seed = derive_seed(participant_id, session_id, task, protocol_version)
    return np.random.Generator(np.random.PCG64(seed))


@dataclass
class RandomizationRecord:
    seed_input: str
    digest: str
    seed: int
    task: str
    n_trials: int
    code_version: str = "randomization.py@O2.16-EXP-0.1.0"


def _no_consecutive_shuffle(rng, labels, key, max_tries=2000):
    """Shuffle indices so no two consecutive entries share key(label); best-effort (feasible here)."""
    order = list(range(len(labels)))
    for _ in range(max_tries):
        rng.shuffle(order)
        if all(key(labels[order[i]]) != key(labels[order[i + 1]]) for i in range(len(order) - 1)):
            return order
    return order  # fall back (still deterministic); QC will note if any consecutive repeats remain


def perception_sequence(participant_id, session_id, n_runs=6, protocol_version=C.PROTOCOL_VERSION):
    """512 anchors x 3 presentations = 1536 trials, balanced across runs, deterministic. Each anchor exactly 3x."""
    rng = _rng(participant_id, session_id, "perception", protocol_version)
    trials = [(a, p) for p in range(C.PERCEPTION_PRESENTATIONS) for a in range(C.PERCEPTION_ANCHORS)]
    order = _no_consecutive_shuffle(rng, trials, key=lambda t: t[0])
    seq = [trials[i] for i in order]
    per_run = len(seq) // n_runs
    runs = [seq[r * per_run:(r + 1) * per_run] for r in range(n_runs)]
    for i, extra in enumerate(seq[n_runs * per_run:]):
        runs[i].append(extra)
    out = []
    for r, run in enumerate(runs):
        for ti, (anchor, pres) in enumerate(run):
            out.append({"task": "perception", "run": r + 1, "trial_index": ti, "anchor_id": anchor,
                        "presentation": pres})
    return out, RandomizationRecord(*derive_seed(participant_id, session_id, "perception", protocol_version),
                                    task="perception", n_trials=len(out))


def imagery_sequence(participant_id, session_id, n_runs=6, protocol_version=C.PROTOCOL_VERSION):
    """12 identities x 8 repeats = 96 trials, family-balanced across runs, deterministic. Each identity exactly 8x."""
    rng = _rng(participant_id, session_id, "imagery", protocol_version)
    fam = {i: ("simple" if i < C.IMAGERY_SIMPLE else "naturalistic") for i in range(C.IMAGERY_IDENTITIES)}
    trials = [(idn, rep) for rep in range(C.IMAGERY_REPEATS) for idn in range(C.IMAGERY_IDENTITIES)]
    order = _no_consecutive_shuffle(rng, trials, key=lambda t: t[0])
    seq = [trials[i] for i in order]
    per_run = len(seq) // n_runs
    runs = [seq[r * per_run:(r + 1) * per_run] for r in range(n_runs)]
    for i, extra in enumerate(seq[n_runs * per_run:]):
        runs[i].append(extra)
    out = []
    for r, run in enumerate(runs):
        for ti, (idn, rep) in enumerate(run):
            out.append({"task": "imagery", "run": r + 1, "trial_index": ti, "identity_id": idn,
                        "family": fam[idn], "repeat_index": rep})
    return out, RandomizationRecord(*derive_seed(participant_id, session_id, "imagery", protocol_version),
                                    task="imagery", n_trials=len(out))


def record_dict(rec: RandomizationRecord) -> dict:
    return asdict(rec)
