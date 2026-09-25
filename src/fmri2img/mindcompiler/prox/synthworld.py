"""Configurable synthetic-world generator with KNOWN ground truth. Every benchmark and identifiability study can
be scored against truth. No historical or real data; purely generative."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import statistics as ST


@dataclass
class WorldConfig:
    n_subjects: int = 12
    dim: int = 40
    shared_rank: int = 3           # cross-subject shared scaffold rank
    private_rank: int = 2          # subject-specific correction rank
    support_overlap: float = 0.5   # fraction of target energy inside perception support
    snr: float = 2.0
    n_identities: int = 12
    n_repeats: int = 8
    anisotropy: float = 0.0        # 0 = isotropic; >0 = dominant private direction
    session_drift: float = 0.0     # cross-session rotation magnitude
    site_drift: float = 0.0
    heavy_tails: bool = False
    behavior_coupling: float = 0.0
    seed_tag: str = "world"

    def to_dict(self):
        return asdict(self)


def _rot(dim, mag, r):
    if mag <= 0:
        return np.eye(dim)
    A = r.standard_normal((dim, dim)); A = mag * (A - A.T)         # skew-symmetric
    # small-rotation via (I - A/2)^-1 (I + A/2) Cayley
    I = np.eye(dim)
    return np.linalg.solve(I - A / 2, I + A / 2)


def generate(cfg: WorldConfig):
    """Returns a dict with per-subject target-state samples (imagery + recall) and the GROUND TRUTH bases."""
    r = ST.rng("synthworld|%s|%s" % (cfg.seed_tag, ST.seed_uint64(str(cfg.to_dict()))))
    shared = np.linalg.qr(r.standard_normal((cfg.dim, cfg.shared_rank)))[0]   # dim x shared_rank
    support = np.linalg.qr(r.standard_normal((cfg.dim, cfg.shared_rank + 2)))[0]
    subjects = []
    for s in range(cfg.n_subjects):
        priv = np.linalg.qr(r.standard_normal((cfg.dim, cfg.private_rank)))[0]
        behavior = float(r.standard_normal())
        def state_samples(state_tag, drift_mag):
            Rd = _rot(cfg.dim, drift_mag, r)
            samples = []
            n = cfg.n_identities * cfg.n_repeats
            for _ in range(n):
                cs = r.standard_normal(cfg.shared_rank)
                cp = r.standard_normal(cfg.private_rank)
                if cfg.anisotropy > 0:
                    cp[0] *= (1.0 + cfg.anisotropy)
                sig = shared @ cs + priv @ cp
                noise = r.standard_normal(cfg.dim) / max(cfg.snr, 1e-6)
                if cfg.heavy_tails:
                    noise *= (1.0 + np.abs(r.standard_t(3)))
                samples.append(Rd @ (sig + noise))
            return np.array(samples)
        subjects.append({
            "subject_id": "syn%03d" % (s + 1), "behavior": behavior,
            "private_basis_true": priv.T, "imagery": state_samples("imagery", 0.0),
            "recall": state_samples("recall", cfg.session_drift),
            "session2_imagery": state_samples("imagery", cfg.session_drift),
        })
    return {"config": cfg.to_dict(), "shared_basis_true": shared.T, "support_basis": support.T,
            "shared_rank": cfg.shared_rank, "private_rank": cfg.private_rank, "subjects": subjects}
