"""Prospective sample-complexity + phase-transition study (SYNTHETIC only). How many target observations recover
a d-dimensional private correction under varying rank/SNR/anisotropy/etc. Produces phase diagrams. These are
method-feasibility results, NOT biological laws."""
from __future__ import annotations

import numpy as np

from . import synthworld as SW
from . import metrics as MX

RECOVER_THRESHOLD = 0.8         # subspace overlap deemed "reliably recoverable"
PARTIAL_THRESHOLD = 0.5


def _sub(samples, k):
    Vt = np.linalg.svd(np.asarray(samples) - np.asarray(samples).mean(0, keepdims=True), full_matrices=False)[2]
    return Vt[:k]


def recovery_curve(private_rank=2, dim=40, snr=2.0, n_max=16, anisotropy=0.0, reps=8):
    """Overlap of the n-observation subspace estimate vs the true private subspace, as n grows."""
    cfg = SW.WorldConfig(n_subjects=1, dim=dim, private_rank=private_rank, snr=snr, anisotropy=anisotropy,
                         n_identities=1, n_repeats=n_max + reps, seed_tag="sc|%d|%.2f|%.2f" % (private_rank, snr, anisotropy))
    world = SW.generate(cfg)
    s = world["subjects"][0]; truth = s["private_basis_true"]; shared = world["shared_basis_true"]
    X = s["imagery"]; Xc = X - X.mean(0, keepdims=True)
    resid_full = Xc - Xc @ shared.T @ shared
    curve = []
    for n in range(private_rank + 1, n_max + 1):
        est = _sub(resid_full[:n], private_rank)
        curve.append((n, MX.subspace_overlap(est, truth)))
    return curve


def phase_point(private_rank, snr, dim=40, n_obs=8, anisotropy=0.0):
    curve = recovery_curve(private_rank, dim, snr, n_max=max(n_obs, private_rank + 2), anisotropy=anisotropy)
    ov = dict(curve).get(n_obs, curve[-1][1] if curve else 0.0)
    regime = ("RELIABLY_RECOVERABLE" if ov >= RECOVER_THRESHOLD else
              "PARTIALLY_RECOVERABLE" if ov >= PARTIAL_THRESHOLD else "UNRECOVERABLE")
    return {"private_rank": private_rank, "snr": snr, "n_obs": n_obs, "anisotropy": anisotropy,
            "overlap": float(ov), "regime": regime}


def phase_diagram(ranks=(1, 2, 3, 4), snrs=(0.5, 1.0, 2.0, 4.0), n_obs=8):
    grid = []
    for pr in ranks:
        for snr in snrs:
            grid.append(phase_point(pr, snr, n_obs=n_obs))
    return {"n_obs": n_obs, "recover_threshold": RECOVER_THRESHOLD, "partial_threshold": PARTIAL_THRESHOLD,
            "grid": grid, "note": "synthetic method-feasibility phase diagram; NOT a biological law"}


def sample_complexity_summary():
    """Simulation-based: minimum n reaching RECOVER_THRESHOLD across rank/SNR."""
    out = []
    for pr in (1, 2, 3):
        for snr in (1.0, 2.0, 4.0):
            curve = recovery_curve(pr, snr=snr, n_max=16)
            n_star = next((n for n, ov in curve if ov >= RECOVER_THRESHOLD), None)
            out.append({"private_rank": pr, "snr": snr, "n_star_for_reliable_recovery": n_star})
    return {"rows": out, "monotone_expectation": "n* increases with rank and decreases with SNR (verify in rows)",
            "caveat": "synthetic; establishes method feasibility + expected scaling only"}
