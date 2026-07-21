"""Least-favourable null family and calibrated detector (Gate M1.2).

The Gate M1.1 detector calibrated against a *pooled* W0+W1 null and reported
~0.10 false-positive rate against W1 -- short of a 0.05 confirmatory standard.
It also reported power 1.00 at n_id=64 where the W2 effect mean was -0.002,
i.e. it "detected" a transformation because W2 was merely *less negative* than
the null. Both faults are fixed here.

Two corrections:

1. **Least-favourable calibration.** The threshold is the max over per-null 95th
   percentiles, not a pooled quantile, so ``max_j P_{H0j}(reject) <= 0.05``.
2. **A positive smallest effect of interest.** Rejection requires BOTH
   ``T > c_alpha`` AND ``delta_R2 > delta_min`` with ``delta_min > 0``. Being
   less-negative than a null is not detection.

The scientifically decisive world is **W1c (incomplete observed features)**:
source and target are both driven by latent features, but the analyst observes
only a *subset*. Source-state neural activity then carries information about the
hidden features that the feature battery lacks, producing **positive incremental
source-state value with no state transformation whatsoever**. W1c is the boundary
of what E-M1 can identify, and it must be visible rather than hidden.

Canonical estimand name:
    Incremental source-state predictive value beyond the preregistered
    stimulus-feature battery,  dR2_source|F = R2(Y|F,X) - R2(Y|F).
Positive values mean the source-state measurements add predictive information
beyond the *measured* battery. They do **not** prove a direct neural mechanism,
and do not exclude unmeasured visual, semantic, attentional, or subject-specific
common causes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np

logger = logging.getLogger(__name__)

NullWorld = Literal["W0", "W1a", "W1b", "W1c", "W1d", "W1f"]


@dataclass
class NullCfg:
    n_identities: int = 12
    n_voxels: int = 120
    n_features_observed: int = 8
    n_features_hidden: int = 6
    rank: int = 6
    target_noise: float = 0.5
    feature_noise: float = 0.5


def _lr(rng, o, i, r):
    return (rng.standard_normal((o, r)) @ rng.standard_normal((r, i))) / np.sqrt(r * i)


def simulate_null(cfg: NullCfg, world: NullWorld, seed: int = 0):
    """Generate condition-level (muX, muY, F_observed) under a null world.

    Every world here contains **no state transformation**: muY is never a
    function of muX except through shared stimulus/nuisance causes.
    """
    rng = np.random.default_rng(seed)
    K, V = cfg.n_identities, cfg.n_voxels
    Fo, Fh = cfg.n_features_observed, cfg.n_features_hidden

    f_obs = rng.standard_normal((K, Fo))
    f_hid = rng.standard_normal((K, Fh))
    A_o, B_o = _lr(rng, V, Fo, cfg.rank), _lr(rng, V, Fo, cfg.rank)
    A_h, B_h = _lr(rng, V, Fh, cfg.rank), _lr(rng, V, Fh, cfg.rank)

    if world == "W0":
        muX = rng.standard_normal((K, V))
        muY = rng.standard_normal((K, V))
        F = f_obs
    elif world == "W1a":                      # complete linear mediation
        muX, muY, F = f_obs @ A_o.T, f_obs @ B_o.T, f_obs
    elif world == "W1b":                      # nonlinear mediation
        g = np.tanh(f_obs)
        muX, muY, F = g @ A_o.T, np.tanh(g @ B_o.T), f_obs
    elif world == "W1c":                      # INCOMPLETE observed features
        muX = f_obs @ A_o.T + f_hid @ A_h.T
        muY = f_obs @ B_o.T + f_hid @ B_h.T
        F = f_obs                              # hidden features NOT exposed
    elif world == "W1d":                      # measured features are noisy
        muX, muY = f_obs @ A_o.T, f_obs @ B_o.T
        F = f_obs + cfg.feature_noise * rng.standard_normal((K, Fo))
    elif world == "W1f":                      # shared attentional nuisance
        a = rng.standard_normal((K, 1))
        wx, wy = rng.standard_normal((1, V)), rng.standard_normal((1, V))
        muX = f_obs @ A_o.T + a @ wx
        muY = f_obs @ B_o.T + a @ wy
        F = f_obs
    else:
        raise ValueError(world)

    muY = muY + cfg.target_noise * rng.standard_normal((K, V))
    return muX, muY, F


def simulate_alt(cfg: NullCfg, seed: int = 0, strength: float = 1.0):
    """W2 alternative: a genuine condition-level transform outside feature span."""
    rng = np.random.default_rng(10_000 + seed)
    K, V, Fo = cfg.n_identities, cfg.n_voxels, cfg.n_features_observed
    f = rng.standard_normal((K, Fo))
    A = _lr(rng, V, Fo, cfg.rank)
    muX = f @ A.T + 0.5 * _lr(rng, K, V, cfg.rank)
    T = _lr(rng, V, V, cfg.rank) * strength
    eta = 0.7 * _lr(rng, K, V, cfg.rank)          # residual not in span(f)
    muY = muX @ T.T + eta + cfg.target_noise * rng.standard_normal((K, V))
    return muX, muY, f


def _r2_loio(muX, muY, F, use_source: bool, lam: float = 10.0) -> float:
    """Leave-one-identity-out R2 for a feature-only or feature+source model."""
    K = muY.shape[0]
    preds, truth = [], []
    for k in range(K):
        tr = np.arange(K) != k
        Xtr = np.hstack([F[tr], muX[tr]]) if use_source else F[tr]
        xk = np.hstack([F[k], muX[k]]) if use_source else F[k]
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ muY[tr])
        preds.append(xk @ W)
        truth.append(muY[k])
    truth, preds = np.stack(truth), np.stack(preds)
    ss_res = ((truth - preds) ** 2).sum()
    ss_tot = ((truth - truth.mean(0)) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def delta_r2_source_given_F(muX, muY, F) -> float:
    """dR2_source|F = R2(Y|F,X) - R2(Y|F), leave-one-identity-out.

    A cross-validated **predictive contrast**, not an information quantity.
    Negative values are legitimate and mean the source model generalizes worse.
    """
    return _r2_loio(muX, muY, F, True) - _r2_loio(muX, muY, F, False)


def calibrate_and_evaluate(
    cfg: NullCfg,
    null_worlds: List[NullWorld],
    n_cal: int = 60,
    n_eval: int = 60,
    alpha: float = 0.05,
    delta_min: float = 0.02,
) -> Dict[str, object]:
    """Least-favourable calibration with a positive smallest effect of interest.

    Threshold = max over null worlds of the (1-alpha) quantile, using
    **calibration seeds disjoint from evaluation seeds**. Rejection requires both
    ``dR2 > threshold`` and ``dR2 > delta_min``.

    Returns per-null FPR (not just pooled), power, and the effect means.
    """
    # Calibration seeds: 0..n_cal-1. Evaluation seeds: offset by 5000. Disjoint.
    per_null_q = {}
    for w in null_worlds:
        vals = [delta_r2_source_given_F(*simulate_null(cfg, w, seed=s)) for s in range(n_cal)]
        per_null_q[w] = float(np.quantile(vals, 1 - alpha))
    threshold = max(max(per_null_q.values()), delta_min)

    fpr = {}
    for w in null_worlds:
        vals = np.array([delta_r2_source_given_F(*simulate_null(cfg, w, seed=5000 + s))
                         for s in range(n_eval)])
        rej = (vals > threshold) & (vals > delta_min)
        fpr[w] = {"fpr": float(rej.mean()), "mean_dr2": float(vals.mean())}

    alt = np.array([delta_r2_source_given_F(*simulate_alt(cfg, seed=5000 + s))
                    for s in range(n_eval)])
    rej_alt = (alt > threshold) & (alt > delta_min)

    return {
        "threshold": threshold,
        "delta_min": delta_min,
        "per_null_q95": per_null_q,
        "per_null_fpr": fpr,
        "max_fpr": max(v["fpr"] for v in fpr.values()),
        "power": float(rej_alt.mean()),
        "alt_mean_dr2": float(alt.mean()),
    }
