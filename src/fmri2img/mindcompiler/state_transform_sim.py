"""Identifiability simulation for condition-level state transformations (Gate M1.1).

Corrects the estimand of the Gate M1 simulation. The earlier version modelled
**single-trial** perception->imagery coupling, which the Roy design cannot
observe: vision and imagery were acquired in **separate runs** and paired
**randomly within stimulus identity**. Single-trial coupling therefore averages
away and is not identifiable here (retained as W4, explicitly non-primary).

The scientific target is a **condition-level** transformation that generalizes
to unseen content. Four worlds with known ground truth:

* **W0 condition-template null** -- mu^Y_k is arbitrary, unrelated to mu^X_k. A
  known-condition lookup can succeed; held-out-content prediction must fail.
* **W1 shared-feature mediation** -- mu^X_k = A f_k, mu^Y_k = B f_k. Vision and
  imagery are related *only* because both encode observable features f_k. **A
  condition-level linear map from mu^X to mu^Y still exists (T = B A^+), so
  neural-source prediction generalizes -- but it adds NOTHING beyond f_k.** This
  is the world that a naive "neural source predicts imagery" claim cannot rule
  out, and the reason the estimand must be *unique neural contribution beyond
  features*.
* **W2 generalizable state transform** -- mu^Y_k = T(mu^X_k) + eta_k, with eta_k
  a condition-specific residual NOT expressible in f_k. Neural source adds a
  unique contribution beyond features. The primary positive world.
* **W3 mixture** -- mu^Y_k = T(mu^X_k) + B f_k + u_k.

The primary estimand is the **unique neural contribution beyond stimulus
features**, under complete content hold-out (LOIO). It must be ~0 in W0 and W1
and > 0 in W2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

logger = logging.getLogger(__name__)

World = Literal["W0", "W1", "W2", "W3", "W4"]


@dataclass
class Cfg:
    n_identities: int = 12
    n_repeats: int = 8
    n_voxels: int = 120
    n_features: int = 10
    neural_rank: int = 6
    source_snr: float = 1.0
    target_snr: float = 1.0
    transform_strength: float = 1.0
    feature_strength: float = 1.0


def _low_rank(rng, out_dim, in_dim, rank):
    return (rng.standard_normal((out_dim, rank)) @ rng.standard_normal((rank, in_dim))) / np.sqrt(rank * in_dim)


def simulate(cfg: Cfg, world: World, seed: int = 0):
    """Generate condition templates, features, and independently-noised repeats.

    Returns a dict with per-trial ``perception``/``imagery`` (T, V), ``identity``
    (T,), condition means ``muX``/``muY`` (K, V), and ``features`` (K, F).
    Perception and imagery repeats carry **independent** trial noise, and are
    paired **randomly within identity** -- reproducing Roy's cross-run pairing.
    """
    rng = np.random.default_rng(seed)
    K, V, F = cfg.n_identities, cfg.n_voxels, cfg.n_features
    f = rng.standard_normal((K, F))                      # stimulus features
    A = _low_rank(rng, V, F, min(F, cfg.neural_rank))
    muX = cfg.feature_strength * (f @ A.T)               # perception templates from features
    # add neural structure not in features so W2's transform has something to carry
    muX = muX + _low_rank(rng, K, V, cfg.neural_rank) * 0.5

    T = _low_rank(rng, V, V, cfg.neural_rank) * cfg.transform_strength
    B = _low_rank(rng, V, F, min(F, cfg.neural_rank)) * cfg.feature_strength

    if world == "W0":
        muY = rng.standard_normal((K, V))
    elif world == "W1":
        muY = f @ B.T                                    # imagery = features only
    elif world == "W2":
        eta = _low_rank(rng, K, V, cfg.neural_rank) * 0.7   # residual NOT in features
        muY = muX @ T.T + eta
    elif world == "W3":
        u = rng.standard_normal((K, V)) * 0.3
        muY = muX @ T.T + f @ B.T + u
    elif world == "W4":
        muY = None                                       # single-trial only; see below
    else:
        raise ValueError(world)

    ident = np.repeat(np.arange(K), cfg.n_repeats)
    Xn = (1.0 / cfg.source_snr) * rng.standard_normal((K * cfg.n_repeats, V))
    perception = muX[ident] + Xn

    if world == "W4":
        # Single-trial coupling: imagery built from THIS perception trial.
        Wtr = _low_rank(rng, V, V, cfg.neural_rank) * cfg.transform_strength
        imagery = perception @ Wtr.T + (1.0 / cfg.target_snr) * rng.standard_normal((K * cfg.n_repeats, V))
    else:
        Yn = (1.0 / cfg.target_snr) * rng.standard_normal((K * cfg.n_repeats, V))
        imagery = muY[ident] + Yn

    # Random within-identity pairing: permute imagery repeats within each identity
    # so no imagery trial stays aligned to the perception trial it was generated
    # from (matters only for W4; harmless elsewhere).
    order = np.arange(len(ident))
    for k in range(K):
        idx = np.where(ident == k)[0]
        order[idx] = rng.permutation(idx)
    imagery = imagery[order]

    return dict(perception=perception, imagery=imagery, identity=ident,
                muX=muX, muY=(muY if world != "W4" else None), features=f)


def _ridge_fit(X, Y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def _r2(Yt, Yp):
    ss_res = ((Yt - Yp) ** 2).sum()
    ss_tot = ((Yt - Yt.mean(0)) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def loio_variance_partition(cfg: Cfg, world: World, seed: int = 0, lam: float = 10.0) -> Dict[str, float]:
    """Leave-one-identity-out unique-neural-beyond-features partition.

    For each held-out identity, predict its (condition-mean) imagery from
    (a) stimulus features, (b) perception neural source, (c) both -- all fit on
    the OTHER identities only. Reports held-out R2 for each and the unique neural
    contribution R2(both) - R2(features).

    The primary number is ``unique_neural``: ~0 under W0/W1, > 0 under W2/W3.
    """
    d = simulate(cfg, world, seed=seed)
    P, Y, ident, feat = d["perception"], d["imagery"], d["identity"], d["features"]
    K = cfg.n_identities

    # Condition-mean representations (the level at which Roy's transform lives).
    Pm = np.stack([P[ident == k].mean(0) for k in range(K)])
    Ym = np.stack([Y[ident == k].mean(0) for k in range(K)])

    preds = {"features": [], "neural": [], "both": []}
    truth = []
    for k in range(K):
        tr = np.arange(K) != k
        Ff, Nf = feat[tr], Pm[tr]
        Both = np.hstack([Ff, Nf])
        for name, Xtr, xk in [("features", Ff, feat[k]),
                              ("neural", Nf, Pm[k]),
                              ("both", Both, np.hstack([feat[k], Pm[k]]))]:
            W = _ridge_fit(Xtr, Ym[tr], lam)
            preds[name].append(xk @ W)
        truth.append(Ym[k])

    truth = np.stack(truth)
    r2 = {k: _r2(truth, np.stack(v)) for k, v in preds.items()}
    return {
        "r2_features": r2["features"],
        "r2_neural": r2["neural"],
        "r2_both": r2["both"],
        "unique_neural": r2["both"] - r2["features"],
        "unique_features": r2["both"] - r2["neural"],
    }
