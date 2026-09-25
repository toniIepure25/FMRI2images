"""Participant-level inference: exact sign-flip, Holm, permutation nulls, bootstrap. Participant is always the
inferential unit; schedules/folds/repeats are never independent N."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


def seed_uint64(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def rng(text: str):
    return np.random.Generator(np.random.PCG64(seed_uint64(text)))


def signflip_p_onesided(effects) -> float:
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


def permutation_p(observed, null_samples, one_sided_high=True):
    null = np.asarray(null_samples, np.float64); n = len(null)
    if one_sided_high:
        return float((np.sum(null >= observed) + 1) / (n + 1))
    return float((np.sum(null <= observed) + 1) / (n + 1))


def bootstrap_ci(values, n_boot=2000, alpha=0.05, name="boot"):
    v = np.asarray(values, np.float64); r = rng("bootstrap|%s|%d" % (name, len(v)))
    stats = [float(np.median(v[r.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"median": float(np.median(v)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def min_signflip_p(n):
    return 1.0 / (2 ** n)
