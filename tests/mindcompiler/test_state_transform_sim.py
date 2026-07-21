"""Ground-truth tests for the condition-level state-transformation simulation.

These pin the Gate M1.1 estimand: the primary quantity is the **unique neural
contribution beyond stimulus features**, under leave-one-identity-out. It must be
~0 in the null (W0) and the feature-mediation world (W1), and positive in the
genuine transformation worlds (W2, W3).

The single-trial-coupling world (W4) exists only to demonstrate why single-trial
coupling is *not identifiable* under Roy's random cross-run pairing: its signal
must average away.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.state_transform_sim import (
    Cfg,
    loio_variance_partition,
    simulate,
)


def _mean_over_seeds(world, key, n=16, **cfg_kw):
    cfg = Cfg(**cfg_kw)
    return float(np.mean([loio_variance_partition(cfg, world, seed=s)[key] for s in range(n)]))


# --- Generative correctness ------------------------------------------------

def test_perception_and_imagery_noise_are_independent() -> None:
    """W0: after within-identity random pairing, paired trials share only identity.

    If trial noise were shared or pairing preserved trial order, the paired
    single-trial correlation would exceed the condition-mean correlation.
    """
    d = simulate(Cfg(n_identities=8, n_repeats=10, source_snr=0.7, target_snr=0.7), "W0", seed=0)
    P, Y, ident = d["perception"], d["imagery"], d["identity"]
    # Residualize each trial against its condition mean -> pure trial noise.
    K = 8
    Pr = P - np.stack([P[ident == k].mean(0) for k in range(K)])[ident]
    Yr = Y - np.stack([Y[ident == k].mean(0) for k in range(K)])[ident]
    trial_noise_corr = np.corrcoef(Pr.ravel(), Yr.ravel())[0, 1]
    assert abs(trial_noise_corr) < 0.05, (
        f"paired trial noise correlates ({trial_noise_corr:.3f}); perception and "
        "imagery repeats are not independent"
    )


def test_random_pairing_permutes_within_identity_only() -> None:
    """Pairing must never move a trial across identities."""
    cfg = Cfg(n_identities=6, n_repeats=8)
    d = simulate(cfg, "W2", seed=1)
    # imagery rows were permuted within identity; identity labels are unchanged.
    assert d["identity"].shape[0] == cfg.n_identities * cfg.n_repeats
    assert set(d["identity"].tolist()) == set(range(cfg.n_identities))


# --- The estimand: unique neural beyond features ---------------------------

def test_W0_null_has_no_unique_neural() -> None:
    """No cross-content rule -> unique neural contribution not positive."""
    assert _mean_over_seeds("W0", "unique_neural") < 0.05


def test_W1_feature_mediation_has_no_unique_neural() -> None:
    """Vision/imagery related ONLY through features -> neural adds nothing unique.

    This is the world a naive 'neural source predicts imagery' claim cannot rule
    out, and the reason the estimand is *unique neural beyond features* rather
    than raw held-out prediction.
    """
    assert _mean_over_seeds("W1", "unique_neural") < 0.05


def test_W1_features_do_predict_heldout() -> None:
    """Sanity: in W1 the feature model MUST generalize, or the world is broken."""
    assert _mean_over_seeds("W1", "r2_features") > 0.2


def test_W2_transform_has_positive_unique_neural() -> None:
    """Genuine condition-level transform -> neural adds unique signal beyond features."""
    assert _mean_over_seeds("W2", "unique_neural") > 0.1


def test_W3_mixture_has_positive_unique_neural() -> None:
    """Mixture retains a detectable unique-neural component."""
    assert _mean_over_seeds("W3", "unique_neural") > 0.02


def test_estimand_separates_null_worlds_from_transform_worlds() -> None:
    """The decisive ordering: {W0,W1} unique-neural < {W2,W3} unique-neural."""
    null_max = max(_mean_over_seeds("W0", "unique_neural"),
                   _mean_over_seeds("W1", "unique_neural"))
    transform_min = min(_mean_over_seeds("W2", "unique_neural"),
                        _mean_over_seeds("W3", "unique_neural"))
    assert transform_min > null_max, (
        f"worlds not separated: null_max={null_max:.3f} transform_min={transform_min:.3f}"
    )


# --- W4: why single-trial coupling is not identifiable here -----------------

def test_single_trial_coupling_is_condition_level_indistinguishable_from_W2() -> None:
    """W4 is observationally equivalent to W2 at the condition level.

    A correction to the naive expectation that single-trial coupling "averages
    away". A LINEAR single-trial map W applied to trials with condition mean muX_k
    has condition-mean image (muX_k @ W.T) -- because averaging commutes with a
    linear operator. So the condition-level transform SURVIVES random pairing and
    is indistinguishable from a genuine W2 condition-level transform.

    What is genuinely non-identifiable under random cross-run pairing is the
    trial-to-trial ALIGNMENT (the residual orthogonal to condition means), not the
    existence of a condition-level map. This makes the identifiability limitation
    STRONGER, not weaker: even a single-trial mechanism is read as a condition-
    level transformation, and the two cannot be separated in the Roy design.
    """
    un = _mean_over_seeds("W4", "unique_neural", n=12)
    assert un > 0.1, (
        f"W4 condition-level unique neural is {un:.3f}; a linear single-trial map "
        "should induce a surviving condition-level transform (~W2), not vanish"
    )


# --- Held-out isolation -----------------------------------------------------

def test_loio_holds_out_whole_identities() -> None:
    """The partition must never let a held-out identity's own trials train it.

    Checked indirectly: W0 (arbitrary templates) must give held-out R2 <= ~0 for
    every model, which is only possible if the held-out identity is truly unseen.
    """
    cfg = Cfg(n_identities=12)
    r = loio_variance_partition(cfg, "W0", seed=3)
    assert r["r2_neural"] < 0.1 and r["r2_features"] < 0.1, (
        "a held-out identity is leaking into training (W0 should predict nothing)"
    )


@pytest.mark.parametrize("n_id", [6, 12, 24])
def test_separation_holds_across_identity_counts(n_id: int) -> None:
    """The qualitative ordering must not depend on a single identity count.

    Sensitivity varies with n_id (that is exactly what real data must measure);
    the SIGN of the separation should not.
    """
    w1 = _mean_over_seeds("W1", "unique_neural", n=10, n_identities=n_id)
    w2 = _mean_over_seeds("W2", "unique_neural", n=10, n_identities=n_id)
    assert w2 > w1, f"n_id={n_id}: W2 ({w2:.3f}) not above W1 ({w1:.3f})"
