"""Simulation-validated tests for the cross-shift partial-conjunction test.

The statistical plan requires the analysis pipeline to be validated on synthetic
data with known ground truth before it touches real results (SAP section 7). That
rule exists because ``pcd_neuroscience_analysis.py`` once produced entirely
plausible figures from untrained weights (F-002).

The decisive test here is ``test_type1_error_with_exactly_one_true_shift``. For
r = 2, a world with **exactly one** real effect is still under the null, and a
valid test must not reject it. That is the configuration in which a single lucky
shift would otherwise masquerade as multi-shift generalization -- and it is
precisely what the discarded "count >= 2 BH discoveries" rule fails to control,
as ``test_counting_bh_discoveries_does_not_calibrate_the_compound_claim``
demonstrates.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmri2img.stats.partial_conjunction import (
    partial_conjunction_pvalue,
    partial_conjunction_report,
)

N_SIM = 20000
ALPHA = 0.05
RNG = np.random.default_rng(20260716)


def _one_sided_p(z: np.ndarray) -> np.ndarray:
    """One-sided p-values in the preregistered direction (larger z = improvement)."""
    from math import erf, sqrt

    vec = np.vectorize(lambda x: 0.5 * (1.0 - erf(x / sqrt(2.0))))
    return vec(z)


def _simulate(n_true: int, n: int = 5, effect: float = 3.0, rho: float = 0.0,
              n_sim: int = N_SIM, effects: np.ndarray | None = None) -> np.ndarray:
    """Simulate per-shift z-statistics and return partial-conjunction p-values."""
    mu = np.zeros(n)
    if effects is not None:
        mu = effects
    elif n_true > 0:
        mu[:n_true] = effect

    if rho == 0.0:
        z = RNG.standard_normal((n_sim, n)) + mu
    else:
        # Equicorrelated statistics via a shared latent factor.
        shared = RNG.standard_normal((n_sim, 1))
        indep = RNG.standard_normal((n_sim, n))
        z = np.sqrt(rho) * shared + np.sqrt(1 - rho) * indep + mu

    p = _one_sided_p(z)
    return np.array([partial_conjunction_pvalue(row, r=2, method="bonferroni") for row in p])


# ---------------------------------------------------------------------------
# Type-I error under the r=2 null: at most ONE shift is truly non-null
# ---------------------------------------------------------------------------

def test_type1_error_all_null() -> None:
    """No true effects: rejection rate must be <= alpha."""
    pc = _simulate(n_true=0)
    rate = float((pc <= ALPHA).mean())
    assert rate <= ALPHA + 0.01, f"all-null Type-I = {rate:.4f}, exceeds alpha={ALPHA}"


def test_type1_error_with_exactly_one_true_shift() -> None:
    """THE decisive calibration test.

    With exactly one true effect, H_0^{2/5} still holds ("at most 1 non-null").
    A valid r=2 test must not reject, no matter how strong that one effect is.
    This is what stops a single lucky shift from being sold as multi-shift
    generalization.
    """
    for effect in [2.0, 3.0, 5.0, 8.0]:
        pc = _simulate(n_true=1, effect=effect)
        rate = float((pc <= ALPHA).mean())
        assert rate <= ALPHA + 0.01, (
            f"one true shift (effect={effect}): Type-I = {rate:.4f} > alpha. The "
            "test is rejecting the r=2 null when only ONE shift is real."
        )


def test_type1_error_with_one_enormous_true_shift_stays_controlled() -> None:
    """Even an overwhelming single effect must not trigger the r=2 claim."""
    pc = _simulate(n_true=1, effect=20.0, n_sim=5000)
    rate = float((pc <= ALPHA).mean())
    assert rate <= ALPHA + 0.01, (
        f"a single overwhelming effect produced rejection rate {rate:.4f}; the "
        "r=2 compound claim is not calibrated."
    )


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_type1_error_under_correlated_statistics(rho: float) -> None:
    """Our five shifts share subjects, a model and a codebase, so they are dependent.

    The Bonferroni PC construction is valid under arbitrary dependence; this
    confirms it empirically at realistic correlations.
    """
    pc = _simulate(n_true=1, rho=rho)
    rate = float((pc <= ALPHA).mean())
    assert rate <= ALPHA + 0.01, (
        f"rho={rho}: Type-I = {rate:.4f} > alpha under correlated statistics"
    )


def test_type1_error_heterogeneous_effects_still_one_true() -> None:
    """One real effect plus four small-but-nonzero nuisances stays under control."""
    effects = np.array([4.0, 0.3, 0.2, 0.1, 0.25])  # only the first is meaningful
    pc = _simulate(n_true=0, effects=effects)
    rate = float((pc <= ALPHA).mean())
    # Nuisances are non-zero so this is technically not the exact null; the point
    # is that near-null drift must not manufacture a 2-of-5 rejection.
    assert rate <= 0.10, (
        f"heterogeneous near-null effects produced rejection rate {rate:.4f}; "
        "small drifts across shifts are being read as multi-shift generalization."
    )


# ---------------------------------------------------------------------------
# Power under the alternative: >= 2 truly non-null
# ---------------------------------------------------------------------------

def test_power_with_two_true_shifts() -> None:
    """With two real effects the test should reject often; this is the alternative."""
    pc = _simulate(n_true=2, effect=4.0)
    rate = float((pc <= ALPHA).mean())
    assert rate > 0.5, (
        f"power with two strong true shifts is only {rate:.3f}; the test is too "
        "conservative to be usable as the primary global test."
    )


def test_power_increases_with_number_of_true_shifts() -> None:
    """Monotonicity: more true shifts, more power."""
    rates = [float((_simulate(n_true=k, effect=3.5, n_sim=4000) <= ALPHA).mean())
             for k in [2, 3, 4, 5]]
    assert all(b >= a - 0.05 for a, b in zip(rates, rates[1:])), (
        f"power not monotone in the number of true shifts: {rates}"
    )


def test_power_is_lower_under_strong_correlation() -> None:
    """Sanity: dependence costs power. Documented, not corrected."""
    indep = float((_simulate(n_true=2, effect=3.5, rho=0.0, n_sim=4000) <= ALPHA).mean())
    dep = float((_simulate(n_true=2, effect=3.5, rho=0.9, n_sim=4000) <= ALPHA).mean())
    assert dep <= indep + 0.05, f"expected dependence to not increase power: {indep} vs {dep}"


# ---------------------------------------------------------------------------
# The rule this replaces
# ---------------------------------------------------------------------------

def test_counting_bh_discoveries_does_not_calibrate_the_compound_claim() -> None:
    """Why the SAP's original ">= 2 BH discoveries" rule was wrong.

    Under exactly ONE true effect (still the r=2 null), the discarded rule rejects
    far more often than the calibrated PC test. BH controls the false-discovery
    proportion among rejections; it grants no error control to the derived
    assertion "at least 2 are non-null".
    """
    def bh_reject_count(p: np.ndarray, alpha: float = 0.05) -> int:
        n = p.size
        ps = np.sort(p)
        thresh = alpha * np.arange(1, n + 1) / n
        below = np.nonzero(ps <= thresh)[0]
        return int(below[-1] + 1) if below.size else 0

    n_sim = 8000
    z = RNG.standard_normal((n_sim, 5))
    z[:, 0] += 5.0  # exactly one true, overwhelming effect
    p = _one_sided_p(z)

    bh_rule = np.array([bh_reject_count(row) >= 2 for row in p])
    pc_rule = np.array([partial_conjunction_pvalue(row, r=2) <= ALPHA for row in p])

    bh_rate, pc_rate = float(bh_rule.mean()), float(pc_rule.mean())
    assert pc_rate <= ALPHA + 0.01, "PC test must stay calibrated"
    assert bh_rate > pc_rate, (
        f"expected the BH-count rule to be more liberal than PC "
        f"(BH={bh_rate:.4f}, PC={pc_rate:.4f}); if not, the demonstration is void"
    )


# ---------------------------------------------------------------------------
# Construction correctness
# ---------------------------------------------------------------------------

def test_bonferroni_pc_matches_closed_form() -> None:
    """p^{r/n} = (n - r + 1) * p_(r)."""
    p = [0.30, 0.01, 0.20, 0.04, 0.50]
    # sorted: 0.01, 0.04, 0.20, 0.30, 0.50 -> p_(2) = 0.04; (5-2+1)*0.04 = 0.16
    assert partial_conjunction_pvalue(p, r=2) == pytest.approx(0.16)


def test_r_equals_1_reduces_to_bonferroni_global_test() -> None:
    p = [0.30, 0.01, 0.20, 0.04, 0.50]
    assert partial_conjunction_pvalue(p, r=1) == pytest.approx(5 * 0.01)


def test_r_equals_n_reduces_to_max_p_intersection_union() -> None:
    p = [0.30, 0.01, 0.20, 0.04, 0.50]
    assert partial_conjunction_pvalue(p, r=5) == pytest.approx(0.50)


def test_simes_is_never_more_conservative_than_bonferroni() -> None:
    for _ in range(200):
        p = RNG.uniform(size=5)
        b = partial_conjunction_pvalue(p, r=2, method="bonferroni")
        s = partial_conjunction_pvalue(p, r=2, method="simes")
        assert s <= b + 1e-12, "Simes PC should be at least as powerful as Bonferroni PC"


def test_pvalue_is_clipped_to_unit_interval() -> None:
    assert partial_conjunction_pvalue([0.9, 0.9, 0.9, 0.9, 0.9], r=2) == 1.0


@pytest.mark.parametrize("bad_r", [0, 6, -1])
def test_invalid_r_rejected(bad_r: int) -> None:
    with pytest.raises(ValueError, match="1 <= r <= n"):
        partial_conjunction_pvalue([0.1] * 5, r=bad_r)


def test_invalid_pvalues_rejected() -> None:
    with pytest.raises(ValueError, match="within"):
        partial_conjunction_pvalue([0.1, 1.5, 0.2], r=2)


def test_report_flags_sensitivity_disagreement() -> None:
    """A Bonferroni/Simes disagreement must surface, not be silently resolved."""
    labels = ["in_dist", "synthetic_ood", "reduced_data", "noise", "subject"]
    rep = partial_conjunction_report([0.30, 0.02, 0.03, 0.04, 0.50], labels, r=2)

    assert rep["test"] == "partial_conjunction_2_of_5"
    assert rep["primary_method"] == "bonferroni"
    assert rep["pc_pvalue_simes"] <= rep["pc_pvalue_bonferroni"]
    assert set(rep["per_shift"]) == set(labels)
    assert isinstance(rep["sensitivity_agrees"], bool)
