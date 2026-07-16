"""Partial-conjunction tests for the cross-shift global hypothesis.

The primary scientific statement of the NCD program is compound:

    ARM-B improves generalization in at least 2 of 5 preregistered shift regimes,
    relative to ARM-F.

An earlier version of the statistical plan proposed to test this by requiring
">= 2 BH-FDR discoveries among the 5 shifts". **That does not formally calibrate
the compound statement.** BH controls the expected proportion of false discoveries
among rejections; it says nothing about the error rate of the derived assertion
"at least 2 are non-null". Counting BH discoveries and thresholding the count is a
post-hoc rule with no Type-I guarantee for the compound claim.

The correct instrument is the **partial conjunction (PC) test** of
Benjamini & Heller (2008), which tests

    H_0^{r/n} : at most (r - 1) of the n hypotheses are non-null
    H_1^{r/n} : at least r of the n hypotheses are non-null

directly, with a valid p-value.

The subtlety that makes this worth doing properly: for r = 2, a configuration with
**exactly one** true effect is still under the null. A valid r=2 test must not
reject when only one shift truly improves. That is precisely the failure mode the
"count BH discoveries" rule is prone to, and it is the failure mode that would let
a single lucky shift masquerade as multi-shift generalization.

References:
    - Benjamini, Y. & Heller, R. (2008) Screening for partial conjunction
      hypotheses. Biometrics 64(4):1215-1222.
    - Simes, R.J. (1986) An improved Bonferroni procedure for multiple tests of
      significance. Biometrika 73(3):751-754.
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)

Method = Literal["bonferroni", "simes"]


def partial_conjunction_pvalue(
    pvalues: Sequence[float],
    r: int,
    method: Method = "bonferroni",
) -> float:
    """p-value for the partial conjunction null H_0^{r/n}.

    Tests "at least ``r`` of the ``n`` hypotheses are non-null" against the null
    that at most ``r - 1`` are.

    Two constructions, from Benjamini & Heller (2008). Let ``p_(1) <= ... <= p_(n)``
    be the ordered p-values:

    * ``bonferroni``: ``p^{r/n} = (n - r + 1) * p_(r)``.
      **Valid under arbitrary dependence** between the per-shift statistics. This is
      the default because our five shifts share subjects, a model, and a codebase,
      so their statistics are certainly dependent and the direction of that
      dependence is not known.
    * ``simes``: ``p^{r/n} = min_{i=r..n} [ (n - i + 1) / (i - r + 1) * p_(i) ]``.
      More powerful, but requires independence or positive regression dependence
      (PRDS). Use only as a sensitivity analysis, never as the primary test.

    Limiting cases worth noting as sanity anchors: ``r = 1`` gives the Bonferroni
    global test ``n * p_(1)``; ``r = n`` gives ``p_(n)``, the classic max-p
    intersection-union test.

    Args:
        pvalues: One-sided p-values, one per hypothesis, each testing the
            preregistered effect direction.
        r: How many must be non-null. Must satisfy ``1 <= r <= n``.
        method: ``"bonferroni"`` (dependence-robust) or ``"simes"`` (needs PRDS).

    Returns:
        The partial-conjunction p-value, clipped to ``[0, 1]``.

    Raises:
        ValueError: If ``r`` is out of range or a p-value is outside ``[0, 1]``.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        raise ValueError("need at least one p-value")
    if not 1 <= r <= n:
        raise ValueError(f"r must satisfy 1 <= r <= n; got r={r}, n={n}")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must be finite and within [0, 1]")

    p_sorted = np.sort(p)

    if method == "bonferroni":
        # Only the r-th smallest matters: if fewer than r are small, p_(r) is large.
        pc = (n - r + 1) * p_sorted[r - 1]
    elif method == "simes":
        idx = np.arange(r, n + 1)  # 1-based positions i = r..n
        pc = float(np.min((n - idx + 1) / (idx - r + 1) * p_sorted[r - 1 :]))
    else:
        raise ValueError(f"unknown method: {method!r}")

    return float(min(max(pc, 0.0), 1.0))


def partial_conjunction_report(
    pvalues: Sequence[float],
    labels: Sequence[str],
    r: int,
    alpha: float = 0.05,
) -> dict:
    """Primary PC test plus a dependence sensitivity analysis, in one record.

    Reports the dependence-robust Bonferroni PC test as primary and the Simes PC
    test as a sensitivity analysis. **A disagreement between them is itself the
    finding**: it means the conclusion rests on an unverified PRDS assumption, and
    it must be reported rather than resolved by picking the smaller p-value.

    Args:
        pvalues: One-sided p-values in the preregistered direction.
        labels: Shift names, aligned with ``pvalues``.
        r: How many shifts must improve for the compound claim.
        alpha: Significance level for the global test.

    Returns:
        A dict suitable for the experiment registry.
    """
    if len(pvalues) != len(labels):
        raise ValueError("pvalues and labels must be the same length")

    pc_bonf = partial_conjunction_pvalue(pvalues, r, "bonferroni")
    pc_simes = partial_conjunction_pvalue(pvalues, r, "simes")
    order = np.argsort(np.asarray(pvalues, dtype=float))

    return {
        "test": f"partial_conjunction_{r}_of_{len(pvalues)}",
        "r": r,
        "n": len(pvalues),
        "alpha": alpha,
        "per_shift": {labels[i]: float(pvalues[i]) for i in order},
        "pc_pvalue_bonferroni": pc_bonf,
        "pc_pvalue_simes": pc_simes,
        "primary_method": "bonferroni",
        "reject_global_null": bool(pc_bonf <= alpha),
        "sensitivity_agrees": bool((pc_bonf <= alpha) == (pc_simes <= alpha)),
        "note": (
            "Bonferroni PC is valid under arbitrary dependence and is primary. "
            "Simes PC requires independence/PRDS and is a sensitivity analysis "
            "only. If the two disagree, the conclusion depends on an unverified "
            "dependence assumption and must be reported as such."
        ),
    }
