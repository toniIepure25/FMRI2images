"""
Conformal Prediction for Neural Image Decoding
================================================

Distribution-free, finite-sample coverage guarantees for fMRI-to-image
retrieval decoders.  Given any trained decoder and a held-out calibration
set, this module constructs **prediction sets** C(x) such that

    P(y_true ∈ C(x)) ≥ 1 − α

for a user-chosen α, under the sole assumption that calibration and test
points are exchangeable (Vovk et al., 2005).

Core ideas
----------
1. **Nonconformity score** s(x, y): measures how "unusual" a (brain, image)
   pair is.  We define several scores tailored to vMF neural decoders:
   - 1/κ  (learned vMF concentration — lower κ ⇒ less confident)
   - 1 − margin  (gap between top-1 and top-2 cosine similarity)
   - 1 − agreement_fraction  (fraction of experts that agree on top-1)
   - Composite weighted combination

2. **Split conformal calibration**: given n calibration examples with known
   ground-truth, compute the (1−α)(1+1/n) quantile of their nonconformity
   scores.  This threshold τ̂ defines the prediction set:

       C(x) = { y ∈ gallery : s(x, y) ≤ τ̂ }

3. **Adaptive set size**: confident trials yield small C(x) (often |C|=1);
   uncertain trials yield larger sets — a natural "abstain or hedge" signal.

References
----------
- Vovk, Gammerman & Shafer (2005). Algorithmic Learning in a Random World.
- Romano, Sesia & Candès (2020). Classification with Valid Adaptive Coverage.
- Angelopoulos et al. (2020). RAPS: Regularized Adaptive Prediction Sets.
- Angelopoulos & Bates (2023). Conformal Prediction: A Gentle Introduction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------

def kappa_nonconformity(
    kappas: NDArray[np.floating],
    *,
    invert: bool = True,
) -> NDArray[np.floating]:
    """1/κ — higher nonconformity when the model is less concentrated.

    Args:
        kappas: Per-image kappa values, shape (n,).
        invert: If True, return 1/κ (default).  If False, return −κ.

    Returns:
        Nonconformity scores, shape (n,).
    """
    kappas = np.asarray(kappas, dtype=np.float64)
    if invert:
        return 1.0 / np.maximum(kappas, 1e-8)
    return -kappas


def margin_nonconformity(
    predictions: NDArray[np.floating],
    gallery: NDArray[np.floating],
    gt_indices: NDArray[np.intp],
) -> NDArray[np.floating]:
    """1 − (cos_sim_top1 − cos_sim_top2): small margin ⇒ high nonconformity.

    Args:
        predictions: Predicted embeddings, shape (n, d), L2-normalised.
        gallery: Gallery embeddings, shape (m, d), L2-normalised.
        gt_indices: Ground-truth gallery index per query, shape (n,).

    Returns:
        Nonconformity scores, shape (n,).
    """
    sim = predictions @ gallery.T  # (n, m)
    n = sim.shape[0]
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = sim[i]
        top2 = np.partition(row, -2)[-2:]
        top2.sort()
        scores[i] = 1.0 - (top2[1] - top2[0])
    return scores


def agreement_nonconformity(
    expert_correct: NDArray[np.bool_],
) -> NDArray[np.floating]:
    """1 − fraction of experts that retrieved the correct image.

    Args:
        expert_correct: Boolean matrix (n_images, n_experts) — True when
            expert k retrieved the correct image at rank 1.

    Returns:
        Nonconformity scores, shape (n,).
    """
    expert_correct = np.asarray(expert_correct)
    frac = expert_correct.mean(axis=1)
    return 1.0 - frac


def rank_nonconformity(
    predictions: NDArray[np.floating],
    gallery: NDArray[np.floating],
    gt_indices: NDArray[np.intp],
) -> NDArray[np.floating]:
    """Rank of the ground-truth image in the sorted similarity list (0-indexed).

    Lower rank = more conforming.  This is the most natural nonconformity
    score for retrieval: s(x,y) = rank(y | x).
    """
    sim = predictions @ gallery.T
    n = sim.shape[0]
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = sim[i]
        gt_sim = row[gt_indices[i]]
        scores[i] = float((row > gt_sim).sum())
    return scores


def composite_nonconformity(
    components: Dict[str, NDArray[np.floating]],
    weights: Optional[Dict[str, float]] = None,
) -> NDArray[np.floating]:
    """Weighted combination of z-scored nonconformity components.

    Each component is z-scored (mean=0, std=1) then combined with the
    given weights (default: equal).
    """
    names = sorted(components.keys())
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}

    n = len(next(iter(components.values())))
    result = np.zeros(n, dtype=np.float64)
    for name in names:
        s = np.asarray(components[name], dtype=np.float64)
        std = s.std()
        if std < 1e-12:
            s_z = s - s.mean()
        else:
            s_z = (s - s.mean()) / std
        result += weights.get(name, 0.0) * s_z
    return result


# ---------------------------------------------------------------------------
# Split Conformal Calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    cal_scores: NDArray[np.floating],
    alpha: float,
) -> float:
    """Compute the conformal threshold τ̂ from calibration scores.

    τ̂ = the ⌈(n+1)(1−α)⌉/n quantile of cal_scores.

    This guarantees P(s_test ≤ τ̂) ≥ 1−α when calibration and test are
    exchangeable (Vovk et al., 2005).

    Args:
        cal_scores: Nonconformity scores on the calibration set, shape (n,).
        alpha: Desired miscoverage rate (e.g. 0.05 for 95% coverage).

    Returns:
        Threshold τ̂.
    """
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_scores, q_level, method="higher"))


def conformal_prediction_sets(
    test_scores_matrix: NDArray[np.floating],
    threshold: float,
) -> List[NDArray[np.intp]]:
    """Build prediction sets from per-gallery nonconformity scores.

    Args:
        test_scores_matrix: Shape (n_test, n_gallery) — nonconformity of
            each gallery item for each test query.
        threshold: Conformal threshold τ̂ from calibration.

    Returns:
        List of arrays, each containing gallery indices in the prediction set.
    """
    sets = []
    for i in range(test_scores_matrix.shape[0]):
        included = np.where(test_scores_matrix[i] <= threshold)[0]
        sets.append(included)
    return sets


# ---------------------------------------------------------------------------
# Retrieval-oriented conformal prediction
# ---------------------------------------------------------------------------

def retrieval_conformal_sets(
    predictions: NDArray[np.floating],
    gallery: NDArray[np.floating],
    gt_indices: NDArray[np.intp],
    confidence_scores: NDArray[np.floating],
    alpha: float = 0.10,
    cal_fraction: float = 0.5,
    seed: int = 42,
) -> "ConformalResult":
    """End-to-end split conformal prediction for retrieval.

    Splits images into calibration/test, calibrates using
    confidence-derived nonconformity scores, and builds prediction sets.

    The nonconformity score for each image is the rank of the ground truth
    among gallery items, but the *ordering* for set inclusion is determined
    by the confidence scores.  Concretely:

    1. On calibration: compute rank of GT for each query.
    2. τ̂ = quantile of calibration ranks at level ⌈(1−α)(1+1/n_cal)⌉.
    3. On test: C(x) = { gallery items with rank ≤ τ̂ in sorted similarities }.

    But we also provide a confidence-aware variant: include all gallery
    items whose similarity exceeds the τ̂-th closest.

    Args:
        predictions: (n, d) L2-normalised predicted embeddings.
        gallery: (m, d) L2-normalised gallery embeddings.
        gt_indices: (n,) ground-truth gallery index for each query.
        confidence_scores: (n,) per-image confidence (higher = more confident).
        alpha: Miscoverage rate.
        cal_fraction: Fraction of images used for calibration.
        seed: Random seed for the calibration/test split.

    Returns:
        ConformalResult with coverage, set sizes, and per-image details.
    """
    n = len(predictions)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_cal = int(n * cal_fraction)
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    sim = predictions @ gallery.T  # (n, m)

    # Rank-based nonconformity: rank of GT in sorted sims (0=best)
    all_ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        gt_sim = sim[i, gt_indices[i]]
        all_ranks[i] = float((sim[i] > gt_sim).sum())

    cal_ranks = all_ranks[cal_idx]
    threshold = calibrate_threshold(cal_ranks, alpha)

    test_ranks = all_ranks[test_idx]
    test_covered = test_ranks <= threshold
    empirical_coverage = float(test_covered.mean())

    # Build prediction sets for test queries
    set_sizes = np.empty(len(test_idx), dtype=np.int64)
    for j, idx in enumerate(test_idx):
        row = sim[idx]
        cutoff_sim = np.partition(row, -int(threshold) - 1)[
            -int(threshold) - 1
        ] if threshold < len(row) - 1 else row.min()
        set_sizes[j] = int((row >= cutoff_sim).sum())

    # Confidence-stratified coverage
    test_conf = confidence_scores[test_idx]
    confidence_quartiles = np.percentile(test_conf, [25, 50, 75])
    strat_coverage = {}
    labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    bounds = [-np.inf] + list(confidence_quartiles) + [np.inf]
    for k in range(4):
        mask = (test_conf >= bounds[k]) & (test_conf < bounds[k + 1])
        if mask.sum() > 0:
            strat_coverage[labels[k]] = {
                "coverage": float(test_covered[mask].mean()),
                "n": int(mask.sum()),
                "mean_set_size": float(set_sizes[mask].mean()),
            }

    return ConformalResult(
        alpha=alpha,
        n_calibration=n_cal,
        n_test=len(test_idx),
        threshold=threshold,
        guaranteed_coverage=1 - alpha,
        empirical_coverage=empirical_coverage,
        mean_set_size=float(set_sizes.mean()),
        median_set_size=float(np.median(set_sizes)),
        max_set_size=int(set_sizes.max()),
        min_set_size=int(set_sizes.min()),
        frac_singleton=float((set_sizes == 1).mean()),
        frac_empty=float((set_sizes == 0).mean()),
        stratified_coverage=strat_coverage,
        cal_indices=cal_idx,
        test_indices=test_idx,
        test_set_sizes=set_sizes,
        test_covered=test_covered,
    )


# ---------------------------------------------------------------------------
# Score-based conformal prediction (simpler, more general)
# ---------------------------------------------------------------------------

def score_based_conformal(
    nonconformity_scores: NDArray[np.floating],
    is_correct: NDArray[np.bool_],
    alpha: float = 0.10,
    cal_fraction: float = 0.5,
    seed: int = 42,
    n_bootstrap: int = 200,
) -> "ScoreConformalResult":
    """Score-based conformal prediction for binary correctness.

    Given per-image nonconformity scores and binary correctness labels,
    calibrate a threshold and evaluate coverage (fraction of correct
    predictions among those included in the prediction set).

    This variant treats retrieval as a binary task: either the top-1
    retrieved image is correct (covered) or not.  The "prediction set"
    is {accept, abstain} — accept if score ≤ τ̂.

    Args:
        nonconformity_scores: (n,) scores — higher means less conforming.
        is_correct: (n,) boolean — True if top-1 retrieval is correct.
        alpha: Miscoverage rate for calibration.
        cal_fraction: Fraction used for calibration.
        seed: Random seed.
        n_bootstrap: Number of bootstrap repetitions for CI estimation.

    Returns:
        ScoreConformalResult with coverage, acceptance rate, and CIs.
    """
    n = len(nonconformity_scores)
    rng = np.random.RandomState(seed)

    # Multiple random splits for stability
    coverages = []
    acceptance_rates = []
    accepted_accuracies = []
    thresholds = []

    for b in range(n_bootstrap):
        perm = rng.permutation(n)
        n_cal = int(n * cal_fraction)
        cal_idx = perm[:n_cal]
        test_idx = perm[n_cal:]

        cal_scores = nonconformity_scores[cal_idx]
        tau = calibrate_threshold(cal_scores, alpha)
        thresholds.append(tau)

        test_scores = nonconformity_scores[test_idx]
        test_correct = is_correct[test_idx]

        accepted = test_scores <= tau
        acceptance_rate = float(accepted.mean())
        acceptance_rates.append(acceptance_rate)

        if accepted.sum() > 0:
            accepted_acc = float(test_correct[accepted].mean())
        else:
            accepted_acc = np.nan
        accepted_accuracies.append(accepted_acc)

        # Coverage = P(correct AND accepted) / P(all test)
        # But for conformal: coverage = P(y_true ∈ C(x)) which means
        # the correct answer is within the prediction set
        # For binary: coverage = P(correct | accepted) * P(accepted)
        # But the guarantee is: P(correct item in set) ≥ 1-alpha
        # which translates to: among ALL test, fraction with correct
        # answer in prediction set ≥ 1-alpha
        # Since prediction set = {top-1 if accepted, empty if rejected}:
        # coverage = P(correct AND accepted) + P(rejected) ... no
        # Actually: coverage = P(y_true ∈ C(x))
        # If we accept, C(x) = {top-1}; coverage for this x = is_correct[x]
        # If we reject, C(x) = {} (or C(x) = gallery); depends on formulation
        # Standard: coverage = fraction of test points where GT is in set
        coverage = float((test_correct & accepted).sum() / len(test_idx))
        coverages.append(coverage)

    return ScoreConformalResult(
        alpha=alpha,
        n_total=n,
        mean_threshold=float(np.mean(thresholds)),
        std_threshold=float(np.std(thresholds)),
        mean_coverage=float(np.mean(coverages)),
        std_coverage=float(np.std(coverages)),
        coverage_ci_95=(
            float(np.percentile(coverages, 2.5)),
            float(np.percentile(coverages, 97.5)),
        ),
        mean_acceptance_rate=float(np.mean(acceptance_rates)),
        std_acceptance_rate=float(np.std(acceptance_rates)),
        mean_accepted_accuracy=float(np.nanmean(accepted_accuracies)),
        std_accepted_accuracy=float(np.nanstd(accepted_accuracies)),
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Multi-alpha sweep
# ---------------------------------------------------------------------------

def conformal_sweep(
    nonconformity_scores: NDArray[np.floating],
    is_correct: NDArray[np.bool_],
    alphas: Sequence[float] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
    cal_fraction: float = 0.5,
    seed: int = 42,
) -> List[Dict]:
    """Run conformal calibration at multiple alpha levels.

    Returns a list of dicts, one per alpha, with coverage and set metrics.
    """
    n = len(nonconformity_scores)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_cal = int(n * cal_fraction)
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    cal_scores = nonconformity_scores[cal_idx]
    test_scores = nonconformity_scores[test_idx]
    test_correct = is_correct[test_idx]

    results = []
    for alpha in alphas:
        tau = calibrate_threshold(cal_scores, alpha)
        accepted = test_scores <= tau
        n_accepted = int(accepted.sum())
        n_test = len(test_idx)

        if n_accepted > 0:
            acc_among_accepted = float(test_correct[accepted].mean())
        else:
            acc_among_accepted = float("nan")

        results.append({
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "threshold": float(tau),
            "acceptance_rate": float(accepted.mean()),
            "n_accepted": n_accepted,
            "n_rejected": n_test - n_accepted,
            "accuracy_among_accepted": acc_among_accepted,
            "accuracy_overall": float(test_correct.mean()),
            "coverage_correct_accepted": float(
                (test_correct & accepted).sum() / n_test
            ),
        })
    return results


# ---------------------------------------------------------------------------
# Cross-subject conformal transfer
# ---------------------------------------------------------------------------

def cross_subject_conformal(
    cal_scores: NDArray[np.floating],
    test_scores_by_subject: Dict[str, NDArray[np.floating]],
    is_correct_by_subject: Dict[str, NDArray[np.bool_]],
    alpha: float = 0.10,
) -> Dict[str, Dict]:
    """Calibrate on one subject, evaluate coverage on others.

    Tests whether the exchangeability assumption holds across subjects
    (it likely doesn't — quantifying the coverage gap is the key result).

    Args:
        cal_scores: Nonconformity scores from the calibration subject.
        test_scores_by_subject: {subject_id: scores} for each test subject.
        is_correct_by_subject: {subject_id: correctness} for each test subject.
        alpha: Miscoverage rate.

    Returns:
        Dict mapping subject_id to coverage statistics.
    """
    tau = calibrate_threshold(cal_scores, alpha)
    results = {}

    for subj, test_scores in test_scores_by_subject.items():
        is_correct = is_correct_by_subject[subj]
        accepted = test_scores <= tau
        n_test = len(test_scores)

        if accepted.sum() > 0:
            acc_accepted = float(is_correct[accepted].mean())
        else:
            acc_accepted = float("nan")

        # Coverage = P(GT in set)
        coverage = float((is_correct & accepted).sum() / n_test)
        coverage_gap = (1 - alpha) - coverage

        results[subj] = {
            "threshold": float(tau),
            "n_test": n_test,
            "acceptance_rate": float(accepted.mean()),
            "accuracy_among_accepted": acc_accepted,
            "empirical_coverage": coverage,
            "target_coverage": 1 - alpha,
            "coverage_gap": float(coverage_gap),
            "exchangeability_holds": coverage >= (1 - alpha - 0.05),
        }

    return results


def weighted_conformal_transfer(
    cal_scores: NDArray[np.floating],
    cal_weights: NDArray[np.floating],
    test_scores: NDArray[np.floating],
    alpha: float = 0.10,
) -> float:
    """Weighted conformal threshold for distribution shift correction.

    Uses importance weights to reweight calibration scores, following
    Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift."

    Args:
        cal_scores: Calibration nonconformity scores, shape (n_cal,).
        cal_weights: Importance weights w(x) = p_test(x)/p_cal(x), shape (n_cal,).
        test_scores: (unused — for signature consistency).
        alpha: Miscoverage rate.

    Returns:
        Weighted conformal threshold.
    """
    cal_weights = np.asarray(cal_weights, dtype=np.float64)
    cal_weights = cal_weights / cal_weights.sum()

    sorted_idx = np.argsort(cal_scores)
    sorted_scores = cal_scores[sorted_idx]
    sorted_weights = cal_weights[sorted_idx]

    cumsum = np.cumsum(sorted_weights)
    target = 1 - alpha
    idx = np.searchsorted(cumsum, target)
    idx = min(idx, len(sorted_scores) - 1)
    return float(sorted_scores[idx])


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConformalResult:
    """Result of retrieval-based split conformal prediction."""
    alpha: float
    n_calibration: int
    n_test: int
    threshold: float
    guaranteed_coverage: float
    empirical_coverage: float
    mean_set_size: float
    median_set_size: float
    max_set_size: int
    min_set_size: int
    frac_singleton: float
    frac_empty: float
    stratified_coverage: Dict[str, Dict]
    cal_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]
    test_set_sizes: NDArray[np.int64]
    test_covered: NDArray[np.bool_]

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
            "n_test": self.n_test,
            "threshold": self.threshold,
            "guaranteed_coverage": self.guaranteed_coverage,
            "empirical_coverage": self.empirical_coverage,
            "mean_set_size": self.mean_set_size,
            "median_set_size": self.median_set_size,
            "max_set_size": self.max_set_size,
            "min_set_size": self.min_set_size,
            "frac_singleton": self.frac_singleton,
            "frac_empty": self.frac_empty,
            "stratified_coverage": self.stratified_coverage,
        }


@dataclass
class ScoreConformalResult:
    """Result of score-based conformal prediction (binary accept/reject)."""
    alpha: float
    n_total: int
    mean_threshold: float
    std_threshold: float
    mean_coverage: float
    std_coverage: float
    coverage_ci_95: Tuple[float, float]
    mean_acceptance_rate: float
    std_acceptance_rate: float
    mean_accepted_accuracy: float
    std_accepted_accuracy: float
    n_bootstrap: int

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "n_total": self.n_total,
            "mean_threshold": self.mean_threshold,
            "std_threshold": self.std_threshold,
            "mean_coverage": self.mean_coverage,
            "std_coverage": self.std_coverage,
            "coverage_ci_95": list(self.coverage_ci_95),
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "std_acceptance_rate": self.std_acceptance_rate,
            "mean_accepted_accuracy": self.mean_accepted_accuracy,
            "std_accepted_accuracy": self.std_accepted_accuracy,
            "n_bootstrap": self.n_bootstrap,
        }
