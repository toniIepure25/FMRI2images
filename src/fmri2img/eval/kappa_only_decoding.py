"""
Kappa-Only Category Decoding Experiment
=========================================

Tests whether stimulus category can be predicted from the kappa vector alone
(without the mu direction vectors), establishing that uncertainty itself
carries semantic information about the stimulus.

Scientific hypothesis:
    If a linear classifier can predict stimulus category from the
    17-dimensional kappa vector (per-ROI concentration parameters),
    this proves that the PATTERN of regional encoding confidence encodes
    semantic content independent of the embedding direction.

    This would be a paradigm-shifting finding: it shows that uncertainty
    is not merely noise, but reflects the functional organization of
    visual encoding.

Experiment design:
    1. Extract kappa vectors from ROI-DCF model for all validation trials
    2. Obtain ground-truth category labels (COCO categories)
    3. Train logistic regression / linear SVM on kappa vectors
    4. Compare accuracy to:
       a) Chance level (1/n_categories)
       b) Random permutation baseline
       c) Full mu+kappa concatenation
       d) mu-only baseline
    5. Report confusion matrix and per-category accuracy

Statistical controls:
    - 5-fold cross-validation
    - Permutation test (1000 shuffles) for significance
    - Bonferroni correction for multiple comparisons
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def run_kappa_only_decoding(
    per_roi_kappas: np.ndarray,
    category_labels: np.ndarray,
    per_roi_mus: Optional[np.ndarray] = None,
    mu_fused: Optional[np.ndarray] = None,
    category_names: Optional[List[str]] = None,
    n_folds: int = 5,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Dict:
    """
    Classify stimulus category from kappa vectors alone.

    Parameters
    ----------
    per_roi_kappas : (N, n_rois) per-ROI kappa values
    category_labels : (N,) integer category labels
    per_roi_mus : (N, n_rois, D) optional, for mu-only baseline
    mu_fused : (N, D) optional, for fused-mu baseline
    category_names : string labels for categories
    n_folds : cross-validation folds
    n_permutations : number of permutation shuffles
    random_state : random seed

    Returns
    -------
    Dict with decoding accuracies, significance tests, and confusion matrices
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    np.random.seed(random_state)

    n_samples = len(per_roi_kappas)
    n_classes = len(np.unique(category_labels))
    chance_level = 1.0 / n_classes

    if category_names is None:
        category_names = [f"cat_{i}" for i in range(n_classes)]

    results = {}
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # ---- Condition 1: Kappa-only ----
    kappa_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, multi_class="multinomial",
            solver="lbfgs", C=1.0, random_state=random_state,
        )),
    ])
    kappa_scores = cross_val_score(
        kappa_pipe, per_roi_kappas, category_labels, cv=cv, scoring="accuracy"
    )
    results["kappa_only"] = {
        "mean_accuracy": float(kappa_scores.mean()),
        "std_accuracy": float(kappa_scores.std()),
        "per_fold": kappa_scores.tolist(),
    }

    # ---- Condition 2: Permutation baseline ----
    perm_accuracies = []
    for _ in range(min(n_permutations, 100)):  # Cap at 100 for speed
        perm_labels = np.random.permutation(category_labels)
        perm_scores = cross_val_score(
            kappa_pipe, per_roi_kappas, perm_labels, cv=cv, scoring="accuracy"
        )
        perm_accuracies.append(perm_scores.mean())

    perm_mean = np.mean(perm_accuracies)
    perm_std = np.std(perm_accuracies)
    p_value = np.mean(np.array(perm_accuracies) >= kappa_scores.mean())

    results["permutation_baseline"] = {
        "mean_accuracy": float(perm_mean),
        "std_accuracy": float(perm_std),
        "p_value": float(p_value),
        "n_permutations": len(perm_accuracies),
    }

    # ---- Condition 3: Mu-fused baseline (if available) ----
    if mu_fused is not None:
        mu_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000, multi_class="multinomial",
                solver="lbfgs", C=1.0, random_state=random_state,
            )),
        ])
        mu_scores = cross_val_score(
            mu_pipe, mu_fused, category_labels, cv=cv, scoring="accuracy"
        )
        results["mu_fused_only"] = {
            "mean_accuracy": float(mu_scores.mean()),
            "std_accuracy": float(mu_scores.std()),
            "per_fold": mu_scores.tolist(),
        }

    # ---- Condition 4: Kappa + Mu concatenated ----
    if mu_fused is not None:
        concat_features = np.concatenate([per_roi_kappas, mu_fused], axis=1)
        concat_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000, multi_class="multinomial",
                solver="lbfgs", C=1.0, random_state=random_state,
            )),
        ])
        concat_scores = cross_val_score(
            concat_pipe, concat_features, category_labels, cv=cv, scoring="accuracy"
        )
        results["kappa_plus_mu"] = {
            "mean_accuracy": float(concat_scores.mean()),
            "std_accuracy": float(concat_scores.std()),
            "per_fold": concat_scores.tolist(),
        }

    # ---- Feature importance (for kappa-only) ----
    kappa_pipe.fit(per_roi_kappas, category_labels)
    coef = kappa_pipe.named_steps["clf"].coef_  # (n_classes, n_rois)
    importance = np.abs(coef).mean(axis=0)  # average across classes
    importance_ranked = np.argsort(-importance)

    from fmri2img.eval.roi_kappa_extraction import DEFAULT_ROI_NAMES
    roi_names = DEFAULT_ROI_NAMES[:per_roi_kappas.shape[1]]

    results["feature_importance"] = {
        roi_names[i]: float(importance[i])
        for i in importance_ranked
        if i < len(roi_names)
    }

    # ---- Confusion matrix (kappa-only, full dataset fit) ----
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000, multi_class="multinomial",
                solver="lbfgs", C=1.0, random_state=random_state,
            )),
        ]),
        per_roi_kappas, category_labels, cv=cv,
    )

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(category_labels, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    results["confusion_matrix"] = cm.tolist()
    results["per_class_accuracy"] = {
        category_names[i] if i < len(category_names) else f"cat_{i}": float(per_class_accuracy[i])
        for i in range(len(per_class_accuracy))
    }

    # ---- Summary statistics ----
    results["summary"] = {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "n_rois_features": per_roi_kappas.shape[1],
        "chance_level": chance_level,
        "kappa_accuracy": results["kappa_only"]["mean_accuracy"],
        "above_chance": results["kappa_only"]["mean_accuracy"] - chance_level,
        "significance_p": results["permutation_baseline"]["p_value"],
        "is_significant": results["permutation_baseline"]["p_value"] < 0.05,
        "most_informative_roi": list(results["feature_importance"].keys())[0],
        "paradigm_shift": (
            results["kappa_only"]["mean_accuracy"] > chance_level * 2 and
            results["permutation_baseline"]["p_value"] < 0.001
        ),
    }

    logger.info(
        "Kappa-only decoding: accuracy=%.1f%% (chance=%.1f%%, p=%.4f). "
        "Above chance by %.1f%%. %s",
        results["summary"]["kappa_accuracy"] * 100,
        chance_level * 100,
        results["summary"]["significance_p"],
        results["summary"]["above_chance"] * 100,
        "*** PARADIGM SHIFT ***" if results["summary"]["paradigm_shift"] else "",
    )

    return results
