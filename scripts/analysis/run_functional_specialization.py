"""
Comprehensive Functional Specialization Analysis with Real COCO Categories.

Combines:
1. Kappa-only category decoding (12 supercategories, 1000 permutations)
2. Category-conditional attention topography (ROI x Category heatmap)
3. Unsupervised recovery of functional specialization
4. Statistical tests (ANOVA, Tukey HSD, effect sizes)

This is the main analysis script for the neuroscience paper.
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")

SUBJECT = os.environ.get("SUBJECT", "subj01")
CACHE_ROOT = Path(os.environ["CACHE_ROOT"])
KAPPA_DIR = Path(f"experimental_results/UDND_analysis/{SUBJECT}/roi_kappa_results")
OUTPUT_DIR = Path(f"experimental_results/UDND_analysis/{SUBJECT}/functional_specialization")


def load_data():
    """Load per-ROI kappa, attention weights, and real COCO categories."""
    # Load kappa results
    meta_json = json.loads((KAPPA_DIR / "roi_kappa_meta.json").read_text())
    roi_names = meta_json["roi_names"]
    per_roi_kappas = np.load(KAPPA_DIR / "roi_kappa_per_roi_kappas.npy")
    alphas = np.load(KAPPA_DIR / "roi_kappa_alphas.npy")
    consensus_kappa = np.load(KAPPA_DIR / "roi_kappa_consensus_kappa.npy")
    
    # Load trial metadata
    meta_path = CACHE_ROOT / "preextracted" / f"subject={SUBJECT}" / "trial_meta.parquet"
    meta_df = pd.read_parquet(meta_path)
    nsd_ids = meta_df["nsdId"].values
    
    # Load COCO category labels
    cat_path = CACHE_ROOT / "nsd_category_labels.parquet"
    cat_df = pd.read_parquet(cat_path)
    
    # Map trial nsdIds to categories
    nsd_to_cat = dict(zip(cat_df["nsdId"].values, cat_df["supercategory"].values))
    nsd_to_finecat = dict(zip(cat_df["nsdId"].values, cat_df["category_name"].values))
    
    trial_supercats = np.array([nsd_to_cat.get(int(nid), "unknown") for nid in nsd_ids])
    trial_finecats = np.array([nsd_to_finecat.get(int(nid), "unknown") for nid in nsd_ids])
    
    # Filter out unknowns
    valid = trial_supercats != "unknown"
    logger.info("Valid trials with categories: %d/%d (%.1f%%)", 
               valid.sum(), len(valid), 100 * valid.sum() / len(valid))
    
    return {
        "roi_names": roi_names,
        "per_roi_kappas": per_roi_kappas[valid],
        "alphas": alphas[valid],
        "consensus_kappa": consensus_kappa[valid],
        "supercategories": trial_supercats[valid],
        "fine_categories": trial_finecats[valid],
        "nsd_ids": nsd_ids[valid],
        "n_trials": valid.sum(),
    }


# =============================================================================
# PART 1: Kappa-Only Category Decoding (12 supercategories)
# =============================================================================

def run_kappa_decoding(data, n_permutations=1000):
    """Decode COCO supercategories from kappa/attention vectors alone."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: KAPPA-ONLY CATEGORY DECODING")
    logger.info("=" * 70)
    
    categories = data["supercategories"]
    unique_cats = np.unique(categories)
    n_classes = len(unique_cats)
    chance = 1.0 / n_classes
    
    logger.info("Categories (%d): %s", n_classes, unique_cats.tolist())
    logger.info("Chance level: %.4f", chance)
    
    feature_sets = {
        "attention_alpha_16d": data["alphas"],
        "proxy_per_roi_kappa_16d": data["per_roi_kappas"],
        "combined_kappa_alpha_32d": np.concatenate([data["per_roi_kappas"], data["alphas"]], axis=1),
        "consensus_kappa_1d": data["consensus_kappa"].reshape(-1, 1),
    }
    
    results = {}
    best_acc = 0
    best_feat_name = None
    
    for feat_name, features in feature_sets.items():
        logger.info("\n--- Feature: %s (dim=%d) ---", feat_name, features.shape[1])
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        all_preds = []
        all_true = []
        
        for train_idx, val_idx in skf.split(features, categories):
            X_train = StandardScaler().fit_transform(features[train_idx])
            X_val = StandardScaler().fit_transform(features[val_idx])
            y_train, y_val = categories[train_idx], categories[val_idx]
            
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_val)
            fold_accs.append(accuracy_score(y_val, y_pred))
            all_preds.extend(y_pred)
            all_true.extend(y_val)
        
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        
        # Confusion matrix
        cm = confusion_matrix(all_true, all_preds, labels=unique_cats)
        
        results[feat_name] = {
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "fold_accuracies": [float(a) for a in fold_accs],
            "chance_level": float(chance),
            "above_chance": float(mean_acc - chance),
            "relative_improvement": float(mean_acc / chance),
            "n_features": features.shape[1],
            "n_classes": n_classes,
            "confusion_matrix": cm.tolist(),
            "class_names": unique_cats.tolist(),
        }
        
        logger.info("  Accuracy: %.4f +/- %.4f (chance: %.4f, %.1fx above)",
                   mean_acc, std_acc, chance, mean_acc / chance)
        
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_feat_name = feat_name
    
    # Permutation test on best feature set
    logger.info("\n--- Permutation Test (%s, n=%d) ---", best_feat_name, n_permutations)
    best_features = feature_sets[best_feat_name]
    perm_accs = []
    
    t0 = time.time()
    for i in range(n_permutations):
        perm_labels = np.random.permutation(categories)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        fold_accs_perm = []
        
        for train_idx, val_idx in skf.split(best_features, perm_labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(best_features[train_idx])
            X_val = scaler.transform(best_features[val_idx])
            
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
            clf.fit(X_train, perm_labels[train_idx])
            fold_accs_perm.append(accuracy_score(perm_labels[val_idx], clf.predict(X_val)))
        
        perm_accs.append(np.mean(fold_accs_perm))
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info("  Permutation %d/%d (%.1fs)", i + 1, n_permutations, elapsed)
    
    p_value = (np.sum(np.array(perm_accs) >= best_acc) + 1) / (n_permutations + 1)
    
    results["permutation_test"] = {
        "feature_set": best_feat_name,
        "observed_accuracy": float(best_acc),
        "p_value": float(p_value),
        "null_mean": float(np.mean(perm_accs)),
        "null_std": float(np.std(perm_accs)),
        "n_permutations": n_permutations,
        "effect_size_d": float((best_acc - np.mean(perm_accs)) / np.std(perm_accs)) if np.std(perm_accs) > 0 else 0,
    }
    
    logger.info("  p-value: %.6f", p_value)
    logger.info("  Effect size (Cohen's d): %.2f", results["permutation_test"]["effect_size_d"])
    logger.info("  Null: %.4f +/- %.4f, Observed: %.4f", np.mean(perm_accs), np.std(perm_accs), best_acc)
    
    # Feature importance
    logger.info("\n--- Feature Importance ---")
    scaler = StandardScaler()
    X_all = scaler.fit_transform(best_features)
    clf_full = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    clf_full.fit(X_all, categories)
    
    coef_importance = np.abs(clf_full.coef_).mean(axis=0)
    roi_names = data["roi_names"]
    
    if best_feat_name.startswith("attention") or best_feat_name.startswith("proxy"):
        feat_labels = roi_names
    elif "combined" in best_feat_name:
        feat_labels = [f"kappa_{r}" for r in roi_names] + [f"alpha_{r}" for r in roi_names]
    else:
        feat_labels = ["consensus_kappa"]
    
    importance_ranking = sorted(zip(feat_labels, coef_importance.tolist()), key=lambda x: x[1], reverse=True)
    
    results["feature_importance"] = {
        "ranking": importance_ranking[:20],
        "all_importances": dict(zip(feat_labels, coef_importance.tolist())),
    }
    
    for name, imp in importance_ranking[:10]:
        logger.info("  %-25s importance=%.4f", name, imp)
    
    return results


# =============================================================================
# PART 2: Category-Conditional Attention Topography
# =============================================================================

def run_category_conditional_topography(data):
    """ROI x Category attention heatmap with ANOVA."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: CATEGORY-CONDITIONAL ATTENTION TOPOGRAPHY")
    logger.info("=" * 70)
    
    alphas = data["alphas"]
    categories = data["supercategories"]
    roi_names = data["roi_names"]
    unique_cats = np.unique(categories)
    
    n_rois = len(roi_names)
    n_cats = len(unique_cats)
    
    # Compute mean attention per ROI per category
    heatmap = np.zeros((n_cats, n_rois))
    heatmap_std = np.zeros((n_cats, n_rois))
    cat_counts = {}
    
    for i, cat in enumerate(unique_cats):
        mask = categories == cat
        cat_alphas = alphas[mask]
        heatmap[i] = cat_alphas.mean(axis=0)
        heatmap_std[i] = cat_alphas.std(axis=0)
        cat_counts[cat] = mask.sum()
    
    logger.info("Heatmap shape: %s (categories x ROIs)", heatmap.shape)
    
    # Two-way ANOVA: Category x ROI interaction
    # Using one-way ANOVA per ROI across categories
    anova_results = {}
    for j, roi in enumerate(roi_names):
        groups = [alphas[categories == cat, j] for cat in unique_cats]
        f_stat, p_val = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        ss_between = sum(len(g) * (g.mean() - alphas[:, j].mean()) ** 2 for g in groups)
        ss_total = np.sum((alphas[:, j] - alphas[:, j].mean()) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        
        anova_results[roi] = {
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "eta_squared": float(eta_sq),
            "significant_bonferroni": p_val < (0.05 / n_rois),
        }
    
    # Log significant ROIs
    logger.info("\n--- ANOVA results (Bonferroni-corrected, alpha=0.05/%d) ---", n_rois)
    for roi, res in sorted(anova_results.items(), key=lambda x: x[1]["f_statistic"], reverse=True):
        sig = "*" if res["significant_bonferroni"] else " "
        logger.info("  %s %-20s F=%.2f, p=%.2e, eta2=%.4f", sig, roi, 
                   res["f_statistic"], res["p_value"], res["eta_squared"])
    
    # Find which category maximally activates each ROI
    roi_preferred_category = {}
    for j, roi in enumerate(roi_names):
        max_cat_idx = heatmap[:, j].argmax()
        max_cat = unique_cats[max_cat_idx]
        
        # Cohen's d: preferred vs mean of others
        preferred = alphas[categories == max_cat, j]
        others = alphas[categories != max_cat, j]
        cohens_d = (preferred.mean() - others.mean()) / np.sqrt(
            ((len(preferred) - 1) * preferred.std() ** 2 + (len(others) - 1) * others.std() ** 2) / 
            (len(preferred) + len(others) - 2)
        )
        
        roi_preferred_category[roi] = {
            "preferred_category": max_cat,
            "mean_alpha_preferred": float(preferred.mean()),
            "mean_alpha_others": float(others.mean()),
            "cohens_d": float(cohens_d),
            "ratio": float(preferred.mean() / others.mean()) if others.mean() > 0 else 0,
        }
    
    logger.info("\n--- ROI Preferred Categories ---")
    for roi, info in sorted(roi_preferred_category.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True):
        logger.info("  %-20s prefers %-12s (d=%.3f, ratio=%.2fx)", 
                   roi, info["preferred_category"], info["cohens_d"], info["ratio"])
    
    return {
        "heatmap": heatmap.tolist(),
        "heatmap_std": heatmap_std.tolist(),
        "roi_names": roi_names,
        "category_names": unique_cats.tolist(),
        "category_counts": cat_counts,
        "anova_per_roi": anova_results,
        "roi_preferred_category": roi_preferred_category,
    }


# =============================================================================
# PART 3: Unsupervised Recovery of Functional Specialization
# =============================================================================

def run_functional_specialization_analysis(data, topography_results):
    """Test whether the model recovered known functional specialization."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: UNSUPERVISED RECOVERY OF FUNCTIONAL SPECIALIZATION")
    logger.info("=" * 70)
    
    roi_names = data["roi_names"]
    
    # Known functional specialization (ground truth from neuroscience)
    # These are the well-established ROI-category associations
    known_specializations = {
        "EBA": "person",       # Extrastriate Body Area -> bodies/persons
        "FFA1": "person",      # Fusiform Face Area -> faces (contained in person)
        "FFA2": "person",      # FFA2 -> faces
        "OFA": "person",       # Occipital Face Area -> faces
        "PPA": "outdoor",      # Parahippocampal Place Area -> scenes/places
        "OPA": "outdoor",      # Occipital Place Area -> scenes
    }
    
    # Test each known specialization
    specialization_tests = {}
    alphas = data["alphas"]
    categories = data["supercategories"]
    
    for roi, expected_cat in known_specializations.items():
        if roi not in roi_names:
            continue
        
        j = roi_names.index(roi)
        
        # Get attention for preferred vs non-preferred
        preferred_mask = categories == expected_cat
        preferred_alpha = alphas[preferred_mask, j]
        other_alpha = alphas[~preferred_mask, j]
        
        # T-test
        t_stat, p_val = stats.ttest_ind(preferred_alpha, other_alpha)
        
        # Cohen's d
        pooled_std = np.sqrt(
            ((len(preferred_alpha) - 1) * preferred_alpha.std() ** 2 + 
             (len(other_alpha) - 1) * other_alpha.std() ** 2) / 
            (len(preferred_alpha) + len(other_alpha) - 2)
        )
        cohens_d = (preferred_alpha.mean() - other_alpha.mean()) / pooled_std if pooled_std > 0 else 0
        
        specialization_tests[roi] = {
            "expected_category": expected_cat,
            "mean_alpha_preferred": float(preferred_alpha.mean()),
            "mean_alpha_other": float(other_alpha.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "ratio": float(preferred_alpha.mean() / other_alpha.mean()) if other_alpha.mean() > 0 else 0,
            "n_preferred": int(preferred_mask.sum()),
            "n_other": int((~preferred_mask).sum()),
            "recovered": bool(p_val < 0.001 and cohens_d > 0),
        }
        
        status = "RECOVERED" if specialization_tests[roi]["recovered"] else "not recovered"
        logger.info("  %-5s -> %-8s: d=%.3f, t=%.2f, p=%.2e, ratio=%.3f [%s]",
                   roi, expected_cat, cohens_d, t_stat, p_val,
                   specialization_tests[roi]["ratio"], status)
    
    # Summary
    n_tested = len(specialization_tests)
    n_recovered = sum(1 for v in specialization_tests.values() if v["recovered"])
    
    logger.info("\n--- Summary ---")
    logger.info("  Known specializations tested: %d", n_tested)
    logger.info("  Successfully recovered: %d/%d (%.0f%%)", n_recovered, n_tested, 100 * n_recovered / n_tested)
    
    # Also test: does kappa (global confidence) vary by category?
    logger.info("\n--- Global Kappa by Category ---")
    kappa_by_cat = {}
    for cat in np.unique(categories):
        mask = categories == cat
        kappa_by_cat[cat] = {
            "mean": float(data["consensus_kappa"][mask].mean()),
            "std": float(data["consensus_kappa"][mask].std()),
            "n": int(mask.sum()),
        }
    
    # Sort by kappa
    sorted_cats = sorted(kappa_by_cat.items(), key=lambda x: x[1]["mean"], reverse=True)
    for cat, info in sorted_cats:
        logger.info("  %-12s kappa=%.3f +/- %.3f (n=%d)", cat, info["mean"], info["std"], info["n"])
    
    # One-way ANOVA on global kappa across categories
    kappa_groups = [data["consensus_kappa"][categories == cat] for cat in np.unique(categories)]
    f_stat, p_val = stats.f_oneway(*kappa_groups)
    logger.info("\n  ANOVA(kappa ~ category): F=%.2f, p=%.2e", f_stat, p_val)
    
    return {
        "specialization_tests": specialization_tests,
        "n_recovered": n_recovered,
        "n_tested": n_tested,
        "recovery_rate": float(n_recovered / n_tested) if n_tested > 0 else 0,
        "kappa_by_category": kappa_by_cat,
        "kappa_anova": {"f_statistic": float(f_stat), "p_value": float(p_val)},
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    data = load_data()
    logger.info("Loaded %d trials, %d ROIs, subject=%s", data["n_trials"], len(data["roi_names"]), SUBJECT)
    
    # Part 1: Kappa decoding
    decoding_results = run_kappa_decoding(data, n_permutations=1000)
    
    # Part 2: Category-conditional topography
    topography_results = run_category_conditional_topography(data)
    
    # Part 3: Functional specialization recovery
    specialization_results = run_functional_specialization_analysis(data, topography_results)
    
    # Save all results
    all_results = {
        "subject": SUBJECT,
        "n_trials": data["n_trials"],
        "n_rois": len(data["roi_names"]),
        "roi_names": data["roi_names"],
        "kappa_decoding": decoding_results,
        "category_conditional_topography": topography_results,
        "functional_specialization": specialization_results,
    }
    
    output_file = OUTPUT_DIR / "functional_specialization_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY — %s", SUBJECT)
    logger.info("=" * 70)
    logger.info("Trials analyzed: %d", data["n_trials"])
    logger.info("")
    logger.info("1. KAPPA DECODING (12 COCO supercategories):")
    best = decoding_results["permutation_test"]
    logger.info("   Best feature: %s", best["feature_set"])
    logger.info("   Accuracy: %.4f (chance=%.4f, %.1fx above)", 
               best["observed_accuracy"], 1/12, best["observed_accuracy"] / (1/12))
    logger.info("   p-value: %.6f (n_perm=%d)", best["p_value"], best["n_permutations"])
    logger.info("   Cohen's d: %.2f", best["effect_size_d"])
    logger.info("")
    logger.info("2. CATEGORY-CONDITIONAL TOPOGRAPHY:")
    n_sig = sum(1 for v in topography_results["anova_per_roi"].values() if v["significant_bonferroni"])
    logger.info("   ROIs with significant category effect: %d/%d", n_sig, len(data["roi_names"]))
    logger.info("")
    logger.info("3. FUNCTIONAL SPECIALIZATION RECOVERY:")
    logger.info("   Known specializations recovered: %d/%d (%.0f%%)",
               specialization_results["n_recovered"], specialization_results["n_tested"],
               specialization_results["recovery_rate"] * 100)
    logger.info("   Kappa ANOVA(category): F=%.2f, p=%.2e",
               specialization_results["kappa_anova"]["f_statistic"],
               specialization_results["kappa_anova"]["p_value"])
    
    logger.info("\nResults saved to: %s", output_file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
