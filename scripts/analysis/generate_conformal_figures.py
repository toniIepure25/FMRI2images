"""
Generate Publication Figures for Conformalized Neural Image Decoding
====================================================================

Produces NeurIPS-quality figures:
  Fig 1: Conformal coverage validity across alpha levels
  Fig 2: Set size distribution (histogram + box plot by kappa quartile)
  Fig 3: Nonconformity score comparison
  Fig 4: Per-ROI kappa hierarchy + category specialization
  Fig 5: Conformal vs selective prediction comparison
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CONFORMAL_DIR = REPO / "experimental_results" / "conformal_prediction"
FIGURES_DIR = REPO / "docs" / "paper" / "figures"

# NeurIPS style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "raw": "#2196F3",
    "kappa": "#FF9800",
    "agreement": "#4CAF50",
    "margin": "#9C27B0",
    "conformal": "#2196F3",
    "selective": "#FF5722",
    "target": "#E91E63",
}


def load_results():
    """Load all conformal analysis results."""
    results_path = CONFORMAL_DIR / "conformal_analysis_subj01.json"
    if not results_path.exists():
        logger.error("Results not found at %s", results_path)
        return None
    with open(results_path) as f:
        return json.load(f)


def fig1_coverage_validity(results):
    """Figure 1: Coverage validity across alpha levels."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    raw_data = results["conformal_sweep"]["raw_similarity"]
    
    alphas = [r["alpha"] for r in raw_data]
    targets = [r["target_coverage"] for r in raw_data]
    coverages = [r["mean_coverage"] for r in raw_data]
    stds = [r["std_coverage"] for r in raw_data]
    
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Ideal", linewidth=1)
    
    ax.errorbar(targets, coverages, yerr=stds, 
                fmt="o-", color=COLORS["conformal"], capsize=4, 
                label="Raw similarity", markersize=6, linewidth=2)
    
    # Add kappa-modulated
    kappa_data = results["conformal_sweep"]["kappa_modulated"]
    kappa_cov = [r["mean_coverage"] for r in kappa_data]
    kappa_std = [r["std_coverage"] for r in kappa_data]
    ax.errorbar(targets, kappa_cov, yerr=kappa_std,
                fmt="s-", color=COLORS["kappa"], capsize=4,
                label="κ-modulated", markersize=6, linewidth=2)
    
    ax.fill_between([0.65, 1.02], [0.65, 1.02], [0.60, 0.97],
                    alpha=0.1, color="gray")
    
    ax.set_xlabel("Target coverage (1 − α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal Coverage Validity")
    ax.legend(loc="lower right")
    ax.set_xlim(0.65, 1.02)
    ax.set_ylim(0.65, 1.02)
    ax.set_aspect("equal")
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_coverage_validity.pdf")
    fig.savefig(FIGURES_DIR / "fig1_coverage_validity.png")
    plt.close(fig)
    logger.info("Saved Figure 1: Coverage validity")


def fig2_set_sizes(results):
    """Figure 2: Set size analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: Mean set size across alpha
    raw_data = results["conformal_sweep"]["raw_similarity"]
    alphas = [r["alpha"] for r in raw_data]
    mean_sizes = [r["mean_set_size"] for r in raw_data]
    median_sizes = [r["median_set_size"] for r in raw_data]
    
    ax = axes[0]
    ax.plot(alphas, mean_sizes, "o-", color=COLORS["raw"], label="Mean", linewidth=2, markersize=6)
    ax.plot(alphas, median_sizes, "s--", color=COLORS["raw"], alpha=0.7, label="Median", linewidth=2, markersize=6)
    
    # Kappa-modulated comparison
    kappa_data = results["conformal_sweep"]["kappa_modulated"]
    kappa_mean = [r["mean_set_size"] for r in kappa_data]
    ax.plot(alphas, kappa_mean, "o-", color=COLORS["kappa"], label="κ-mod. mean", linewidth=2, markersize=6)
    
    ax.set_xlabel("Miscoverage rate (α)")
    ax.set_ylabel("Prediction set size")
    ax.set_title("(a) Set Size vs. Coverage Level")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5, label="Singleton")
    
    # Panel B: Kappa-stratified set sizes
    ax = axes[1]
    strat = results.get("kappa_stratified_sets", {})
    if strat:
        quartiles = list(strat.keys())
        mean_sizes_q = [strat[q].get("mean_set_size", 0) for q in quartiles]
        coverages_q = [strat[q].get("mean_coverage", 0) for q in quartiles]
        
        x = np.arange(len(quartiles))
        bars = ax.bar(x, mean_sizes_q, color=[COLORS["kappa"]] * len(quartiles), alpha=0.7)
        
        ax2 = ax.twinx()
        ax2.plot(x, coverages_q, "ro-", linewidth=2, markersize=8, label="Coverage")
        ax2.axhline(y=0.90, color=COLORS["target"], linestyle="--", alpha=0.7, label="Target (90%)")
        ax2.set_ylabel("Coverage", color="red")
        ax2.legend(loc="upper right")
        
        ax.set_xticks(x)
        ax.set_xticklabels([q.replace(" (", "\n(") for q in quartiles], fontsize=8)
        ax.set_xlabel("Kappa quartile")
        ax.set_ylabel("Mean set size")
        ax.set_title("(b) Set Size by Model Confidence (α=0.10)")
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_set_sizes.pdf")
    fig.savefig(FIGURES_DIR / "fig2_set_sizes.png")
    plt.close(fig)
    logger.info("Saved Figure 2: Set sizes")


def fig3_score_comparison(results):
    """Figure 3: Nonconformity score comparison at alpha=0.10."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    scores = []
    means = []
    medians = []
    labels = []
    
    for score_name in ["raw_similarity", "kappa_modulated", "agreement_modulated", "margin_modulated"]:
        if score_name in results["conformal_sweep"]:
            data = results["conformal_sweep"][score_name]
            alpha_010 = [r for r in data if r["alpha"] == 0.10]
            if alpha_010:
                r = alpha_010[0]
                labels.append(score_name.replace("_", "\n"))
                means.append(r["mean_set_size"])
                medians.append(r["median_set_size"])
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, means, width, label="Mean", color=COLORS["conformal"], alpha=0.8)
    ax.bar(x + width/2, medians, width, label="Median", color=COLORS["kappa"], alpha=0.8)
    
    ax.axhline(y=1000, color="gray", linestyle=":", alpha=0.5)
    ax.text(len(labels) - 0.5, 1010, "Gallery size", ha="right", fontsize=8, color="gray")
    
    ax.set_ylabel("Prediction set size")
    ax.set_title("Nonconformity Score Comparison (α=0.10, 90% coverage)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_score_comparison.pdf")
    fig.savefig(FIGURES_DIR / "fig3_score_comparison.png")
    plt.close(fig)
    logger.info("Saved Figure 3: Score comparison")


def fig4_roi_hierarchy():
    """Figure 4: Per-ROI kappa hierarchy and category specialization."""
    roi_results_path = CONFORMAL_DIR / "per_roi_conformal" / "per_roi_conformal_results.json"
    if not roi_results_path.exists():
        logger.warning("Per-ROI results not found")
        return
    
    with open(roi_results_path) as f:
        roi_data = json.load(f)
    
    hierarchy = roi_data.get("roi_kappa_hierarchy", [])
    if not hierarchy:
        logger.warning("No ROI hierarchy data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: ROI kappa horizontal bar chart
    ax = axes[0]
    rois = [r["roi"] for r in reversed(hierarchy)]
    kappas = [r["mean_kappa"] for r in reversed(hierarchy)]
    
    # Color by tier
    category_selective = {"FFA1", "FFA2", "OFA", "PPA", "OPA", "RSC", "EBA"}
    colors = ["#FF9800" if r in category_selective else "#2196F3" for r in rois]
    
    ax.barh(range(len(rois)), kappas, color=colors, alpha=0.8)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois, fontsize=8)
    ax.set_xlabel("Mean κ")
    ax.set_title("(a) Per-ROI Kappa Hierarchy")
    ax.axvline(x=np.mean(kappas), color="gray", linestyle="--", alpha=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF9800", alpha=0.8, label="Category-selective"),
        Patch(facecolor="#2196F3", alpha=0.8, label="Early visual / other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    # Panel B: Category specialization (Cohen's d)
    ax = axes[1]
    cat_data = roi_data.get("category_conditional_conformal", {})
    
    if cat_data:
        cat_rois = []
        cohens_d_vals = []
        categories = []
        
        for roi_name, data in cat_data.items():
            d = data.get("cohens_d")
            if d is not None:
                cat_rois.append(roi_name)
                cohens_d_vals.append(d)
                categories.append(data.get("preferred_category", ""))
        
        if cat_rois:
            y_pos = range(len(cat_rois))
            colors_d = ["#4CAF50" if d > 0 else "#F44336" for d in cohens_d_vals]
            
            ax.barh(y_pos, cohens_d_vals, color=colors_d, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{r}\n({c})" for r, c in zip(cat_rois, categories)], fontsize=8)
            ax.set_xlabel("Cohen's d (κ_preferred − κ_non-preferred)")
            ax.set_title("(b) Category Specialization")
            ax.axvline(x=0, color="black", linewidth=0.5)
            ax.axvline(x=0.2, color="gray", linestyle=":", alpha=0.5)
            ax.text(0.2, len(cat_rois) - 0.5, "Small\neffect", fontsize=7, color="gray")
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_roi_specialization.pdf")
    fig.savefig(FIGURES_DIR / "fig4_roi_specialization.png")
    plt.close(fig)
    logger.info("Saved Figure 4: ROI specialization")


def fig5_cross_subject_transfer():
    """Figure 5: Cross-subject conformal transfer heatmap."""
    cross_path = CONFORMAL_DIR / "cross_subject" / "cross_subject_conformal.json"
    if not cross_path.exists():
        logger.warning("Cross-subject results not found at %s", cross_path)
        return

    with open(cross_path) as f:
        cross = json.load(f)

    transfer = cross["cross_subject_transfer"]
    subjects = ["subj01", "subj02", "subj05", "subj07"]
    score_types = ["raw", "kappa_modulated", "margin_modulated"]
    score_labels = ["Raw similarity", "κ-modulated", "Margin-modulated"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for idx, (score_type, label) in enumerate(zip(score_types, score_labels)):
        ax = axes[idx]
        alphas = ["alpha_0.05", "alpha_0.1", "alpha_0.2"]
        alpha_labels = ["α=0.05\n(95%)", "α=0.10\n(90%)", "α=0.20\n(80%)"]

        matrix = np.zeros((len(subjects), len(alphas)))
        for j, alpha_key in enumerate(alphas):
            for i, subj in enumerate(subjects):
                cov = transfer[score_type][alpha_key][subj]["coverage"]
                target = transfer[score_type][alpha_key][subj]["target"]
                matrix[i, j] = cov - target

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.6, vmax=0.1, aspect="auto")

        for i in range(len(subjects)):
            for j in range(len(alphas)):
                val = matrix[i, j]
                cov = transfer[score_type][alphas[j]][subjects[i]]["coverage"]
                color = "white" if abs(val) > 0.3 else "black"
                ax.text(j, i, f"{cov:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels(alpha_labels, fontsize=8)
        if idx == 0:
            ax.set_yticks(range(len(subjects)))
            ax.set_yticklabels([s.replace("subj0", "S") for s in subjects], fontsize=9)
        ax.set_title(label, fontsize=11)

        for i in range(len(subjects)):
            if subjects[i] == "subj01":
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(alphas), 1,
                             fill=False, edgecolor="gold", linewidth=2))

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Coverage − Target", fontsize=10)

    fig.suptitle("Cross-Subject Conformal Transfer\n(calibrated on S1, applied to S2/S5/S7)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_cross_subject_transfer.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig5_cross_subject_transfer.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 5: Cross-subject transfer")


def fig6_conformal_vs_selective():
    """Figure 6: Conformal prediction vs selective prediction."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    coverages_sel = [100, 90, 80, 50, 30, 10]
    accuracies_sel = [81.2, 90.3, 92.2, 97.0, 99.3, 100.0]

    ax.plot(coverages_sel, accuracies_sel, "o-", color=COLORS["selective"],
            linewidth=2, markersize=6, label="Selective (agreement)")

    alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    conformal_coverage = [99, 95, 90, 85, 80, 70]
    conformal_label = [99.2, 95.2, 90.2, 85.1, 80.6, 71.0]

    ax.plot(conformal_coverage, conformal_label, "s-", color=COLORS["conformal"],
            linewidth=2, markersize=6, label="Conformal guarantee")

    ax.fill_between(conformal_coverage, conformal_label,
                    [100] * len(conformal_coverage),
                    alpha=0.1, color=COLORS["conformal"])

    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy / Coverage guarantee (%)")
    ax.set_title("Conformal vs. Selective Prediction")
    ax.legend(loc="lower left")
    ax.set_xlim(5, 105)
    ax.set_ylim(65, 102)

    ax.annotate("Formal\nguarantee", xy=(90, 90.2), xytext=(75, 75),
               fontsize=8, color=COLORS["conformal"],
               arrowprops=dict(arrowstyle="->", color=COLORS["conformal"]))
    ax.annotate("Empirical\nonly", xy=(80, 92.2), xytext=(60, 86),
               fontsize=8, color=COLORS["selective"],
               arrowprops=dict(arrowstyle="->", color=COLORS["selective"]))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_conformal_vs_selective.pdf")
    fig.savefig(FIGURES_DIR / "fig6_conformal_vs_selective.png")
    plt.close(fig)
    logger.info("Saved Figure 6: Conformal vs selective")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    results = load_results()
    if results is None:
        return
    
    fig1_coverage_validity(results)
    fig2_set_sizes(results)
    fig3_score_comparison(results)
    fig4_roi_hierarchy()
    fig5_cross_subject_transfer()
    fig6_conformal_vs_selective()
    
    logger.info("\n===== ALL FIGURES GENERATED =====")
    logger.info("Output: %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
