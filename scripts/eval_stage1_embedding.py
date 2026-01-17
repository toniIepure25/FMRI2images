#!/usr/bin/env python3
"""
Stage 1 Embedding Evaluation Script

Evaluates fMRI-to-CLIP embedding predictions with:
- Retrieval metrics (R@K, MeanR, MRR, nDCG)
- Identification metrics (2AFC, AUC, Cohen's d)
- RSA and CKA
- Gallery size curves
- Comprehensive plots

Usage:
    python scripts/eval_stage1_embedding.py \
        --checkpoint checkpoints/exp001/best.ckpt \
        --output experimental_results/exp001/evaluation \
        --split test
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.eval.embedding_metrics import EmbeddingEvaluator
from fmri2img.embedding_preproc import EmbeddingPreprocessor
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_predictions(checkpoint_path: Path, split: str = "test") -> dict:
    """Load model predictions from checkpoint or cache."""
    # This is a placeholder - implement based on your checkpoint format
    # Should return: {
    #   'pred_embeddings': np.ndarray (N, D),
    #   'gt_embeddings': np.ndarray (N, D),
    #   'ids': np.ndarray (N,),
    # }
    raise NotImplementedError(
        "Implement load_predictions() based on your checkpoint format. "
        "Should extract predictions, ground truth, and IDs."
    )


def generate_summary_report(results: dict, output_path: Path):
    """Generate markdown summary report."""
    with open(output_path, "w") as f:
        f.write("# Stage 1 Embedding Evaluation Report\n\n")
        
        # Oracle check
        f.write("## Oracle Sanity Check\n\n")
        oracle = results["oracle"]
        passed = results["oracle_passed"]
        status = "✅ PASSED" if passed else "❌ FAILED"
        f.write(f"**Status:** {status}\n\n")
        f.write(f"- Oracle R@1: {oracle['oracle_r@1']:.4f}\n")
        f.write(f"- Oracle Mean Rank: {oracle['oracle_mean_rank']:.2f}\n")
        f.write(f"- Oracle MRR: {oracle['oracle_mrr']:.4f}\n\n")
        
        if not passed:
            f.write("⚠️ **WARNING:** Oracle check failed! Check for ID mismatches or normalization issues.\n\n")
        
        # Full retrieval
        f.write("## Full Retrieval Metrics\n\n")
        retrieval = results["retrieval_full"]
        f.write(f"- **R@1:** {retrieval['r@1']:.4f} (chance: {retrieval['chance_r@1']:.4f})\n")
        f.write(f"- **R@5:** {retrieval['r@5']:.4f}\n")
        f.write(f"- **R@10:** {retrieval['r@10']:.4f}\n")
        f.write(f"- **Mean Rank:** {retrieval['mean_rank']:.2f}\n")
        f.write(f"- **Median Rank:** {retrieval['median_rank']:.2f}\n")
        f.write(f"- **MRR:** {retrieval['mrr']:.4f}\n")
        f.write(f"- **nDCG@10:** {retrieval['ndcg@10']:.4f}\n")
        f.write(f"- **Gallery Size:** {retrieval['gallery_size']}\n\n")
        
        # Identification
        f.write("## Pairwise Identification\n\n")
        ident = results["identification"]
        f.write(f"- **2AFC Accuracy:** {ident['2afc_accuracy']:.4f} ")
        f.write(f"(95% CI: [{ident['2afc_ci_lower']:.4f}, {ident['2afc_ci_upper']:.4f}])\n")
        f.write(f"- **AUC (pos vs neg):** {ident['auc_pos_neg']:.4f}\n")
        f.write(f"- **Cohen's d:** {ident['cohens_d']:.4f}\n")
        f.write(f"- **d-prime:** {ident['d_prime']:.4f}\n")
        f.write(f"- **Number of pairs:** {ident['n_pairs']}\n\n")
        
        # Structure
        f.write("## Structural Similarity\n\n")
        if "rsa_spearman" in results:
            f.write(f"- **RSA (Spearman):** {results['rsa_spearman']:.4f}\n")
        if "linear_cka" in results:
            f.write(f"- **Linear CKA:** {results['linear_cka']:.4f}\n")
        f.write("\n")
        
        # Gallery size curves
        f.write("## Retrieval vs Gallery Size\n\n")
        f.write("| Gallery Size | R@1 | R@5 | R@10 | Mean Rank | Chance |\n")
        f.write("|--------------|-----|-----|------|-----------|--------|\n")
        
        for size in sorted(results["retrieval_curves"].keys()):
            metrics = results["retrieval_curves"][size]
            f.write(
                f"| {size:,} | {metrics['r@1']:.4f} | {metrics['r@5']:.4f} | "
                f"{metrics['r@10']:.4f} | {metrics['mean_rank']:.2f} | "
                f"{metrics['chance_r@1']:.4f} |\n"
            )
        
        f.write("\n")
    
    logger.info(f"Saved summary report to {output_path}")


def plot_retrieval_curves(results: dict, output_dir: Path):
    """Plot retrieval metrics vs gallery size."""
    curves = results["retrieval_curves"]
    sizes = sorted(curves.keys())
    
    r1_values = [curves[s]["r@1"] for s in sizes]
    r5_values = [curves[s]["r@5"] for s in sizes]
    r10_values = [curves[s]["r@10"] for s in sizes]
    chance_values = [curves[s]["chance_r@1"] for s in sizes]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: R@K curves
    axes[0].plot(sizes, r1_values, marker='o', label='R@1', linewidth=2)
    axes[0].plot(sizes, r5_values, marker='s', label='R@5', linewidth=2)
    axes[0].plot(sizes, r10_values, marker='^', label='R@10', linewidth=2)
    axes[0].plot(sizes, chance_values, linestyle='--', color='gray', label='Chance (R@1)', alpha=0.7)
    axes[0].set_xlabel("Gallery Size")
    axes[0].set_ylabel("Recall@K")
    axes[0].set_title("Retrieval Performance vs Gallery Size")
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Mean Rank
    mean_ranks = [curves[s]["mean_rank"] for s in sizes]
    axes[1].plot(sizes, mean_ranks, marker='o', color='#e74c3c', linewidth=2)
    axes[1].set_xlabel("Gallery Size")
    axes[1].set_ylabel("Mean Rank")
    axes[1].set_title("Mean Rank vs Gallery Size")
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved retrieval_curves.png")


def plot_identification_summary(results: dict, output_dir: Path):
    """Plot identification metrics summary."""
    ident = results["identification"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 2AFC with CI
    axes[0].bar([0], [ident["2afc_accuracy"]], color='#3498db', width=0.5)
    axes[0].errorbar(
        [0],
        [ident["2afc_accuracy"]],
        yerr=[[ident["2afc_accuracy"] - ident["2afc_ci_lower"]],
              [ident["2afc_ci_upper"] - ident["2afc_accuracy"]]],
        fmt='none',
        color='black',
        capsize=10,
        linewidth=2,
    )
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("2-Alternative Forced Choice")
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["2AFC"])
    axes[0].set_ylim([0, 1.0])
    axes[0].legend()
    
    # AUC
    axes[1].bar([0], [ident["auc_pos_neg"]], color='#27ae60', width=0.5)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Discriminability (Pos vs Neg)")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["AUC"])
    axes[1].set_ylim([0, 1.0])
    axes[1].legend()
    
    # Effect sizes
    effect_sizes = {
        "Cohen's d": ident["cohens_d"],
        "d-prime": ident["d_prime"],
    }
    axes[2].bar(range(len(effect_sizes)), list(effect_sizes.values()), color='#e74c3c', width=0.6)
    axes[2].set_ylabel("Effect Size")
    axes[2].set_title("Effect Sizes")
    axes[2].set_xticks(range(len(effect_sizes)))
    axes[2].set_xticklabels(list(effect_sizes.keys()))
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "identification_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved identification_summary.png")


def plot_structure_metrics(results: dict, output_dir: Path):
    """Plot structural similarity metrics."""
    metrics = {}
    if "rsa_spearman" in results:
        metrics["RSA\n(Spearman)"] = results["rsa_spearman"]
    if "linear_cka" in results:
        metrics["Linear\nCKA"] = results["linear_cka"]
    
    if not metrics:
        logger.info("No structure metrics to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(metrics)), list(metrics.values()), color='#9b59b6', width=0.6)
    ax.set_ylabel("Correlation / Similarity")
    ax.set_title("Structural Similarity Metrics")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim([-0.1, 1.0])
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Annotate with values
    for i, (name, value) in enumerate(metrics.items()):
        ax.text(i, value + 0.02, f"{value:.4f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "structure_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved structure_metrics.png")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 embedding predictions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default=None,
        help="Optional: path to embedding preprocessor artifacts",
    )
    parser.add_argument(
        "--gallery_sizes",
        type=int,
        nargs="+",
        default=[2, 10, 50, 100, 500, 1000],
        help="Gallery sizes for retrieval curves",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Stage 1 Embedding Evaluation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output: {output_dir}")
    
    # Load predictions
    logger.info("\nLoading predictions...")
    # TODO: Implement load_predictions based on your checkpoint format
    # For now, create dummy data for demonstration
    logger.warning("Using dummy data - implement load_predictions() for real evaluation")
    
    np.random.seed(args.seed)
    N = 100
    D = 768
    pred_embeddings = np.random.randn(N, D).astype(np.float32)
    gt_embeddings = np.random.randn(N, D).astype(np.float32)
    ids = np.arange(N)
    
    # Load and apply preprocessor if provided
    if args.preprocessor:
        logger.info(f"\nLoading preprocessor from {args.preprocessor}...")
        preprocessor = EmbeddingPreprocessor()
        preprocessor.load(args.preprocessor)
        
        logger.info("Applying preprocessing to predictions and GT...")
        pred_embeddings = preprocessor.transform(pred_embeddings)
        gt_embeddings = preprocessor.transform(gt_embeddings)
    
    # Run evaluation
    logger.info("\nRunning evaluation...")
    evaluator = EmbeddingEvaluator(
        gallery_sizes=args.gallery_sizes,
        n_identification_pairs=5000,
        seed=args.seed,
    )
    
    results = evaluator.evaluate(
        pred_embeddings=pred_embeddings,
        gt_embeddings=gt_embeddings,
        ids=ids,
        compute_rsa=True,
        compute_cka=True,
    )
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Generate summary report
    report_path = output_dir / "summary_report.md"
    generate_summary_report(results, report_path)
    
    # Generate plots
    logger.info("\nGenerating plots...")
    plot_retrieval_curves(results, plots_dir)
    plot_identification_summary(results, plots_dir)
    plot_structure_metrics(results, plots_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
