#!/usr/bin/env python3
"""
Build and cache embedding preprocessor from training split.

Usage:
    python scripts/build_embedding_preproc.py \
        --subject subj01 \
        --mode center_pcr \
        --k_components 8 \
        --output cache/embedding_preproc/subj01_center_pcr_k8.pkl
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.embedding_preproc import (
    EmbeddingPreprocessor,
    compute_separation_histograms,
)
from fmri2img.data.nsd_dataset import NSDDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Build embedding preprocessor from training data"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="subj01",
        help="NSD subject (default: subj01)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["center_pcr", "center_whiten"],
        default="center_pcr",
        help="Preprocessing mode (default: center_pcr)",
    )
    parser.add_argument(
        "--k_components",
        type=int,
        default=8,
        help="Number of top PCs to remove (for center_pcr, default: 8)",
    )
    parser.add_argument(
        "--whiten_eps",
        type=float,
        default=1e-5,
        help="Whitening epsilon (for center_whiten, default: 1e-5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for preprocessor artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Data config path (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Optional directory to save diagnostic plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Building Embedding Preprocessor")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"K components: {args.k_components}")
    logger.info(f"Output: {args.output}")
    
    # Load config
    config = load_config(Path(args.config))
    
    # Load training data
    logger.info("\nLoading training dataset...")
    try:
        dataset = NSDDataset(
            subject=args.subject,
            split="train",
            roi="nsdgeneral",
            **config.get("dataset", {}),
        )
        logger.info(f"Loaded {len(dataset)} training samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Attempting alternative loading method...")
        # Fallback: load from cache if dataset class fails
        raise NotImplementedError("Implement fallback loading from cache")
    
    # Extract CLIP embeddings
    logger.info("\nExtracting CLIP embeddings from training split...")
    embeddings_list = []
    labels_list = []
    
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            # Assuming sample has 'clip_embedding' or 'target' field
            if 'clip_embedding' in sample:
                emb = sample['clip_embedding']
            elif 'target' in sample:
                emb = sample['target']
            else:
                raise KeyError("No embedding field found in sample")
            
            # Convert to numpy
            if isinstance(emb, np.ndarray):
                pass
            else:
                emb = emb.cpu().numpy() if hasattr(emb, 'cpu') else np.array(emb)
            
            embeddings_list.append(emb)
            
            # Store image ID or index for separation analysis
            image_id = sample.get('image_id', idx)
            labels_list.append(image_id)
            
        except Exception as e:
            logger.warning(f"Failed to extract embedding at index {idx}: {e}")
            continue
    
    embeddings = np.stack(embeddings_list, axis=0)
    labels = np.array(labels_list)
    
    logger.info(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Fit preprocessor
    logger.info("\nFitting preprocessor...")
    preprocessor = EmbeddingPreprocessor(
        mode=args.mode,
        k_components=args.k_components,
        whiten_eps=args.whiten_eps,
        seed=args.seed,
    )
    
    preprocessor.fit(embeddings)
    
    # Compute diagnostics
    logger.info("\nComputing diagnostics...")
    diagnostics = preprocessor.compute_diagnostics(embeddings)
    
    logger.info("\nDiagnostics:")
    for key, value in diagnostics.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save artifacts
    output_path = Path(args.output)
    preprocessor.save(output_path)
    
    # Save diagnostics
    diagnostics_path = output_path.parent / f"{output_path.stem}_diagnostics.yaml"
    with open(diagnostics_path, "w") as f:
        yaml.dump(diagnostics, f)
    logger.info(f"Saved diagnostics to {diagnostics_path}")
    
    # Generate plots if requested
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nGenerating diagnostic plots in {plot_dir}...")
        
        # 1. Anisotropy comparison
        if "anisotropy_score_after" in diagnostics:
            fig, ax = plt.subplots(figsize=(8, 6))
            scores = [
                diagnostics["anisotropy_score"],
                diagnostics["anisotropy_score_after"],
            ]
            ax.bar(["Before", "After"], scores, color=["#e74c3c", "#27ae60"])
            ax.set_ylabel("Mean Cosine Similarity (Random Pairs)")
            ax.set_title("Anisotropy Reduction")
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_ylim([-0.1, max(scores) * 1.2])
            
            # Annotate with values
            for i, (label, score) in enumerate(zip(["Before", "After"], scores)):
                ax.text(i, score + 0.01, f"{score:.4f}", ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plot_dir / "anisotropy_comparison.png", dpi=150)
            plt.close()
            logger.info("  Saved anisotropy_comparison.png")
        
        # 2. Positive/Negative separation histograms
        logger.info("  Computing separation histograms...")
        pos_before, neg_before, pos_after, neg_after = compute_separation_histograms(
            embeddings,
            labels,
            preprocessor=preprocessor,
            n_pairs=5000,
            seed=args.seed,
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Before
        axes[0].hist(neg_before, bins=50, alpha=0.6, label="Negative pairs", color="#e74c3c")
        axes[0].hist(pos_before, bins=50, alpha=0.6, label="Positive pairs", color="#27ae60")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Before Preprocessing")
        axes[0].legend()
        axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # After
        if len(pos_after) > 0:
            axes[1].hist(neg_after, bins=50, alpha=0.6, label="Negative pairs", color="#e74c3c")
            axes[1].hist(pos_after, bins=50, alpha=0.6, label="Positive pairs", color="#27ae60")
            axes[1].set_xlabel("Cosine Similarity")
            axes[1].set_ylabel("Count")
            axes[1].set_title("After Preprocessing")
            axes[1].legend()
            axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            
            # Compute and display AUC/Cohen's d
            from scipy.stats import mannwhitneyu
            from sklearn.metrics import roc_auc_score
            
            # AUC before
            y_true_before = np.concatenate([np.ones(len(pos_before)), np.zeros(len(neg_before))])
            y_score_before = np.concatenate([pos_before, neg_before])
            auc_before = roc_auc_score(y_true_before, y_score_before)
            
            # AUC after
            y_true_after = np.concatenate([np.ones(len(pos_after)), np.zeros(len(neg_after))])
            y_score_after = np.concatenate([pos_after, neg_after])
            auc_after = roc_auc_score(y_true_after, y_score_after)
            
            # Cohen's d
            def cohens_d(x1, x2):
                nx1, nx2 = len(x1), len(x2)
                dof = nx1 + nx2 - 2
                return (np.mean(x1) - np.mean(x2)) / np.sqrt(
                    ((nx1 - 1) * np.std(x1, ddof=1) ** 2 + (nx2 - 1) * np.std(x2, ddof=1) ** 2) / dof
                )
            
            d_before = cohens_d(pos_before, neg_before)
            d_after = cohens_d(pos_after, neg_after)
            
            axes[0].text(
                0.05, 0.95, 
                f"AUC: {auc_before:.3f}\nCohen's d: {d_before:.3f}",
                transform=axes[0].transAxes,
                va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
            axes[1].text(
                0.05, 0.95,
                f"AUC: {auc_after:.3f}\nCohen's d: {d_after:.3f}",
                transform=axes[1].transAxes,
                va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
            logger.info(f"    AUC: {auc_before:.4f} → {auc_after:.4f}")
            logger.info(f"    Cohen's d: {d_before:.4f} → {d_after:.4f}")
        
        plt.tight_layout()
        plt.savefig(plot_dir / "separation_histograms.png", dpi=150)
        plt.close()
        logger.info("  Saved separation_histograms.png")
    
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS: Preprocessor built and saved")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
