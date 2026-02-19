#!/usr/bin/env python3
"""
Preprocessing and Ridge Ablation Study
======================================

Sweeps over reliability threshold and PCA dimensionality grids to evaluate
their impact on Ridge baseline performance.

Scientific Rationale:
- Reliability sweep follows NSD practice to trade voxel count vs. SNR
  (GLMsingle/NSD reliability literature: PMC)
- Dimensionality sweep (PCA) mirrors principal-component regression used
  in encoding/decoding work (standard in vision-fMRI)
- Train/val/test separation and retrain on train+val before test is
  required to avoid leakage

Pipeline:
1. Load subject index and split train/val/test
2. Optional: rebuild/validate CLIP cache for all nsdIds
3. For each reliability threshold in grid:
   - Fit T1 scaler with reliability mask on train only
   - For each k in PCA grid:
     - Fit T2 PCA (auto-capped to available variance)
     - Save artifacts to outputs/preproc/{subject}/rel={rel}_k={k}/
     - Train Ridge with alpha grid on train/val
     - Retrain on train+val, evaluate on test
     - Record metrics to DataFrame
4. Save summary to outputs/reports/{subject}/ablation_ridge.csv
5. Save individual JSON reports for each setting

Usage:
    # Quick test with small grids
    python scripts/ablate_preproc_and_ridge.py \\
        --subject subj01 \\
        --rel-grid "0.1,0.2" \\
        --k-grid "512,1024" \\
        --limit 256
    
    # Full ablation with cache rebuild
    python scripts/ablate_preproc_and_ridge.py \\
        --index-root data/indices/nsd_index \\
        --subject subj01 \\
        --rel-grid "0.05,0.1,0.2" \\
        --k-grid "512,1024,4096" \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --rebuild-cache \\
        --limit 4096
    
    # Via Makefile
    make ablate
"""

import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

# Silence nibabel warnings
logging.getLogger("nibabel.global").setLevel(logging.WARNING)

from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.models.train_utils import run_ridge_experiment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test with guaranteed minimum samples.
    
    Returns:
        train_df, val_df, test_df
    """
    # Shuffle with fixed seed
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    
    # Ensure minimum samples for each split (at least 1 for val/test if n >= 3)
    if n_total < 3:
        raise ValueError(f"Need at least 3 samples for train/val/test split, got {n_total}")
    
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val
    
    if n_test < 1:
        # Adjust to ensure at least 1 test sample
        n_test = 1
        n_val = max(1, n_total - n_train - n_test)
        n_train = n_total - n_val - n_test
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train + n_val]
    test_df = df_shuffled[n_train + n_val:]
    
    logger.info(f"Split {n_total} trials: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def rebuild_clip_cache(
    index_file: str,
    index_root: str,
    subject: str,
    clip_cache_path: str,
    device: str = "cuda"
) -> None:
    """
    Rebuild/extend CLIP cache by calling build_clip_cache.py script.
    
    Args:
        index_file: Path to single index file (or None)
        index_root: Path to index root directory (or None)
        subject: Subject ID
        clip_cache_path: Path to CLIP cache file
        device: Device for CLIP model
    """
    logger.info("=" * 80)
    logger.info("REBUILDING CLIP CACHE")
    logger.info("=" * 80)
    
    cmd = ["python", "scripts/build_clip_cache.py"]
    
    if index_file:
        cmd.extend(["--index-file", index_file])
    elif index_root:
        cmd.extend(["--index-root", index_root])
        cmd.extend(["--subject", subject])
    else:
        raise ValueError("Must provide either --index-file or --index-root")
    
    cmd.extend([
        "--cache", clip_cache_path,
        "--device", device,
        "--batch", "64"
    ])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        raise RuntimeError("CLIP cache rebuild failed")
    
    logger.info("✅ CLIP cache rebuild complete")


def fit_preprocessing(
    subject: str,
    train_df: pd.DataFrame,
    nifti_loader: NIfTILoader,
    reliability_threshold: float,
    pca_k: int,
    out_dir: str,
    min_variance: float = 1e-6
) -> NSDPreprocessor:
    """
    Fit preprocessing pipeline (T1 + T2) on training data.
    
    Args:
        subject: Subject ID
        train_df: Training DataFrame
        nifti_loader: NIfTI data loader
        reliability_threshold: Reliability threshold for T1 masking
        pca_k: Number of PCA components (auto-capped)
        out_dir: Output directory for artifacts
        min_variance: Minimum variance threshold
    
    Returns:
        Fitted NSDPreprocessor
    """
    logger.info(f"Fitting preprocessing: rel={reliability_threshold:.3f}, k={pca_k}")
    
    # Initialize preprocessor with custom output directory
    preprocessor = NSDPreprocessor(subject=subject, out_dir="outputs/preproc")
    preprocessor.set_out_dir(out_dir)
    
    # Define loader factory (returns loader and get_volume function)
    def loader_factory():
        def get_volume(loader, row):
            """Extract volume from DataFrame row."""
            try:
                beta_path = row["beta_path"]
                beta_index = int(row.get("beta_index", 0))
                img = loader.load(beta_path)
                data_4d = img.get_fdata()
                vol = data_4d[..., beta_index].astype(np.float32)
                return vol
            except Exception as e:
                logger.warning(f"Failed to load volume: {e}")
                return None
        return nifti_loader, get_volume
    
    # Fit T1 scaler + reliability mask
    preprocessor.fit(
        train_df,
        loader_factory,
        reliability_threshold=reliability_threshold,
        min_variance=min_variance,
        min_repeat_ids=20,  # Use default for ablation
        seed=42  # Fixed seed for reproducibility
    )
    
    # Fit T2 PCA
    preprocessor.fit_pca(
        train_df,
        loader_factory,
        k=pca_k
    )
    
    # Artifacts are saved automatically during fit/fit_pca
    
    summary = preprocessor.summary()
    logger.info(f"✅ Preprocessing fitted: {summary['n_voxels_kept']:,} voxels, k_eff={summary['pca_components']}")
    
    return preprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: reliability threshold × PCA dimensionality"
    )
    
    # Data paths
    parser.add_argument("--index-root", help="NSD index root directory")
    parser.add_argument("--index-file", help="Path to single index file")
    parser.add_argument("--subject", default="subj01", help="Subject to process")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet",
                       help="Path to CLIP cache")
    
    # Ablation grids
    parser.add_argument("--rel-grid", default="0.05,0.1,0.2",
                       help="Comma-separated reliability thresholds")
    parser.add_argument("--k-grid", default="512,1024,4096",
                       help="Comma-separated PCA k values")
    
    # Training config
    parser.add_argument("--alpha-grid", default="0.1,1,3,10,30,100",
                       help="Comma-separated alpha values for Ridge")
    parser.add_argument("--limit", type=int, help="Limit number of samples (for testing)")
    # Model selection: ridge or mlp
    parser.add_argument("--model", choices=["ridge", "mlp"], default="ridge",
                        help="Which model to run for each setting (ridge or mlp)")

    # MLP passthrough args (only used when --model mlp)
    parser.add_argument("--hidden", type=int, default=1024, help="MLP hidden size")
    parser.add_argument("--dropout", type=float, default=0.1, help="MLP dropout")
    parser.add_argument("--lr", type=float, default=1e-3, help="MLP learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="MLP weight decay")
    parser.add_argument("--epochs", type=int, default=50, help="MLP max epochs")
    parser.add_argument("--patience", type=int, default=7, help="MLP early stopping patience")
    parser.add_argument("--batch-size", type=int, default=256, help="MLP batch size")
    
    # Output paths
    parser.add_argument("--preproc-root", default="outputs/preproc",
                       help="Root directory for preprocessing artifacts")
    parser.add_argument("--checkpoint-root", default="checkpoints/ridge_ablation",
                       help="Root directory for model checkpoints")
    parser.add_argument("--report-root", default="outputs/reports",
                       help="Root directory for evaluation reports")
    
    # Optional cache rebuild
    parser.add_argument("--rebuild-cache", action="store_true",
                       help="Rebuild/extend CLIP cache before training")
    parser.add_argument("--device", default="cuda", help="Device for CLIP (if rebuilding)")
    
    # Config file
    parser.add_argument("--config", default="configs/data.yaml",
                       help="Path to data config file")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.index_root and not args.index_file:
        logger.error("Must provide either --index-root or --index-file")
        return 1
    
    try:
        # Load config
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        splits = config.get("preprocessing", {}).get("splits", {})
        
        # Parse grids
        rel_grid = [float(x.strip()) for x in args.rel_grid.split(",")]
        k_grid = [int(x.strip()) for x in args.k_grid.split(",")]
        alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",")]
        
        logger.info("=" * 80)
        logger.info("ABLATION STUDY: Reliability × PCA Dimensionality")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Reliability grid: {rel_grid}")
        logger.info(f"PCA k grid: {k_grid}")
        logger.info(f"Alpha grid: {alpha_grid}")
        if args.limit:
            logger.info(f"Sample limit: {args.limit}")
        
        # Load subject index
        if args.index_file:
            logger.info(f"Loading index from {args.index_file}")
            df = pd.read_parquet(args.index_file)
        else:
            logger.info(f"Loading index for {args.subject} from {args.index_root}")
            df = read_subject_index(args.index_root, args.subject)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Split data (fixed splits for entire ablation)
        train_df, val_df, test_df = split_dataframe(
            df,
            train_ratio=splits.get("train_ratio", 0.8),
            val_ratio=splits.get("val_ratio", 0.1),
            test_ratio=splits.get("test_ratio", 0.1),
            random_seed=splits.get("random_seed", 42)
        )
        
        # Rebuild CLIP cache if requested
        if args.rebuild_cache:
            rebuild_clip_cache(
                args.index_file,
                args.index_root,
                args.subject,
                args.clip_cache,
                args.device
            )
        
        # Load CLIP cache
        logger.info(f"Loading CLIP cache from {args.clip_cache}")
        clip_cache = CLIPCache(args.clip_cache).load()
        stats = clip_cache.stats()
        logger.info(f"✅ CLIP cache loaded: {stats['cache_size']} embeddings")
        
        # Initialize NIfTI loader
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        # Results accumulator
        results = []
        
        # Ablation loop: reliability × PCA k
        for rel in rel_grid:
            for k in k_grid:
                logger.info("\n" + "=" * 80)
                logger.info(f"EXPERIMENT: model={args.model}, rel={rel:.3f}, k={k}")
                logger.info("=" * 80)
                
                # Create unique output directory for this setting
                setting_name = f"rel={rel:.3f}_k={k}"
                preproc_dir = Path(args.preproc_root) / args.subject / setting_name
                checkpoint_path = Path(args.checkpoint_root) / args.subject / setting_name / "ridge.pkl"
                report_path = Path(args.report_root) / args.subject / f"ridge_{setting_name}.json"
                
                try:
                    # Fit preprocessing
                    preprocessor = fit_preprocessing(
                        args.subject,
                        train_df,
                        nifti_loader,
                        reliability_threshold=rel,
                        pca_k=k,
                        out_dir=str(preproc_dir)
                    )
                    
                    # Get preprocessing summary
                    summary = preprocessor.summary()
                    k_eff = summary.get("pca_components", 0)
                    n_voxels_kept = summary.get("n_voxels_kept", 0)
                    var_explained = summary.get("pca_explained_variance", 0.0)
                    
                    if args.model == "ridge":
                        # Train Ridge baseline
                        report = run_ridge_experiment(
                            train_df=train_df,
                            val_df=val_df,
                            test_df=test_df,
                            nifti_loader=nifti_loader,
                            preprocessor=preprocessor,
                            clip_cache=clip_cache,
                            alpha_grid=alpha_grid,
                            subject=args.subject,
                            checkpoint_path=checkpoint_path,
                            report_path=report_path
                        )
                        
                        # Extract key metrics
                        val_metrics = report["validation_metrics"]
                        test_metrics = report["test_metrics"]
                        
                        # Record results (Ridge)
                        results.append({
                            "model": "Ridge",
                            "subject": args.subject,
                            "rel_threshold": rel,
                            "k_requested": k,
                            "k_eff": k_eff,
                            "n_voxels_kept": n_voxels_kept,
                            "var_explained": var_explained,
                            "best_alpha": report["hyperparameters"]["best_alpha"],
                            "val_cosine": val_metrics["cosine"],
                            "val_mse": val_metrics["mse"],
                            "test_cosine": test_metrics["cosine"],
                            "test_cosine_std": test_metrics["cosine_std"],
                            "test_mse": test_metrics["mse"],
                            "R@1": test_metrics.get("R@1", np.nan),
                            "R@5": test_metrics.get("R@5", np.nan),
                            "R@10": test_metrics.get("R@10", np.nan),
                            "mean_rank": test_metrics.get("mean_rank", np.nan),
                            "mrr": test_metrics.get("mrr", np.nan),
                            "n_train": report["data_splits"]["n_train_valid"],
                            "n_val": report["data_splits"]["n_val_valid"],
                            "n_test": report["data_splits"]["n_test_valid"],
                            "checkpoint": str(checkpoint_path),
                            "report": str(report_path)
                        })
                        
                        logger.info(f"✅ Experiment complete: val_cos={val_metrics['cosine']:.4f}, test_cos={test_metrics['cosine']:.4f}")
                    else:
                        # Run MLP as external subprocess. We rely on the already-fitted
                        # preprocessing artifacts (same splits and T0/T1/T2 ensure
                        # apples-to-apples Ridge vs MLP).
                        mlp_cmd = [
                            "python", "scripts/train_mlp.py",
                            "--subject", args.subject,
                            "--use-preproc",
                            "--clip-cache", args.clip_cache,
                            "--hidden", str(args.hidden),
                            "--dropout", str(args.dropout),
                            "--lr", str(args.lr),
                            "--wd", str(args.wd),
                            "--epochs", str(args.epochs),
                            "--patience", str(args.patience),
                            "--batch-size", str(args.batch_size)
                        ]

                        if args.limit:
                            mlp_cmd.extend(["--limit", str(args.limit)])

                        logger.info(f"Running MLP subprocess: {' '.join(mlp_cmd)}")
                        subprocess.run(mlp_cmd, check=True)

                        # Read MLP report and extract metrics
                        mlp_report_path = Path(args.report_root) / args.subject / "mlp_eval.json"
                        if not mlp_report_path.exists():
                            raise FileNotFoundError(f"Expected MLP report not found: {mlp_report_path}")

                        with open(mlp_report_path, "r") as fh:
                            mlp_report = json.load(fh)

                        val_metrics = mlp_report.get("validation_metrics", {})
                        test_metrics = mlp_report.get("test_metrics", {})

                        # Typical MLP checkpoint path (train_mlp writes to checkpoints/mlp/{subject}/mlp.pt)
                        mlp_checkpoint = Path("checkpoints") / "mlp" / args.subject / "mlp.pt"

                        # Record results (MLP)
                        results.append({
                            "model": "MLP",
                            "subject": args.subject,
                            "rel_threshold": rel,
                            "k_requested": k,
                            "k_eff": k_eff,
                            "n_voxels_kept": n_voxels_kept,
                            "var_explained": var_explained,
                            "best_alpha": np.nan,
                            "val_cosine": val_metrics.get("best_cosine", val_metrics.get("cosine", np.nan)),
                            "val_mse": val_metrics.get("mse", np.nan),
                            "test_cosine": test_metrics.get("cosine", np.nan),
                            "test_cosine_std": test_metrics.get("cosine_std", np.nan),
                            "test_mse": test_metrics.get("mse", np.nan),
                            "R@1": test_metrics.get("R@1", np.nan),
                            "R@5": test_metrics.get("R@5", np.nan),
                            "R@10": test_metrics.get("R@10", np.nan),
                            "mean_rank": test_metrics.get("mean_rank", np.nan),
                            "mrr": test_metrics.get("mrr", np.nan),
                            "n_train": mlp_report.get("data_splits", {}).get("n_train", len(train_df)),
                            "n_val": mlp_report.get("data_splits", {}).get("n_val", len(val_df)),
                            "n_test": mlp_report.get("data_splits", {}).get("n_test", len(test_df)),
                            "checkpoint": str(mlp_checkpoint),
                            "report": str(mlp_report_path)
                        })

                        logger.info(f"✅ MLP experiment complete: val_cos={results[-1]['val_cosine']:.4f}, test_cos={results[-1]['test_cosine']:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Experiment failed (rel={rel:.3f}, k={k}): {e}")
                    # Record failure
                    results.append({
                        "model": args.model.upper(),
                        "subject": args.subject,
                        "rel_threshold": rel,
                        "k_requested": k,
                        "k_eff": np.nan,
                        "n_voxels_kept": np.nan,
                        "var_explained": np.nan,
                        "best_alpha": np.nan,
                        "val_cosine": np.nan,
                        "val_mse": np.nan,
                        "test_cosine": np.nan,
                        "test_cosine_std": np.nan,
                        "test_mse": np.nan,
                        "R@1": np.nan,
                        "R@5": np.nan,
                        "R@10": np.nan,
                        "mean_rank": np.nan,
                        "mrr": np.nan,
                        "n_train": len(train_df),
                        "n_val": len(val_df),
                        "n_test": len(test_df),
                        "checkpoint": "",
                        "report": "",
                        "error": str(e)
                    })
        
        # Save summary CSV (append to existing file; write header only if new)
        results_df = pd.DataFrame(results)
        summary_path = Path(args.report_root) / args.subject / "ablation_ridge.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not summary_path.exists()
        # Use append mode when file exists to accumulate Ridge and MLP rows
        results_df.to_csv(summary_path, mode="a", header=write_header, index=False)

        logger.info("\n" + "=" * 80)
        logger.info("ABLATION STUDY COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"\nResults preview:")
        print(results_df[["rel_threshold", "k_eff", "n_voxels_kept", "test_cosine", "R@1", "R@5"]].to_string(index=False))
        
        return 0
        
    except Exception as e:
        logger.error(f"Ablation study failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
