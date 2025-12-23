"""
Ridge Training Utilities
========================

Reusable training/evaluation logic for Ridge baseline.
Factored out from scripts/train_ridge.py for use in ablation studies.
"""

import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

from fmri2img.models.ridge import RidgeEncoder, evaluate_predictions
from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import NIfTILoader

logger = logging.getLogger(__name__)


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test with guaranteed minimum samples.
    
    Shared by Ridge and MLP to ensure identical experimental splits.
    
    Args:
        df: Input DataFrame
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        random_seed: Random seed for reproducibility
    
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


def torch_seed_all(seed: int) -> None:
    """
    Set all random seeds for reproducibility (NumPy, Python, PyTorch).
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not available


def cosine_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """
    Cosine distance loss: 1 - cosine_similarity(pred, target).
    
    Both pred and target should be L2-normalized for proper cosine computation.
    
    Args:
        pred: Predicted embeddings (B, D), L2-normalized
        target: Target embeddings (B, D), L2-normalized
    
    Returns:
        Scalar loss (averaged over batch)
    """
    import torch
    # Cosine similarity: dot product of normalized vectors
    cos_sim = (pred * target).sum(dim=-1)  # (B,)
    # Cosine loss: 1 - similarity (minimizing distance)
    return (1.0 - cos_sim).mean()


def compose_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    cosine_weight: float = 1.0,
    mse_weight: float = 0.0,
    infonce_weight: float = 0.0,
    temperature: float = 0.07,
    return_components: bool = False
) -> "torch.Tensor | tuple[torch.Tensor, dict]":
    """
    Multi-objective loss composition for CLIP alignment.
    
    NOVEL CONTRIBUTION: Supports InfoNCE contrastive loss for direct
    retrieval optimization in addition to cosine and MSE losses.
    
    Loss components:
    - Cosine: Directional alignment (1 - cosine_similarity)
    - MSE: Magnitude alignment
    - InfoNCE: Contrastive learning for retrieval (NOVEL)
    
    Backward compatible: Default weights (cosine=1.0, others=0.0) 
    reproduce legacy cosine-only behavior.
    
    Args:
        pred: Predicted embeddings (B, D), L2-normalized
        target: Target embeddings (B, D), L2-normalized
        cosine_weight: Weight for cosine loss (default: 1.0)
        mse_weight: Weight for MSE loss (default: 0.0)
        infonce_weight: Weight for InfoNCE contrastive loss (default: 0.0)
        temperature: Temperature for InfoNCE softmax (default: 0.07)
        return_components: If True, return (loss, components_dict)
    
    Returns:
        If return_components=False: Scalar loss tensor
        If return_components=True: (loss, dict with component losses)
    
    Example:
        >>> # Legacy behavior (cosine only)
        >>> loss = compose_loss(pred, target)
        >>> 
        >>> # Add InfoNCE for retrieval optimization
        >>> loss, components = compose_loss(
        ...     pred, target, 
        ...     cosine_weight=1.0,
        ...     infonce_weight=0.3,
        ...     return_components=True
        ... )
        >>> logger.info(f"Cosine: {components['cosine']:.4f}, InfoNCE: {components['infonce']:.4f}")
    """
    # Import the new comprehensive loss module
    from fmri2img.models.losses import compose_loss as new_compose_loss

    # New compose_loss returns (loss, components) by default. To remain
    # backward-compatible with callers that request only the scalar loss,
    # we unwrap or forward the components depending on `return_components`.
    total_loss, components = new_compose_loss(
        pred, target,
        cosine_weight=cosine_weight,
        mse_weight=mse_weight,
        infonce_weight=infonce_weight,
        temperature=temperature
    )

    if return_components:
        return total_loss, components
    return total_loss


def extract_features_and_targets(
    df: pd.DataFrame,
    nifti_loader: NIfTILoader,
    preprocessor: NSDPreprocessor,
    clip_cache: CLIPCache,
    desc: str = "data"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fMRI features and CLIP targets from DataFrame.
    
    Args:
        df: DataFrame with columns ['beta_path', 'nsdId', ...]
        nifti_loader: NIfTI data loader
        preprocessor: Preprocessing pipeline (or None)
        clip_cache: CLIP embeddings cache
        desc: Description for logging
    
    Returns:
        X: fMRI features (n_samples, n_features)
        Y: CLIP embeddings (n_samples, 512), L2-normalized
        nsd_ids: NSD stimulus IDs (n_samples,)
    """
    logger.info(f"Extracting {desc}: {len(df)} samples")
    
    X_list = []
    Y_list = []
    nsd_ids = []
    
    # OPTIMIZATION: Group samples by beta_path to load each file only once
    from collections import defaultdict
    from tqdm import tqdm
    
    samples_by_file = defaultdict(list)
    for idx, row in df.iterrows():
        beta_path = row["beta_path"]
        samples_by_file[beta_path].append({
            'beta_index': int(row["beta_index"]),
            'nsdId': int(row["nsdId"]),
            'row_idx': idx
        })
    
    logger.info(f"Loading from {len(samples_by_file)} unique beta files")
    
    # Process each beta file once
    for beta_path, samples in tqdm(samples_by_file.items(), desc=f"Loading {desc}"):
        try:
            # Load the 4D beta file ONCE
            logger.debug(f"Loading {beta_path} for {len(samples)} samples")
            img = nifti_loader.load(beta_path)
            data_4d = img.get_fdata()
            
            # Extract all volumes needed from this file
            for sample in samples:
                try:
                    beta_index = sample['beta_index']
                    nsd_id = sample['nsdId']
                    
                    # Extract single volume
                    vol = data_4d[..., beta_index].astype(np.float32)
                    
                    # Apply preprocessing if available
                    if preprocessor and preprocessor.is_fitted_:
                        # T0: z-score (online)
                        vol_z = preprocessor.transform_T0(vol)
                        # T1 + T2: scaler + reliability mask + PCA
                        features = preprocessor.transform(vol_z)
                    else:
                        # No preprocessing: flatten
                        features = vol.flatten()
                    
                    # Get CLIP embedding
                    clip_dict = clip_cache.get([nsd_id])
                    if nsd_id not in clip_dict:
                        logger.warning(f"CLIP embedding missing for nsdId={nsd_id}, skipping")
                        continue
                    
                    clip_emb = clip_dict[nsd_id]  # Already L2-normalized
                    
                    X_list.append(features)
                    Y_list.append(clip_emb)
                    nsd_ids.append(nsd_id)
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample nsdId={nsd_id}, beta_index={beta_index}: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"Failed to load beta file {beta_path}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError(f"No valid samples extracted from {desc}")
    
    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    nsd_ids = np.array(nsd_ids)
    
    logger.info(f"✅ Extracted {desc}: X {X.shape}, Y {Y.shape}")
    
    return X, Y, nsd_ids


def select_alpha(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    alpha_grid: list
) -> Tuple[float, Dict]:
    """
    Select best alpha by validation cosine similarity.
    
    Train/val/test separation and retrain on train+val before test
    is required to avoid leakage (standard NSD practice).
    
    Args:
        X_train: Training fMRI features
        Y_train: Training CLIP embeddings
        X_val: Validation fMRI features
        Y_val: Validation CLIP embeddings
        alpha_grid: List of alpha values to test
    
    Returns:
        best_alpha: Best alpha value
        results: Dict of alpha -> validation metrics
    """
    logger.info(f"Alpha selection: testing {len(alpha_grid)} values on validation set")
    
    results = {}
    best_alpha = None
    best_cosine = -np.inf
    
    for alpha in alpha_grid:
        # Train model
        model = RidgeEncoder(alpha=alpha)
        model.fit(X_train, Y_train)
        
        # Evaluate on validation
        Y_val_pred = model.predict(X_val, normalize=True)
        metrics = evaluate_predictions(Y_val, Y_val_pred, normalize=True)
        
        results[alpha] = metrics
        logger.info(f"  α={alpha:8.3f}: cosine={metrics['cosine']:.4f} ± {metrics['cosine_std']:.4f}, MSE={metrics['mse']:.4f}")
        
        if metrics['cosine'] > best_cosine:
            best_cosine = metrics['cosine']
            best_alpha = alpha
    
    logger.info(f"✅ Best α={best_alpha:.3f} (val cosine={best_cosine:.4f})")
    
    return best_alpha, results


def run_ridge_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    nifti_loader: NIfTILoader,
    preprocessor: NSDPreprocessor,
    clip_cache: CLIPCache,
    alpha_grid: list,
    subject: str,
    checkpoint_path: Optional[Path] = None,
    report_path: Optional[Path] = None
) -> Dict:
    """
    Complete Ridge training experiment: extract data, select alpha, train, evaluate.
    
    Dimensionality sweep (PCA) mirrors principal-component regression
    used in encoding/decoding work (standard in vision-fMRI).
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        nifti_loader: NIfTI data loader
        preprocessor: Preprocessing pipeline
        clip_cache: CLIP embeddings cache
        alpha_grid: List of alpha values for grid search
        subject: Subject ID
        checkpoint_path: Optional path to save model
        report_path: Optional path to save JSON report
    
    Returns:
        Dict with evaluation metrics and model info
    """
    # Extract features
    X_train, Y_train, train_ids = extract_features_and_targets(
        train_df, nifti_loader, preprocessor, clip_cache, "train"
    )
    X_val, Y_val, val_ids = extract_features_and_targets(
        val_df, nifti_loader, preprocessor, clip_cache, "validation"
    )
    X_test, Y_test, test_ids = extract_features_and_targets(
        test_df, nifti_loader, preprocessor, clip_cache, "test"
    )
    
    # Alpha selection on validation set
    logger.info("=" * 80)
    logger.info("ALPHA SELECTION (validation set)")
    logger.info("=" * 80)
    best_alpha, alpha_results = select_alpha(X_train, Y_train, X_val, Y_val, alpha_grid)
    
    # Retrain on train+val with best alpha
    logger.info("=" * 80)
    logger.info(f"RETRAINING on train+val with α={best_alpha:.3f}")
    logger.info("=" * 80)
    X_trainval = np.vstack([X_train, X_val])
    Y_trainval = np.vstack([Y_train, Y_val])
    
    final_model = RidgeEncoder(alpha=best_alpha)
    final_model.fit(X_trainval, Y_trainval)
    
    # Evaluate on test set
    logger.info("=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)
    
    Y_test_pred = final_model.predict(X_test, normalize=True)
    test_metrics = evaluate_predictions(Y_test, Y_test_pred, normalize=True)
    
    logger.info(f"Cosine: {test_metrics['cosine']:.4f} ± {test_metrics['cosine_std']:.4f}")
    logger.info(f"MSE: {test_metrics['mse']:.4f}")
    
    # Retrieval evaluation (test set as gallery)
    logger.info("\nRetrieval evaluation (test set as gallery)...")
    gt_indices = np.arange(len(Y_test))  # Each query matches its own index
    
    retrieval_metrics = retrieval_at_k(
        Y_test_pred, Y_test, gt_indices, ks=(1, 5, 10)
    )
    
    for k, v in retrieval_metrics.items():
        logger.info(f"{k}: {v:.4f} ({v*100:.2f}%)")
    
    # Additional ranking metrics
    ranking_metrics = compute_ranking_metrics(Y_test_pred, Y_test, gt_indices)
    logger.info(f"Mean rank: {ranking_metrics['mean_rank']:.2f}")
    logger.info(f"Median rank: {ranking_metrics['median_rank']:.2f}")
    logger.info(f"MRR: {ranking_metrics['mrr']:.4f}")
    
    # Combine metrics
    test_metrics.update(retrieval_metrics)
    test_metrics.update(ranking_metrics)
    
    # Get preprocessing summary
    preproc_summary = preprocessor.summary() if preprocessor.is_fitted_ else {}
    
    # Build evaluation report
    report = {
        "subject": subject,
        "preprocessing": {
            "used": preprocessor.is_fitted_,
            "pca_k": preproc_summary.get("pca_components", None),
            "n_voxels_kept": preproc_summary.get("n_voxels_kept", None),
        },
        "data_splits": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_train_valid": len(X_train),
            "n_val_valid": len(X_val),
            "n_test_valid": len(X_test),
        },
        "hyperparameters": {
            "alpha_grid": alpha_grid,
            "best_alpha": best_alpha,
            "alpha_selection_results": {str(k): v for k, v in alpha_results.items()},
        },
        "validation_metrics": alpha_results[best_alpha],
        "test_metrics": test_metrics,
        "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }
    
    # Save model if path provided
    if checkpoint_path:
        final_model.save(checkpoint_path)
    
    # Save report if path provided
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"✅ Report saved to {report_path}")
    
    return report
