"""
Monte Carlo Dropout Uncertainty Estimation for fMRI-to-CLIP models.

Novel contribution: Quantifies prediction uncertainty using MC dropout to
identify when reconstructions are reliable. Enables:
- Calibration analysis (uncertainty vs actual error)
- Error correlation studies
- Model confidence estimates

Key features:
- MC dropout inference (multiple forward passes with dropout enabled)
- Variance-based uncertainty quantification
- Correlation with reconstruction error
- Calibration curve generation

Scientific justification:
    MC dropout approximates Bayesian inference by treating dropout as
    approximate variational inference (Gal & Ghahramani, 2016).
    Prediction variance across MC samples estimates epistemic uncertainty.
    
Reference:
    Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation"
    ICML 2016
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def enable_dropout(model: nn.Module) -> nn.Module:
    """
    Enable dropout layers during inference for MC dropout.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model with dropout enabled (in-place modification)
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Enable dropout
    return model


def predict_with_mc_dropout(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    enable_dropout_fn: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    device: str = "cuda",
    return_all_samples: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Perform MC dropout inference to estimate prediction uncertainty.
    
    Runs multiple forward passes with dropout enabled to approximate
    Bayesian posterior predictive distribution.
    
    Args:
        model: PyTorch model with dropout layers
        x: Input tensor, shape (B, D_in) or (D_in,) for single sample
        n_samples: Number of MC samples (default: 20)
        enable_dropout_fn: Optional custom function to enable dropout
                          Default: enable_dropout()
        batch_size: Optional batch size for processing x (if x is large)
        device: Device for inference
        return_all_samples: If True, return all MC samples
        
    Returns:
        Dictionary containing:
            - "mean": Mean prediction (B, D_out) or (D_out,)
            - "variance": Prediction variance (B, D_out) or (D_out,)
            - "std": Standard deviation (B, D_out) or (D_out,)
            - "uncertainty": Mean std across dimensions (B,) or scalar
            - "samples": All MC samples (n_samples, B, D_out) [if return_all_samples=True]
    
    Example:
        >>> model = FMRIToClipModel(...)
        >>> fmri = torch.randn(16, 512)  # Batch of 16 samples
        >>> 
        >>> # MC dropout inference
        >>> result = predict_with_mc_dropout(model, fmri, n_samples=20)
        >>> mean_pred = result["mean"]        # (16, 512)
        >>> uncertainty = result["uncertainty"]  # (16,)
        >>> 
        >>> # High uncertainty samples
        >>> uncertain_idx = torch.topk(uncertainty, k=5).indices
        >>> print(f"Most uncertain samples: {uncertain_idx}")
    """
    model.eval()  # Set to eval mode first
    
    # Handle single sample case
    single_sample = (x.ndim == 1)
    if single_sample:
        x = x.unsqueeze(0)  # (D_in,) -> (1, D_in)
    
    x = x.to(device)
    
    # Enable dropout
    if enable_dropout_fn is None:
        enable_dropout_fn = enable_dropout
    enable_dropout_fn(model)
    
    # Collect MC samples
    samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            if batch_size is not None and x.size(0) > batch_size:
                # Process in batches
                outputs = []
                for i in range(0, x.size(0), batch_size):
                    batch = x[i:i+batch_size]
                    out = model(batch)
                    outputs.append(out.cpu())
                output = torch.cat(outputs, dim=0).to(device)
            else:
                output = model(x)
            
            samples.append(output)
    
    # Stack samples: (n_samples, B, D_out)
    samples_tensor = torch.stack(samples, dim=0)
    
    # Compute statistics
    mean = samples_tensor.mean(dim=0)  # (B, D_out)
    variance = samples_tensor.var(dim=0, unbiased=True)  # (B, D_out)
    std = torch.sqrt(variance + 1e-10)  # (B, D_out)
    
    # Aggregate uncertainty (mean std across dimensions)
    uncertainty = std.mean(dim=-1)  # (B,)
    
    # Handle single sample case
    if single_sample:
        mean = mean.squeeze(0)
        variance = variance.squeeze(0)
        std = std.squeeze(0)
        uncertainty = uncertainty.squeeze(0)
        if return_all_samples:
            samples_tensor = samples_tensor.squeeze(1)
    
    result = {
        "mean": mean,
        "variance": variance,
        "std": std,
        "uncertainty": uncertainty
    }
    
    if return_all_samples:
        result["samples"] = samples_tensor
    
    # Restore model to eval mode with dropout disabled
    model.eval()
    
    return result


def compute_uncertainty_error_correlation(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10,
    metric: str = "pearson"
) -> Dict:
    """
    Compute correlation between prediction uncertainty and actual error.
    
    High correlation indicates well-calibrated uncertainty estimates.
    
    Args:
        uncertainties: Prediction uncertainties, shape (N,)
        errors: Actual errors (e.g., MSE, cosine distance), shape (N,)
        n_bins: Number of bins for calibration curve
        metric: Correlation metric ("pearson", "spearman", "kendall")
        
    Returns:
        Dictionary containing:
            - "correlation": Correlation coefficient
            - "p_value": Statistical significance
            - "calibration_curve": Dict with "bin_uncertainties" and "bin_errors"
            - "n_samples": Number of samples
    
    Example:
        >>> uncertainties = mc_results["uncertainty"].cpu().numpy()
        >>> errors = torch.norm(pred - target, dim=-1).cpu().numpy()
        >>> 
        >>> calib = compute_uncertainty_error_correlation(uncertainties, errors)
        >>> print(f"Correlation: {calib['correlation']:.3f}")
        >>> print(f"p-value: {calib['p_value']:.4f}")
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    uncertainties = np.asarray(uncertainties).flatten()
    errors = np.asarray(errors).flatten()
    
    if len(uncertainties) != len(errors):
        raise ValueError(f"Shape mismatch: uncertainties {len(uncertainties)} vs errors {len(errors)}")
    
    # Remove NaN/Inf
    valid_mask = np.isfinite(uncertainties) & np.isfinite(errors)
    uncertainties = uncertainties[valid_mask]
    errors = errors[valid_mask]
    
    if len(uncertainties) < 3:
        logger.warning("Insufficient samples for correlation analysis")
        return {
            "correlation": np.nan,
            "p_value": np.nan,
            "calibration_curve": {"bin_uncertainties": [], "bin_errors": []},
            "n_samples": len(uncertainties)
        }
    
    # Compute correlation
    if metric == "pearson":
        corr, p_value = pearsonr(uncertainties, errors)
    elif metric == "spearman":
        corr, p_value = spearmanr(uncertainties, errors)
    elif metric == "kendall":
        corr, p_value = kendalltau(uncertainties, errors)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Compute calibration curve (binned)
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_uncertainties = []
    bin_errors = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = mask | (uncertainties == bin_edges[i+1])
        
        if mask.sum() > 0:
            bin_uncertainties.append(uncertainties[mask].mean())
            bin_errors.append(errors[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_uncertainties.append(np.nan)
            bin_errors.append(np.nan)
            bin_counts.append(0)
    
    return {
        "correlation": float(corr),
        "p_value": float(p_value),
        "calibration_curve": {
            "bin_uncertainties": bin_uncertainties,
            "bin_errors": bin_errors,
            "bin_counts": bin_counts,
            "n_bins": n_bins
        },
        "n_samples": int(len(uncertainties)),
        "metric": metric
    }


def evaluate_uncertainty_calibration(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_mc_samples: int = 20,
    error_fn: Optional[Callable] = None,
    device: str = "cuda",
    n_bins: int = 10,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Evaluate uncertainty calibration on a dataset.
    
    Computes MC dropout uncertainty and compares with actual errors
    to assess calibration quality.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader yielding (fmri, target_embedding) tuples
        n_mc_samples: Number of MC dropout samples
        error_fn: Function to compute error (pred, target) -> scalar
                 Default: L2 distance
        device: Device for inference
        n_bins: Number of bins for calibration curve
        max_samples: Maximum samples to process (for speed)
        
    Returns:
        Dictionary containing:
            - "uncertainties": All prediction uncertainties (N,)
            - "errors": All errors (N,)
            - "correlation": Correlation coefficient
            - "p_value": Statistical significance
            - "calibration_curve": Calibration curve data
            - "mean_uncertainty": Mean uncertainty
            - "mean_error": Mean error
            - "n_samples": Number of samples processed
    
    Example:
        >>> model = load_checkpoint("best_model.pth")
        >>> val_loader = DataLoader(val_dataset, batch_size=32)
        >>> 
        >>> results = evaluate_uncertainty_calibration(
        ...     model, val_loader, n_mc_samples=20
        ... )
        >>> 
        >>> print(f"Uncertainty-Error Correlation: {results['correlation']:.3f}")
        >>> print(f"Mean Uncertainty: {results['mean_uncertainty']:.4f}")
        >>> print(f"Mean Error: {results['mean_error']:.4f}")
    """
    if error_fn is None:
        # Default: L2 distance
        error_fn = lambda pred, target: torch.norm(pred - target, p=2, dim=-1)
    
    model.eval()
    model.to(device)
    
    all_uncertainties = []
    all_errors = []
    
    n_processed = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating uncertainty")
        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                fmri, target = batch
            else:
                raise ValueError("DataLoader should yield (fmri, target) tuples")
            
            fmri = fmri.to(device)
            target = target.to(device)
            
            # MC dropout inference
            mc_result = predict_with_mc_dropout(
                model, fmri, n_samples=n_mc_samples, device=device
            )
            
            # Compute error
            errors_batch = error_fn(mc_result["mean"], target)
            
            all_uncertainties.append(mc_result["uncertainty"].cpu().numpy())
            all_errors.append(errors_batch.cpu().numpy())
            
            n_processed += len(fmri)
            
            if max_samples is not None and n_processed >= max_samples:
                break
            
            pbar.set_postfix({
                "n_samples": n_processed,
                "mean_unc": np.mean(all_uncertainties[-1])
            })
    
    # Concatenate results
    uncertainties = np.concatenate(all_uncertainties)
    errors = np.concatenate(all_errors)
    
    # Compute correlation and calibration
    calib_stats = compute_uncertainty_error_correlation(
        uncertainties, errors, n_bins=n_bins
    )
    
    results = {
        "uncertainties": uncertainties,
        "errors": errors,
        "correlation": calib_stats["correlation"],
        "p_value": calib_stats["p_value"],
        "calibration_curve": calib_stats["calibration_curve"],
        "mean_uncertainty": float(np.mean(uncertainties)),
        "std_uncertainty": float(np.std(uncertainties)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "n_samples": int(len(uncertainties)),
        "n_mc_samples": n_mc_samples
    }
    
    logger.info(f"Uncertainty evaluation complete:")
    logger.info(f"  Correlation: {results['correlation']:.3f} (p={results['p_value']:.4f})")
    logger.info(f"  Mean uncertainty: {results['mean_uncertainty']:.4f}")
    logger.info(f"  Mean error: {results['mean_error']:.4f}")
    
    return results


def compute_confidence_intervals(
    samples: torch.Tensor,
    confidence: float = 0.95,
    axis: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute confidence intervals from MC samples.
    
    Args:
        samples: MC samples, shape (n_samples, ...) or (..., n_samples)
        confidence: Confidence level (default: 0.95 for 95% CI)
        axis: Axis along which samples are stacked
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    
    Example:
        >>> mc_result = predict_with_mc_dropout(model, x, n_samples=100, return_all_samples=True)
        >>> samples = mc_result["samples"]  # (100, B, D)
        >>> lower, upper = compute_confidence_intervals(samples, confidence=0.95)
        >>> ci_width = upper - lower  # (B, D)
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = torch.quantile(samples, lower_percentile / 100, dim=axis)
    upper = torch.quantile(samples, upper_percentile / 100, dim=axis)
    
    return lower, upper


def entropy_of_prediction(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Shannon entropy of softmax predictions (for classification tasks).
    
    Args:
        probs: Softmax probabilities, shape (..., n_classes)
        dim: Dimension to compute entropy over
        
    Returns:
        Entropy values, shape (...)
    
    Note:
        For regression tasks (CLIP embeddings), use variance-based uncertainty instead.
    """
    # Clip to avoid log(0)
    probs = torch.clamp(probs, min=1e-10, max=1.0)
    entropy = -torch.sum(probs * torch.log(probs), dim=dim)
    return entropy
