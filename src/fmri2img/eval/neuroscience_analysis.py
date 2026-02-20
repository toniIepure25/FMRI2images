"""
Neuroscience Analysis Module for ROI-Transformer Interpretability
=================================================================

Provides tools for:
1. ROI contribution analysis via attention weights and ablation
2. Category-level performance breakdown using COCO labels
3. Uncertainty-neural reliability correlation
4. Hubness analysis and mutual information estimation

These analyses connect ML performance to neuroscience hypotheses,
which is essential for MICCAI and strengthens a NeurIPS submission.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  ROI Contribution Analysis
# ---------------------------------------------------------------------------

def compute_roi_attention_importance(
    model: nn.Module,
    dataloader,
    roi_names: List[str],
    device: str = "cuda",
    max_batches: int = 100,
) -> Dict[str, float]:
    """
    Compute average [CLS]-to-ROI attention across the test set.

    For each Transformer layer, extract the attention weight from
    the [CLS] token (position 0) to every ROI token (positions 1..R).
    The final importance is the mean across layers and samples.

    Args:
        model: trained UnifiedModel with an ROITransformerEncoder
        dataloader: test set loader yielding (fmri, embedding) pairs
        roi_names: list of ROI names matching the encoder's tokenisation
        device: compute device
        max_batches: limit for speed

    Returns:
        roi_importance: {roi_name: mean_attention_weight}
    """
    encoder = model.encoder
    if not hasattr(encoder, "get_attention_weights"):
        logger.warning("Encoder has no get_attention_weights method; skipping.")
        return {}

    model.eval()
    all_weights = []

    with torch.no_grad():
        for i, (fmri, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            fmri = fmri.to(device)
            attn_list = encoder.get_attention_weights(fmri)
            # attn_list[layer]: (B, n_heads, seq, seq)
            # We want row 0 (CLS) -> columns 1..R (ROIs)
            layer_avg = torch.stack([
                a[:, :, 0, 1:].mean(dim=1)  # avg over heads -> (B, n_rois)
                for a in attn_list
            ]).mean(dim=0)  # avg over layers -> (B, n_rois)
            all_weights.append(layer_avg.cpu())

    if not all_weights:
        return {}

    all_weights = torch.cat(all_weights, dim=0).numpy()  # (N, n_rois)
    mean_weights = all_weights.mean(axis=0)

    importance = {
        name: float(mean_weights[i])
        for i, name in enumerate(roi_names)
    }
    return importance


def roi_ablation_study(
    model: nn.Module,
    dataloader,
    roi_names: List[str],
    device: str = "cuda",
    metric_fn=None,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Mask out one ROI at a time and measure the R@1 drop.

    Sets the voxels belonging to a target ROI to zero and re-runs
    inference.  The drop in performance indicates how much that ROI
    contributes.

    Args:
        model: trained UnifiedModel
        dataloader: test set
        roi_names: ROI names in order
        device: device
        metric_fn: callable(preds, targets) -> float  (e.g. R@1)
        max_batches: limit

    Returns:
        {roi_name: metric_drop}
    """
    model.eval()

    # First compute baseline metric
    preds_base, targets_base = _collect_predictions(
        model, dataloader, device, max_batches
    )
    if metric_fn is None:
        metric_fn = _default_r1
    baseline = metric_fn(preds_base, targets_base)

    roi_sizes = model.encoder.roi_sizes if hasattr(model.encoder, "roi_sizes") else None
    if roi_sizes is None:
        logger.warning("Encoder doesn't expose roi_sizes; skipping ablation.")
        return {}

    drops = {}
    for roi_idx, roi_name in enumerate(roi_names):
        offset = sum(roi_sizes[:roi_idx])
        size = roi_sizes[roi_idx]

        preds_abl = []
        tgts_abl = []
        with torch.no_grad():
            for i, (fmri, emb) in enumerate(dataloader):
                if i >= max_batches:
                    break
                fmri = fmri.to(device)
                fmri_ablated = fmri.clone()
                fmri_ablated[:, offset:offset + size] = 0.0
                pred, _ = model(fmri_ablated)
                preds_abl.append(pred.cpu())
                tgts_abl.append(emb)

        preds_abl = torch.cat(preds_abl, 0).numpy()
        tgts_abl = torch.cat(tgts_abl, 0).numpy()
        ablated_metric = metric_fn(preds_abl, tgts_abl)
        drops[roi_name] = baseline - ablated_metric

    return drops


# ---------------------------------------------------------------------------
# 2.  Category-Level Performance Breakdown
# ---------------------------------------------------------------------------

def category_performance_breakdown(
    predictions: np.ndarray,
    targets: np.ndarray,
    category_labels: np.ndarray,
    metric_fn=None,
) -> Dict[str, Dict[str, float]]:
    """
    Break down performance by COCO super-category.

    Args:
        predictions: (N, D)
        targets: (N, D)
        category_labels: (N,) string labels like "person", "animal", ...
        metric_fn: callable(preds, targets) -> float

    Returns:
        {category: {"metric": value, "n_samples": count, "mean_uncertainty": float}}
    """
    if metric_fn is None:
        metric_fn = _default_cosine_sim
    unique_cats = np.unique(category_labels)
    results = {}
    for cat in unique_cats:
        mask = category_labels == cat
        n = mask.sum()
        if n < 5:
            continue
        score = metric_fn(predictions[mask], targets[mask])
        results[str(cat)] = {"metric": score, "n_samples": int(n)}
    return results


# ---------------------------------------------------------------------------
# 3.  Uncertainty vs Neural Reliability Correlation
# ---------------------------------------------------------------------------

def uncertainty_reliability_correlation(
    uncertainty: np.ndarray,
    reliability_per_trial: np.ndarray,
) -> Dict[str, float]:
    """
    Correlate model uncertainty with per-trial neural reliability.

    Hypothesis: trials from noisy fMRI data (low reliability) should
    produce higher model uncertainty.  A significant negative correlation
    connects model behaviour to neural measurement noise.

    Args:
        uncertainty: (N,) per-trial model uncertainty (logvar mean, or 1/kappa)
        reliability_per_trial: (N,) per-trial reliability estimate

    Returns:
        dict with pearson_r, spearman_rho, p_values
    """
    from scipy import stats

    valid = np.isfinite(uncertainty) & np.isfinite(reliability_per_trial)
    u = uncertainty[valid]
    r = reliability_per_trial[valid]

    pearson_r, pearson_p = stats.pearsonr(u, r)
    spearman_rho, spearman_p = stats.spearmanr(u, r)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n_valid": int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# 4.  Hubness Analysis
# ---------------------------------------------------------------------------

def compute_hubness_metrics(
    embeddings: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """
    Measure hubness in an embedding space.

    Hubness (Radovanovic et al. 2010) occurs when a few "hub" points
    appear as nearest neighbours of many others, degrading retrieval.

    Args:
        embeddings: (N, D) L2-normalised embeddings
        k: number of neighbours

    Returns:
        dict with skewness of k-occurrence, mean/std N_k, and hub fraction
    """
    from scipy.spatial.distance import cdist

    N = embeddings.shape[0]
    dists = cdist(embeddings, embeddings, metric="cosine")
    np.fill_diagonal(dists, np.inf)

    # k-nearest neighbour indices
    knn_indices = np.argsort(dists, axis=1)[:, :k]

    # N_k: how many times each point appears as a k-NN of others
    n_k = np.zeros(N)
    for row in knn_indices:
        for idx in row:
            n_k[idx] += 1

    from scipy.stats import skew
    return {
        "skewness_Nk": float(skew(n_k)),
        "mean_Nk": float(n_k.mean()),
        "std_Nk": float(n_k.std()),
        "hub_fraction": float((n_k > 2 * k).mean()),
        "antihub_fraction": float((n_k == 0).mean()),
    }


# ---------------------------------------------------------------------------
# 5.  Mutual Information Estimation (InfoNCE bound)
# ---------------------------------------------------------------------------

def estimate_mi_infonce(
    query_embeddings: np.ndarray,
    key_embeddings: np.ndarray,
    temperature: float = 0.07,
    n_samples: int = 1000,
) -> float:
    """
    Estimate mutual information I(fMRI; CLIP) via the InfoNCE lower bound.

    I >= log(N) - L_NCE

    where L_NCE is the InfoNCE loss evaluated on N samples.

    Args:
        query_embeddings: (N, D) predicted embeddings
        key_embeddings: (N, D) ground-truth embeddings
        temperature: InfoNCE temperature
        n_samples: number of samples to use

    Returns:
        MI lower bound in nats
    """
    import torch.nn.functional as F

    N = min(n_samples, query_embeddings.shape[0])
    q = torch.tensor(query_embeddings[:N], dtype=torch.float32)
    k = torch.tensor(key_embeddings[:N], dtype=torch.float32)

    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)

    logits = (q @ k.T) / temperature
    labels = torch.arange(N)
    loss = F.cross_entropy(logits, labels).item()

    mi_lower = np.log(N) - loss
    return float(mi_lower)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_predictions(model, dataloader, device, max_batches):
    preds, tgts = [], []
    with torch.no_grad():
        for i, (fmri, emb) in enumerate(dataloader):
            if i >= max_batches:
                break
            fmri = fmri.to(device)
            pred, _ = model(fmri)
            preds.append(pred.cpu())
            tgts.append(emb)
    return torch.cat(preds, 0).numpy(), torch.cat(tgts, 0).numpy()


def _default_r1(preds, targets):
    from scipy.spatial.distance import cdist
    sims = 1 - cdist(preds, targets, metric="cosine")
    ranks = (sims >= np.diag(sims)[:, None]).sum(axis=1)
    return float((ranks == 1).mean())


def _default_cosine_sim(preds, targets):
    preds_n = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    targets_n = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(preds_n * targets_n, axis=1)))
