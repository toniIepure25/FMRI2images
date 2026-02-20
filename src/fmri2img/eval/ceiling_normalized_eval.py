"""
Noise-Ceiling Normalized Evaluation for fMRI Brain Decoding
===========================================================

Expresses all decoding metrics as a fraction of the theoretical maximum
achievable given fMRI measurement noise.  This is essential for:
    1. Fair comparison across subjects with different SNR
    2. Fair comparison across ROIs with different reliability
    3. Understanding how close a decoder is to the information limit

Uses the Spearman-Brown prophecy formula on NSD's 3-repeat structure
to estimate the upper bound on achievable embedding similarity.

Implementation:
    ceiling_score = raw_score / noise_ceiling

    where noise_ceiling is estimated from inter-repetition consistency
    of the decoder's predictions (or from NCSNR voxel reliability).

References:
    - Allen et al. (2022) Natural Scenes Dataset
    - Schoppe et al. (2016) Measuring neural model performance
    - Naselaris et al. (2011) Encoding and decoding review
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fmri2img.reliability.noise_ceiling import (
    load_ncsnr,
    compute_voxel_noise_ceiling_from_ncsnr,
    aggregate_roi_ceiling,
    compute_ceiling_normalized_score,
    spearman_brown_noise_ceiling,
)

logger = logging.getLogger(__name__)


@dataclass
class CeilingNormalizedMetrics:
    """Container for ceiling-normalized metrics."""
    raw: Dict[str, float]
    ceiling: Dict[str, float]
    normalized: Dict[str, float]
    ceiling_info: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "raw": self.raw,
            "ceiling": self.ceiling,
            "normalized": self.normalized,
            "ceiling_info": self.ceiling_info,
        }


def estimate_embedding_ceiling(
    predictions_per_rep: List[np.ndarray],
    metric: str = "cosine",
) -> Dict[str, float]:
    """
    Estimate noise ceiling for embedding-space metrics from NSD repeats.

    Given predictions from 3 independent fMRI repetitions of the same
    stimulus, the inter-prediction consistency sets an upper bound on
    achievable performance.

    Args:
        predictions_per_rep: List of (N, D) prediction arrays, one per
                            repetition of the same N stimuli.
        metric: "cosine" or "l2" for measuring inter-rep consistency.

    Returns:
        Dict with keys:
            upper: Spearman-Brown corrected ceiling
            lower: Single-repeat vs mean-of-others ceiling
            raw_consistency: Average pairwise consistency
    """
    n_reps = len(predictions_per_rep)
    if n_reps < 2:
        logger.warning("Need >=2 reps for ceiling estimation")
        return {"upper": 1.0, "lower": 1.0, "raw_consistency": 1.0}

    pairwise_scores = []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            pred_i = predictions_per_rep[i]
            pred_j = predictions_per_rep[j]

            if metric == "cosine":
                pi_norm = pred_i / (np.linalg.norm(pred_i, axis=1, keepdims=True) + 1e-8)
                pj_norm = pred_j / (np.linalg.norm(pred_j, axis=1, keepdims=True) + 1e-8)
                per_sample = np.sum(pi_norm * pj_norm, axis=1)
            else:
                per_sample = -np.linalg.norm(pred_i - pred_j, axis=1)

            pairwise_scores.append(per_sample.mean())

    raw_r = np.mean(pairwise_scores)
    raw_r_clipped = np.clip(raw_r, 0.0, 1.0)

    # Spearman-Brown: corrected for n_reps
    sb_upper = n_reps * raw_r_clipped / (1 + (n_reps - 1) * raw_r_clipped + 1e-10)

    # Lower bound: single rep vs mean of others
    lower_scores = []
    for i in range(n_reps):
        others = np.mean(
            [predictions_per_rep[j] for j in range(n_reps) if j != i],
            axis=0,
        )
        pi = predictions_per_rep[i]
        if metric == "cosine":
            pi_n = pi / (np.linalg.norm(pi, axis=1, keepdims=True) + 1e-8)
            ot_n = others / (np.linalg.norm(others, axis=1, keepdims=True) + 1e-8)
            lower_scores.append(np.sum(pi_n * ot_n, axis=1).mean())
        else:
            lower_scores.append(-np.linalg.norm(pi - others, axis=1).mean())

    return {
        "upper": float(sb_upper),
        "lower": float(np.mean(lower_scores)),
        "raw_consistency": float(raw_r),
    }


def normalize_metrics_by_ceiling(
    metrics: Dict[str, float],
    ceiling: float,
    metrics_to_normalize: Optional[List[str]] = None,
    invert_metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Normalize a dictionary of metrics by the noise ceiling.

    Args:
        metrics: Raw metric values.
        ceiling: Noise ceiling estimate (0 to 1).
        metrics_to_normalize: List of metric keys to normalize.
            Default: cosine_similarity, r_at_1, r_at_5, mrr, pixel_corr.
        invert_metrics: Metrics where lower is better (normalize differently).

    Returns:
        Dict with same keys, values = raw / ceiling.
    """
    if metrics_to_normalize is None:
        metrics_to_normalize = [
            "cosine_similarity", "r_at_1", "r_at_5", "r_at_10",
            "mrr", "ndcg_at_10", "pixel_corr", "ssim",
            "alexnet_2", "alexnet_5", "clip_score",
        ]
    if invert_metrics is None:
        invert_metrics = ["lpips", "mse", "mean_rank", "median_rank"]

    normalized = {}
    for key, value in metrics.items():
        if value is None:
            normalized[key] = None
            continue

        if key in metrics_to_normalize and ceiling > 1e-8:
            normalized[key] = value / ceiling
        elif key in invert_metrics and ceiling > 1e-8:
            # For lower-is-better: normalize by (1 - ceiling) floor
            normalized[key] = value  # Keep raw; annotate separately
        else:
            normalized[key] = value

    return normalized


class CeilingNormalizedEvaluator:
    """
    Evaluator that reports all metrics both raw and ceiling-normalized.

    Usage:
        evaluator = CeilingNormalizedEvaluator(subject="subj01")
        evaluator.set_ceiling_from_repeats(preds_rep0, preds_rep1, preds_rep2)
        results = evaluator.evaluate(predictions, targets, raw_metrics)
    """

    def __init__(
        self,
        subject: str = "subj01",
        data_root: str = "data",
        roi: str = "nsdgeneral",
    ):
        self.subject = subject
        self.data_root = data_root
        self.roi = roi
        self.embedding_ceiling: Optional[Dict[str, float]] = None
        self.voxel_ceiling: Optional[float] = None
        self._load_voxel_ceiling()

    def _load_voxel_ceiling(self) -> None:
        """Attempt to load voxel-level noise ceiling from NCSNR."""
        ncsnr = load_ncsnr(self.subject, roi=self.roi, data_root=self.data_root)
        if ncsnr is not None:
            ceiling_map = compute_voxel_noise_ceiling_from_ncsnr(ncsnr)
            self.voxel_ceiling = aggregate_roi_ceiling(ceiling_map)
            logger.info(
                "Loaded voxel ceiling for %s/%s: %.3f",
                self.subject, self.roi, self.voxel_ceiling,
            )

    def set_ceiling_from_repeats(
        self,
        *prediction_reps: np.ndarray,
        metric: str = "cosine",
    ) -> None:
        """
        Estimate embedding-space ceiling from repeated predictions.

        Args:
            *prediction_reps: 2 or 3 arrays of shape (N, D), one per
                             fMRI repetition of the same stimuli.
            metric: Consistency metric ("cosine" or "l2").
        """
        self.embedding_ceiling = estimate_embedding_ceiling(
            list(prediction_reps), metric=metric,
        )
        logger.info(
            "Embedding ceiling for %s: upper=%.3f, lower=%.3f, raw=%.3f",
            self.subject,
            self.embedding_ceiling["upper"],
            self.embedding_ceiling["lower"],
            self.embedding_ceiling["raw_consistency"],
        )

    def evaluate(
        self,
        raw_metrics: Dict[str, float],
    ) -> CeilingNormalizedMetrics:
        """
        Produce ceiling-normalized metrics from raw evaluation results.

        Args:
            raw_metrics: Dictionary of raw metric values.

        Returns:
            CeilingNormalizedMetrics with raw, ceiling, and normalized dicts.
        """
        ceiling_upper = 1.0
        ceiling_lower = 1.0
        source = "none"

        if self.embedding_ceiling is not None:
            ceiling_upper = self.embedding_ceiling["upper"]
            ceiling_lower = self.embedding_ceiling["lower"]
            source = "embedding_repeats"
        elif self.voxel_ceiling is not None:
            ceiling_upper = self.voxel_ceiling
            ceiling_lower = self.voxel_ceiling
            source = "ncsnr"

        normalized = normalize_metrics_by_ceiling(raw_metrics, ceiling_upper)

        ceiling_dict = {}
        for key in raw_metrics:
            ceiling_dict[key] = ceiling_upper

        return CeilingNormalizedMetrics(
            raw=raw_metrics,
            ceiling=ceiling_dict,
            normalized=normalized,
            ceiling_info={
                "ceiling_upper": ceiling_upper,
                "ceiling_lower": ceiling_lower,
                "source": source,
                "subject": self.subject,
                "roi": self.roi,
            },
        )

    def format_table(
        self, metrics: CeilingNormalizedMetrics, keys: Optional[List[str]] = None,
    ) -> str:
        """Format metrics as a markdown table with both raw and normalized."""
        if keys is None:
            keys = list(metrics.raw.keys())

        lines = [
            "| Metric | Raw | Ceiling | Normalized (% of ceiling) |",
            "|--------|-----|---------|--------------------------|",
        ]
        for key in keys:
            raw = metrics.raw.get(key)
            norm = metrics.normalized.get(key)
            ceil_val = metrics.ceiling.get(key, 1.0)
            if raw is None:
                continue
            pct = f"{norm / ceil_val * 100:.1f}%" if norm is not None and ceil_val > 0 else "N/A"
            lines.append(f"| {key} | {raw:.4f} | {ceil_val:.4f} | {pct} |")

        return "\n".join(lines)
