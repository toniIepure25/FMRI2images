"""
Reverse Encoding Model: CLIP --> Predicted fMRI per ROI
========================================================

Trains the reverse direction (CLIP embedding -> predicted fMRI activity per ROI)
to enable bidirectional consistency analysis.

Neuroscience motivation:
    If the decoding model's uncertainty (kappa_r) for ROI r correlates with
    the encoding model's prediction error (R^2_r) for the same ROI, this
    provides strong evidence that uncertainty reflects genuine encoding
    fidelity rather than model artifacts.

Architecture:
    CLIP embedding (768-D) -> Per-ROI Ridge/MLP -> predicted voxel activity per ROI

This is a standard linearizing encoding model (Naselaris et al., 2011),
but applied per-ROI to enable regional comparison with decoder kappa.

References:
    - Naselaris et al. (2011) Encoding and decoding in fMRI
    - Allen et al. (2022) NSD: "A massive 7T fMRI dataset..."
    - Kay et al. (2008) "Identifying natural images from human brain activity"
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class EncodingModelResult:
    """Results from training the encoding model."""
    r_squared_per_roi: Dict[str, float]        # R^2 per ROI
    r_squared_per_voxel: Dict[str, np.ndarray]  # R^2 per voxel within each ROI
    correlation_per_roi: Dict[str, float]       # Pearson r per ROI
    n_voxels_per_roi: Dict[str, int]
    roi_names: List[str]
    subject: str
    n_train: int
    n_test: int


def train_encoding_model_per_roi(
    clip_embeddings: np.ndarray,
    fmri_features: np.ndarray,
    roi_indices: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float = 1000.0,
    subject: str = "unknown",
) -> EncodingModelResult:
    """
    Train ridge regression encoding model per ROI.

    For each ROI r: fit Ridge(CLIP --> fMRI_r) on training data,
    evaluate R^2 on test data.

    Parameters
    ----------
    clip_embeddings : (N, D) CLIP embeddings (D=768)
    fmri_features : (N, V_total) full fMRI voxel matrix
    roi_indices : {roi_name: voxel_indices_array} mapping
    train_mask : (N,) boolean training mask
    test_mask : (N,) boolean test mask
    alpha : Ridge regularization strength
    subject : subject identifier for logging

    Returns
    -------
    EncodingModelResult with per-ROI R^2 and correlations
    """
    from sklearn.linear_model import Ridge

    X_train = clip_embeddings[train_mask]
    X_test = clip_embeddings[test_mask]

    r_squared_per_roi = {}
    r_squared_per_voxel = {}
    correlation_per_roi = {}
    n_voxels_per_roi = {}

    for roi_name, voxel_idx in roi_indices.items():
        if len(voxel_idx) == 0:
            continue

        Y_train = fmri_features[train_mask][:, voxel_idx]
        Y_test = fmri_features[test_mask][:, voxel_idx]

        # Standardize targets (per-voxel z-score on training)
        y_mean = Y_train.mean(axis=0)
        y_std = Y_train.std(axis=0) + 1e-8
        Y_train_z = (Y_train - y_mean) / y_std
        Y_test_z = (Y_test - y_mean) / y_std

        # Fit Ridge regression
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, Y_train_z)

        # Predict on test
        Y_pred = model.predict(X_test)

        # Compute R^2 per voxel
        ss_res = np.sum((Y_test_z - Y_pred) ** 2, axis=0)
        ss_tot = np.sum((Y_test_z - Y_test_z.mean(axis=0)) ** 2, axis=0)
        r2_voxels = 1 - ss_res / (ss_tot + 1e-8)
        r2_voxels = np.clip(r2_voxels, -1.0, 1.0)

        # Mean R^2 across voxels (weighted by voxel reliability could be added)
        mean_r2 = float(np.mean(r2_voxels[r2_voxels > 0])) if (r2_voxels > 0).any() else 0.0

        # Correlation between predicted and actual (flattened)
        pred_flat = Y_pred.flatten()
        true_flat = Y_test_z.flatten()
        if len(pred_flat) > 2:
            corr, _ = sp_stats.pearsonr(pred_flat, true_flat)
        else:
            corr = 0.0

        r_squared_per_roi[roi_name] = mean_r2
        r_squared_per_voxel[roi_name] = r2_voxels
        correlation_per_roi[roi_name] = float(corr)
        n_voxels_per_roi[roi_name] = len(voxel_idx)

        logger.debug(
            "Encoding %s [%s]: R²=%.4f (mean over %d voxels with R²>0), r=%.4f",
            roi_name, subject, mean_r2, (r2_voxels > 0).sum(), corr,
        )

    logger.info(
        "Encoding model for %s: %d ROIs trained. "
        "Mean R²=%.4f, Best ROI=%s (R²=%.4f)",
        subject, len(r_squared_per_roi),
        np.mean(list(r_squared_per_roi.values())),
        max(r_squared_per_roi, key=r_squared_per_roi.get),
        max(r_squared_per_roi.values()),
    )

    return EncodingModelResult(
        r_squared_per_roi=r_squared_per_roi,
        r_squared_per_voxel=r_squared_per_voxel,
        correlation_per_roi=correlation_per_roi,
        n_voxels_per_roi=n_voxels_per_roi,
        roi_names=list(roi_indices.keys()),
        subject=subject,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
    )


def compute_bidirectional_consistency(
    decoder_kappa_per_roi: Dict[str, float],
    encoder_r2_per_roi: Dict[str, float],
    roi_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Compute correlation between decoder kappa and encoder R^2 across ROIs.

    This is the key bidirectional consistency analysis:
    If kappa_r (decoder confidence) correlates with R^2_r (encoder accuracy),
    then uncertainty genuinely reflects encoding fidelity.

    Parameters
    ----------
    decoder_kappa_per_roi : {roi_name: mean_kappa} from decoder
    encoder_r2_per_roi : {roi_name: R^2} from encoding model
    roi_names : ROIs to include (intersection if not specified)

    Returns
    -------
    Dict with correlation statistics
    """
    if roi_names is None:
        roi_names = sorted(
            set(decoder_kappa_per_roi.keys()) & set(encoder_r2_per_roi.keys())
        )

    kappas = np.array([decoder_kappa_per_roi[r] for r in roi_names])
    r2s = np.array([encoder_r2_per_roi[r] for r in roi_names])

    # Pearson correlation
    if len(kappas) >= 3:
        pearson_r, pearson_p = sp_stats.pearsonr(kappas, r2s)
        spearman_rho, spearman_p = sp_stats.spearmanr(kappas, r2s)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = float("nan")

    # Per-ROI comparison
    per_roi = {
        roi: {"kappa": float(decoder_kappa_per_roi[roi]),
               "encoder_r2": float(encoder_r2_per_roi[roi])}
        for roi in roi_names
    }

    report = {
        "pearson": {"r": float(pearson_r), "p": float(pearson_p)},
        "spearman": {"rho": float(spearman_rho), "p": float(spearman_p)},
        "n_rois": len(roi_names),
        "per_roi": per_roi,
        "interpretation": (
            "Strong bidirectional consistency: decoder confidence "
            "tracks encoding model accuracy"
            if pearson_r > 0.5 and pearson_p < 0.05 else
            "Moderate bidirectional consistency"
            if pearson_r > 0.3 else
            "Weak or no bidirectional consistency"
        ),
    }

    logger.info(
        "Bidirectional consistency: Pearson r=%.3f (p=%.4f), "
        "Spearman rho=%.3f (p=%.4f) across %d ROIs",
        pearson_r, pearson_p, spearman_rho, spearman_p, len(roi_names),
    )

    return report


def build_roi_index_from_mask(
    roi_mask_dir: Path,
    subject: str,
    roi_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build ROI voxel index from NSD atlas files.

    Parameters
    ----------
    roi_mask_dir : directory containing ROI NIfTI masks
    subject : subject identifier
    roi_names : ROIs to extract (default: standard 17)

    Returns
    -------
    {roi_name: voxel_indices} where indices are into the nsdgeneral mask
    """
    import nibabel as nib

    if roi_names is None:
        from fmri2img.eval.roi_kappa_extraction import DEFAULT_ROI_NAMES
        roi_names = DEFAULT_ROI_NAMES

    # Load nsdgeneral mask as the base
    nsdgeneral_path = roi_mask_dir / "nsdgeneral.nii.gz"
    if not nsdgeneral_path.exists():
        raise FileNotFoundError(f"nsdgeneral mask not found: {nsdgeneral_path}")

    nsdgeneral = nib.load(str(nsdgeneral_path)).get_fdata()
    nsdgeneral_indices = np.where(nsdgeneral > 0)
    n_voxels = len(nsdgeneral_indices[0])

    # Create coordinate-to-linear-index mapping
    coord_to_idx = {}
    for idx in range(n_voxels):
        coord = (nsdgeneral_indices[0][idx],
                 nsdgeneral_indices[1][idx],
                 nsdgeneral_indices[2][idx])
        coord_to_idx[coord] = idx

    roi_index = {}

    # Load Kastner2015 for V1-V4
    kastner_path = roi_mask_dir / "Kastner2015.nii.gz"
    kastner_map = {
        1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d",
        5: "V3v", 6: "V3d", 7: "V3A", 8: "V3B", 9: "V4",
    }
    if kastner_path.exists():
        kastner = nib.load(str(kastner_path)).get_fdata()
        for label_val, roi_name in kastner_map.items():
            if roi_name in roi_names:
                roi_coords = np.where(kastner == label_val)
                indices = []
                for i in range(len(roi_coords[0])):
                    coord = (roi_coords[0][i], roi_coords[1][i], roi_coords[2][i])
                    if coord in coord_to_idx:
                        indices.append(coord_to_idx[coord])
                roi_index[roi_name] = np.array(indices, dtype=np.int64)

    # Load category-selective ROIs
    floc_rois = {
        "floc-faces.nii.gz": {"FFA1": 1, "FFA2": 2, "OFA": 3},
        "floc-places.nii.gz": {"PPA": 1, "OPA": 2, "RSC": 3},
        "floc-bodies.nii.gz": {"EBA": 1},
    }
    for mask_file, roi_labels in floc_rois.items():
        mask_path = roi_mask_dir / mask_file
        if mask_path.exists():
            mask_data = nib.load(str(mask_path)).get_fdata()
            for roi_name, label_val in roi_labels.items():
                if roi_name in roi_names:
                    roi_coords = np.where(mask_data == label_val)
                    indices = []
                    for i in range(len(roi_coords[0])):
                        coord = (roi_coords[0][i], roi_coords[1][i], roi_coords[2][i])
                        if coord in coord_to_idx:
                            indices.append(coord_to_idx[coord])
                    roi_index[roi_name] = np.array(indices, dtype=np.int64)

    # Remaining voxels go to nsdgeneral_other
    if "nsdgeneral_other" in roi_names:
        assigned = set()
        for indices in roi_index.values():
            assigned.update(indices.tolist())
        remaining = [i for i in range(n_voxels) if i not in assigned]
        roi_index["nsdgeneral_other"] = np.array(remaining, dtype=np.int64)

    logger.info(
        "Built ROI index for %s: %d ROIs, %d/%d voxels assigned",
        subject, len(roi_index),
        sum(len(v) for v in roi_index.values()), n_voxels,
    )

    return roi_index
