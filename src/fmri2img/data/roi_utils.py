"""
ROI Index Utilities for NSD
===========================

Builds per-ROI voxel index arrays from NSD anatomical masks, enabling the
ROITransformerEncoder to map a flat nsdgeneral feature vector to
brain-region-specific tokens at runtime.

NSD atlas files referenced (Allen et al., 2022):
  - Kastner2015.nii.gz     : V1v … V3A, V3B  (retinotopic, labels 1-17)
  - prf-visualrois.nii.gz  : V1v … hV4        (retinotopic, labels 1-7)
  - floc-faces.nii.gz      : OFA, FFA1, FFA2   (category-selective)
  - floc-bodies.nii.gz     : EBA               (category-selective)
  - floc-places.nii.gz     : OPA, PPA, RSC     (category-selective)
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# ── NSD ROI label map ────────────────────────────────────────────────────
# Mapping: roi_name -> list of (nifti_filename, integer_label) candidates.
# We try files in order so that the atlas with the widest coverage
# (Kastner2015) is preferred for retinotopic areas.
NSD_ROI_LABEL_MAP: Dict[str, List[Tuple[str, int]]] = {
    "V1v":  [("Kastner2015.nii.gz", 1),  ("prf-visualrois.nii.gz", 1)],
    "V1d":  [("Kastner2015.nii.gz", 2),  ("prf-visualrois.nii.gz", 2)],
    "V2v":  [("Kastner2015.nii.gz", 3),  ("prf-visualrois.nii.gz", 3)],
    "V2d":  [("Kastner2015.nii.gz", 4),  ("prf-visualrois.nii.gz", 4)],
    "V3v":  [("Kastner2015.nii.gz", 5),  ("prf-visualrois.nii.gz", 5)],
    "V3d":  [("Kastner2015.nii.gz", 6),  ("prf-visualrois.nii.gz", 6)],
    "V4":   [("Kastner2015.nii.gz", 7),  ("prf-visualrois.nii.gz", 7)],
    "V3A":  [("Kastner2015.nii.gz", 17)],
    "V3B":  [("Kastner2015.nii.gz", 16)],
    "FFA1": [("floc-faces.nii.gz", 2)],
    "FFA2": [("floc-faces.nii.gz", 3)],
    "OFA":  [("floc-faces.nii.gz", 1)],
    "EBA":  [("floc-bodies.nii.gz", 1)],
    "PPA":  [("floc-places.nii.gz", 2)],
    "OPA":  [("floc-places.nii.gz", 1)],
    "RSC":  [("floc-places.nii.gz", 3)],
}


def _resolve_roi_dir(subject: str) -> Path:
    """Locate the ROI mask directory for a subject."""
    nsd_root = os.environ.get("NSD_DATA_ROOT", "data/nsd")
    candidates = [
        Path(nsd_root) / "nsddata" / "ppdata" / subject / "func1pt8mm" / "roi",
        Path(nsd_root) / "ppdata" / subject / "func1pt8mm" / "roi",
        Path("data") / "nsd" / "ppdata" / subject / "func1pt8mm" / "roi",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def _load_nsdgeneral_mask(subject: str) -> Tuple[np.ndarray, Path]:
    """Load the nsdgeneral boolean mask for *subject*.

    Returns:
        (mask_3d, mask_path)  where mask_3d is bool dtype.
    """
    roi_dir = _resolve_roi_dir(subject)
    mask_path = roi_dir / "nsdgeneral.nii.gz"
    if not mask_path.exists():
        parent = roi_dir.parent
        alt = parent / "nsdgeneral.nii.gz"
        if alt.exists():
            mask_path = alt
        else:
            raise FileNotFoundError(
                f"nsdgeneral.nii.gz not found at {mask_path} or {alt}"
            )
    return nib.load(str(mask_path)).get_fdata() > 0.5, mask_path


def _load_label_map(roi_dir: Path, filename: str) -> Optional[np.ndarray]:
    """Try to load an ROI label map NIfTI, returning None if missing."""
    path = roi_dir / filename
    if not path.exists():
        return None
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.int32)


def build_roi_index(
    subject: str,
    roi_names: List[str],
) -> Tuple[OrderedDict, OrderedDict]:
    """Compute per-ROI voxel indices into the flat nsdgeneral feature vector.

    For each ROI listed in *roi_names* (excluding ``"nsdgeneral_other"``),
    the function loads the corresponding NSD atlas, intersects with
    nsdgeneral, and records which positions in the flat nsdgeneral vector
    belong to that ROI.  Any remaining nsdgeneral voxels are assigned to
    ``"nsdgeneral_other"``.

    Args:
        subject: NSD subject id (e.g. ``"subj01"``).
        roi_names: Ordered list of ROI names as they appear in the config
            ``encoder.roi_dims`` keys.  ``"nsdgeneral_other"`` may be
            present; it is always recomputed from the residual.

    Returns:
        ``(roi_dims, roi_indices)`` where both are OrderedDicts keyed by
        ROI name.  ``roi_dims`` maps to ``int`` (voxel count) and
        ``roi_indices`` maps to ``np.ndarray[int64]`` of indices into the
        flat nsdgeneral vector.
    """
    nsdgen_3d, mask_path = _load_nsdgeneral_mask(subject)
    nsdgen_flat = np.flatnonzero(nsdgen_3d)
    n_gen = len(nsdgen_flat)
    logger.info("nsdgeneral mask: %s (%d voxels)", mask_path, n_gen)

    nsdgen_set = set(nsdgen_flat.tolist())
    nsdgen_flat_to_idx = {v: i for i, v in enumerate(nsdgen_flat)}

    roi_dir = _resolve_roi_dir(subject)
    label_cache: Dict[str, Optional[np.ndarray]] = {}

    roi_dims: OrderedDict = OrderedDict()
    roi_indices: OrderedDict = OrderedDict()
    claimed = np.zeros(n_gen, dtype=bool)

    named_rois = [r for r in roi_names if r != "nsdgeneral_other"]

    for roi_name in named_rois:
        candidates = NSD_ROI_LABEL_MAP.get(roi_name)
        if candidates is None:
            logger.warning("ROI '%s' has no entry in NSD_ROI_LABEL_MAP — skipping", roi_name)
            continue

        found = False
        for filename, label_val in candidates:
            if filename not in label_cache:
                label_cache[filename] = _load_label_map(roi_dir, filename)
            label_map = label_cache[filename]
            if label_map is None:
                continue

            roi_voxels_3d = np.flatnonzero(label_map == label_val)
            intersection = np.array(
                sorted(nsdgen_flat_to_idx[v] for v in roi_voxels_3d if v in nsdgen_set),
                dtype=np.int64,
            )

            if len(intersection) == 0:
                continue

            overlap_with_prev = claimed[intersection].sum()
            if overlap_with_prev > 0:
                intersection = intersection[~claimed[intersection]]

            if len(intersection) == 0:
                continue

            claimed[intersection] = True
            roi_dims[roi_name] = len(intersection)
            roi_indices[roi_name] = intersection
            logger.info("  ROI %-20s : %5d voxels (from %s label=%d)",
                        roi_name, len(intersection), filename, label_val)
            found = True
            break

        if not found:
            logger.warning("ROI '%s': no atlas file found in %s — skipping", roi_name, roi_dir)

    other_indices = np.where(~claimed)[0].astype(np.int64)
    roi_dims["nsdgeneral_other"] = len(other_indices)
    roi_indices["nsdgeneral_other"] = other_indices
    logger.info("  ROI %-20s : %5d voxels (residual)", "nsdgeneral_other", len(other_indices))

    total = sum(roi_dims.values())
    assert total == n_gen, (
        f"ROI partition sanity check failed: {total} != {n_gen}"
    )
    logger.info("ROI index built: %d ROIs, %d total voxels", len(roi_dims), total)
    return roi_dims, roi_indices
