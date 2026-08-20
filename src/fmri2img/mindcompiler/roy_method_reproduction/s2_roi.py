"""S2.2B ROI definitions + outcome-independent voxel selection (subj01).

Seven project ROIs. V1 is the S2.1B DISCOVERY_ROI; the other six are the
PROSPECTIVE_CONFIRMATION_ROIS. Voxel selection uses ONLY the shared NSD-core
``betas_fithrf/ncsnr`` (independent of any B0/B1 NSD-Imagery outcome): ROI mask ∩
valid_nsdimagery ∩ finite ncsnr, then strict ``ncsnr > ROI-specific 98th pct``. B0
and B1 use the identical selected voxel indices within an ROI.
"""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np

#: (source NIfTI, integer labels). prf-visualrois: V1v/V1d/V2v/V2d/V3v/V3d/hV4 = 1..7;
#: streams: ventral/lateral/parietal = 5/6/7.
ROI_DEFS: Dict[str, tuple] = {
    "V1": ("prf-visualrois", (1, 2)),        # DISCOVERY
    "V2": ("prf-visualrois", (3, 4)),
    "V3": ("prf-visualrois", (5, 6)),
    "hV4": ("prf-visualrois", (7,)),
    "ventral": ("streams", (5,)),
    "lateral": ("streams", (6,)),
    "parietal": ("streams", (7,)),
}
DISCOVERY_ROI = "V1"
PROSPECTIVE_ROIS = ("V2", "V3", "hV4", "ventral", "lateral", "parietal")
VOXEL_SNR_PERCENTILE = 98.0


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def select_roi_voxels(roi_name: str, roi_dir: str, ppdata_dir: str, ncsnr_path: str) -> dict:
    """Outcome-independent selection for one ROI. Returns xyz + full QC record.

    Mirrors the V1 policy exactly (regression-checked in tests): strict ``>`` the
    ROI-specific 98th percentile of NSD-core ncsnr within ROI ∩ valid ∩ finite.
    """
    import nibabel as nib
    if roi_name not in ROI_DEFS:
        raise ValueError(f"unknown ROI {roi_name!r}")
    source, labels = ROI_DEFS[roi_name]
    src_img = nib.load(f"{roi_dir}/{source}.nii.gz")
    src = np.round(src_img.get_fdata()).astype(int)
    valid = nib.load(f"{ppdata_dir}/valid_nsdimagery.nii.gz").get_fdata()
    ncsnr = nib.load(ncsnr_path).get_fdata()

    roi_mask = np.isin(src, list(labels))
    valid_finite = roi_mask & (valid > 0) & np.isfinite(ncsnr)
    xs, ys, zs = np.where(valid_finite)
    snr = ncsnr[valid_finite]
    thr = float(np.percentile(snr, VOXEL_SNR_PERCENTILE)) if snr.size else float("nan")
    keep = snr > thr
    xyz = np.stack([xs[keep], ys[keep], zs[keep]])
    return dict(
        roi=roi_name, source=source, labels=list(labels),
        candidate_voxels=int(roi_mask.sum()),
        valid_finite_voxels=int(valid_finite.sum()),
        threshold=thr,
        selected_voxels=int(keep.sum()),
        ge_sensitivity_count=int((snr >= thr).sum()) if snr.size else 0,
        threshold_ties=int((snr == thr).sum()) if snr.size else 0,
        # 12-char to match the V1 discovery convention (select_v1_voxels)
        voxel_hash=(hashlib.sha256(xyz.tobytes()).hexdigest()[:12] if xyz.size else ""),
        affine_hash=_h(src_img.affine),
        mask_hash=_h(valid_finite.astype(np.uint8)),
        xyz=xyz,
    )
