from __future__ import annotations
import logging, numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem
from fmri2img.io.nsd_layout import NSDLayout
import re

log = logging.getLogger(__name__)

@dataclass
class ROIDef:
    name: str
    mask_indices: np.ndarray  # flat indices into (H,W,D).ravel()

class ROIPooler:
    """
    Loads ROI masks for a subject and pools fMRI volumes into ROI means.
    Assumes masks are aligned with beta volumes (native func space).
    """
    def __init__(self, subject: str, min_voxels: int = 50):
        self.subject = subject
        self.min_voxels = min_voxels
        self.rois: List[ROIDef] = []
        self.shape: Optional[Tuple[int,int,int]] = None

    def fit(self, sample_beta_path: str):
        """
        Discover masks via NSDLayout roi_masks pattern and prepare index arrays.
        Uses `sample_beta_path` to infer target shape for sanity.
        
        Falls back gracefully to whole-brain (no ROI) if masks are not found.
        """
        layout = NSDLayout("configs/data.yaml")
        fs = get_s3_filesystem().fs
        nifti = NIfTILoader()

        # infer functional shape from one beta file (header-only)
        if sample_beta_path.startswith("s3://"):
            img = nifti.load(sample_beta_path, validate=False)
        else:
            # Local file - load directly with nibabel
            import nibabel as nib
            img = nib.load(sample_beta_path)
        if len(img.shape) < 3:
            raise RuntimeError("Invalid beta file shape")
        self.shape = tuple(img.shape[:3])
        flat_size = int(np.prod(self.shape))

        # Try alternate ROI mask patterns
        alt_patterns = [
            layout.roi_masks_path(self.subject, full_url=True),
            layout.fsaverage_roi_masks_path(self.subject, full_url=True),
            layout.mni_roi_masks_path(self.subject, full_url=True),
        ]
        
        roi_files = []
        for pattern in alt_patterns:
            if pattern.startswith("s3://"):
                files = fs.glob(pattern)
            else:
                # Local files - use glob directly
                import glob
                files = glob.glob(pattern)
            
            if files:
                roi_files = files
                log.info(f"Found {len(roi_files)} ROI masks at {pattern}")
                break
        
        if not roi_files:
            # Graceful fallback: warn once and continue without ROI
            subj_path = self.subject if not self.subject.startswith("subj") else self.subject[4:]
            log.warning(
                f"ROI masks not found for {self.subject} → using whole-brain fallback. "
                f"To use ROI pooling, provide masks under ppdata/subj{subj_path}/anat/*roi*.nii.gz"
            )
            self.rois = []  # Empty ROI list triggers whole-brain mode
            return self

        # load each ROI mask, build index list
        rois: List[ROIDef] = []
        for p in roi_files:
            try:
                if isinstance(p, str) and p.startswith("s3://"):
                    mimg = nifti.load(p, validate=False)
                elif isinstance(p, str):
                    # Local file
                    import nibabel as nib
                    mimg = nib.load(p)
                else:
                    # Assume it's already an S3 path from glob
                    mimg = nifti.load(f"s3://{p}" if not str(p).startswith("s3://") else str(p), validate=False)
                mdata = mimg.get_fdata()  # masks are usually tiny; OK to load
                if mdata.shape[:3] != self.shape:
                    log.warning(f"ROI {p} shape {mdata.shape[:3]} != func shape {self.shape}, skipping")
                    continue
                mask = (mdata > 0).astype(np.bool_)
                count = int(mask.sum())
                if count < self.min_voxels:
                    continue
                # derive a stable roi name
                name = p.split("/")[-1]
                name = re.sub(r"\.nii(\.gz)?$", "", name)
                idx = np.flatnonzero(mask.ravel())
                rois.append(ROIDef(name=name, mask_indices=idx))
            except Exception as e:
                log.warning(f"Failed ROI load {p}: {e}")

        # de-duplicate by name
        seen = set()
        uniq: List[ROIDef] = []
        for r in rois:
            if r.name not in seen:
                seen.add(r.name)
                uniq.append(r)
        self.rois = uniq
        
        if len(self.rois) == 0:
            log.warning(
                f"No valid ROI masks loaded for {self.subject} → using whole-brain fallback"
            )
        else:
            log.info(f"ROI pooler for {self.subject}: {len(self.rois)} masks ready")
        
        return self

    def pool(self, vol3d: np.ndarray) -> np.ndarray:
        """
        vol3d: float32 (H,W,D). Returns (n_roi,) float32 with mean per ROI.
        """
        assert self.shape, "Call fit() first"
        assert vol3d.shape == self.shape, f"Unexpected shape {vol3d.shape} vs {self.shape}"
        
        if not self.rois:
            # Return empty array if no ROIs found
            return np.array([], dtype=np.float32)
            
        flat = vol3d.ravel()
        out = np.empty((len(self.rois),), dtype=np.float32)
        for i, r in enumerate(self.rois):
            if r.mask_indices.size == 0:
                out[i] = np.nan
            else:
                out[i] = float(np.nanmean(flat[r.mask_indices]))
        return out

    def names(self) -> List[str]:
        return [r.name for r in self.rois]