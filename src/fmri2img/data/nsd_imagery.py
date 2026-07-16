"""
NSD-Imagery Dataset Integration
================================

Preprocessing pipeline for the NSD-Imagery benchmark (Kneeland et al., 2025),
enabling cross-cognitive-state evaluation (perception vs mental imagery).

NSD-Imagery provides fMRI recordings of the SAME subjects (subj01, subj02,
subj05, subj07) performing mental imagery of previously seen images,
enabling direct comparison of perception vs imagery decoding.

Dataset source: OpenNeuro ds005614
Paper: Kneeland et al. (2025) "NSD-Imagery: A benchmark dataset for extending
       fMRI vision decoding methods to mental imagery" (CVPR 2025)

Integration:
    - Uses same ROI masks (nsdgeneral) as NSD perception data
    - Compatible with PreextractedNSDDataset format
    - Outputs: imagery_features.npy + imagery_meta.parquet

Key insight from NSD-Imagery paper:
    "Performance on vision reconstruction does not guarantee similar
     performance on mental imagery [...] models employing simple linear
     decoding architectures and multimodal feature decoding generalize
     better to mental imagery."

Our hypothesis: Uncertainty-aware models (vMF with calibrated kappa) will
show LOWER kappa for imagery trials (correct! brain encodes less) and
therefore produce appropriately uncertain reconstructions rather than
confident hallucinations.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NSD-Imagery OpenNeuro dataset ID
OPENNEURO_DATASET_ID = "ds005614"
OPENNEURO_URL = f"https://openneuro.org/datasets/{OPENNEURO_DATASET_ID}"

# Subjects available in NSD-Imagery (subset of NSD)
IMAGERY_SUBJECTS = ["subj01", "subj02", "subj05", "subj07"]

# Expected directory structure after download
# {data_root}/nsd_imagery/
#   sub-01/func/  -- imagery fMRI betas
#   sub-01/anat/  -- same anatomy as NSD
#   stimuli/      -- imagery stimulus images (cued images)
#   participants.tsv


@dataclass
class ImageryTrialInfo:
    """Metadata for a single imagery trial."""
    trial_idx: int
    subject: str
    nsd_id: int
    session: int
    run: int
    condition: str  # 'imagery' or 'perception' (if perception runs included)
    cue_image_path: Optional[str] = None


def get_imagery_data_root(base_data_root: Optional[Path] = None) -> Path:
    """Resolve the NSD-Imagery data root directory."""
    import os
    if base_data_root is not None:
        return base_data_root / "nsd_imagery"
    env_root = os.environ.get("NSD_IMAGERY_ROOT")
    if env_root:
        return Path(env_root)
    dataset_root = os.environ.get("DATASET_ROOT", "/home/jovyan/work/data")
    return Path(dataset_root) / "nsd_imagery"


def download_nsd_imagery(
    output_dir: Optional[Path] = None,
    subjects: Optional[List[str]] = None,
) -> Path:
    """
    Download NSD-Imagery dataset from OpenNeuro.

    Uses datalad or openneuro-py CLI if available, otherwise provides
    instructions for manual download.

    Parameters
    ----------
    output_dir : target directory (default: auto from env)
    subjects : which subjects to download (default: all 4)

    Returns
    -------
    Path to the downloaded dataset root
    """
    data_root = output_dir or get_imagery_data_root()
    data_root.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = IMAGERY_SUBJECTS

    logger.info(
        "NSD-Imagery download target: %s (subjects: %s)",
        data_root, subjects,
    )

    # Check if already downloaded
    marker = data_root / ".download_complete"
    if marker.exists():
        logger.info("NSD-Imagery already downloaded (marker found)")
        return data_root

    # Try openneuro-py first
    try:
        import subprocess
        cmd = [
            "openneuro-py", "download",
            "--dataset", OPENNEURO_DATASET_ID,
            "--target_dir", str(data_root),
        ]
        for subj in subjects:
            sub_id = subj.replace("subj0", "sub-0")
            cmd.extend(["--include", f"{sub_id}/"])

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode == 0:
            marker.touch()
            logger.info("NSD-Imagery download complete")
            return data_root
        else:
            logger.warning("openneuro-py failed: %s", result.stderr[:500])
    except (ImportError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("openneuro-py not available: %s", e)

    # Provide manual instructions
    instructions = f"""
    NSD-Imagery dataset not found at {data_root}.
    
    Download options:
    1. pip install openneuro-py && openneuro-py download --dataset {OPENNEURO_DATASET_ID} --target_dir {data_root}
    2. aws s3 sync --no-sign-request s3://openneuro.org/{OPENNEURO_DATASET_ID} {data_root}
    3. Manual download from {OPENNEURO_URL}
    
    After download, create marker: touch {marker}
    """
    logger.info(instructions)
    return data_root


def preprocess_imagery_betas(
    subject: str,
    imagery_root: Optional[Path] = None,
    nsd_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    roi_mask: str = "nsdgeneral",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract and preprocess imagery fMRI betas using same pipeline as NSD.

    Applies the SAME nsdgeneral ROI mask used for perception data,
    ensuring the feature spaces are directly comparable.

    Parameters
    ----------
    subject : e.g., 'subj01'
    imagery_root : NSD-Imagery dataset root
    nsd_root : NSD dataset root (for ROI masks)
    output_dir : where to save preprocessed features
    roi_mask : which ROI mask to apply (default: nsdgeneral)

    Returns
    -------
    features : (N_imagery_trials, N_voxels) float32 array
    meta_df : DataFrame with trial metadata (nsd_id, session, condition)
    """
    import os
    import nibabel as nib

    imagery_root = imagery_root or get_imagery_data_root()
    nsd_root = Path(os.environ.get("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd"))

    if output_dir is None:
        cache_root = Path(os.environ.get("CACHE_ROOT", "cache"))
        output_dir = cache_root / "preextracted_imagery" / f"subject={subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sub_id = subject.replace("subj0", "sub-0")

    # Load ROI mask (same as perception pipeline)
    mask_path = nsd_root / "nsddata" / "ppdata" / subject / "func1pt8mm" / "roi" / f"{roi_mask}.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")

    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata()
    roi_indices = np.where(mask_data > 0)
    n_voxels = len(roi_indices[0])
    logger.info("ROI mask %s for %s: %d voxels", roi_mask, subject, n_voxels)

    # Find imagery beta files
    imagery_func_dir = imagery_root / sub_id / "func"
    if not imagery_func_dir.exists():
        raise FileNotFoundError(
            f"Imagery functional data not found: {imagery_func_dir}. "
            f"Run download_nsd_imagery() first."
        )

    beta_files = sorted(imagery_func_dir.glob("*_betas*.nii.gz"))
    if not beta_files:
        beta_files = sorted(imagery_func_dir.glob("*_bold*.nii.gz"))

    if not beta_files:
        raise FileNotFoundError(f"No beta/bold files in {imagery_func_dir}")

    logger.info("Found %d imagery beta files for %s", len(beta_files), subject)

    all_features = []
    trial_meta = []
    trial_idx = 0

    for beta_file in beta_files:
        beta_img = nib.load(str(beta_file))
        beta_data = beta_img.get_fdata()

        # Handle 4D (multiple trials per file) or 3D (single trial)
        if beta_data.ndim == 4:
            n_trials_in_file = beta_data.shape[3]
        elif beta_data.ndim == 3:
            beta_data = beta_data[..., np.newaxis]
            n_trials_in_file = 1
        else:
            logger.warning("Unexpected shape %s in %s, skipping", beta_data.shape, beta_file)
            continue

        for t in range(n_trials_in_file):
            vol = beta_data[:, :, :, t]
            masked_voxels = vol[roi_indices].astype(np.float32)
            all_features.append(masked_voxels)

            trial_meta.append({
                "trial_idx": trial_idx,
                "subject": subject,
                "condition": "imagery",
                "source_file": beta_file.name,
                "volume_idx": t,
            })
            trial_idx += 1

    features = np.array(all_features, dtype=np.float32)
    meta_df = pd.DataFrame(trial_meta)

    # Load stimulus mapping if available
    events_files = sorted(imagery_func_dir.glob("*_events.tsv"))
    if events_files:
        events_dfs = []
        for ef in events_files:
            try:
                events_dfs.append(pd.read_csv(ef, sep="\t"))
            except Exception as e:
                logger.warning("Could not read events file %s: %s", ef, e)
        if events_dfs:
            events_combined = pd.concat(events_dfs, ignore_index=True)
            if "nsd_id" in events_combined.columns:
                meta_df["nsd_id"] = events_combined["nsd_id"].values[:len(meta_df)]
            elif "nsdId" in events_combined.columns:
                meta_df["nsd_id"] = events_combined["nsdId"].values[:len(meta_df)]

    # Save
    features_path = output_dir / "imagery_features.npy"
    meta_path = output_dir / "imagery_meta.parquet"
    np.save(features_path, features)
    meta_df.to_parquet(meta_path, index=False)

    logger.info(
        "Preprocessed NSD-Imagery for %s: %d trials, %d voxels -> %s",
        subject, len(features), n_voxels, output_dir,
    )

    return features, meta_df


def load_imagery_features(
    subject: str,
    cache_root: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load pre-extracted imagery features (analogous to perception features)."""
    import os
    if cache_root is None:
        cache_root = Path(os.environ.get("CACHE_ROOT", "cache"))

    output_dir = cache_root / "preextracted_imagery" / f"subject={subject}"
    features_path = output_dir / "imagery_features.npy"
    meta_path = output_dir / "imagery_meta.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Imagery features not found at {features_path}. "
            f"Run preprocess_imagery_betas('{subject}') first."
        )

    features = np.load(features_path)
    meta_df = pd.read_parquet(meta_path)

    logger.info(
        "Loaded imagery features for %s: %d trials x %d voxels",
        subject, features.shape[0], features.shape[1],
    )
    return features, meta_df


class NSDImageryDataset:
    """
    PyTorch Dataset for NSD-Imagery, compatible with evaluation pipeline.

    Mirrors the interface of PreextractedNSDDataset but for imagery trials.
    Supports paired evaluation: same nsd_ids across perception and imagery.
    """

    def __init__(
        self,
        features: np.ndarray,
        meta_df: pd.DataFrame,
        embeddings_df: Optional[pd.DataFrame] = None,
        embedding_column: str = "fused",
    ):
        """
        Parameters
        ----------
        features : (N, V) imagery fMRI features
        meta_df : imagery trial metadata (must have 'nsd_id' column for CLIP lookup)
        embeddings_df : CLIP embedding lookup (same as perception pipeline)
        embedding_column : which column in embeddings_df contains the embedding
        """
        self.features = features
        self.meta_df = meta_df
        self.embeddings_df = embeddings_df
        self.embedding_column = embedding_column

        self._has_embeddings = (
            embeddings_df is not None and
            "nsd_id" in meta_df.columns
        )

        if self._has_embeddings:
            self._embedding_lookup = {
                row.get("nsdId", row.get("nsd_id")): idx
                for idx, (_, row) in enumerate(embeddings_df.iterrows())
            }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict:
        import torch

        fmri = torch.from_numpy(self.features[idx]).float()
        meta = self.meta_df.iloc[idx]

        item = {
            "fmri": fmri,
            "condition": "imagery",
            "trial_idx": idx,
        }

        if "nsd_id" in meta.index:
            item["nsd_id"] = int(meta["nsd_id"])

        if self._has_embeddings and "nsd_id" in meta.index:
            nsd_id = int(meta["nsd_id"])
            emb_idx = self._embedding_lookup.get(nsd_id)
            if emb_idx is not None:
                emb_row = self.embeddings_df.iloc[emb_idx]
                if self.embedding_column in emb_row.index:
                    emb = emb_row[self.embedding_column]
                    if isinstance(emb, np.ndarray):
                        item["clip_target"] = torch.from_numpy(emb).float()
                    elif hasattr(emb, "values"):
                        item["clip_target"] = torch.tensor(emb.values, dtype=torch.float32)

        return item


def create_paired_perception_imagery_split(
    perception_meta: pd.DataFrame,
    imagery_meta: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find paired trials: same nsd_ids present in both perception and imagery.

    Returns indices into each DataFrame for matched pairs.
    """
    if "nsd_id" not in imagery_meta.columns or "nsdId" not in perception_meta.columns:
        raise ValueError("Both DataFrames need nsd_id/nsdId columns for pairing")

    perception_ids = set(perception_meta["nsdId"].values)
    imagery_ids = set(imagery_meta["nsd_id"].values)
    shared_ids = perception_ids & imagery_ids

    perc_indices = []
    img_indices = []

    for nsd_id in sorted(shared_ids):
        perc_mask = perception_meta["nsdId"] == nsd_id
        img_mask = imagery_meta["nsd_id"] == nsd_id

        perc_idx = perception_meta[perc_mask].index[0]
        img_idx = imagery_meta[img_mask].index[0]

        perc_indices.append(perc_idx)
        img_indices.append(img_idx)

    logger.info(
        "Paired perception-imagery split: %d shared nsd_ids "
        "(from %d perception, %d imagery)",
        len(shared_ids), len(perception_ids), len(imagery_ids),
    )

    return np.array(perc_indices), np.array(img_indices)


def compute_perception_imagery_kappa_comparison(
    perception_kappas: np.ndarray,
    imagery_kappas: np.ndarray,
    roi_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Compare kappa distributions between perception and imagery.

    Core hypothesis: Imagery trials should have LOWER kappa (less signal),
    and this drop should vary across ROIs (early visual > higher areas).

    Parameters
    ----------
    perception_kappas : (N_paired, n_rois) perception trial kappas
    imagery_kappas : (N_paired, n_rois) matched imagery trial kappas
    roi_names : ROI labels

    Returns
    -------
    Dict with statistical comparisons per ROI and overall
    """
    from scipy import stats as sp_stats

    if roi_names is None:
        from fmri2img.eval.roi_kappa_extraction import DEFAULT_ROI_NAMES
        roi_names = DEFAULT_ROI_NAMES[:perception_kappas.shape[1]]

    n_rois = perception_kappas.shape[1]

    overall_perc_mean = perception_kappas.mean()
    overall_img_mean = imagery_kappas.mean()
    overall_drop = (overall_perc_mean - overall_img_mean) / overall_perc_mean

    per_roi_results = {}
    for roi_idx in range(n_rois):
        perc_k = perception_kappas[:, roi_idx]
        img_k = imagery_kappas[:, roi_idx]

        t_stat, p_val = sp_stats.ttest_rel(perc_k, img_k)
        effect_size = (perc_k.mean() - img_k.mean()) / np.sqrt(
            (perc_k.std() ** 2 + img_k.std() ** 2) / 2
        )

        roi_name = roi_names[roi_idx] if roi_idx < len(roi_names) else f"roi_{roi_idx}"
        per_roi_results[roi_name] = {
            "perception_mean_kappa": float(perc_k.mean()),
            "imagery_mean_kappa": float(img_k.mean()),
            "kappa_drop_fraction": float((perc_k.mean() - img_k.mean()) / perc_k.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(effect_size),
            "significant": p_val < 0.05 / n_rois,  # Bonferroni correction
        }

    report = {
        "overall": {
            "perception_mean_kappa": float(overall_perc_mean),
            "imagery_mean_kappa": float(overall_img_mean),
            "kappa_drop_fraction": float(overall_drop),
        },
        "per_roi": per_roi_results,
        "n_pairs": len(perception_kappas),
        "hypothesis_confirmed": overall_img_mean < overall_perc_mean,
    }

    n_sig = sum(1 for v in per_roi_results.values() if v["significant"])
    logger.info(
        "Perception vs Imagery kappa: %.2f vs %.2f (%.1f%% drop). "
        "%d/%d ROIs significant after Bonferroni.",
        overall_perc_mean, overall_img_mean, overall_drop * 100,
        n_sig, n_rois,
    )

    return report
