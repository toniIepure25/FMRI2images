"""
Multi-Subject Pre-extracted fMRI Dataset
========================================

Concatenates pre-extracted fMRI features from multiple NSD subjects into a
single PyTorch Dataset.  Each sample returns ``(fmri, clip_embedding,
subject_id_int)`` so the model can route through per-subject ROI projections.

CLIP embeddings are keyed by ``nsdId`` and shared across subjects (same
stimulus images).  Subject identifiers are mapped to contiguous integers
0..N-1 for use with ``nn.Embedding``.

Usage::

    from fmri2img.data.multi_subject_dataset import MultiSubjectPreextractedDataset

    ds = MultiSubjectPreextractedDataset(
        subjects=["subj01", "subj02", "subj05", "subj07"],
        cache_root=Path("cache/preextracted"),
        index_root=Path("data/indices/nsd_index"),
        embeddings_df=clip_df,
    )
    fmri, emb, subj_id = ds[0]
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultiSubjectPreextractedDataset(Dataset):
    """Concatenated multi-subject dataset backed by per-subject ``.npy`` files.

    Args:
        subjects:     Ordered list of subject IDs (e.g. ``["subj01", "subj02"]``).
        cache_root:   Root directory containing ``subject={subj}/fmri_features.npy``.
        index_root:   Root directory containing ``subject={subj}/index.parquet``.
        embeddings_df: CLIP embedding DataFrame with ``nsdId`` column.
        embedding_column: Which column to use (``"fused"``, ``"final"``, etc.).
                          If ``None``, auto-resolved via priority list.
        exclude_shared1000: If True, remove shared1000 stimuli (for clean eval).
        split_by_image: If True, split by unique ``nsdId`` (no image leakage).
        val_ratio:    Fraction of unique images for validation.
        seed:         Random seed for reproducible splits.
        average_repetitions: If True, average fMRI across repetitions of the
                             same ``nsdId`` within each subject before training.
                             Reduces noise by ~sqrt(n_reps).
    """

    SUBJECT_TO_INT: Dict[str, int] = {}

    # Multi-layer CLIP columns used for hierarchical alignment
    HIER_COLUMNS: List[str] = ["layer_12_proj", "layer_18_proj", "final"]

    def __init__(
        self,
        subjects: List[str],
        cache_root: Path,
        index_root: Path,
        embeddings_df: pd.DataFrame,
        embedding_column: Optional[str] = None,
        exclude_shared1000: bool = True,
        split_by_image: bool = True,
        val_ratio: float = 0.10,
        seed: int = 42,
        average_repetitions: bool = False,
        token_cache=None,
    ):
        super().__init__()
        self.subjects = list(subjects)
        self.subject_to_int = {s: i for i, s in enumerate(self.subjects)}
        MultiSubjectPreextractedDataset.SUBJECT_TO_INT = self.subject_to_int

        self.token_cache = token_cache
        self.embeddings_df = embeddings_df

        from fmri2img.data.multi_subject_dataset import _resolve_emb_col
        self._emb_col = _resolve_emb_col(embeddings_df, embedding_column)
        logger.info("Embedding column: %s", self._emb_col)

        # Detect available hierarchical CLIP columns
        self._hier_cols = [
            c for c in self.HIER_COLUMNS if c in embeddings_df.columns
        ]
        if self._hier_cols:
            logger.info("Hierarchical CLIP columns available: %s", self._hier_cols)
        else:
            logger.info("No hierarchical CLIP columns found — hierarchical loss disabled")

        self.embedding_lookup: Dict[int, int] = {}
        if "nsdId" in embeddings_df.columns:
            for i, nid in enumerate(embeddings_df["nsdId"].values):
                self.embedding_lookup[int(nid)] = i

        all_features: List[np.ndarray] = []
        all_rows: List[pd.DataFrame] = []

        for subj in self.subjects:
            feat_path = cache_root / f"subject={subj}" / "fmri_features.npy"
            idx_path = index_root / f"subject={subj}" / "index.parquet"

            if not feat_path.exists():
                raise FileNotFoundError(
                    f"Pre-extracted features not found: {feat_path}. "
                    f"Run: make preextract SUBJECT={subj}"
                )
            if not idx_path.exists():
                raise FileNotFoundError(
                    f"Index not found: {idx_path}. "
                    f"Run: make index SUBJECT={subj}"
                )

            feats = np.load(feat_path, mmap_mode=None)
            idx_df = pd.read_parquet(idx_path).reset_index(drop=True)

            if len(feats) != len(idx_df):
                raise ValueError(
                    f"{subj}: features ({len(feats)}) != index ({len(idx_df)}). "
                    f"Re-run: make preextract SUBJECT={subj}"
                )

            if average_repetitions and "nsdId" in idx_df.columns:
                n_raw = len(idx_df)
                nsd_vals = idx_df["nsdId"].values
                unique_ids = np.unique(nsd_vals)
                avg_feats = np.zeros(
                    (len(unique_ids), feats.shape[1]), dtype=np.float32,
                )
                new_rows: List[dict] = []
                for i, nsd_id in enumerate(unique_ids):
                    mask = nsd_vals == nsd_id
                    avg_feats[i] = feats[mask].mean(axis=0)
                    new_rows.append(idx_df[mask].iloc[0].to_dict())
                feats = avg_feats
                idx_df = pd.DataFrame(new_rows).reset_index(drop=True)
                logger.info(
                    "%s: rep-averaging %d trials -> %d images (SNR ~%.2fx)",
                    subj, n_raw, len(unique_ids),
                    np.sqrt(n_raw / max(len(unique_ids), 1)),
                )

            idx_df["_subject"] = subj
            idx_df["_subject_int"] = self.subject_to_int[subj]
            idx_df["_local_feat_idx"] = np.arange(len(idx_df), dtype=np.int64)

            all_features.append(feats)
            all_rows.append(idx_df)

            logger.info(
                "%s: %d trials, %d voxels",
                subj, len(idx_df), feats.shape[1],
            )

        self.features_list = all_features
        self.voxel_counts = {
            subj: f.shape[1] for subj, f in zip(self.subjects, all_features)
        }

        combined_df = pd.concat(all_rows, ignore_index=True)

        if exclude_shared1000 and "shared1000" in combined_df.columns:
            n_before = len(combined_df)
            combined_df = combined_df[~combined_df["shared1000"]].reset_index(drop=True)
            logger.info("Excluded shared1000: %d -> %d trials", n_before, len(combined_df))

        # Filter to nsdIds present in embedding cache (CLS) or token cache
        if self.token_cache is not None:
            cached_nsd_ids = set(int(nid) for nid in self.token_cache._nsd_ids)
            logger.info(
                "Token cache mode: filtering to %d nsdIds in token cache",
                len(cached_nsd_ids),
            )
        else:
            cached_nsd_ids = set(self.embedding_lookup.keys())
        n_before_filter = len(combined_df)
        combined_df = combined_df[combined_df["nsdId"].isin(cached_nsd_ids)].reset_index(drop=True)
        n_filtered = n_before_filter - len(combined_df)
        if n_filtered > 0:
            logger.warning(
                "Filtered %d trials with nsdIds not in CLIP cache (%d -> %d)",
                n_filtered, n_before_filter, len(combined_df),
            )

        self.index_df = combined_df
        self._build_feature_index()

        self.split_by_image = split_by_image
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None

        if split_by_image:
            self._split_by_image()

        total_trials = len(self.index_df)
        total_images = self.index_df["nsdId"].nunique() if "nsdId" in self.index_df.columns else "?"
        logger.info(
            "MultiSubjectPreextractedDataset: %d subjects, %d total trials, "
            "%s unique images",
            len(self.subjects), total_trials, total_images,
        )

    def _build_feature_index(self) -> None:
        """Build per-row pointers into the per-subject feature arrays.

        Uses the ``_local_feat_idx`` column stored at load time so that
        pointers remain correct even after rows have been filtered out
        (e.g. shared1000 exclusion, missing CLIP cache entries).
        """
        self._feat_subj = self.index_df["_subject_int"].values.astype(np.int32)
        self._feat_local_idx = self.index_df["_local_feat_idx"].values.astype(np.int64)

    def _split_by_image(self) -> None:
        """Image-level train/val split (no stimulus leakage).

        Uses sorted nsdIds + ``np.random.default_rng`` (PCG64) to match the
        single-subject split path in ``train_unified.py``, ensuring identical
        val sets across cross-subject (V29a) and fine-tune (V29b) phases.
        """
        rng = np.random.default_rng(self.seed)
        unique_nsd = np.sort(self.index_df["nsdId"].unique())
        rng.shuffle(unique_nsd)

        n_val = max(1, int(len(unique_nsd) * self.val_ratio))
        val_images = set(unique_nsd[:n_val])
        train_images = set(unique_nsd[n_val:])

        self.train_indices = np.where(
            self.index_df["nsdId"].isin(train_images)
        )[0]
        self.val_indices = np.where(
            self.index_df["nsdId"].isin(val_images)
        )[0]

        logger.info(
            "Image-level split: %d unique -> %d train (%d trials), %d val (%d trials)",
            len(unique_nsd),
            len(train_images), len(self.train_indices),
            len(val_images), len(self.val_indices),
        )

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int):
        """Returns (fmri, clip_embedding, subject_int) or
        (fmri, clip_embedding, subject_int, hier_targets_dict) when
        hierarchical CLIP columns are available.

        When ``token_cache`` is provided, ``clip_embedding`` is a flat
        (num_tokens * token_dim,) vector from the HDF5 token cache
        instead of the 768-D CLS embedding from ``embeddings_df``.
        """
        subj_int = int(self._feat_subj[idx])
        local_idx = int(self._feat_local_idx[idx])

        fmri = self.features_list[subj_int][local_idx]

        nsd_id = int(self.index_df.iloc[idx]["nsdId"])

        # Token-level targets (197376-D) override CLS embedding (768-D)
        if self.token_cache is not None:
            embedding = self.token_cache.get_flat(nsd_id)  # (num_tokens * token_dim,)
            fmri_t = torch.from_numpy(np.asarray(fmri, dtype=np.float32))
            emb_t = torch.from_numpy(embedding)  # already float32
            return fmri_t, emb_t, subj_int

        emb_idx = self.embedding_lookup.get(nsd_id)
        if emb_idx is None:
            raise KeyError(f"nsdId={nsd_id} not in CLIP cache")
        row = self.embeddings_df.iloc[emb_idx]
        embedding = row[self._emb_col]

        fmri_t = torch.from_numpy(np.asarray(fmri, dtype=np.float32))
        emb_t = torch.from_numpy(np.asarray(embedding, dtype=np.float32))

        if self._hier_cols:
            hier = {}
            for col in self._hier_cols:
                hier[col] = torch.from_numpy(
                    np.asarray(row[col], dtype=np.float32)
                )
            return fmri_t, emb_t, subj_int, hier

        return fmri_t, emb_t, subj_int

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)


def _resolve_emb_col(df: pd.DataFrame, override: Optional[str]) -> str:
    """Resolve the embedding column name."""
    _PRIORITY = ["fused", "final", "embedding", "clip_embedding", "clip512"]
    if override and override in df.columns:
        return override
    for col in _PRIORITY:
        if col in df.columns:
            return col
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if emb_cols:
        return emb_cols[0]
    raise ValueError(f"No embedding column found. Available: {list(df.columns)}")
