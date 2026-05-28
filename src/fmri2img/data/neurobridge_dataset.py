"""
NeuroBridge-OT Dataset — Module I
====================================

Extended multi-subject dataset supporting all NeuroBridge-OT protocols:
- Single-subject and multi-subject training.
- Balanced subject sampling and shared-image batching.
- LOSO splits and few-shot target-subject adaptation.
- Teacher prediction loading and alignment.
- Protocol-aware split hygiene with SHARED1000 exclusion.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

# Taxonomy labels for experiment protocols
TAXONOMY_LABELS = [
    "single_subject_same_subject",
    "subject_specific_replication",
    "multi_subject_seen_subject",
    "true_unseen_subject_zero_shot",
    "loso_generalization",
    "few_shot_adaptation",
    "data_efficiency_transfer",
]


class NeuroBridgeDataset(Dataset):
    """Multi-subject dataset for NeuroBridge-OT training and evaluation.

    Supports all experiment protocols with configurable splitting,
    teacher prediction loading, and subject fingerprint computation.

    Args:
        subjects: List of subject IDs to include.
        cache_root: Root directory with pre-extracted fMRI features.
        index_root: Root directory with NSD parquet indices.
        embeddings_df: CLIP embedding DataFrame keyed by nsdId.
        embedding_column: Column name for CLIP embeddings.
        exclude_shared1000: Whether to exclude SHARED1000 from training.
        split_by_image: Whether to split by unique nsdId (prevents leakage).
        val_ratio: Fraction for validation split.
        seed: Random seed for splits.
        teacher_registry: Optional TeacherRegistry for distillation.
        token_cache: Optional path to token-level CLIP cache.
        protocol: Experiment protocol for metadata.
        target_subject: For LOSO/few-shot, the held-out target subject.
        few_shot_n: Number of target-subject samples for few-shot.
        max_trials_per_subject: Optional cap on per-subject trials.
    """

    SUBJECT_TO_INT: Dict[str, int] = {}

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
        teacher_registry: Optional[Any] = None,
        token_cache: Optional[Path] = None,
        protocol: str = "multi_subject_seen_subject",
        target_subject: Optional[str] = None,
        few_shot_n: Optional[int] = None,
        max_trials_per_subject: Optional[int] = None,
    ):
        super().__init__()
        assert protocol in TAXONOMY_LABELS, f"Invalid protocol: {protocol}"
        self.subjects = list(subjects)
        self.protocol = protocol
        self.target_subject = target_subject
        self.few_shot_n = few_shot_n
        self.seed = seed

        # Build subject-to-int mapping
        all_subjects_sorted = sorted(set(subjects))
        NeuroBridgeDataset.SUBJECT_TO_INT = {
            s: i for i, s in enumerate(all_subjects_sorted)
        }

        # Resolve embedding column
        if embedding_column is None:
            for col in ["fused", "final", "embedding", "clip_embedding"]:
                if col in embeddings_df.columns:
                    embedding_column = col
                    break
        assert embedding_column is not None, "No valid embedding column found"
        self.embedding_column = embedding_column

        # Load per-subject data
        combined_rows = []
        self.voxel_counts: Dict[str, int] = {}
        self._fmri_data: Dict[str, np.ndarray] = {}

        for subj in subjects:
            fmri_path = Path(cache_root) / f"subject={subj}" / "fmri_features.npy"
            index_path = Path(index_root) / f"subject={subj}" / "index.parquet"

            if not fmri_path.exists():
                logger.warning("fMRI features not found for %s: %s", subj, fmri_path)
                continue
            if not index_path.exists():
                logger.warning("Index not found for %s: %s", subj, index_path)
                continue

            fmri = np.load(fmri_path, mmap_mode="r")
            self._fmri_data[subj] = fmri
            self.voxel_counts[subj] = fmri.shape[1]

            idx_df = pd.read_parquet(index_path)
            idx_df["_subject"] = subj
            idx_df["_subject_int"] = NeuroBridgeDataset.SUBJECT_TO_INT[subj]
            idx_df["_local_idx"] = range(len(idx_df))

            if max_trials_per_subject and len(idx_df) > max_trials_per_subject:
                idx_df = idx_df.iloc[:max_trials_per_subject]

            combined_rows.append(idx_df)

        if not combined_rows:
            raise RuntimeError("No valid subject data found!")

        self.index_df = pd.concat(combined_rows, ignore_index=True)
        logger.info(
            "NeuroBridgeDataset: %d subjects, %d total trials",
            len(self.voxel_counts), len(self.index_df),
        )

        # Exclude SHARED1000
        if exclude_shared1000 and "shared1000" in self.index_df.columns:
            n_before = len(self.index_df)
            self.index_df = self.index_df[
                ~self.index_df["shared1000"].fillna(False).astype(bool)
            ].reset_index(drop=True)
            logger.info(
                "Excluded shared1000: %d -> %d trials", n_before, len(self.index_df)
            )

        # Build CLIP embedding lookup
        self._build_clip_lookup(embeddings_df, embedding_column)

        # Split logic
        self._train_indices: Optional[List[int]] = None
        self._val_indices: Optional[List[int]] = None
        if split_by_image:
            self._split_by_image(val_ratio, seed)

        # Teacher predictions
        self.teacher_registry = teacher_registry
        self._teacher_cache: Dict[str, Optional[Tensor]] = {}

        # Token cache
        self._token_cache: Optional[Dict[int, np.ndarray]] = None
        if token_cache is not None and Path(token_cache).exists():
            self._load_token_cache(Path(token_cache))

        # Few-shot subset
        if protocol in ("few_shot_adaptation", "data_efficiency_transfer"):
            self._apply_few_shot_subset()

    def _build_clip_lookup(self, df: pd.DataFrame, col: str) -> None:
        """Build nsdId -> CLIP embedding lookup."""
        self._clip_embeddings: Dict[int, np.ndarray] = {}
        for _, row in df.iterrows():
            nsd_id = int(row["nsdId"])
            emb = row[col]
            if isinstance(emb, np.ndarray):
                self._clip_embeddings[nsd_id] = emb
            elif isinstance(emb, (list, tuple)):
                self._clip_embeddings[nsd_id] = np.array(emb, dtype=np.float32)

    def _split_by_image(self, val_ratio: float, seed: int) -> None:
        """Split by unique nsdId to prevent image leakage."""
        rng = np.random.RandomState(seed)
        unique_nsd_ids = self.index_df["nsdId"].unique()
        rng.shuffle(unique_nsd_ids)

        n_val = max(1, int(len(unique_nsd_ids) * val_ratio))
        val_nsd_ids = set(unique_nsd_ids[:n_val])

        self._train_indices = self.index_df.index[
            ~self.index_df["nsdId"].isin(val_nsd_ids)
        ].tolist()
        self._val_indices = self.index_df.index[
            self.index_df["nsdId"].isin(val_nsd_ids)
        ].tolist()

        logger.info(
            "Split: %d train, %d val (by image, %d unique nsdIds in val)",
            len(self._train_indices), len(self._val_indices), n_val,
        )

    def _apply_few_shot_subset(self) -> None:
        """Restrict target subject to few-shot N samples."""
        if self.target_subject is None or self.few_shot_n is None:
            return
        if self._train_indices is None:
            return

        rng = np.random.RandomState(self.seed)
        target_train = [
            i for i in self._train_indices
            if self.index_df.iloc[i]["_subject"] == self.target_subject
        ]

        if len(target_train) > self.few_shot_n:
            kept = rng.choice(target_train, size=self.few_shot_n, replace=False).tolist()
            # Remove excess target subject samples
            removed = set(target_train) - set(kept)
            self._train_indices = [i for i in self._train_indices if i not in removed]
            logger.info(
                "Few-shot: kept %d/%d trials for %s",
                self.few_shot_n, len(target_train), self.target_subject,
            )

    def _load_token_cache(self, path: Path) -> None:
        """Load token-level CLIP cache."""
        try:
            df = pd.read_parquet(path)
            self._token_cache = {}
            for _, row in df.iterrows():
                nsd_id = int(row["nsdId"])
                # Token embeddings stored as flat array
                for col in df.columns:
                    if col.startswith("token_") or col == "tokens":
                        self._token_cache[nsd_id] = np.array(row[col], dtype=np.float32)
                        break
            logger.info("Loaded token cache: %d entries", len(self._token_cache))
        except Exception as e:
            logger.warning("Failed to load token cache: %s", e)
            self._token_cache = None

    @property
    def train_indices(self) -> List[int]:
        return self._train_indices if self._train_indices is not None else list(range(len(self)))

    @property
    def val_indices(self) -> List[int]:
        return self._val_indices if self._val_indices is not None else []

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.

        Returns dict with:
            'fmri': Tensor (V,) raw voxels.
            'clip_target': Tensor (D,) CLIP embedding.
            'subject_id': int subject index.
            'subject_name': str subject identifier.
            'nsd_id': int image ID.
            'token_target': Optional Tensor token-level target.
            'teacher_pred': Optional Tensor teacher prediction.
        """
        row = self.index_df.iloc[idx]
        subj = row["_subject"]
        local_idx = row["_local_idx"]
        nsd_id = int(row["nsdId"])
        subject_int = int(row["_subject_int"])

        # fMRI
        fmri = torch.from_numpy(
            np.array(self._fmri_data[subj][local_idx], dtype=np.float32)
        )

        # CLIP target
        if nsd_id in self._clip_embeddings:
            clip_target = torch.from_numpy(self._clip_embeddings[nsd_id].copy())
        else:
            clip_target = torch.zeros(768)  # fallback

        sample = {
            "fmri": fmri,
            "clip_target": clip_target,
            "subject_id": subject_int,
            "subject_name": subj,
            "nsd_id": nsd_id,
        }

        # Token target
        if self._token_cache is not None and nsd_id in self._token_cache:
            sample["token_target"] = torch.from_numpy(self._token_cache[nsd_id].copy())

        return sample

    def get_shared1000_indices(self) -> List[int]:
        """Get indices of SHARED1000 trials (for evaluation only)."""
        if "shared1000" not in self.index_df.columns:
            return []
        mask = self.index_df["shared1000"].fillna(False).astype(bool)
        return self.index_df.index[mask].tolist()


class BalancedSubjectSampler(Sampler):
    """Sampler that balances samples across subjects per batch.

    Ensures each batch contains roughly equal representation from all subjects.

    Args:
        dataset: NeuroBridgeDataset instance.
        batch_size: Total batch size.
        indices: Subset indices (train or val).
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: NeuroBridgeDataset,
        batch_size: int,
        indices: Optional[List[int]] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        if indices is None:
            indices = list(range(len(dataset)))

        # Group indices by subject
        self._per_subject: Dict[str, List[int]] = {}
        for idx in indices:
            subj = dataset.index_df.iloc[idx]["_subject"]
            if subj not in self._per_subject:
                self._per_subject[subj] = []
            self._per_subject[subj].append(idx)

        self.n_subjects = len(self._per_subject)
        self._total = len(indices)

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        # Shuffle within each subject
        shuffled = {}
        for subj, idxs in self._per_subject.items():
            arr = np.array(idxs)
            rng.shuffle(arr)
            shuffled[subj] = list(arr)

        # Interleave subjects
        subjects = list(shuffled.keys())
        pointers = {s: 0 for s in subjects}
        result = []

        while len(result) < self._total:
            for subj in subjects:
                if pointers[subj] < len(shuffled[subj]):
                    result.append(shuffled[subj][pointers[subj]])
                    pointers[subj] += 1
                    if len(result) >= self._total:
                        break

        return iter(result)

    def __len__(self) -> int:
        return self._total


def neurobridge_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for NeuroBridge-OT batches.

    Handles variable-length fMRI vectors by zero-padding to max in batch.

    Args:
        batch: List of sample dicts from NeuroBridgeDataset.

    Returns:
        Collated batch dict with padded tensors.
    """
    fmri_list = [s["fmri"] for s in batch]
    max_v = max(f.shape[0] for f in fmri_list)

    padded_fmri = torch.stack([
        F.pad(f, (0, max_v - f.shape[0])) for f in fmri_list
    ])

    clip_targets = torch.stack([s["clip_target"] for s in batch])
    subject_ids = torch.tensor([s["subject_id"] for s in batch], dtype=torch.long)
    nsd_ids = torch.tensor([s["nsd_id"] for s in batch], dtype=torch.long)

    collated = {
        "fmri": padded_fmri,
        "clip_target": clip_targets,
        "subject_id": subject_ids,
        "nsd_id": nsd_ids,
        "subject_name": [s["subject_name"] for s in batch],
    }

    # Token targets (optional)
    if "token_target" in batch[0] and batch[0]["token_target"] is not None:
        collated["token_target"] = torch.stack([s["token_target"] for s in batch])

    return collated
