from __future__ import annotations
import io, math, random
import numpy as np
import pandas as pd
from typing import Dict, Iterator, Optional, Sequence, Union
from torch.utils.data import IterableDataset, get_worker_info

from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem
from fmri2img.io.nsd_layout import NSDLayout

# Optional preprocessor import
try:
    from fmri2img.data.preprocess import NSDPreprocessor
except ImportError:
    NSDPreprocessor = None

# Optional CLIP cache import
try:
    from fmri2img.data.clip_cache import CLIPCache
except ImportError:
    CLIPCache = None

class NSDIterableDataset(IterableDataset):
    """
    Streams rows from the canonical NSD Parquet index and yields:
      { "fmri": (H,W,D) float32 or (k,) if PCA, "nsdId": int, "beta_path": str, "beta_index": int }
    Optionally yields 'image_meta' (COCO ids etc.) if present in index.
    
    If preprocessor is provided, applies T0/T1/T2 transformations:
    - T0: Online z-score normalization
    - T1: Subject-level scaler + reliability mask
    - T2: PCA dimensionality reduction
    
    If clip_cache is provided, adds 'clip' key with (512,) float32 embedding.
    """
    def __init__(
        self,
        index_path_or_root: str,
        subject: str = "subj01",
        session: Optional[int] = None,
        shuffle: bool = False,
        limit: Optional[int] = None,
        seed: int = 0,
        preprocessor: Optional["NSDPreprocessor"] = None,
        clip_cache: Union["CLIPCache", str, None] = None,
    ):
        super().__init__()
        self.index_path_or_root = index_path_or_root
        self.subject = subject
        self.session = session
        self.shuffle = shuffle
        self.limit = limit
        self.seed = seed
        self.preprocessor = preprocessor
        self._preproc_logged = False  # Track if we've logged preprocessing status
        
        # Handle clip_cache initialization
        if clip_cache is not None:
            if isinstance(clip_cache, str):
                # Path string provided - instantiate and load
                if CLIPCache is None:
                    raise ImportError("CLIPCache not available. Install required dependencies.")
                self.clip_cache = CLIPCache(clip_cache).load()
            else:
                # CLIPCache instance provided - ensure it's loaded
                self.clip_cache = clip_cache
                if not self.clip_cache.is_loaded:
                    self.clip_cache.load()
        else:
            self.clip_cache = None

        # Eagerly read the partition for simplicity (it is metadata-scale, not GBs)
        if index_path_or_root.endswith(".parquet") and "subject=" in index_path_or_root:
            df = pd.read_parquet(index_path_or_root)
        elif index_path_or_root.endswith(".parquet"):
            df = pd.read_parquet(index_path_or_root)
        else:
            df = pd.read_parquet(f"{index_path_or_root.rstrip('/')}/subject={subject}/index.parquet")

        df = df[df["subject"] == subject]
        if session is not None and "session" in df.columns:
            df = df[df["session"] == session]

        self.df = df.reset_index(drop=True)

    def __iter__(self) -> Iterator[Dict]:
        # Worker-aware sharding
        info = get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))

        indices = list(range(len(self.df)))
        if self.shuffle:
            rng.shuffle(indices)

        if self.limit is not None:
            indices = indices[: self.limit]

        # Simple round-robin sharding across workers
        if info and info.num_workers > 1:
            indices = indices[info.id :: info.num_workers]

        s3_fs = get_s3_filesystem()
        nifti = NIfTILoader(s3_fs)
        
        # Log preprocessing status once using summary() API (no sklearn access)
        logger = __import__('logging').getLogger(__name__)
        if self.preprocessor is not None and not self._preproc_logged:
            if self.preprocessor.is_fitted_:
                summary = self.preprocessor.summary()
                pca_status = ""
                if summary.get("pca_fitted", False):
                    k = summary.get("pca_components", 0)
                    var_explained = summary.get("explained_variance_ratio", 0.0)
                    pca_status = f", T2 PCA (k={k}, {var_explained:.1%} var)"
                logger.info(f"✅ Loaded preprocessing: T0 (z-score) + T1 (scaler+mask, {summary['n_voxels_kept']:,} voxels){pca_status}")
            else:
                logger.info("⚠️  Preprocessing artifacts not found; applying T0 z-score only")
            self._preproc_logged = True
        
        # Pre-fetch CLIP embeddings if cache is available (batch lookup for efficiency)
        clip_embeddings = {}
        clip_missing_logged = False
        if self.clip_cache is not None:
            nsd_ids_to_fetch = [int(self.df.iloc[i]["nsdId"]) for i in indices]
            clip_embeddings = self.clip_cache.get(nsd_ids_to_fetch)

        for i in indices:
            row = self.df.iloc[i]
            beta_path = row["beta_path"]
            beta_index = int(row["beta_index"])
            nsd_id = int(row["nsdId"])
            
            try:
                img = nifti.load(beta_path)  
                # Get the full 4D data and slice it - this ensures proper file handling
                data_4d = img.get_fdata()
                vol = data_4d[..., beta_index].astype("float32")  # load only 3D
                
                # Apply preprocessing if available
                if self.preprocessor is not None:
                    vol = self.preprocessor.transform(vol)
                
                # Build output dict
                out = {
                    "fmri": vol,
                    "nsdId": nsd_id,
                    "beta_path": beta_path,
                    "beta_index": beta_index,
                }
                
                # Add CLIP embedding if available
                if self.clip_cache is not None:
                    if nsd_id in clip_embeddings:
                        out["clip"] = clip_embeddings[nsd_id]
                    elif not clip_missing_logged:
                        logger = __import__('logging').getLogger(__name__)
                        logger.warning(f"CLIP embedding missing for nsdId={nsd_id} (further warnings suppressed)")
                        clip_missing_logged = True
                
                # Optionally expose mapping fields if present
                if "cocoId" in row:
                    out["cocoId"] = int(row["cocoId"])
                
                yield out
            except Exception as e:
                # Skip failed loads (this is a smoke test)
                logger = __import__('logging').getLogger(__name__)
                logger.warning(f"Skipping trial {i} due to load error: {e}")
                continue