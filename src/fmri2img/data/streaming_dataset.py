"""
Streaming dataset for memory-efficient multi-layer training.

Loads fMRI and CLIP data on-demand during training instead of all at once.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import logging
from tqdm import tqdm
from threading import Lock

logger = logging.getLogger(__name__)


class StreamingMultiLayerDataset(Dataset):
    """
    Memory-efficient streaming dataset for multi-layer CLIP supervision.
    
    Instead of loading all 30K samples into RAM (~15-20 GB), this dataset:
    1. Stores only metadata (DataFrame with file paths and indices)
    2. Loads fMRI volumes on-demand from disk during __getitem__
    3. Loads CLIP embeddings on-demand from cache files
    
    Memory usage: ~100 MB (metadata only) vs ~15-20 GB (eager loading)
    Trade-off: Slightly slower per batch due to disk I/O
    """
    
    def __init__(
        self,
        df,
        nifti_loader,
        preprocessor,
        multilayer_cache_path: str,
        text_clip_cache_path: Optional[str] = None,
        desc: str = "data"
    ):
        """
        Args:
            df: DataFrame with beta_path, beta_index, nsdId
            nifti_loader: NIfTI file loader
            preprocessor: Preprocessing pipeline
            multilayer_cache_path: Path to multilayer CLIP cache parquet
            text_clip_cache_path: Optional path to text-CLIP cache parquet
            desc: Description for logging
        """
        import pandas as pd
        
        self.df = df.reset_index(drop=True)
        self.nifti_loader = nifti_loader
        self.preprocessor = preprocessor
        self.multilayer_cache_path = multilayer_cache_path
        self.text_clip_cache_path = text_clip_cache_path
        self.desc = desc
        
        # Load cache files once (small metadata)
        logger.info(f"StreamingMultiLayerDataset ({desc}): Loading cache metadata...")
        self.multilayer_cache = self._load_multilayer_cache()
        self.text_clip_cache = self._load_text_cache() if text_clip_cache_path else None
        
        # Filter DataFrame to only include samples with available cache
        logger.info(f"  Filtering {desc} for cache availability...")
        valid_indices = []
        for idx, row in df.iterrows():
            nsd_id = int(row["nsdId"])
            if nsd_id not in self.multilayer_cache:
                continue
            if self.text_clip_cache is not None and nsd_id not in self.text_clip_cache:
                continue
            valid_indices.append(idx)
        
        self.df = df.loc[valid_indices].reset_index(drop=True)
        logger.info(f"  {desc}: {len(valid_indices)}/{len(df)} samples have cache available")
        
        # Cache for loaded NIfTI files (LRU-like, keep more files in memory)
        self.nifti_cache = {}
        self.nifti_cache_order = []
        self.max_nifti_cache = 30  # Cache ~30 files = ~4-5 GB
        self.cache_lock = Lock()  # Thread safety for multi-worker DataLoader
        
    def _load_multilayer_cache(self) -> Dict:
        """Load multilayer CLIP cache into memory (small)."""
        import pandas as pd
        
        df = pd.read_parquet(self.multilayer_cache_path)
        
        # Handle both nsd_id and nsdId column names (backward compatibility)
        nsd_col = 'nsd_id' if 'nsd_id' in df.columns else 'nsdId'
        
        cache = {}
        for _, row in df.iterrows():
            nsd_id = int(row[nsd_col])
            cache[nsd_id] = {
                'layer_4': np.array(row['layer_4'], dtype=np.float32),
                'layer_8': np.array(row['layer_8'], dtype=np.float32),
                'layer_12': np.array(row['layer_12'], dtype=np.float32),
                'final': np.array(row['final'], dtype=np.float32)
            }
        logger.info(f"    Loaded {len(cache)} multilayer embeddings")
        return cache
    
    def _load_text_cache(self) -> Dict:
        """Load text-CLIP cache into memory (small)."""
        import pandas as pd
        
        df = pd.read_parquet(self.text_clip_cache_path)
        nsd_col = 'nsd_id' if 'nsd_id' in df.columns else 'nsdId'
        
        cache = {}
        for _, row in df.iterrows():
            nsd_id = int(row[nsd_col])
            cache[nsd_id] = np.array(row['text_clip_embedding'], dtype=np.float32)
        logger.info(f"    Loaded {len(cache)} text-CLIP embeddings")
        return cache
    
    def _load_nifti(self, beta_path: str):
        """Load NIfTI file with thread-safe caching."""
        with self.cache_lock:
            if beta_path in self.nifti_cache:
                return self.nifti_cache[beta_path]
            
            # Load new file
            img = self.nifti_loader.load(beta_path)
            data_4d = img.get_fdata()
            
            # Cache management (LRU-like)
            if len(self.nifti_cache) >= self.max_nifti_cache:
                # Remove oldest
                oldest_path = self.nifti_cache_order.pop(0)
                del self.nifti_cache[oldest_path]
            
            self.nifti_cache[beta_path] = data_4d
            self.nifti_cache_order.append(beta_path)
            
            return data_4d
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Load single sample on-demand."""
        row = self.df.iloc[idx]
        
        beta_path = row["beta_path"]
        beta_index = int(row["beta_index"])
        nsd_id = int(row["nsdId"])
        
        # Load fMRI volume from NIfTI file
        data_4d = self._load_nifti(beta_path)
        vol = data_4d[..., beta_index].astype(np.float32)
        
        # Apply preprocessing
        if self.preprocessor and self.preprocessor.is_fitted_:
            vol_z = self.preprocessor.transform_T0(vol)
            features = self.preprocessor.transform(vol_z)
        else:
            features = vol.flatten()
        
        # Get CLIP embeddings from cache (already in memory)
        y_dict = self.multilayer_cache[nsd_id].copy()
        
        if self.text_clip_cache is not None:
            y_dict['text'] = self.text_clip_cache[nsd_id]
        
        # Convert to tensors
        x_tensor = torch.from_numpy(features).float()
        y_tensors = {k: torch.from_numpy(v).float() for k, v in y_dict.items()}
        
        return x_tensor, y_tensors
