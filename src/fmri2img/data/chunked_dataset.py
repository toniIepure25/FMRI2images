"""
Chunked dataset loader for memory-efficient training.

Loads data in chunks (e.g., 5K samples) instead of all at once,
balancing memory usage and training speed.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
import logging
import gc
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChunkedMultiLayerDataset(Dataset):
    """
    Memory-efficient dataset that loads data in chunks.
    
    Strategy:
    - Divide dataset into chunks (e.g., 5K samples each)
    - Load one chunk at a time into memory
    - When batch indices span multiple chunks, load next chunk
    - Much faster than per-sample lazy loading
    - Uses ~2-4x less memory than full eager loading
    """
    
    def __init__(
        self,
        df,
        nifti_loader,
        preprocessor,
        multilayer_cache_path: str,
        text_clip_cache_path: Optional[str] = None,
        chunk_size: int = 5000,
        desc: str = "data"
    ):
        """
        Args:
            df: DataFrame with beta_path, beta_index, nsdId
            nifti_loader: NIfTI file loader
            preprocessor: Preprocessing pipeline
            multilayer_cache_path: Path to multilayer CLIP cache parquet
            text_clip_cache_path: Optional path to text-CLIP cache parquet
            chunk_size: Number of samples per chunk
            desc: Description for logging
        """
        import pandas as pd
        
        self.df = df.reset_index(drop=True)
        self.nifti_loader = nifti_loader
        self.preprocessor = preprocessor
        self.multilayer_cache_path = multilayer_cache_path
        self.text_clip_cache_path = text_clip_cache_path
        self.chunk_size = chunk_size
        self.desc = desc
        
        # We'll load cache on-demand per chunk (not upfront!)
        self.multilayer_cache = None
        self.text_clip_cache = None
        
        # Calculate chunks
        self.total_samples = len(self.df)
        self.n_chunks = (self.total_samples + chunk_size - 1) // chunk_size
        
        # Current chunk in memory
        self.current_chunk_idx = -1
        self.chunk_X = None
        self.chunk_Y_dict = None
        self.chunk_start_idx = 0
        self.chunk_end_idx = 0
        
        logger.info(f"ChunkedMultiLayerDataset ({desc}): {self.total_samples} samples, "
                   f"{self.n_chunks} chunks of {chunk_size}")
        logger.info(f"  Cache will be loaded on-demand (NOT pre-loaded)")
    
    def _extract_chunk_data(
        self, 
        chunk_df
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Extract fMRI features and multi-layer CLIP targets for chunk.
        DataFrame is pre-filtered, so all samples should have cache.
        """
        X_list = []
        Y_dict_lists = {'layer_4': [], 'layer_8': [], 'layer_12': [], 'final': []}
        
        if self.text_clip_cache is not None:
            Y_dict_lists['text'] = []
        
        nsd_ids_list = []
        
        # Group samples by beta_path to load each file only once
        samples_by_file = defaultdict(list)
        for idx, row in chunk_df.iterrows():
            nsd_id = int(row["nsdId"])
            beta_path = row["beta_path"]
            samples_by_file[beta_path].append({
                'beta_index': int(row["beta_index"]),
                'nsdId': nsd_id,
                'row_idx': idx
            })
        
        # Process each beta file once
        pbar = tqdm(samples_by_file.items(), desc=f"Loading {self.desc}", leave=False)
        for beta_path, samples in pbar:
            try:
                img = self.nifti_loader.load(beta_path)
                data_4d = img.get_fdata()
                
                for sample in samples:
                    try:
                        beta_index = sample['beta_index']
                        nsd_id = sample['nsdId']
                        
                        vol = data_4d[..., beta_index].astype(np.float32)
                        
                        if self.preprocessor and self.preprocessor.is_fitted_:
                            vol_z = self.preprocessor.transform_T0(vol)
                            features = self.preprocessor.transform(vol_z)
                        else:
                            features = vol.flatten()
                        
                        y_dict = self.multilayer_cache[nsd_id]
                        
                        if self.text_clip_cache is not None:
                            y_dict = {**y_dict, 'text': self.text_clip_cache[nsd_id]}
                        
                        X_list.append(features)
                        for layer_name in Y_dict_lists:
                            Y_dict_lists[layer_name].append(y_dict[layer_name])
                        nsd_ids_list.append(nsd_id)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract sample {nsd_id}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to load {beta_path}: {e}")
                continue
        
        # Convert to arrays
        X = np.vstack(X_list).astype(np.float32) if X_list else np.array([], dtype=np.float32)
        del X_list
        gc.collect()
        
        Y_dict = {}
        for k, v in Y_dict_lists.items():
            Y_dict[k] = np.vstack(v).astype(np.float32) if v else np.array([], dtype=np.float32)
        del Y_dict_lists
        gc.collect()
        
        nsd_ids = np.array(nsd_ids_list, dtype=np.int64)
        del nsd_ids_list
        gc.collect()
        
        return X, Y_dict, nsd_ids
    
    def _load_chunk(self, chunk_idx: int):
        """Load a specific chunk into memory, including its cache entries."""
        if chunk_idx == self.current_chunk_idx:
            return  # Already loaded
        
        # Clear previous chunk
        if self.chunk_X is not None:
            del self.chunk_X
            del self.chunk_Y_dict
            if self.multilayer_cache is not None:
                del self.multilayer_cache
                self.multilayer_cache = None
            if self.text_clip_cache is not None:
                del self.text_clip_cache
                self.text_clip_cache = None
            gc.collect()
        
        # Calculate chunk boundaries
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        logger.info(f"  Loading {self.desc} chunk {chunk_idx+1}/{self.n_chunks} "
                   f"(samples {start_idx}-{end_idx})...")
        
        # Get chunk DataFrame
        chunk_df = self.df.iloc[start_idx:end_idx]
        
        # Get NSD IDs for this chunk
        chunk_nsd_ids = set(chunk_df["nsdId"].astype(int).values)
        
        # Load ONLY the cache entries needed for this chunk
        logger.info(f"    Loading cache for {len(chunk_nsd_ids)} unique stimuli...")
        import pandas as pd
        
        # Load multilayer cache (only needed entries)
        ml_df = pd.read_parquet(self.multilayer_cache_path)
        ml_df_filtered = ml_df[ml_df['nsd_id'].isin(chunk_nsd_ids)]
        
        self.multilayer_cache = {}
        for _, row in ml_df_filtered.iterrows():
            nsd_id = int(row['nsd_id'])
            self.multilayer_cache[nsd_id] = {
                'layer_4': np.array(row['layer_4'], dtype=np.float32),
                'layer_8': np.array(row['layer_8'], dtype=np.float32),
                'layer_12': np.array(row['layer_12'], dtype=np.float32),
                'final': np.array(row['final'], dtype=np.float32)
            }
        
        logger.info(f"    Loaded {len(self.multilayer_cache)} multilayer cache entries")
        
        # Load text-CLIP cache if needed (only needed entries)
        if self.text_clip_cache_path:
            text_df = pd.read_parquet(self.text_clip_cache_path)
            nsd_col = 'nsd_id' if 'nsd_id' in text_df.columns else 'nsdId'
            text_df_filtered = text_df[text_df[nsd_col].isin(chunk_nsd_ids)]
            
            self.text_clip_cache = {}
            for _, row in text_df_filtered.iterrows():
                nsd_id = int(row[nsd_col])
                self.text_clip_cache[nsd_id] = np.array(row['text_clip_embedding'], dtype=np.float32)
            
            logger.info(f"    Loaded {len(self.text_clip_cache)} text-CLIP cache entries")
        
        # Load chunk data using inlined extraction
        self.chunk_X, self.chunk_Y_dict, _ = self._extract_chunk_data(chunk_df)
        
        self.current_chunk_idx = chunk_idx
        self.chunk_start_idx = start_idx
        self.chunk_end_idx = end_idx
        
        # Clear cache after extraction to save memory
        del self.multilayer_cache
        self.multilayer_cache = None
        if self.text_clip_cache is not None:
            del self.text_clip_cache
            self.text_clip_cache = None
        
        gc.collect()
        
        logger.info(f"    Chunk loaded: {len(self.chunk_X)} samples extracted")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """Get sample, loading chunk if necessary."""
        # Determine which chunk this index belongs to
        chunk_idx = idx // self.chunk_size
        
        # Load chunk if needed
        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)
        
        # Get sample from current chunk
        chunk_local_idx = idx - self.chunk_start_idx
        
        # Convert to tensors
        x_tensor = torch.from_numpy(self.chunk_X[chunk_local_idx]).float()
        y_tensors = {k: torch.from_numpy(v[chunk_local_idx]).float() 
                    for k, v in self.chunk_Y_dict.items()}
        
        return x_tensor, y_tensors
