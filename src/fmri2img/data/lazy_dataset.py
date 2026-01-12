"""
Lazy-loading dataset for NSD multi-layer CLIP training.

Loads fMRI and CLIP data on-demand from disk to minimize memory usage.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LazyMultiLayerDataset(Dataset):
    """
    Memory-efficient dataset that loads data on-demand.
    
    Instead of loading all data into RAM, this dataset:
    1. Stores only metadata (file paths, indices)
    2. Loads individual samples on __getitem__
    3. Caches recently accessed files
    
    This allows training on large datasets (30K+ samples) with limited RAM.
    """
    
    def __init__(
        self,
        df,
        nifti_loader,
        preprocessor,
        multilayer_cache: Dict,
        text_clip_cache: Optional[Dict] = None,
        file_cache_size: int = 5
    ):
        """
        Args:
            df: DataFrame with beta_path, beta_index, nsdId
            nifti_loader: NIfTI file loader
            preprocessor: Preprocessing pipeline
            multilayer_cache: Multi-layer CLIP embeddings
            text_clip_cache: Optional text-CLIP embeddings
            file_cache_size: Number of NIfTI files to keep in memory
        """
        self.nifti_loader = nifti_loader
        self.preprocessor = preprocessor
        self.multilayer_cache = multilayer_cache
        self.text_clip_cache = text_clip_cache
        
        # Store sample metadata (lightweight)
        self.samples = []
        for idx, row in df.iterrows():
            nsd_id = int(row["nsdId"])
            
            # Skip if no CLIP embeddings
            if nsd_id not in multilayer_cache:
                continue
            if text_clip_cache is not None and nsd_id not in text_clip_cache:
                continue
            
            self.samples.append({
                'beta_path': row["beta_path"],
                'beta_index': int(row["beta_index"]),
                'nsd_id': nsd_id
            })
        
        # Simple file cache (LRU-like)
        self.file_cache = {}
        self.file_cache_size = file_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"LazyMultiLayerDataset initialized: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_nifti_file(self, beta_path: str):
        """Load NIfTI file with simple caching."""
        if beta_path in self.file_cache:
            self.cache_hits += 1
            return self.file_cache[beta_path]
        
        self.cache_misses += 1
        
        # Load file
        img = self.nifti_loader.load(beta_path)
        data_4d = img.get_fdata()
        
        # Update cache (simple FIFO)
        if len(self.file_cache) >= self.file_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.file_cache))
            del self.file_cache[oldest_key]
        
        self.file_cache[beta_path] = data_4d
        return data_4d
    
    def __getitem__(self, idx):
        """Load single sample on-demand."""
        sample = self.samples[idx]
        
        try:
            # Load NIfTI volume
            data_4d = self._load_nifti_file(sample['beta_path'])
            vol = data_4d[..., sample['beta_index']].astype(np.float32)
            
            # Apply preprocessing
            if self.preprocessor and self.preprocessor.is_fitted_:
                vol_z = self.preprocessor.transform_T0(vol)
                features = self.preprocessor.transform(vol_z)
            else:
                features = vol.flatten()
            
            # Get CLIP targets
            nsd_id = sample['nsd_id']
            y_dict = self.multilayer_cache[nsd_id]
            
            # Add text-CLIP if available
            if self.text_clip_cache is not None:
                y_dict = {**y_dict, 'text': self.text_clip_cache[nsd_id]}
            
            # Convert to tensors
            x_tensor = torch.from_numpy(features).float()
            y_tensors = {k: torch.from_numpy(v).float() for k, v in y_dict.items()}
            
            return x_tensor, y_tensors
            
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            # Return dummy data to avoid breaking training
            dummy_x = torch.zeros(features.shape[0] if 'features' in locals() else 512)
            dummy_y = {
                'layer_4': torch.zeros(768),
                'layer_8': torch.zeros(768),
                'layer_12': torch.zeros(768),
                'final': torch.zeros(512)
            }
            if self.text_clip_cache is not None:
                dummy_y['text'] = torch.zeros(512)
            return dummy_x, dummy_y
    
    def get_cache_stats(self):
        """Return cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.file_cache)
        }
