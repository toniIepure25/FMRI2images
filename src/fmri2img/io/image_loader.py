"""
Robust Image Loading with Fallback Chain

Load order:
1. Local HDF5 ($NSD_HDF5 or cache/nsd_hdf5/nsd_stimuli.hdf5)
2. S3 HDF5 (s3://natural-scenes-dataset/.../nsd_stimuli.hdf5)
3. COCO HTTP with local cache (.cache/coco/{cocoId}.jpg)

Features:
- Environment variable support for local HDF5
- Automatic caching of COCO images
- Single-warning-per-error pattern (no spam)
- Graceful degradation on partial/truncated files
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from io import BytesIO
import hashlib

import numpy as np
from PIL import Image
import pandas as pd

# Try h5py import
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False

# Try requests import
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

from fmri2img.io.s3 import HDF5Loader
from fmri2img.io.nsd_layout import NSDLayout

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    """Raised when all image loading methods fail"""
    pass


class RobustImageLoader:
    """
    Robust image loader with fallback chain and caching.
    
    Features:
    - Tries local HDF5 first (fastest)
    - Falls back to S3 HDF5 (moderate)
    - Falls back to COCO HTTP (slowest but reliable)
    - Caches COCO images locally
    - Single warning per error type
    """
    
    def __init__(
        self,
        local_hdf5_path: Optional[str] = None,
        s3_hdf5_path: Optional[str] = None,
        coco_cache_dir: str = ".cache/coco",
        enable_warnings: bool = True
    ):
        """
        Initialize robust image loader.
        
        Args:
            local_hdf5_path: Path to local HDF5 file (or None to check $NSD_HDF5)
            s3_hdf5_path: S3 path to HDF5 file
            coco_cache_dir: Directory for caching COCO images
            enable_warnings: Whether to print warnings
        """
        self.enable_warnings = enable_warnings
        self._warnings_shown = set()  # Track which warnings we've shown
        
        # Resolve local HDF5 path
        self.local_hdf5_path = self._resolve_local_hdf5(local_hdf5_path)
        self.s3_hdf5_path = s3_hdf5_path
        
        # Setup COCO caching
        self.coco_cache_dir = Path(coco_cache_dir)
        self.coco_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loaders
        self.hdf5_loader = HDF5Loader() if HAS_H5PY else None
        self.layout = NSDLayout()
        
        # Stats tracking
        self.stats = {
            'local_hdf5': 0,
            's3_hdf5': 0,
            'coco_cached': 0,
            'coco_http': 0,
            'failed': 0
        }
        
        # Log initial configuration
        if self.local_hdf5_path and self.local_hdf5_path.exists():
            logger.info(f"✓ Local HDF5 found: {self.local_hdf5_path}")
        else:
            self._warn_once('no_local_hdf5', 
                f"⚠️  No local HDF5 found. Set NSD_HDF5=cache/nsd_hdf5/nsd_stimuli.hdf5 for faster loading.")
    
    def _resolve_local_hdf5(self, path: Optional[str]) -> Optional[Path]:
        """Resolve local HDF5 path from argument or environment."""
        if path:
            p = Path(path)
            if p.exists():
                return p
        
        # Check environment variable
        env_path = os.getenv('NSD_HDF5')
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p
        
        # Check default location
        default_path = Path("cache/nsd_hdf5/nsd_stimuli.hdf5")
        if default_path.exists():
            return default_path
        
        return None
    
    def _warn_once(self, key: str, message: str):
        """Print warning only once per key."""
        if self.enable_warnings and key not in self._warnings_shown:
            logger.warning(message)
            self._warnings_shown.add(key)
    
    def _load_from_local_hdf5(self, nsd_id: int) -> Optional[Image.Image]:
        """Try loading from local HDF5 file."""
        if not self.local_hdf5_path or not HAS_H5PY:
            return None
        
        try:
            with h5py.File(self.local_hdf5_path, 'r') as hf:
                if "imgBrick" not in hf:
                    self._warn_once('no_imgbrick', "⚠️  'imgBrick' dataset not found in local HDF5")
                    return None
                
                img_arr = hf["imgBrick"][nsd_id]
                
                # Convert to PIL
                if img_arr.ndim == 2:
                    img = Image.fromarray(img_arr.astype(np.uint8), mode='L').convert('RGB')
                elif img_arr.ndim == 3:
                    img = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')
                else:
                    logger.debug(f"Unexpected shape for nsdId={nsd_id}: {img_arr.shape}")
                    return None
                
                self.stats['local_hdf5'] += 1
                logger.debug(f"✓ Loaded nsdId={nsd_id} from local HDF5")
                return img
                
        except OSError as e:
            # Truncated file error - log once and continue
            self._warn_once('local_hdf5_truncated', 
                f"⚠️  Local HDF5 corrupted/truncated (will use fallbacks): {e}")
            return None
        except Exception as e:
            logger.debug(f"Local HDF5 error for nsdId={nsd_id}: {e}")
            return None
    
    def _load_from_s3_hdf5(self, nsd_id: int) -> Optional[Image.Image]:
        """Try loading from S3 HDF5 file."""
        if not self.s3_hdf5_path or not self.hdf5_loader:
            return None
        
        try:
            with self.hdf5_loader.open(self.s3_hdf5_path) as hf:
                if "imgBrick" not in hf:
                    self._warn_once('no_imgbrick_s3', "⚠️  'imgBrick' dataset not found in S3 HDF5")
                    return None
                
                img_arr = hf["imgBrick"][nsd_id]
                
                # Convert to PIL
                if img_arr.ndim == 2:
                    img = Image.fromarray(img_arr.astype(np.uint8), mode='L').convert('RGB')
                elif img_arr.ndim == 3:
                    img = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')
                else:
                    logger.debug(f"Unexpected shape for nsdId={nsd_id}: {img_arr.shape}")
                    return None
                
                self.stats['s3_hdf5'] += 1
                logger.debug(f"✓ Loaded nsdId={nsd_id} from S3 HDF5")
                return img
                
        except OSError as e:
            # Truncated file error - log once and continue
            self._warn_once('s3_hdf5_truncated', 
                f"⚠️  S3 HDF5 corrupted/truncated (will use COCO fallback): {e}")
            return None
        except Exception as e:
            logger.debug(f"S3 HDF5 error for nsdId={nsd_id}: {e}")
            return None
    
    def _load_from_coco(self, coco_id: int, coco_split: str = "train2017") -> Optional[Image.Image]:
        """Try loading from COCO with local caching."""
        if not HAS_REQUESTS:
            return None
        
        # Check cache first
        cache_file = self.coco_cache_dir / f"{coco_id}_{coco_split}.jpg"
        if cache_file.exists():
            try:
                img = Image.open(cache_file).convert('RGB')
                self.stats['coco_cached'] += 1
                logger.debug(f"✓ Loaded cocoId={coco_id} from cache")
                return img
            except Exception as e:
                logger.debug(f"Cache read error for cocoId={coco_id}: {e}")
                # Continue to HTTP fetch
        
        # Fetch from HTTP
        try:
            url = self.layout.coco_http_url(coco_id, coco_split)
            logger.debug(f"Fetching COCO image from {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Cache for next time
            try:
                img.save(cache_file, 'JPEG', quality=95)
            except Exception as e:
                logger.debug(f"Failed to cache cocoId={coco_id}: {e}")
            
            self.stats['coco_http'] += 1
            logger.debug(f"✓ Loaded cocoId={coco_id} from COCO HTTP")
            return img
            
        except Exception as e:
            logger.debug(f"COCO HTTP error for cocoId={coco_id}: {e}")
            return None
    
    def load(self, row: pd.Series) -> Optional[Image.Image]:
        """
        Load image with full fallback chain.
        
        Args:
            row: DataFrame row with 'nsdId' (required) and optionally 'cocoId', 'cocoSplit'
        
        Returns:
            PIL Image or None if all methods fail
        """
        nsd_id = int(row.get("nsdId", row.get("nsd_id", -1)))
        if nsd_id < 0:
            logger.debug("No valid nsd_id in row")
            self.stats['failed'] += 1
            return None
        
        # Try 1: Local HDF5 (fastest)
        img = self._load_from_local_hdf5(nsd_id)
        if img is not None:
            return img
        
        # Try 2: S3 HDF5 (moderate)
        img = self._load_from_s3_hdf5(nsd_id)
        if img is not None:
            return img
        
        # Try 3: COCO HTTP with caching (slowest but reliable)
        if "cocoId" in row or "coco_id" in row:
            coco_id = int(row.get("cocoId", row.get("coco_id", -1)))
            coco_split = row.get("cocoSplit", row.get("coco_split", "train2017"))
            
            if coco_id >= 0:
                # Only warn once about fallback
                self._warn_once('using_coco_fallback',
                    f"⚠️  HDF5 methods failed for nsdId={nsd_id}, using COCO HTTP fallback")
                
                img = self._load_from_coco(coco_id, coco_split)
                if img is not None:
                    return img
        
        # All methods failed
        self.stats['failed'] += 1
        logger.debug(f"All load methods failed for nsdId={nsd_id}")
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get loading statistics."""
        return self.stats.copy()
