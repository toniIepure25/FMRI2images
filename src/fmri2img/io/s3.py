"""
Robust S3 Data Loaders for Natural Scenes Dataset

This module provides memory-safe, cached loaders for NIfTI and HDF5 files
from S3 storage. Handles large files efficiently with proper error handling.

Key Features:
- Memory-safe streaming of large files
- Automatic caching with fsspec
- Proper error handling and retries
- Support for NIfTI and HDF5 formats
- Context managers for resource cleanup
"""

from __future__ import annotations
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, BinaryIO, Generator
import tempfile
import os

import fsspec
import numpy as np
import pandas as pd

# Optional imports with graceful fallbacks
try:
    import nibabel as nib
    from nibabel.filebasedimages import FileBasedImage
    HAS_NIBABEL = True
except ImportError:
    nib = None
    FileBasedImage = None
    HAS_NIBABEL = False
    warnings.warn("nibabel not available - NIfTI loading disabled")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False
    warnings.warn("h5py not available - HDF5 loading disabled")

logger = logging.getLogger(__name__)

class S3LoadError(Exception):
    """Raised when S3 loading fails"""
    pass

class S3FileSystem:
    """
    Wrapper around fsspec S3 filesystem with NSD-specific configurations.
    """
    
    def __init__(
        self, 
        anon: bool = True, 
        cache_storage: Optional[str] = None,
        cache_type: str = "simplecache"
    ):
        """
        Initialize S3 filesystem.
        
        Args:
            anon: Use anonymous access
            cache_storage: Local cache directory
            cache_type: Type of caching ('simplecache', 'blockcache', etc.)
        """
        self.anon = anon
        self.cache_storage = cache_storage or "cache/s3_cache"
        self.cache_type = cache_type
        
        # Ensure cache directory exists
        Path(self.cache_storage).mkdir(parents=True, exist_ok=True)
        
        self._fs = None
    
    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        """Lazy initialization of filesystem"""
        if self._fs is None:
            if self.cache_type and self.cache_storage:
                self._fs = fsspec.filesystem(
                    "simplecache",
                    target_protocol="s3",
                    cache_storage=self.cache_storage,
                    target_options={"anon": self.anon}
                )
            else:
                self._fs = fsspec.filesystem("s3", anon=self.anon)
        return self._fs
    
    def exists(self, path: str) -> bool:
        """Check if S3 path exists"""
        try:
            return self.fs.exists(path)
        except Exception as e:
            logger.warning(f"Error checking if {path} exists: {e}")
            return False
    
    def glob(self, pattern: str) -> List[str]:
        """Glob pattern matching on S3"""
        try:
            return self.fs.glob(pattern)
        except Exception as e:
            logger.error(f"Error globbing {pattern}: {e}")
            return []
    
    def info(self, path: str) -> Dict[str, Any]:
        """Get file info from S3"""
        try:
            return self.fs.info(path)
        except Exception as e:
            logger.error(f"Error getting info for {path}: {e}")
            raise S3LoadError(f"Cannot get info for {path}: {e}")
    
    @contextmanager
    def open(
        self, 
        path: str, 
        mode: str = 'rb',
        **kwargs
    ) -> Generator[BinaryIO, None, None]:
        """
        Context manager for opening S3 files.
        
        Args:
            path: S3 path or URL
            mode: File open mode
            **kwargs: Additional arguments for fsspec.open
            
        Yields:
            File-like object
        """
        try:
            with self.fs.open(path, mode, **kwargs) as f:
                yield f
        except Exception as e:
            logger.error(f"Error opening {path}: {e}")
            raise S3LoadError(f"Cannot open {path}: {e}")


# Global filesystem instance
_default_fs = None

def get_s3_filesystem(
    cache_storage: Optional[str] = None,
    reset: bool = False
) -> S3FileSystem:
    """
    Get default S3 filesystem instance.
    
    Args:
        cache_storage: Cache directory (if None, uses default)
        reset: Force creation of new filesystem
        
    Returns:
        S3FileSystem instance
    """
    global _default_fs
    if _default_fs is None or reset:
        _default_fs = S3FileSystem(cache_storage=cache_storage)
    return _default_fs


# Legacy function for compatibility
def s3_ls(url: str, anon: bool = True) -> List[str]:
    """List S3 objects matching URL pattern"""
    fs = fsspec.filesystem("s3", anon=anon)
    return [f"s3://{p}" for p in fs.glob(url)]


class NIfTILoader:
    """
    Memory-safe loader for NIfTI files from S3.
    """
    
    def __init__(self, s3_fs: Optional[S3FileSystem] = None):
        """
        Initialize NIfTI loader.
        
        Args:
            s3_fs: S3 filesystem instance (if None, uses default)
        """
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required for NIfTI loading")
        
        self.s3_fs = s3_fs or get_s3_filesystem()
    
    def load(
        self, 
        s3_path: str,
        mmap: bool = False,  # Changed default to False for S3
        validate: bool = True
    ) -> FileBasedImage:
        """
        Load NIfTI file from S3.
        
        Args:
            s3_path: S3 path to NIfTI file
            mmap: Use memory mapping (not recommended for S3)
            validate: Validate file format
            
        Returns:
            nibabel image object
            
        Raises:
            S3LoadError: If loading fails
        """
        logger.debug(f"Loading NIfTI from {s3_path}")
        
        try:
            # For S3 files, we need to download to a temporary file first
            # since nibabel needs a real file path for many operations
            with self.s3_fs.open(s3_path) as s3_file:
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                    # Copy S3 data to temp file
                    temp_file.write(s3_file.read())
                    temp_file.flush()
                    temp_path = temp_file.name
                
                try:
                    # Load with nibabel using the temp file path
                    img = nib.load(temp_path, mmap=mmap)
                    
                    if validate:
                        # Basic validation
                        if img.header is None:
                            raise ValueError("Invalid NIfTI header")
                        if img.get_fdata().size == 0:
                            raise ValueError("Empty NIfTI data")
                    
                    logger.debug(f"Loaded NIfTI shape: {img.shape}")
                    return img
                    
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                
        except Exception as e:
            logger.error(f"Failed to load NIfTI from {s3_path}: {e}")
            raise S3LoadError(f"Cannot load NIfTI from {s3_path}: {e}")
    
    def load_data(
        self, 
        s3_path: str,
        dtype: Optional[np.dtype] = None
    ) -> np.ndarray:
        """
        Load only the data array from NIfTI file.
        
        Args:
            s3_path: S3 path to NIfTI file
            dtype: Convert to specific dtype
            
        Returns:
            NumPy array with image data
        """
        img = self.load(s3_path)
        data = img.get_fdata()
        
        if dtype is not None:
            data = data.astype(dtype)
        
        return data
    
    def get_header(self, s3_path: str) -> Dict[str, Any]:
        """
        Get NIfTI header information without loading full data.
        
        Args:
            s3_path: S3 path to NIfTI file
            
        Returns:
            Dictionary with header information
        """
        img = self.load(s3_path)
        header = img.header
        
        return {
            'shape': img.shape,
            'dtype': img.get_data_dtype(),
            'affine': img.affine.tolist(),
            'voxel_size': header.get_zooms(),
            'units': header.get_xyzt_units()
        }


class HDF5Loader:
    """
    Memory-safe loader for HDF5 files from S3.
    """
    
    def __init__(self, s3_fs: Optional[S3FileSystem] = None):
        """
        Initialize HDF5 loader.
        
        Args:
            s3_fs: S3 filesystem instance (if None, uses default)
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 loading")
        
        self.s3_fs = s3_fs or get_s3_filesystem()
    
    @contextmanager
    def open(self, s3_path: str, mode: str = 'r') -> Generator[h5py.File, None, None]:
        """
        Context manager for opening HDF5 files from S3.
        
        Args:
            s3_path: S3 path to HDF5 file
            mode: File open mode
            
        Yields:
            h5py.File object
        """
        logger.debug(f"Opening HDF5 from {s3_path}")
        
        try:
            with self.s3_fs.open(s3_path, 'rb') as s3_file:
                # Create temporary file for h5py (which needs seekable file)
                with tempfile.NamedTemporaryFile() as temp_file:
                    # Copy S3 data to temp file
                    temp_file.write(s3_file.read())
                    temp_file.flush()
                    
                    # Open with h5py
                    with h5py.File(temp_file.name, mode) as hf:
                        yield hf
                        
        except Exception as e:
            logger.error(f"Failed to open HDF5 from {s3_path}: {e}")
            raise S3LoadError(f"Cannot open HDF5 from {s3_path}: {e}")
    
    def load_dataset(
        self, 
        s3_path: str, 
        dataset_name: str,
        slice_obj: Optional[Union[slice, tuple]] = None
    ) -> np.ndarray:
        """
        Load specific dataset from HDF5 file.
        
        Args:
            s3_path: S3 path to HDF5 file
            dataset_name: Name of dataset within HDF5
            slice_obj: Optional slice to load partial data
            
        Returns:
            NumPy array with dataset data
        """
        with self.open(s3_path) as hf:
            if dataset_name not in hf:
                raise KeyError(f"Dataset '{dataset_name}' not found in {s3_path}")
            
            dataset = hf[dataset_name]
            
            if slice_obj is not None:
                return dataset[slice_obj]
            else:
                return dataset[:]
    
    def list_datasets(self, s3_path: str) -> List[str]:
        """
        List all datasets in HDF5 file.
        
        Args:
            s3_path: S3 path to HDF5 file
            
        Returns:
            List of dataset names
        """
        datasets = []
        
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)
        
        with self.open(s3_path) as hf:
            hf.visititems(collect_datasets)
        
        return datasets
    
    def get_info(self, s3_path: str) -> Dict[str, Any]:
        """
        Get information about HDF5 file structure.
        
        Args:
            s3_path: S3 path to HDF5 file
            
        Returns:
            Dictionary with file information
        """
        info = {
            'datasets': {},
            'groups': [],
            'attributes': {}
        }
        
        with self.open(s3_path) as hf:
            # Get root attributes
            info['attributes'] = dict(hf.attrs)
            
            # Walk through file structure
            def collect_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info['datasets'][name] = {
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'size_mb': obj.size * obj.dtype.itemsize / (1024**2)
                    }
                elif isinstance(obj, h5py.Group):
                    info['groups'].append(name)
            
            hf.visititems(collect_info)
        
        return info


class CSVLoader:
    """
    Loader for CSV files from S3.
    """
    
    def __init__(self, s3_fs: Optional[S3FileSystem] = None):
        """
        Initialize CSV loader.
        
        Args:
            s3_fs: S3 filesystem instance (if None, uses default)
        """
        self.s3_fs = s3_fs or get_s3_filesystem()
    
    def load(
        self, 
        s3_path: str,
        **pandas_kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file from S3 into pandas DataFrame.
        
        Args:
            s3_path: S3 path to CSV file
            **pandas_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pandas DataFrame
        """
        logger.debug(f"Loading CSV from {s3_path}")
        
        try:
            with self.s3_fs.open(s3_path, 'r') as f:
                df = pd.read_csv(f, **pandas_kwargs)
                logger.debug(f"Loaded CSV shape: {df.shape}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to load CSV from {s3_path}: {e}")
            raise S3LoadError(f"Cannot load CSV from {s3_path}: {e}")


# Convenience functions for direct loading
def load_nifti(s3_path: str, **kwargs) -> FileBasedImage:
    """Convenience function to load NIfTI file"""
    loader = NIfTILoader()
    return loader.load(s3_path, **kwargs)

def load_hdf5_dataset(s3_path: str, dataset_name: str, **kwargs) -> np.ndarray:
    """Convenience function to load HDF5 dataset"""
    loader = HDF5Loader()
    return loader.load_dataset(s3_path, dataset_name, **kwargs)

def load_csv(s3_path: str, **kwargs) -> pd.DataFrame:
    """Convenience function to load CSV file"""
    loader = CSVLoader()
    return loader.load(s3_path, **kwargs)
