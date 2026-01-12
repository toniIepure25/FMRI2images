"""
Data Loading Utilities
======================

Standardized data loading infrastructure for fMRI experiments.

Features:
- DataLoaderFactory for consistent data loader creation
- Train/val/test split utilities
- fMRI feature extraction with preprocessing
- CLIP target loading and caching
- Batch collation with proper device handling

Usage:
    # Simple loading
    loaders = DataLoaderFactory.create_loaders(
        df,
        subject="subj01",
        batch_size=32,
        preprocessor=preprocessor,
        clip_cache=clip_cache
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # Custom split ratios
    loaders = DataLoaderFactory.create_loaders(
        df,
        subject="subj01",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem

logger = logging.getLogger(__name__)


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    stratify_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test with guaranteed minimum samples.
    
    Ensures reproducible splits across experiments for fair comparison.
    
    Args:
        df: Input DataFrame
        train_ratio: Training split ratio (default: 0.8)
        val_ratio: Validation split ratio (default: 0.1)
        test_ratio: Test split ratio (default: 0.1)
        random_seed: Random seed for reproducibility
        stratify_column: Optional column name for stratified splitting
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    
    Raises:
        ValueError: If insufficient samples or invalid ratios
    
    Example:
        >>> train_df, val_df, test_df = train_val_test_split(
        ...     df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
        ... )
        >>> print(len(train_df), len(val_df), len(test_df))
        700 200 100
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )
    
    # Shuffle with fixed seed
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    
    # Ensure minimum samples for each split
    if n_total < 3:
        raise ValueError(
            f"Need at least 3 samples for train/val/test split, got {n_total}"
        )
    
    # Compute split sizes
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val
    
    # Ensure at least 1 test sample
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_total - n_train - n_test)
        n_train = n_total - n_val - n_test
    
    # Split DataFrame
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train + n_val]
    test_df = df_shuffled[n_train + n_val:]
    
    logger.info(
        f"Split {n_total} samples: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    return train_df, val_df, test_df


def extract_features_and_targets(
    df: pd.DataFrame,
    nifti_loader: NIfTILoader,
    preprocessor: NSDPreprocessor,
    clip_cache: CLIPCache,
    split_name: str = "data",
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extract preprocessed fMRI features and CLIP targets from index DataFrame.
    
    Args:
        df: Index DataFrame with 'nsdId', 's3_nifti_path', and other columns
        nifti_loader: NIfTI loader for loading fMRI volumes
        preprocessor: Preprocessor for fMRI data (T0+T1+T2 transforms)
        clip_cache: CLIP cache for loading embeddings
        split_name: Name for logging (e.g., 'train', 'validation', 'test')
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (features, targets, nsd_ids):
        - features: NumPy array of shape (n_samples, n_features)
        - targets: NumPy array of shape (n_samples, embedding_dim)
        - nsd_ids: List of NSD IDs for each sample
    
    Example:
        >>> X_train, Y_train, train_ids = extract_features_and_targets(
        ...     train_df, nifti_loader, preprocessor, clip_cache, "train"
        ... )
        >>> print(X_train.shape, Y_train.shape)
        (800, 512) (800, 768)
    """
    X_list = []
    Y_list = []
    nsd_ids = []
    
    iterator = tqdm(df.itertuples(), total=len(df), desc=f"Loading {split_name}") if show_progress else df.itertuples()
    
    for row in iterator:
        nsd_id = row.nsdId
        s3_path = row.s3_nifti_path
        
        try:
            # Load and preprocess fMRI volume
            vol = nifti_loader.load_volume(s3_path)
            x = preprocessor.transform(vol)  # T0+T1+T2 pipeline
            
            # Load CLIP embedding
            y = clip_cache.get_embedding(nsd_id)
            
            X_list.append(x)
            Y_list.append(y)
            nsd_ids.append(nsd_id)
        
        except Exception as e:
            logger.warning(f"Failed to load sample {nsd_id}: {e}")
            continue
    
    # Convert to arrays
    X = np.vstack(X_list).astype(np.float32)
    Y = np.vstack(Y_list).astype(np.float32)
    
    logger.info(
        f"✅ Loaded {split_name}: "
        f"X.shape={X.shape}, Y.shape={Y.shape}"
    )
    
    return X, Y, nsd_ids


class FMRIDataset(Dataset):
    """
    PyTorch Dataset for fMRI → CLIP mapping.
    
    Features:
    - Lazy loading of fMRI volumes
    - On-the-fly preprocessing
    - CLIP embedding caching
    - Configurable transforms
    
    Args:
        df: Index DataFrame
        nifti_loader: NIfTI loader
        preprocessor: fMRI preprocessor
        clip_cache: CLIP cache
        transform: Optional transform function
    
    Example:
        >>> dataset = FMRIDataset(train_df, nifti_loader, preprocessor, clip_cache)
        >>> features, targets = dataset[0]
        >>> print(features.shape, targets.shape)
        torch.Size([512]) torch.Size([768])
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        nifti_loader: NIfTILoader,
        preprocessor: NSDPreprocessor,
        clip_cache: CLIPCache,
        transform: Optional[callable] = None
    ):
        self.df = df.reset_index(drop=True)
        self.nifti_loader = nifti_loader
        self.preprocessor = preprocessor
        self.clip_cache = clip_cache
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample.
        
        Returns:
            Tuple of (features, targets) as PyTorch tensors
        """
        row = self.df.iloc[idx]
        nsd_id = row['nsdId']
        s3_path = row['s3_nifti_path']
        
        # Load and preprocess fMRI
        vol = self.nifti_loader.load_volume(s3_path)
        features = self.preprocessor.transform(vol)
        features = torch.from_numpy(features).float()
        
        # Load CLIP embedding
        targets = self.clip_cache.get_embedding(nsd_id)
        targets = torch.from_numpy(targets).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            features, targets = self.transform(features, targets)
        
        return features, targets


class DataLoaderFactory:
    """
    Factory for creating standardized data loaders.
    
    Provides consistent interface for creating train/val/test loaders
    with proper configuration.
    
    Example:
        >>> loaders = DataLoaderFactory.create_loaders(
        ...     df,
        ...     subject="subj01",
        ...     batch_size=32,
        ...     preprocessor=preprocessor,
        ...     clip_cache=clip_cache
        ... )
        >>> for batch in loaders['train']:
        ...     features, targets = batch
        ...     print(features.shape)
    """
    
    @staticmethod
    def create_loaders(
        df: pd.DataFrame,
        subject: str,
        batch_size: int = 32,
        preprocessor: Optional[NSDPreprocessor] = None,
        clip_cache: Optional[CLIPCache] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
        preload: bool = True,
        show_progress: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create train/val/test data loaders.
        
        Args:
            df: Index DataFrame
            subject: Subject identifier
            batch_size: Batch size for training
            preprocessor: Optional preprocessor (if None, uses default)
            clip_cache: Optional CLIP cache (if None, loads default)
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            random_seed: Random seed for splits
            num_workers: Number of data loader workers
            pin_memory: Whether to pin memory for faster GPU transfer
            preload: Whether to preload all data into memory
            show_progress: Whether to show progress bars
        
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing DataLoaders
        """
        # Split data
        train_df, val_df, test_df = train_val_test_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )
        
        # Setup preprocessor if not provided
        if preprocessor is None:
            preprocessor = NSDPreprocessor(subject)
            if not preprocessor.load_artifacts():
                raise ValueError(
                    f"Preprocessor artifacts not found for {subject}. "
                    "Run preprocessing first."
                )
        
        # Setup CLIP cache if not provided
        if clip_cache is None:
            clip_cache = CLIPCache().load()
        
        # Setup NIfTI loader
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        # Create data loaders
        loaders = {}
        
        for split_name, split_df in [
            ('train', train_df),
            ('val', val_df),
            ('test', test_df)
        ]:
            if preload:
                # Preload all data into memory
                X, Y, _ = extract_features_and_targets(
                    split_df,
                    nifti_loader,
                    preprocessor,
                    clip_cache,
                    split_name=split_name,
                    show_progress=show_progress
                )
                
                # Convert to tensors
                X = torch.from_numpy(X).float()
                Y = torch.from_numpy(Y).float()
                
                # Create TensorDataset
                dataset = TensorDataset(X, Y)
            else:
                # Use lazy loading
                dataset = FMRIDataset(
                    split_df,
                    nifti_loader,
                    preprocessor,
                    clip_cache
                )
            
            # Create DataLoader
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),  # Only shuffle training
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            loaders[split_name] = loader
        
        logger.info(
            f"✅ Created data loaders: "
            f"train={len(loaders['train'].dataset)}, "
            f"val={len(loaders['val'].dataset)}, "
            f"test={len(loaders['test'].dataset)}"
        )
        
        return loaders
    
    @staticmethod
    def create_single_loader(
        df: pd.DataFrame,
        subject: str,
        batch_size: int = 32,
        preprocessor: Optional[NSDPreprocessor] = None,
        clip_cache: Optional[CLIPCache] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        preload: bool = True,
        show_progress: bool = True,
        split_name: str = "data"
    ) -> DataLoader:
        """
        Create a single data loader (no train/val/test split).
        
        Useful for inference or when split already performed.
        
        Args:
            df: Index DataFrame
            subject: Subject identifier
            batch_size: Batch size
            preprocessor: Optional preprocessor
            clip_cache: Optional CLIP cache
            shuffle: Whether to shuffle data
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            preload: Preload all data into memory
            show_progress: Show progress bars
            split_name: Name for logging
        
        Returns:
            DataLoader instance
        """
        # Setup preprocessor if not provided
        if preprocessor is None:
            preprocessor = NSDPreprocessor(subject)
            if not preprocessor.load_artifacts():
                raise ValueError(
                    f"Preprocessor artifacts not found for {subject}. "
                    "Run preprocessing first."
                )
        
        # Setup CLIP cache if not provided
        if clip_cache is None:
            clip_cache = CLIPCache().load()
        
        # Setup NIfTI loader
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        if preload:
            # Preload all data
            X, Y, _ = extract_features_and_targets(
                df,
                nifti_loader,
                preprocessor,
                clip_cache,
                split_name=split_name,
                show_progress=show_progress
            )
            
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float()
            dataset = TensorDataset(X, Y)
        else:
            # Lazy loading
            dataset = FMRIDataset(df, nifti_loader, preprocessor, clip_cache)
        
        # Create loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        logger.info(f"✅ Created data loader: {len(dataset)} samples")
        
        return loader
