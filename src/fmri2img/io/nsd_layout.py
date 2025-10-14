"""
NSD Dataset Path Layout Management

This module centralizes all path patterns and URL generation for the Natural Scenes Dataset.
It provides a clean interface for generating S3 URLs and handles path validation.

Key Features:
- Centralized path pattern management
- S3 URL generation with validation
- Support for different preprocessing pipelines
- COCO dataset fallback URLs
- Path existence checking
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass
import yaml
import pandas as pd
import fsspec
import io

logger = logging.getLogger(__name__)

# Type definitions
SubjectId = Union[int, str]
SessionId = Union[int, str] 
CocoSplit = Literal['train2017', 'val2017', 'test2017']
Resolution = Literal['func1mm', 'func1pt8mm', 'MNI', 'fsaverage', 'nativesurface']
PreprocessingPipeline = Literal[
    'betas_assumehrf', 
    'betas_fithrf', 
    'betas_fithrf_GLMdenoise_RR'
]

@dataclass
class NSDPaths:
    """Container for NSD dataset path patterns"""
    bucket: str
    base_url: str
    
    # Metadata paths
    stim_info: str
    experiment_design: str
    
    # Stimuli paths
    stimuli_hdf5: str
    
    # fMRI path patterns
    session_pattern: str
    single_trial_design: str
    roi_masks: str
    
    # Processing parameters
    default_resolution: Resolution
    default_preprocessing: PreprocessingPipeline


class NSDLayout:
    """
    Centralized path management for Natural Scenes Dataset.
    
    Provides methods to generate S3 URLs for all dataset components
    with proper validation and error handling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize with configuration file or use defaults.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default paths.
        """
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.paths = self._load_from_config(config)
        else:
            self.paths = self._get_default_paths()
            
        self._validate_paths()
    
    def _load_from_config(self, config: Dict) -> NSDPaths:
        """Load paths from configuration dictionary"""
        s3_config = config['s3']
        nsd_config = config['nsd']
        
        return NSDPaths(
            bucket=s3_config['bucket'],
            base_url=s3_config['base_url'],
            stim_info=nsd_config['metadata_files']['stim_info'],
            experiment_design=nsd_config['metadata_files']['experiment_design'],
            stimuli_hdf5=nsd_config['stimuli']['hdf5_file'],
            session_pattern=nsd_config['fmri']['session_pattern'],
            single_trial_design=nsd_config['fmri']['single_trial_design'],
            roi_masks=nsd_config['fmri']['roi_masks'],
            default_resolution=nsd_config['fmri']['resolution'],
            default_preprocessing=nsd_config['fmri']['preprocessing']
        )
    
    def _get_default_paths(self) -> NSDPaths:
        """Get default path configuration"""
        return NSDPaths(
            bucket="natural-scenes-dataset",
            base_url="s3://natural-scenes-dataset",
            stim_info="nsddata/experiments/nsd/nsd_stim_info_merged.csv",
            experiment_design="nsddata/experiments/nsd/nsd_expdesign.mat",
            stimuli_hdf5="nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
            session_pattern="nsddata_betas/ppdata/subj{subject:02d}/{resolution}/{preprocessing}/betas_session{session:02d}.nii.gz",
            single_trial_design="nsddata/ppdata/subj{subject:02d}/func/design_matrices_single_trial.hdf5",
            roi_masks="nsddata/ppdata/subj{subject:02d}/anat/*roi*.nii.gz",
            default_resolution="func1pt8mm",
            default_preprocessing="betas_fithrf_GLMdenoise_RR"
        )
    
    def _validate_paths(self):
        """Validate that all required path patterns are present"""
        required_attrs = [
            'bucket', 'base_url', 'stim_info', 'experiment_design', 
            'stimuli_hdf5', 'session_pattern', 'default_resolution', 
            'default_preprocessing'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.paths, attr) or getattr(self.paths, attr) is None:
                raise ValueError(f"Missing required path configuration: {attr}")
    
    def _normalize_subject_id(self, subject: SubjectId) -> int:
        """Normalize subject ID to integer"""
        if isinstance(subject, str):
            # Handle "subj01" format
            if subject.startswith("subj"):
                return int(subject[4:])
            # Handle string numbers
            return int(subject)
        return int(subject)
    
    def _normalize_session_id(self, session: SessionId) -> int:
        """Normalize session ID to integer"""
        if isinstance(session, str):
            # Handle "session01" format  
            if session.startswith("session"):
                return int(session[7:])
            # Handle string numbers
            return int(session)
        return int(session)
    
    # Core path generation methods
    
    def beta_path(
        self, 
        subject: SubjectId, 
        session: SessionId,
        resolution: Optional[Resolution] = None,
        preprocessing: Optional[PreprocessingPipeline] = None,
        full_url: bool = True
    ) -> str:
        """
        Generate S3 path for beta (fMRI) files.
        
        Args:
            subject: Subject ID (int or string)
            session: Session ID (int or string) 
            resolution: Spatial resolution ('func1mm' or 'func1pt8mm')
            preprocessing: Preprocessing pipeline
            full_url: If True, return full S3 URL; if False, return relative path
            
        Returns:
            S3 URL or relative path to beta file
            
        Example:
            >>> layout.beta_path(1, 1)
            's3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz'
        """
        subject_num = self._normalize_subject_id(subject)
        session_num = self._normalize_session_id(session)
        
        resolution = resolution or self.paths.default_resolution
        preprocessing = preprocessing or self.paths.default_preprocessing
        
        # Format the path pattern
        relative_path = self.paths.session_pattern.format(
            subject=subject_num,
            session=session_num,
            resolution=resolution,
            preprocessing=preprocessing
        )
        
        if full_url:
            return f"{self.paths.base_url}/{relative_path}"
        return relative_path
    
    def stim_hdf5_path(self, full_url: bool = True) -> str:
        """
        Generate S3 path for stimuli HDF5 file.
        
        Args:
            full_url: If True, return full S3 URL; if False, return relative path
            
        Returns:
            S3 URL or relative path to stimuli file
            
        Example:
            >>> layout.stim_hdf5_path()
            's3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'
        """
        if full_url:
            return f"{self.paths.base_url}/{self.paths.stimuli_hdf5}"
        return self.paths.stimuli_hdf5
    
    def stim_info_path(self, full_url: bool = True) -> str:
        """
        Generate S3 path for stimulus metadata CSV.
        
        Args:
            full_url: If True, return full S3 URL; if False, return relative path
            
        Returns:
            S3 URL or relative path to stimulus info file
        """
        if full_url:
            return f"{self.paths.base_url}/{self.paths.stim_info}"
        return self.paths.stim_info
    
    def experiment_design_path(self, full_url: bool = True) -> str:
        """
        Generate S3 path for experiment design file.
        
        Args:
            full_url: If True, return full S3 URL; if False, return relative path
            
        Returns:
            S3 URL or relative path to experiment design file
        """
        if full_url:
            return f"{self.paths.base_url}/{self.paths.experiment_design}"
        return self.paths.experiment_design
    
    def single_trial_design_path(
        self, 
        subject: SubjectId, 
        full_url: bool = True
    ) -> str:
        """
        Generate S3 path for single trial design matrices.
        
        Args:
            subject: Subject ID (int or string)
            full_url: If True, return full S3 URL; if False, return relative path
            
        Returns:
            S3 URL or relative path to single trial design file
        """
        subject_num = self._normalize_subject_id(subject)
        
        relative_path = self.paths.single_trial_design.format(subject=subject_num)
        
        if full_url:
            return f"{self.paths.base_url}/{relative_path}"
        return relative_path
    
    def roi_masks_path(
        self, 
        subject: SubjectId, 
        full_url: bool = True
    ) -> str:
        """
        Generate S3 path pattern for ROI masks.
        
        Args:
            subject: Subject ID (int or string)
            full_url: If True, return full S3 URL; if False, return relative path
            
        Returns:
            S3 URL or relative path pattern for ROI masks
        """
        subject_num = self._normalize_subject_id(subject)
        
        relative_path = self.paths.roi_masks.format(subject=subject_num)
        
        if full_url:
            return f"{self.paths.base_url}/{relative_path}"
        return relative_path
    
    def fsaverage_roi_masks_path(
        self,
        subject: SubjectId,
        full_url: bool = True
    ) -> str:
        """
        Generate S3 path pattern for fsaverage ROI masks.
        
        Args:
            subject: Subject ID (int or string)
            full_url: If True, return full S3 URL; if False, return relative path
            
        Returns:
            S3 URL or relative path pattern for fsaverage ROI masks
        """
        subject_num = self._normalize_subject_id(subject)
        pat = f"nsddata/ppdata/subj{subject_num:02d}/fsaverage/*roi*.nii.gz"
        
        if full_url:
            return f"{self.paths.base_url}/{pat}"
        return pat
    
    def mni_roi_masks_path(
        self,
        subject: SubjectId,
        full_url: bool = True
    ) -> str:
        """
        Generate S3 path pattern for MNI ROI masks.
        
        Args:
            subject: Subject ID (int or string)
            full_url: If True, return full S3 URL; if False, return relative path
            
        Returns:
            S3 URL or relative path pattern for MNI ROI masks
        """
        subject_num = self._normalize_subject_id(subject)
        pat = f"nsddata/ppdata/subj{subject_num:02d}/MNI/*roi*.nii.gz"
        
        if full_url:
            return f"{self.paths.base_url}/{pat}"
        return pat
    
    # COCO dataset fallback URLs
    
    def coco_http_url(
        self, 
        coco_id: int, 
        coco_split: CocoSplit = 'train2017'
    ) -> str:
        """
        Generate HTTP URL for COCO images as fallback.
        
        Args:
            coco_id: COCO image ID
            coco_split: COCO dataset split
            
        Returns:
            HTTP URL to COCO image
            
        Example:
            >>> layout.coco_http_url(391895)
            'http://images.cocodataset.org/train2017/000000391895.jpg'
        """
        return f"http://images.cocodataset.org/{coco_split}/{coco_id:012d}.jpg"
    
    def coco_download_url(self, coco_split: CocoSplit = 'train2017') -> str:
        """
        Generate download URL for COCO dataset archives.
        
        Args:
            coco_split: COCO dataset split
            
        Returns:
            HTTP URL to COCO dataset archive
        """
        return f"http://images.cocodataset.org/zips/{coco_split}.zip"
    
    # Utility methods
    
    def get_available_resolutions(self) -> List[Resolution]:
        """Get list of available spatial resolutions"""
        return ['func1mm', 'func1pt8mm', 'MNI', 'fsaverage', 'nativesurface']
    
    def get_available_preprocessing(self) -> List[PreprocessingPipeline]:
        """Get list of available preprocessing pipelines"""
        return ['betas_assumehrf', 'betas_fithrf', 'betas_fithrf_GLMdenoise_RR']
    
    def get_subject_session_range(self, subject: SubjectId) -> Tuple[int, int]:
        """
        Get expected session range for a subject.
        
        Args:
            subject: Subject ID
            
        Returns:
            Tuple of (min_session, max_session)
            
        Note:
            This returns approximate ranges. Actual session availability 
            should be checked by listing S3 files.
        """
        subject_num = self._normalize_subject_id(subject)
        
        # Approximate session counts based on NSD documentation
        session_counts = {
            1: 40, 2: 40, 3: 32, 4: 30,
            5: 40, 6: 32, 7: 40, 8: 30
        }
        
        max_sessions = session_counts.get(subject_num, 40)
        return (1, max_sessions)
    
    def validate_subject_session(
        self, 
        subject: SubjectId, 
        session: SessionId
    ) -> bool:
        """
        Validate that subject and session are in expected ranges.
        
        Args:
            subject: Subject ID
            session: Session ID
            
        Returns:
            True if valid, False otherwise
        """
        try:
            subject_num = self._normalize_subject_id(subject)
            session_num = self._normalize_session_id(session)
            
            # Check subject range
            if not (1 <= subject_num <= 8):
                return False
            
            # Check session range
            min_session, max_session = self.get_subject_session_range(subject_num)
            if not (min_session <= session_num <= max_session):
                return False
                
            return True
            
        except (ValueError, TypeError):
            return False
    
    def beta_session_pattern(self, subject: SubjectId) -> str:
        """
        Generate glob pattern for all beta session files for a subject
        
        Args:
            subject: Subject ID
            
        Returns:
            Full S3 URL glob pattern suitable for fsspec.glob
        """
        subject_num = self._normalize_subject_id(subject)
        
        # Build pattern for all sessions - use string format for wildcard
        pattern = "nsddata_betas/ppdata/subj{subject:02d}/{resolution}/{preprocessing}/betas_session*.nii.gz".format(
            subject=subject_num,
            resolution=self.paths.default_resolution,
            preprocessing=self.paths.default_preprocessing
        )
        
        return f"{self.paths.base_url}/{pattern}"
        
    def format_s3_path(self, relative_path: str) -> str:
        """
        Convert relative path to full S3 URL
        
        Args:
            relative_path: Relative path within the bucket
            
        Returns:
            Full S3 URL
        """
        if relative_path.startswith(('s3://', 'https://')):
            return relative_path
            
        # Remove leading slash if present
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
            
        return f"{self.paths.base_url}/{relative_path}"
    
    def index_path(
        self, 
        index_name: str = "nsd_canonical_index",
        format: str = "parquet",
        full_url: bool = True,
        subject_partitioned: str = None
    ) -> str:
        """
        Generate S3 path for index files (Parquet or CSV)
        
        Args:
            index_name: Name of the index (default: "nsd_canonical_index") 
            format: File format ("parquet" or "csv")
            full_url: If True, return full S3 URL; if False, return relative path
            subject_partitioned: If provided (e.g., "subj01"), returns partitioned path:
                                'nsd_index/subject=subjXX/index.parquet' for the builder.
                                If None, returns demo format: 'nsd_indices/<name>.parquet'
            
        Returns:
            S3 URL or relative path to index file
        """
        if format not in ["parquet", "csv"]:
            raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'")
        
        if subject_partitioned:
            # Subject-partitioned format for production builder
            relative_path = f"nsd_index/subject={subject_partitioned}/index.{format}"
        else:
            # Demo format for testing
            relative_path = f"nsd_indices/{index_name}.{format}"
        
        if full_url:
            return f"{self.paths.base_url}/{relative_path}"
        return relative_path
    
    def write_parquet_to_s3(
        self, 
        df: pd.DataFrame, 
        s3_path: str,
        **kwargs
    ) -> None:
        """
        Write DataFrame to S3 as Parquet with robust error handling
        
        Args:
            df: DataFrame to write
            s3_path: Full S3 URL (e.g., 's3://bucket/path/file.parquet')
            **kwargs: Additional arguments passed to to_parquet()
            
        Raises:
            ValueError: If s3_path is not a valid S3 URL
            Exception: If write operation fails
        """
        if not s3_path.startswith('s3://'):
            raise ValueError(f"s3_path must start with 's3://': {s3_path}")
        
        logger.info(f"Writing {len(df)} rows to S3 Parquet: {s3_path}")
        
        # Retry logic for transient errors
        for attempt in range(2):
            try:
                # Use fsspec for S3 access (works with anonymous access)
                with fsspec.open(s3_path, 'wb') as f:
                    # Convert to parquet bytes in memory
                    parquet_buffer = io.BytesIO()
                    df.to_parquet(parquet_buffer, index=False, **kwargs)
                    
                    # Write to S3
                    f.write(parquet_buffer.getvalue())
                    
                logger.info(f"Successfully wrote Parquet to S3: {s3_path}")
                return
                
            except Exception as e:
                if attempt == 0:  # First attempt failed, retry once
                    import time
                    logger.warning(f"S3 write attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(1)
                else:  # Second attempt failed, raise
                    logger.error(f"Failed to write Parquet to S3 {s3_path}: {e}")
                    raise
    
    def read_parquet_from_s3(self, s3_path: str, **kwargs) -> pd.DataFrame:
        """
        Read DataFrame from S3 Parquet with robust error handling
        
        Args:
            s3_path: Full S3 URL (e.g., 's3://bucket/path/file.parquet')
            **kwargs: Additional arguments passed to read_parquet()
            
        Returns:
            DataFrame loaded from S3 Parquet
            
        Raises:
            ValueError: If s3_path is not a valid S3 URL
            Exception: If read operation fails
        """
        if not s3_path.startswith('s3://'):
            raise ValueError(f"s3_path must start with 's3://': {s3_path}")
        
        logger.info(f"Reading Parquet from S3: {s3_path}")
        
        # Retry logic for transient errors
        for attempt in range(2):
            try:
                # Use fsspec for S3 access
                with fsspec.open(s3_path, 'rb') as f:
                    df = pd.read_parquet(f, **kwargs)
                    
                logger.info(f"Successfully read {len(df)} rows from S3 Parquet")
                return df
                
            except Exception as e:
                if attempt == 0:  # First attempt failed, retry once
                    import time
                    logger.warning(f"S3 read attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(1)
                else:  # Second attempt failed, raise
                    logger.error(f"Failed to read Parquet from S3 {s3_path}: {e}")
                    raise
    
    def write_parquet_local(self, df: pd.DataFrame, local_path: str, **kwargs) -> None:
        """
        Write DataFrame to local Parquet file with directory creation
        
        Args:
            df: DataFrame to write
            local_path: Local file path
            **kwargs: Additional arguments passed to to_parquet()
        """
        from pathlib import Path
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_path, index=False, **kwargs)
    
    def __repr__(self) -> str:
        return (
            f"NSDLayout(bucket='{self.paths.bucket}', "
            f"resolution='{self.paths.default_resolution}', "
            f"preprocessing='{self.paths.default_preprocessing}')"
        )


# Convenience function for quick access
def get_nsd_layout(config_path: Optional[str] = None) -> NSDLayout:
    """
    Convenience function to get NSD layout instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        NSDLayout instance
    """
    return NSDLayout(config_path)
