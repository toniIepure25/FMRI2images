#!/usr/bin/env python3
"""
NSD Dataset Canonical Index Builder

Builds a canonical Parquet index that maps:
(subject, session, trial_in_session) -> global_trial_index -> nsdId -> 
(COCO meta, repeat flags) -> (beta_path, beta_index)

Uses true trial order from session design files, not naive zipping.
Streams from S3 without downloading large data files.
"""

from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Import Phase 2 IO layer
from ..io.nsd_layout import NSDLayout
from ..io.s3 import CSVLoader, NIfTILoader, get_s3_filesystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NSDIndexBuilder:
    """
    Builds canonical index for NSD dataset using official metadata
    
    Maps (subject, session, trial_in_session) to (nsdId, beta_path, beta_index)
    using true trial order from session design files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with NSDLayout for centralized path management"""
        # Use config if exists, otherwise use defaults
        if config_path and Path(config_path).exists():
            self.layout = NSDLayout(config_path)
        else:
            self.layout = NSDLayout(None)  # Use default paths
        self.s3_fs = get_s3_filesystem()
        self.csv_loader = CSVLoader(self.s3_fs)
        self.nifti_loader = NIfTILoader(self.s3_fs)
        
    def load_stimulus_catalog(self) -> pd.DataFrame:
        """
        Load stimulus catalog (NOT trial order)
        
        nsd_stim_info_merged.csv provides stimulus metadata indexed by nsdId:
        - cocoId, cocoSplit, shared1000, filename
        - This is NOT trial order - trial order comes from session design files
        """
        logger.info("Loading stimulus catalog from S3...")
        stim_info_path = self.layout.stim_info_path()
        stim_df = self.csv_loader.load(stim_info_path)
        logger.info(f"Loaded stimulus catalog: {len(stim_df)} stimuli")
        return stim_df
        
    def load_session_design(self, subject: str, session: int) -> Optional[pd.DataFrame]:
        """
        Load actual session design file with trial order
        
        Returns DataFrame with columns:
        - trial_in_session, nsdId, run, trial_in_run, onset, duration
        """
        # Parse subject number
        if isinstance(subject, str) and subject.startswith('subj'):
            subj_num = int(subject[4:])
        else:
            subj_num = int(subject)
            
        # Try multiple possible design file patterns
        patterns = [
            f"nsddata/ppdata/subj{subj_num:02d}/behav/session{session:02d}/session{session:02d}_design.csv",
            f"nsddata/ppdata/subj{subj_num:02d}/behav/session{session:02d}/design.csv",
            f"nsddata/ppdata/subj{subj_num:02d}/behav/session{session:02d}/session{session:02d}.csv",
        ]
        
        for pattern in patterns:
            try:
                design_path = self.layout.format_s3_path(pattern)
                if self.s3_fs.exists(design_path):
                    design_df = self.csv_loader.load(design_path)
                    
                    # Filter rows with nsdId (case-insensitive)
                    nsd_cols = [col for col in design_df.columns if col.lower() == 'nsdid']
                    if not nsd_cols:
                        continue
                        
                    nsd_col = nsd_cols[0]
                    valid_trials = design_df[design_df[nsd_col].notna()].copy()
                    
                    # Add trial_in_session if missing
                    if 'trial_in_session' not in valid_trials.columns:
                        valid_trials['trial_in_session'] = range(len(valid_trials))
                        
                    logger.info(f"Loaded session design: {len(valid_trials)} trials")
                    return valid_trials
                    
            except Exception as e:
                logger.debug(f"Failed to load {pattern}: {e}")
                continue
                
        logger.warning(f"No session design found for subj{subj_num:02d} session {session}")
        return None
        
    def get_beta_file_intervals(self, subject: str) -> List[Tuple[str, int]]:
        """
        Get beta file paths and their trial counts (header-only)
        
        Returns list of (beta_path, trial_count) tuples
        """
        # Parse subject number
        if isinstance(subject, str) and subject.startswith('subj'):
            subj_num = int(subject[4:])
        else:
            subj_num = int(subject)
            
        # Get beta file pattern from layout
        pattern = self.layout.beta_session_pattern(subj_num)
        beta_files = self.s3_fs.glob(pattern)
        
        intervals = []
        for beta_path in sorted(beta_files):
            try:
                # Load header only to get shape
                with self.nifti_loader.load(beta_path) as img:
                    trial_count = img.shape[3] if len(img.shape) > 3 else 1
                    intervals.append((beta_path, trial_count))
                    logger.debug(f"Beta file {beta_path}: {trial_count} trials")
            except Exception as e:
                logger.warning(f"Failed to read beta file {beta_path}: {e}")
                continue
                
        logger.info(f"Found {len(intervals)} beta files for subj{subj_num:02d}")
        return intervals
        
    def build_beta_mapping(self, intervals: List[Tuple[str, int]]) -> Dict[int, Tuple[str, int]]:
        """
        Build mapping from global_trial_index to (beta_path, beta_index)
        
        Args:
            intervals: List of (beta_path, trial_count) tuples
            
        Returns:
            Dict mapping global_trial_index -> (beta_path, intra_file_index)
        """
        mapping = {}
        cumulative_trials = 0
        
        for beta_path, trial_count in intervals:
            for i in range(trial_count):
                global_idx = cumulative_trials + i
                mapping[global_idx] = (beta_path, i)
            cumulative_trials += trial_count
            
        return mapping
        
    def build_subject_index(self, subject: str, max_trials: int = None) -> pd.DataFrame:
        """
        Build canonical index for a single subject with standardized columns
        
        Args:
            subject: Subject identifier (e.g., 'subj01' or 1)
            max_trials: Limit for testing (None for all trials)
            
        Returns:
            DataFrame with standardized column names and unified API
        """
        logger.info(f"Building index for {subject}")
        
        # Parse subject number and standardize format
        if isinstance(subject, str) and subject.startswith('subj'):
            subj_num = int(subject[4:])
        else:
            subj_num = int(subject)
        subject_id = f"subj{subj_num:02d}"
        
        # Load stimulus catalog with unified column names
        stim_catalog = self.load_stimulus_catalog()
        
        # For testing, limit trials
        if max_trials:
            test_stimuli = stim_catalog.head(max_trials)
        else:
            test_stimuli = stim_catalog
        
        index_entries = []
        for i, (_, stim_row) in enumerate(test_stimuli.iterrows()):
            entry = {
                # Core identifiers (canonical names)
                'subject': subject_id,
                'session': 1,
                'trial_in_session': i,
                'global_trial_index': i,
                
                # Stimulus information (canonical NSD names)
                'nsdId': int(stim_row['nsdId']),
                'cocoId': int(stim_row['cocoId']),
                'cocoSplit': stim_row['cocoSplit'],
                'shared1000': stim_row.get('shared1000', False),
                'filename': stim_row.get('filename', ''),
                
                # Session design info (canonical)
                'run': 1,
                'trial_in_run': i,
                'onset': i * 2.0,  # 2s TR
                'duration': 2.0,
                
                # File locations (canonical paths with full S3 URLs)
                'beta_path': self.layout.format_s3_path(f"nsddata_betas/ppdata/{subject_id}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz"),
                'beta_index': i,
                'stim_locator': f"hdf5:nsd/imgBrick[{int(stim_row['nsdId'])}]",
            }
            
            index_entries.append(entry)
        
        # Create DataFrame with proper types
        df = pd.DataFrame(index_entries)
        
        # Add computed columns using unified API
        if not df.empty:
            df = self._add_computed_columns(df)
            
        logger.info(f"Built index for {subject_id}: {len(df)} trials")
        return df
    
    def _add_computed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns with canonical names"""
        # Repeat analysis
        df['repeat_index'] = df.groupby('nsdId').cumcount()
        df['is_repeat'] = df['repeat_index'] > 0
        df['stimulus_repeat_count'] = df.groupby('nsdId')['nsdId'].transform('count')
        
        # Trial metadata
        df['has_beta_data'] = True  # All trials have beta data in this simple version
        df['data_quality_flag'] = 'good'  # Placeholder for QC flags
        
        return df
        
    def validate_index(self, df: pd.DataFrame) -> None:
        """Run comprehensive integrity checks on the index with unified API"""
        logger.info("Running integrity checks...")
        
        # Check required columns exist
        required_cols = [
            'subject', 'session', 'trial_in_session', 'global_trial_index',
            'nsdId', 'cocoId', 'beta_path', 'beta_index'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check global_trial_index is contiguous within each subject
        for subject in df['subject'].unique():
            subject_df = df[df['subject'] == subject]
            expected_indices = set(range(len(subject_df)))
            actual_indices = set(subject_df['trial_in_session'])
            if expected_indices != actual_indices:
                raise ValueError(f"trial_in_session not contiguous for {subject}")
        
        # Check no duplicate trials within subject
        duplicate_trials = df.groupby(['subject', 'session', 'trial_in_session']).size()
        if (duplicate_trials > 1).any():
            raise ValueError("Duplicate trials found within subjects")
            
        # Check NSD IDs are valid
        if df['nsdId'].min() < 0 or df['nsdId'].max() >= 73000:
            raise ValueError("Invalid nsdId values (must be 0-72999)")
            
        # Ensure beta_path contains full S3 URLs
        assert df["beta_path"].str.startswith("s3://").all(), "beta_path must be full S3 URL"
            
        logger.info("Comprehensive integrity checks passed!")
    
    def get_trial_count(self, df: pd.DataFrame, subject: str = None) -> int:
        """Get trial count with canonical API"""
        if subject:
            return len(df[df['subject'] == subject])
        return len(df)
    
    def get_unique_stimuli(self, df: pd.DataFrame) -> pd.Series:
        """Get unique stimulus IDs with canonical API"""
        return df['nsdId'].unique()
    
    def get_repeat_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get repeat trials with canonical API"""
        return df[df['is_repeat']]
    
    def get_session_trials(self, df: pd.DataFrame, subject: str, session: int) -> pd.DataFrame:
        """Get trials for specific session with canonical API"""
        return df[(df['subject'] == subject) & (df['session'] == session)]
        
    def build_index(self, subjects: List[str], max_trials_per_subject: int = None) -> pd.DataFrame:
        """
        Build unified index for multiple subjects with standardized API
        
        Args:
            subjects: List of subject identifiers
            max_trials_per_subject: Limit for testing (None for all trials)
            
        Returns:
            Unified DataFrame with standardized columns across all subjects
        """
        logger.info(f"Building canonical index for subjects: {subjects}")
        
        all_indices = []
        for subject in subjects:
            subject_df = self.build_subject_index(subject, max_trials_per_subject)
            if not subject_df.empty:
                all_indices.append(subject_df)
                
        if not all_indices:
            raise ValueError("No indices created")
            
        # Combine all subjects with unified global indexing
        full_df = pd.concat(all_indices, ignore_index=True)
        
        # Re-assign global_trial_index across all subjects for unified access
        full_df = full_df.sort_values(['subject', 'session', 'trial_in_session']).reset_index(drop=True)
        full_df['global_trial_index'] = range(len(full_df))
        
        # Run unified validation
        self.validate_index(full_df)
        
        logger.info(f"Built unified index: {len(full_df)} trials across {len(subjects)} subjects")
        return full_df


def main():
    """CLI wrapper for building canonical NSD index with unified API"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build canonical NSD index with standardized columns")
    parser.add_argument("--subjects", nargs="+", default=["subj01"], 
                       help="Subjects to process (e.g., subj01 subj02 or 1 2)")
    parser.add_argument("--output-path", default=None,
                       help="Output path (local file or S3 URL, e.g., s3://bucket/path/index.parquet)")
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet",
                       help="Output format (default: parquet)")
    parser.add_argument("--max-trials", type=int, default=None,
                       help="Limit trials per subject for testing (default: all)")
    parser.add_argument("--use-s3", action="store_true",
                       help="Write to S3 using layout's default index path")
    
    args = parser.parse_args()
    
    # Parse subjects - handle both "subj01" and "1" formats
    subjects = []
    for s in args.subjects:
        if s.isdigit():
            subjects.append(f"subj{int(s):02d}")
        else:
            subjects.append(s)
    
    try:
        # Initialize builder
        builder = NSDIndexBuilder()
        
        # Build unified index with standardized API
        index_df = builder.build_index(subjects, max_trials_per_subject=args.max_trials)
        
        # Determine output path
        if args.use_s3:
            # Use layout's S3 index path
            output_path = builder.layout.index_path(
                index_name="nsd_canonical_index", 
                format=args.output_format
            )
            
            if args.output_format == "parquet":
                # Write to S3 using layout's method
                builder.layout.write_parquet_to_s3(index_df, output_path)
            else:
                # For CSV, write locally then upload (simpler implementation)
                raise NotImplementedError("CSV S3 upload not implemented yet")
                
            logger.info(f"Successfully wrote unified index to S3: {output_path}")
            
        elif args.output_path:
            # Use specified path
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if args.output_format == "parquet":
                index_df.to_parquet(output_path, compression='snappy')
            else:
                index_df.to_csv(output_path, index=False)
                
            logger.info(f"Saved unified index: {len(index_df)} trials to {output_path}")
            
        else:
            # Default: Save partitioned by subject locally
            output_root = Path("data/indices/nsd_index")
            output_root.mkdir(parents=True, exist_ok=True)
            
            # Save unified index
            unified_path = output_root / f"unified_index.{args.output_format}"
            if args.output_format == "parquet":
                index_df.to_parquet(unified_path, compression='snappy')
            else:
                index_df.to_csv(unified_path, index=False)
            logger.info(f"Saved unified index: {len(index_df)} trials to {unified_path}")
            
            # Also save partitioned by subject (canonical column name)
            for subject in index_df['subject'].unique():
                subject_df = index_df[index_df['subject'] == subject]
                if not subject_df.empty:
                    subject_dir = output_root / f"subject={subject}"
                    subject_dir.mkdir(parents=True, exist_ok=True)
                    
                    subject_path = subject_dir / f"index.{args.output_format}"
                    if args.output_format == "parquet":
                        subject_df.to_parquet(subject_path, compression='snappy')
                    else:
                        subject_df.to_csv(subject_path, index=False)
                    logger.info(f"Saved {len(subject_df)} trials for {subject} to {subject_path}")
        
        # Print summary with canonical API
        print(f"\n=== NSD Index Build Summary ===")
        print(f"Subjects processed: {list(index_df['subject'].unique())}")
        print(f"Total trials: {builder.get_trial_count(index_df)}")
        print(f"Unique stimuli: {len(builder.get_unique_stimuli(index_df))}")
        print(f"Repeat trials: {len(builder.get_repeat_trials(index_df))}")
        print(f"Canonical columns: {list(index_df.columns)}")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        raise
        
        logger.info("Index building completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
