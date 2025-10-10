#!/usr/bin/env python3
"""
NSD Dataset Canonical Index Builder

This module builds a canonical index mapping subject/session/trial combinations
to stimulus IDs and file paths. Replaces naive zip-based data access with 
efficient Parquet-based indexing.

Updated for Phase 2: Uses centralized path management from NSDLayout.
"""

from __future__ import annotations
import logging
import fsspec
import h5py
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from tqdm import tqdm

# Import Phase 2 IO layer
from ..io.nsd_layout import NSDLayout
from ..io.s3 import CSVLoader, get_s3_filesystem

import pandas as pd
import numpy as np
import fsspec
import yaml
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSDIndexBuilder:
    """
    Builds canonical index for NSD dataset using official metadata
    
    Updated for Phase 2: Now uses NSDLayout for centralized path management
    and robust S3 loaders from the IO layer.
    """
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Phase 2 components
        self.layout = NSDLayout(config_path)
        self.s3_fs = get_s3_filesystem()
        self.csv_loader = CSVLoader(self.s3_fs)
        
        # Legacy filesystem for compatibility
        self.fs = fsspec.filesystem("s3", anon=True)
        self.bucket = self.config['s3']['bucket']
        
        # Cache directory
        self.cache_dir = Path(self.config['cache']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Output directory
        self.output_dir = Path("data/indices")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_stimulus_metadata(self) -> pd.DataFrame:
        """
        Load the main stimulus metadata file using Phase 2 IO layer
        """
        logger.info("Loading stimulus metadata...")
        
        # Use NSDLayout to get the correct path
        stim_info_url = self.layout.stim_info_path()
        logger.info("Downloading stimulus metadata from S3...")
        
        # Use robust CSV loader from Phase 2
        stim_df = self.csv_loader.load(stim_info_url)
        
        logger.info(f"Loaded stimulus metadata: {stim_df.shape}")
        return stim_df
    
    def get_available_sessions(self, subject: str) -> List[int]:
        """Get list of available sessions for a subject"""
        logger.info(f"Finding available sessions for {subject}...")
        
        # Build path pattern
        fmri_config = self.config['nsd']['fmri']
        base_path = f"{self.bucket}/nsddata_betas/ppdata/{subject}/{fmri_config['resolution']}/{fmri_config['preprocessing']}"
        
        try:
            # List all beta files
            files = self.fs.glob(f"{base_path}/betas_session*.nii.gz")
            
            # Extract session numbers
            sessions = []
            for file_path in files:
                filename = file_path.split('/')[-1]
                if 'betas_session' in filename:
                    # Extract session number from filename like "betas_session01.nii.gz"
                    session_str = filename.split('betas_session')[1].split('.')[0]
                    try:
                        session_num = int(session_str)
                        sessions.append(session_num)
                    except ValueError:
                        continue
            
            sessions = sorted(sessions)
            logger.info(f"Found {len(sessions)} sessions for {subject}: {sessions}")
            return sessions
            
        except Exception as e:
            logger.error(f"Error finding sessions for {subject}: {e}")
            return []
    
    def load_session_design(self, subject: str, session: int) -> Optional[pd.DataFrame]:
        """
        Load session design information (trial order).
        
        Note: The actual NSD dataset might have specific design files per session.
        For now, we'll estimate trial order based on the stimulus metadata.
        In a production system, you'd load the actual session design files.
        """
        logger.debug(f"Creating session design for {subject} session {session}")
        
        # This is a simplified approach - in reality, you'd load actual design files
        # For now, we'll create a reasonable approximation
        
        trials_per_session = self.config['sessions']['trials_per_session']
        
        # Create a simple sequential trial mapping
        trials = []
        for trial_idx in range(trials_per_session):
            trials.append({
                'trial_in_session': trial_idx,
                'volume_index': trial_idx,  # Assuming 1:1 mapping to fMRI volumes
            })
        
        return pd.DataFrame(trials)
    
    def build_subject_index(self, subject: str, sessions: Optional[List[int]] = None) -> pd.DataFrame:
        """Build comprehensive index for a single subject"""
        logger.info(f"Building index for {subject}")
        
        # Load stimulus metadata
        stim_df = self.load_stimulus_metadata()
        
        # Get subject's stimuli
        if isinstance(subject, int):
            subject_num = subject
        else:
            # Extract subject number from "subj01" format
            subject_num = int(subject[-1]) if isinstance(subject, str) else subject
            
        subject_col = f"subject{subject_num}"
        if subject_col not in stim_df.columns:
            logger.error(f"Subject column {subject_col} not found in metadata")
            return pd.DataFrame()
        
        subject_stimuli = stim_df[stim_df[subject_col] == 1].copy()
        logger.info(f"subj{subject_num:02d} viewed {len(subject_stimuli)} stimuli")
        
        # Get available sessions
        if sessions is None:
            available_sessions = self.get_available_sessions(subject)
        else:
            available_sessions = sessions
        
        if not available_sessions:
            logger.error(f"No sessions found for {subject}")
            return pd.DataFrame()
        
        # Build index entries
        index_entries = []
        trials_per_session = self.config['sessions']['trials_per_session']
        
        # Track stimulus assignment across sessions
        stimulus_idx = 0
        
        for session_num in tqdm(available_sessions, desc=f"Processing {subject} sessions"):
            # Load session design
            session_design = self.load_session_design(subject, session_num)
            
            if session_design is None:
                logger.warning(f"Could not load design for {subject} session {session_num}")
                continue
            
            # Build beta file path using Phase 2 layout
            beta_file_path = self.layout.beta_path(
                subject=subject_num,
                session=session_num,
                full_url=False  # Get relative path for index
            )
            
            # Process each trial in the session
            for _, trial_info in session_design.iterrows():
                trial_in_session = trial_info['trial_in_session']
                volume_index = trial_info['volume_index']
                
                # Get corresponding stimulus (if available)
                if stimulus_idx < len(subject_stimuli):
                    stim_row = subject_stimuli.iloc[stimulus_idx]
                    
                    # Check for repetitions
                    rep_cols = [col for col in stim_df.columns if f"{subject_col}_rep" in col]
                    repetition_info = {}
                    for rep_col in rep_cols:
                        rep_num = rep_col.split('_rep')[1]
                        repetition_info[f'repetition_{rep_num}'] = stim_row[rep_col]
                    
                    # Create index entry
                    entry = {
                        # Core identifiers
                        'subject': subject,
                        'session': session_num,
                        'trial_in_session': trial_in_session,
                        'global_trial_id': f"{subject}_s{session_num:02d}_t{trial_in_session:03d}",
                        
                        # Stimulus information
                        'nsd_id': stim_row['nsdId'],
                        'coco_id': stim_row['cocoId'],
                        'coco_split': stim_row['cocoSplit'],
                        'crop_box': stim_row['cropBox'],
                        'flagged': stim_row['flagged'],
                        'shared1000': stim_row.get('shared1000', False),
                        'bold5000': stim_row.get('BOLD5000', False),
                        
                        # File paths
                        'beta_file': beta_file_path,
                        'volume_index': volume_index,
                        'stimulus_file': self.config['nsd']['stimuli']['hdf5_file'],
                        
                        # Quality metrics (if available)
                        'loss': stim_row.get('loss', np.nan),
                        
                        # Session metadata
                        'trials_in_session': len(session_design),
                        'session_complete': True,  # Assume complete for now
                    }
                    
                    # Add repetition information
                    entry.update(repetition_info)
                    
                    index_entries.append(entry)
                    stimulus_idx += 1
                
                else:
                    # No more stimuli for this subject
                    logger.warning(f"No stimulus for {subject} session {session_num} trial {trial_in_session}")
                    break
        
        # Create DataFrame
        index_df = pd.DataFrame(index_entries)
        
        logger.info(f"Created index for {subject}: {len(index_df)} trials across {len(available_sessions)} sessions")
        
        return index_df
    
    def build_full_index(self, subjects: Optional[List[str]] = None, sessions: Optional[List[int]] = None) -> pd.DataFrame:
        """Build comprehensive index for multiple subjects"""
        logger.info("Building full NSD index...")
        
        if subjects is None:
            subjects = self.config['subjects']['default']
        
        all_indices = []
        
        for subject in subjects:
            subject_index = self.build_subject_index(subject, sessions)
            if not subject_index.empty:
                all_indices.append(subject_index)
        
        if not all_indices:
            logger.error("No indices created")
            return pd.DataFrame()
        
        # Combine all subject indices
        full_index = pd.concat(all_indices, ignore_index=True)
        
        # Add global trial ID
        full_index['global_trial_index'] = range(len(full_index))
        
        # Sort by subject, session, trial
        full_index = full_index.sort_values(['subject', 'session', 'trial_in_session']).reset_index(drop=True)
        
        logger.info(f"Full index created: {len(full_index)} trials across {len(subjects)} subjects")
        
        return full_index
    
    def validate_index(self, index_df: pd.DataFrame) -> Dict[str, any]:
        """Validate the created index"""
        logger.info("Validating index...")
        
        validation_results = {
            'total_trials': len(index_df),
            'subjects': index_df['subject'].nunique(),
            'sessions': index_df['session'].nunique(),
            'unique_stimuli': index_df['nsd_id'].nunique(),
            'missing_nsd_ids': index_df['nsd_id'].isna().sum(),
            'missing_coco_ids': index_df['coco_id'].isna().sum(),
            'flagged_stimuli': index_df['flagged'].sum(),
            'shared_stimuli': index_df['shared1000'].sum() if 'shared1000' in index_df.columns else 0,
        }
        
        # Check for duplicates
        duplicates = index_df.duplicated(subset=['subject', 'session', 'trial_in_session']).sum()
        validation_results['duplicate_trials'] = duplicates
        
        # Check session completeness
        sessions_per_subject = index_df.groupby('subject')['session'].nunique()
        validation_results['sessions_per_subject'] = sessions_per_subject.to_dict()
        
        # Log validation results
        logger.info("Validation Results:")
        for key, value in validation_results.items():
            logger.info(f"  {key}: {value}")
        
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate trials!")
        
        return validation_results
    
    def save_index(self, index_df: pd.DataFrame, filename: str = "nsd_canonical_index.parquet"):
        """Save index to Parquet format"""
        output_path = self.output_dir / filename
        
        logger.info(f"Saving index to {output_path}")
        index_df.to_parquet(output_path, index=False)
        
        # Also save as CSV for easy inspection
        csv_path = output_path.with_suffix('.csv')
        index_df.to_csv(csv_path, index=False)
        
        # Save metadata
        metadata = {
            'creation_date': pd.Timestamp.now().isoformat(),
            'total_trials': len(index_df),
            'subjects': sorted(index_df['subject'].unique().tolist()),
            'sessions': sorted(index_df['session'].unique().tolist()),
            'columns': index_df.columns.tolist(),
            'config_used': self.config
        }
        
        metadata_path = output_path.with_suffix('.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Index saved successfully:")
        logger.info(f"  Parquet: {output_path}")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  Metadata: {metadata_path}")
        
        return output_path


def main():
    """Main function to build the canonical index"""
    parser = argparse.ArgumentParser(description="Build canonical NSD index")
    parser.add_argument("--config", default="configs/data.yaml", help="Configuration file path")
    parser.add_argument("--subjects", nargs="+", help="Subjects to process (default: from config)")
    parser.add_argument("--sessions", nargs="+", type=int, help="Sessions to process (default: all available)")
    parser.add_argument("--output", default="nsd_canonical_index.parquet", help="Output filename")
    parser.add_argument("--quick", action="store_true", help="Quick test with limited data")
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = NSDIndexBuilder(args.config)
    
    # Set subjects and sessions
    if args.quick:
        subjects = ["subj01"]
        sessions = [1, 2]
        logger.info("Quick mode: processing subj01, sessions 1-2")
    else:
        subjects = args.subjects
        sessions = args.sessions
    
    # Build index
    index_df = builder.build_full_index(subjects=subjects, sessions=sessions)
    
    if index_df.empty:
        logger.error("Failed to create index")
        return 1
    
    # Validate index
    validation_results = builder.validate_index(index_df)
    
    # Save index
    output_path = builder.save_index(index_df, args.output)
    
    logger.info("Index building completed successfully!")
    logger.info(f"Index shape: {index_df.shape}")
    logger.info(f"Saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())