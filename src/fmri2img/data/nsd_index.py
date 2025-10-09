#!/usr/bin/env python3
"""
NSD Canonical Index utilities

Functions to load and query the canonical NSD index
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class NSDIndex:
    """Interface to the canonical NSD index"""
    
    def __init__(self, index_path: Union[str, Path]):
        """Load the canonical index"""
        self.index_path = Path(index_path)
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        logger.info(f"Loading NSD index from {self.index_path}")
        
        if self.index_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.index_path)
        elif self.index_path.suffix == '.csv':
            self.df = pd.read_csv(self.index_path)
        else:
            raise ValueError(f"Unsupported file format: {self.index_path.suffix}")
        
        logger.info(f"Loaded index: {self.df.shape} trials")
        
        # Set up efficient indexing
        self._setup_indices()
    
    def _setup_indices(self):
        """Set up efficient indexing for common queries"""
        # Create multi-index for fast lookups
        self.df_indexed = self.df.set_index(['subject', 'session', 'trial_in_session'])
        
        # Cache common groupings
        self._subject_groups = self.df.groupby('subject')
        self._session_groups = self.df.groupby(['subject', 'session'])
    
    @property
    def subjects(self) -> List[str]:
        """Get list of all subjects"""
        return sorted(self.df['subject'].unique())
    
    @property 
    def sessions(self) -> Dict[str, List[int]]:
        """Get sessions per subject"""
        return self.df.groupby('subject')['session'].apply(sorted).to_dict()
    
    @property
    def summary(self) -> Dict[str, any]:
        """Get summary statistics"""
        return {
            'total_trials': len(self.df),
            'subjects': len(self.subjects),
            'sessions_per_subject': self.df.groupby('subject')['session'].nunique().to_dict(),
            'trials_per_subject': self.df.groupby('subject').size().to_dict(),
            'unique_stimuli': self.df['nsd_id'].nunique(),
            'shared_stimuli': self.df['shared1000'].sum() if 'shared1000' in self.df.columns else 0,
            'flagged_trials': self.df['flagged'].sum() if 'flagged' in self.df.columns else 0,
        }
    
    def get_trial(self, subject: str, session: int, trial_in_session: int) -> Optional[pd.Series]:
        """Get specific trial information"""
        try:
            return self.df_indexed.loc[(subject, session, trial_in_session)]
        except KeyError:
            return None
    
    def get_session(self, subject: str, session: int) -> pd.DataFrame:
        """Get all trials for a specific session"""
        mask = (self.df['subject'] == subject) & (self.df['session'] == session)
        return self.df[mask].copy()
    
    def get_subject(self, subject: str) -> pd.DataFrame:
        """Get all trials for a specific subject"""
        return self.df[self.df['subject'] == subject].copy()
    
    def get_stimulus_trials(self, nsd_id: int) -> pd.DataFrame:
        """Get all trials showing a specific stimulus"""
        return self.df[self.df['nsd_id'] == nsd_id].copy()
    
    def get_shared_stimuli(self) -> pd.DataFrame:
        """Get trials with shared stimuli across subjects"""
        if 'shared1000' in self.df.columns:
            return self.df[self.df['shared1000'] == True].copy()
        else:
            # Find stimuli seen by multiple subjects
            stimulus_counts = self.df['nsd_id'].value_counts()
            shared_stimuli = stimulus_counts[stimulus_counts > 1].index
            return self.df[self.df['nsd_id'].isin(shared_stimuli)].copy()
    
    def get_clean_trials(self, exclude_flagged: bool = True) -> pd.DataFrame:
        """Get clean trials (optionally excluding flagged stimuli)"""
        df_clean = self.df.copy()
        
        if exclude_flagged and 'flagged' in df_clean.columns:
            df_clean = df_clean[df_clean['flagged'] == False]
        
        # Remove trials with missing essential data
        df_clean = df_clean.dropna(subset=['nsd_id', 'beta_file'])
        
        return df_clean
    
    def create_train_val_split(self, 
                             train_subjects: Optional[List[str]] = None,
                             val_subjects: Optional[List[str]] = None,
                             train_sessions: Optional[List[int]] = None,
                             val_sessions: Optional[List[int]] = None,
                             val_fraction: float = 0.2,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/validation split"""
        
        if train_subjects is not None and val_subjects is not None:
            # Subject-based split
            train_df = self.df[self.df['subject'].isin(train_subjects)].copy()
            val_df = self.df[self.df['subject'].isin(val_subjects)].copy()
            
        elif train_sessions is not None and val_sessions is not None:
            # Session-based split
            train_mask = self.df['session'].isin(train_sessions)
            val_mask = self.df['session'].isin(val_sessions)
            train_df = self.df[train_mask].copy()
            val_df = self.df[val_mask].copy()
            
        else:
            # Random split within each subject
            train_dfs = []
            val_dfs = []
            
            np.random.seed(random_state)
            
            for subject in self.subjects:
                subject_df = self.get_subject(subject)
                n_trials = len(subject_df)
                n_val = int(n_trials * val_fraction)
                
                # Random indices for validation
                val_indices = np.random.choice(n_trials, n_val, replace=False)
                train_indices = np.setdiff1d(np.arange(n_trials), val_indices)
                
                train_dfs.append(subject_df.iloc[train_indices])
                val_dfs.append(subject_df.iloc[val_indices])
            
            train_df = pd.concat(train_dfs, ignore_index=True)
            val_df = pd.concat(val_dfs, ignore_index=True)
        
        logger.info(f"Train/val split: {len(train_df)} train, {len(val_df)} val")
        
        return train_df, val_df
    
    def get_file_paths(self, trial_indices: Optional[List[int]] = None) -> pd.DataFrame:
        """Get file paths for trials"""
        if trial_indices is not None:
            subset_df = self.df.iloc[trial_indices]
        else:
            subset_df = self.df
        
        return subset_df[['subject', 'session', 'trial_in_session', 'nsd_id', 
                         'beta_file', 'volume_index', 'stimulus_file']].copy()
    
    def validate_files_exist(self, check_s3: bool = False) -> Dict[str, any]:
        """Validate that referenced files exist"""
        logger.info("Validating file existence...")
        
        # Get unique file paths
        beta_files = self.df['beta_file'].unique()
        stimulus_files = self.df['stimulus_file'].unique()
        
        results = {
            'unique_beta_files': len(beta_files),
            'unique_stimulus_files': len(stimulus_files),
            'missing_beta_files': [],
            'missing_stimulus_files': []
        }
        
        if check_s3:
            import fsspec
            fs = fsspec.filesystem("s3", anon=True)
            
            # Check beta files
            for beta_file in beta_files:
                if beta_file.startswith('s3://'):
                    s3_path = beta_file[5:]  # Remove 's3://'
                elif beta_file.startswith('natural-scenes-dataset/'):
                    s3_path = beta_file
                else:
                    s3_path = f"natural-scenes-dataset/{beta_file}"
                
                try:
                    fs.info(s3_path)
                except:
                    results['missing_beta_files'].append(beta_file)
            
            # Check stimulus files
            for stim_file in stimulus_files:
                if stim_file.startswith('s3://'):
                    s3_path = stim_file[5:]
                elif stim_file.startswith('natural-scenes-dataset/'):
                    s3_path = stim_file
                else:
                    s3_path = f"natural-scenes-dataset/{stim_file}"
                
                try:
                    fs.info(s3_path)
                except:
                    results['missing_stimulus_files'].append(stim_file)
        
        results['beta_files_missing'] = len(results['missing_beta_files'])
        results['stimulus_files_missing'] = len(results['missing_stimulus_files'])
        
        logger.info(f"File validation complete:")
        logger.info(f"  Beta files: {results['unique_beta_files']} unique, {results['beta_files_missing']} missing")
        logger.info(f"  Stimulus files: {results['unique_stimulus_files']} unique, {results['stimulus_files_missing']} missing")
        
        return results
    
    def export_manifest(self, output_path: Union[str, Path], format: str = 'jsonl') -> Path:
        """Export index as manifest in various formats"""
        output_path = Path(output_path)
        
        if format == 'jsonl':
            # JSONL format for streaming
            with open(output_path, 'w') as f:
                for _, row in self.df.iterrows():
                    record = {
                        'trial_id': row['global_trial_id'],
                        'subject': row['subject'],
                        'session': row['session'],
                        'trial_in_session': row['trial_in_session'],
                        'nsd_id': row['nsd_id'],
                        'coco_id': row['coco_id'],
                        'beta_file': row['beta_file'],
                        'volume_index': row['volume_index'],
                        'stimulus_file': row['stimulus_file'],
                        'flagged': row.get('flagged', False),
                    }
                    f.write(f"{record}\n")
        
        elif format == 'parquet':
            self.df.to_parquet(output_path, index=False)
        
        elif format == 'csv':
            self.df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported manifest to {output_path}")
        return output_path


def load_nsd_index(index_path: Union[str, Path] = "data/indices/nsd_canonical_index.parquet") -> NSDIndex:
    """Convenience function to load the canonical NSD index"""
    return NSDIndex(index_path)


# Example usage and tests
if __name__ == "__main__":
    # This would be run as a test/example
    import sys
    
    if len(sys.argv) > 1:
        index_path = sys.argv[1]
    else:
        index_path = "data/indices/nsd_canonical_index.parquet"
    
    try:
        # Load index
        nsd_idx = load_nsd_index(index_path)
        
        # Print summary
        print("NSD Index Summary:")
        summary = nsd_idx.summary
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Test queries
        print(f"\nSubjects: {nsd_idx.subjects}")
        
        if nsd_idx.subjects:
            subject = nsd_idx.subjects[0]
            subject_data = nsd_idx.get_subject(subject)
            print(f"\n{subject} has {len(subject_data)} trials")
            
            if not subject_data.empty:
                # Get first session
                first_session = subject_data['session'].min()
                session_data = nsd_idx.get_session(subject, first_session)
                print(f"Session {first_session} has {len(session_data)} trials")
                
                # Get a specific trial
                trial = nsd_idx.get_trial(subject, first_session, 0)
                if trial is not None:
                    print(f"\nExample trial: {trial['global_trial_id']}")
                    print(f"  NSD ID: {trial['nsd_id']}")
                    print(f"  COCO ID: {trial['coco_id']}")
                    print(f"  Beta file: {trial['beta_file']}")
        
    except FileNotFoundError:
        print(f"Index file not found: {index_path}")
        print("Run the index builder first:")
        print("python src/fmri2img/data/nsd_index_builder.py --quick")