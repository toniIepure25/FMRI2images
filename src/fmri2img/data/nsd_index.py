#!/usr/bin/env python3
"""
NSD Canonical Index Interface

Simple wrapper to load and query the canonical NSD index.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class NSDIndex:
    """Interface to the canonical NSD index"""
    
    def __init__(self, index_path: Union[str, Path]):
        """Load the canonical index from Parquet or CSV"""
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
        
        logger.info(f"Loaded index: {len(self.df)} trials")
    
    @property
    def subjects(self) -> List[str]:
        """Get list of available subjects"""
        return sorted(self.df['subject'].unique())
    
    @property
    def sessions(self) -> List[int]:
        """Get list of available sessions"""
        return sorted(self.df['session'].unique())
    
    def get_subject(self, subject: str) -> pd.DataFrame:
        """Get all trials for a subject"""
        return self.df[self.df['subject'] == subject].copy()
    
    def get_session(self, subject: str, session: int) -> pd.DataFrame:
        """Get all trials for a specific session"""
        return self.df[
            (self.df['subject'] == subject) & 
            (self.df['session'] == session)
        ].copy()
    
    def get_trial(self, subject: str, session: int, trial_in_session: int) -> Optional[pd.Series]:
        """Get a specific trial"""
        trials = self.df[
            (self.df['subject'] == subject) & 
            (self.df['session'] == session) & 
            (self.df['trial_in_session'] == trial_in_session)
        ]
        
        if trials.empty:
            return None
        
        return trials.iloc[0]
    
    @property
    def summary(self) -> Dict:
        """Get summary statistics of the index"""
        summary = {
            'total_trials': len(self.df),
            'subjects': len(self.subjects),
            'sessions': len(self.sessions),
            'unique_stimuli': self.df['nsdId'].nunique(),
        }
        
        # Add repeat stats if available
        if 'repeat_index' in self.df.columns:
            summary['repeat_trials'] = (self.df['repeat_index'] > 0).sum()
            summary['repeat_fraction'] = (self.df['repeat_index'] > 0).mean()
        
        return summary
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __repr__(self) -> str:
        return f"NSDIndex({len(self.df)} trials, {len(self.subjects)} subjects)"


def load_nsd_index(index_path: Union[str, Path]) -> NSDIndex:
    """Convenience function to load NSD index"""
    return NSDIndex(index_path)
