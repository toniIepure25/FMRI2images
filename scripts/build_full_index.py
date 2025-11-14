#!/usr/bin/env python3
"""
Build Full NSD Index with All Sessions

This script rebuilds the NSD index to include ALL sessions for a subject,
not just session 1. Subject 01 has 40 sessions = 30,000 trials.

Usage:
    python scripts/build_full_index.py --subject subj01 --output data/indices/nsd_index/subject=subj01/index_full.parquet
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from fmri2img.io.s3 import get_s3_filesystem, CSVLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_full_index(subject: str, output_path: Path, max_sessions: int = None):
    """
    Build complete index with all available sessions
    
    Args:
        subject: Subject ID (e.g., 'subj01')
        output_path: Where to save the parquet file
        max_sessions: Limit sessions for testing (None = all)
    """
    logger.info(f"Building full index for {subject}")
    
    # Parse subject number
    if subject.startswith('subj'):
        subj_num = int(subject[4:])
    else:
        subj_num = int(subject)
        subject = f"subj{subj_num:02d}"
    
    # Initialize S3
    s3_fs = get_s3_filesystem()
    csv_loader = CSVLoader(s3_fs)
    
    # Load stimulus catalog
    logger.info("Loading stimulus catalog...")
    stim_info_path = "natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    stim_catalog = csv_loader.load(stim_info_path)
    logger.info(f"Loaded {len(stim_catalog)} stimuli")
    
    # Create stimulus lookup
    stim_lookup = stim_catalog.set_index('nsdId').to_dict('index')
    
    # Load REAL behavioral data (responses.tsv)
    logger.info("Loading behavioral data (responses.tsv)...")
    behav_path = f"natural-scenes-dataset/nsddata/ppdata/{subject}/behav/responses.tsv"
    with s3_fs.open(behav_path, 'r') as f:
        behav_data = pd.read_csv(f, sep='\t')
    logger.info(f"Loaded {len(behav_data)} trials from behavioral data")
    
    # Rename 73KID to nsdId for consistency
    behav_data = behav_data.rename(columns={'73KID': 'nsdId', 'SESSION': 'session', 'RUN': 'run', 'TRIAL': 'trial_in_run'})
    
    # Filter to requested sessions
    if max_sessions:
        behav_data = behav_data[behav_data['session'] <= max_sessions]
    
    logger.info(f"Using {len(behav_data)} trials from sessions {behav_data['session'].min()}-{behav_data['session'].max()}")
    
    # Build index from REAL behavioral data
    all_entries = []
    
    for session_num in tqdm(sorted(behav_data['session'].unique()), desc="Processing sessions"):
        session_trials = behav_data[behav_data['session'] == session_num].copy()
        session_trials = session_trials.sort_values(['run', 'trial_in_run'])
        session_trials['trial_in_session'] = range(len(session_trials))
        
        # Beta file path for this session
        beta_path = f"s3://natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{session_num:02d}.nii.gz"
        
        for idx, row in session_trials.iterrows():
            nsd_id = int(row['nsdId'])
            
            # Get stimulus info from catalog
            stim_info = stim_lookup.get(nsd_id, {})
            
            entry = {
                # Core identifiers
                'subject': subject,
                'session': int(row['session']),
                'trial_in_session': int(row['trial_in_session']),
                'global_trial_index': len(all_entries),
                
                # Stimulus information (REAL from behavioral data)
                'nsdId': nsd_id,
                'cocoId': int(stim_info.get('cocoId', 0)),
                'cocoSplit': stim_info.get('cocoSplit', ''),
                'shared1000': bool(stim_info.get('shared1000', False)),
                'filename': stim_info.get('filename', ''),
                
                # Session design (REAL from behavioral data)
                'run': int(row['run']),
                'trial_in_run': int(row['trial_in_run']),
                'onset': 0.0,  # Not in responses.tsv
                'duration': 2.0,  # Standard NSD trial duration
                
                # Behavioral responses (NEW - real data!)
                'is_old': bool(row.get('ISOLD', False)),
                'is_correct': bool(row.get('ISCORRECT', False)),
                'reaction_time': float(row.get('RT', 0)),
                
                # File locations
                'beta_path': beta_path,
                'beta_index': int(row['trial_in_session']),  # Index within session file
                'stim_locator': f"hdf5:nsd/imgBrick[{nsd_id}]",
            }
            
            all_entries.append(entry)
    
    # Create DataFrame
    logger.info(f"Creating DataFrame with {len(all_entries)} trials...")
    df = pd.DataFrame(all_entries)
    
    # Add computed columns
    logger.info("Adding computed columns...")
    df['repeat_index'] = df.groupby('nsdId').cumcount()
    df['is_repeat'] = df['repeat_index'] > 0
    df['stimulus_repeat_count'] = df.groupby('nsdId')['nsdId'].transform('count')
    df['has_beta_data'] = True
    df['data_quality_flag'] = 'good'
    
    # Validate
    logger.info("Validating index...")
    logger.info(f"  Total trials: {len(df)}")
    logger.info(f"  Sessions: {df['session'].min()}-{df['session'].max()}")
    logger.info(f"  Unique stimuli: {df['nsdId'].nunique()}")
    logger.info(f"  Unique beta files: {df['beta_path'].nunique()}")
    logger.info(f"  Beta index range: {df['beta_index'].min()}-{df['beta_index'].max()}")
    
    # Check for issues
    max_beta_index_per_session = df.groupby('session')['beta_index'].max()
    if (max_beta_index_per_session >= 750).any():
        logger.warning("⚠️  Some sessions have beta_index >= 750! This will cause errors.")
        problem_sessions = max_beta_index_per_session[max_beta_index_per_session >= 750]
        logger.warning(f"   Problem sessions: {problem_sessions.to_dict()}")
    else:
        logger.info("✅ All beta indices are valid (< 750)")
    
    # Save
    logger.info(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"✅ Done! Saved {len(df)} trials to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Build full NSD index with all sessions")
    parser.add_argument(
        '--subject',
        type=str,
        default='subj01',
        help='Subject ID (e.g., subj01)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/indices/nsd_index/subject=subj01/index_full.parquet'),
        help='Output parquet file path'
    )
    parser.add_argument(
        '--max-sessions',
        type=int,
        default=None,
        help='Limit number of sessions for testing (default: all)'
    )
    
    args = parser.parse_args()
    
    df = build_full_index(args.subject, args.output, args.max_sessions)
    
    print("\n" + "="*80)
    print("INDEX SUMMARY")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Output: {args.output}")
    print(f"Total trials: {len(df):,}")
    print(f"Sessions: {df['session'].nunique()}")
    print(f"Unique stimuli: {df['nsdId'].nunique()}")
    print(f"\nSample rows:")
    print(df[['subject', 'session', 'trial_in_session', 'global_trial_index', 'nsdId', 'beta_index']].head(10))
    print("\nLast rows:")
    print(df[['subject', 'session', 'trial_in_session', 'global_trial_index', 'nsdId', 'beta_index']].tail(10))


if __name__ == '__main__':
    main()
