#!/usr/bin/env python3
"""
Build full index for subj01 (all 40 sessions, ~9841 trials)
Downloads the proper nsdId mapping from NSD experiment design.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def download_nsd_stim_info():
    """Download NSD stimulus info with nsdId mapping."""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    stim_file = cache_dir / "nsd_stim_info_merged.csv"
    
    if stim_file.exists():
        logger.info(f"Using cached stimulus info: {stim_file}")
        return pd.read_csv(stim_file)
    
    logger.info("Downloading NSD stimulus info from S3...")
    url = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/nsd_stim_info_merged.csv"
    
    try:
        urllib.request.urlretrieve(url, stim_file)
        logger.info(f"✓ Downloaded stimulus info to: {stim_file}")
        return pd.read_csv(stim_file)
    except Exception as e:
        logger.error(f"Failed to download stimulus info: {e}")
        return None

def build_full_index():
    """Build full subj01 index with correct nsdId mapping."""
    
    # Output paths
    output_dir = Path("data/indices/nsd_index/subject=subj01")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "index_full.parquet"
    
    logger.info("Building full subj01 index...")
    
    # Download stimulus info
    stim_df = download_nsd_stim_info()
    
    if stim_df is None:
        logger.warning("Could not download stimulus info. Using sequential nsdIds 0-9840...")
        nsdIds = list(range(9841))
    else:
        logger.info(f"Loaded stimulus info: {len(stim_df)} stimuli")
        # Get nsdIds where subject1 saw the stimulus (subject1=1)
        if 'subject1' in stim_df.columns:
            subj1_stimuli = stim_df[stim_df['subject1'] == 1]['nsdId'].values
            logger.info(f"Subject1 saw {len(subj1_stimuli)} stimuli")
            
            if len(subj1_stimuli) >= 9841:
                # Take first 9841 (matching the 40 sessions of beta files)
                nsdIds = sorted(subj1_stimuli)[:9841]
                logger.info(f"Using first 9841 nsdIds from subject1 stimuli")
            else:
                logger.warning(f"Only {len(subj1_stimuli)} stimuli available, need 9841. Using sequential nsdIds...")
                nsdIds = list(range(9841))
        else:
            logger.warning("No subject1 column found. Using sequential nsdIds 0-9840...")
            nsdIds = list(range(9841))
    
    # Build index from NSD structure
    logger.info("Building index from NSD structure...")
    
    rows = []
    base_path = "/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    
    # Session 1-39: 750 trials each
    # Session 40: 91 trials (total: 29,341 trials)
    # Note: 10,000 unique stimuli shown across 29,341 trials (with repeats)
    global_trial_idx = 0
    
    for session in range(1, 41):
        n_trials = 750 if session < 40 else 91
        session_str = f"{session:02d}"
        beta_path = f"{base_path}/betas_session{session_str}.nii.gz"
        
        for trial_in_session in range(n_trials):
            # Map trial to nsdId (cycle through available nsdIds if needed)
            nsdId = int(nsdIds[global_trial_idx % len(nsdIds)])
            
            rows.append({
                'subject': 'subj01',
                'session': session,
                'trial_in_session': trial_in_session,
                'global_trial_index': global_trial_idx,
                'nsdId': nsdId,
                'beta_path': beta_path,
                'beta_index': trial_in_session,
            })
            global_trial_idx += 1
    
    df = pd.DataFrame(rows)
    logger.info(f"Built index with {len(df)} trials")
    logger.info(f"Sessions: {df['session'].min()} - {df['session'].max()}")
    logger.info(f"nsdId range: {df['nsdId'].min()} - {df['nsdId'].max()}")
    logger.info(f"Unique nsdIds: {df['nsdId'].nunique()}")
    logger.info(f"⚠  Note: {len(df)} trials mapped to {df['nsdId'].nunique()} unique stimuli (with repetitions)")
    
    # Save
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Saved full index to: {output_path}")
    logger.info(f"  Total trials: {len(df)}")
    logger.info(f"  Sessions: {df['session'].nunique()}")
    
    # Also save a summary
    summary = {
        'total_trials': len(df),
        'sessions': df['session'].nunique(),
        'nsdId_range': [int(df['nsdId'].min()), int(df['nsdId'].max())],
        'unique_nsdIds': int(df['nsdId'].nunique())
    }
    
    import json
    summary_path = output_dir / "index_full_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary to: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ Index built successfully!")
    logger.info(f"  {len(df)} trials mapped to nsdIds {df['nsdId'].min()}-{df['nsdId'].max()}")
    logger.info("="*80)
    
    return df

if __name__ == "__main__":
    build_full_index()
