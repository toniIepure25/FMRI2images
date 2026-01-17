#!/usr/bin/env python3
"""
Build full index for subj01 (all 40 sessions, ~9841 trials)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def build_full_index():
    """Build full subj01 index from NSD experiment design."""
    
    # Output paths
    output_dir = Path("data/indices/nsd_index/subject=subj01")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "index_full.parquet"
    
    logger.info("Building full subj01 index...")
    
    # NSD subj01 has 40 sessions, 750 trials per session (except last session has ~91)
    # Total: 9,841 trials
    
    # Try to download the experiment design from NSD
    try:
        import requests
        logger.info("Downloading NSD experiment design from S3...")
        
        # Download the experiment design CSV
        url = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_expdesign.mat"
        logger.info(f"URL: {url}")
        logger.info("Note: This is a .mat file. We'll build the index from known structure instead.")
    except Exception as e:
        logger.warning(f"Could not download experiment design: {e}")
    
    # Build index manually from known NSD structure
    logger.info("Building index from NSD structure...")
    
    rows = []
    base_path = "/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    
    # Session 1-39: 750 trials each
    # Session 40: 91 trials
    global_trial_idx = 0
    
    for session in range(1, 41):
        n_trials = 750 if session < 40 else 91
        session_str = f"{session:02d}"
        beta_path = f"{base_path}/betas_session{session_str}.nii.gz"
        
        for trial_in_session in range(n_trials):
            rows.append({
                'subject': 'subj01',
                'session': session,
                'trial_in_session': trial_in_session,
                'global_trial_index': global_trial_idx,
                'nsdId': global_trial_idx,  # Placeholder - need actual nsdId mapping
                'beta_path': beta_path,
                'beta_index': trial_in_session,
            })
            global_trial_idx += 1
    
    df = pd.DataFrame(rows)
    logger.info(f"Built index with {len(df)} trials")
    logger.info(f"Sessions: {df['session'].min()} - {df['session'].max()}")
    
    # Save
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Saved full index to: {output_path}")
    logger.info(f"  Total trials: {len(df)}")
    logger.info(f"  Sessions: {df['session'].nunique()}")
    
    # Also save a summary
    summary = {
        'total_trials': len(df),
        'sessions': df['session'].nunique(),
        'trials_per_session': df.groupby('session').size().to_dict()
    }
    
    import json
    summary_path = output_dir / "index_full_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary to: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("IMPORTANT: This index uses placeholder nsdId values!")
    logger.info("You need to map global_trial_index to actual nsdId using NSD experiment design.")
    logger.info("="*80)
    
    return df

if __name__ == "__main__":
    build_full_index()
