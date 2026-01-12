#!/usr/bin/env python3
"""
Quick index builder for subj01 session 1 - just to get started
Creates a minimal parquet index so you can run experiments
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import nibabel as nib

def build_minimal_index(subject='subj01', session=1, output_dir='data/indices/nsd_index'):
    """Build minimal index from local NSD data"""
    
    print(f"Building minimal index for {subject} session {session}...")
    
    # Get data root
    nsd_root = os.getenv('NSD_DATA_ROOT', '/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd')
    
    # Path to beta file
    beta_path = Path(nsd_root) / f"nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{session:02d}.nii.gz"
    
    if not beta_path.exists():
        print(f"❌ Beta file not found: {beta_path}")
        print(f"   Make sure you have downloaded the NSD data to {nsd_root}")
        return False
    
    print(f"✓ Found beta file: {beta_path}")
    
    # Load beta file to get number of trials
    print("Loading beta file to count trials...")
    img = nib.load(str(beta_path))
    n_trials = img.shape[-1]  # Last dimension is trials
    print(f"✓ Found {n_trials} trials in session {session}")
    
    # Create index entries
    records = []
    for trial_idx in range(n_trials):
        # NSD session 1 has trial IDs 0-749 (750 trials)
        # Each session has 750 trials
        nsd_id = (session - 1) * 750 + trial_idx
        
        records.append({
            'subject': subject,
            'session': session,
            'nsdId': nsd_id,
            'beta_path': str(beta_path),
            'beta_index': trial_idx,
        })
    
    df = pd.DataFrame(records)
    
    # Create output directory
    output_path = Path(output_dir) / f"subject={subject}" / "index.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    print(f"Saving index to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Index created successfully!")
    print(f"   Path: {output_path}")
    print(f"   Entries: {len(df)}")
    print(f"   NSD IDs: {df['nsdId'].min()} - {df['nsdId'].max()}")
    
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Build minimal NSD index")
    parser.add_argument('--subject', default='subj01', help='Subject ID')
    parser.add_argument('--session', type=int, default=1, help='Session number')
    parser.add_argument('--output', default='data/indices/nsd_index', help='Output directory')
    
    args = parser.parse_args()
    
    success = build_minimal_index(args.subject, args.session, args.output)
    sys.exit(0 if success else 1)
