#!/usr/bin/env python3
"""
NSD Dataset Streaming Explorer
This script demonstrates how to stream and examine NSD data from AWS S3 anonymously
"""

import fsspec
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import io
import tempfile
from pathlib import Path

class NSDDataExplorer:
    def __init__(self):
        self.fs = fsspec.filesystem("s3", anon=True)
        self.bucket = "natural-scenes-dataset"
        
    def explore_subjects(self):
        """Explore available subjects"""
        print("=== Available Subjects ===")
        subjects = self.fs.ls(f"{self.bucket}/nsddata_betas/ppdata")
        subject_names = [s.split('/')[-1] for s in subjects]
        print(f"Available subjects: {subject_names}")
        return subject_names
    
    def explore_sessions(self, subject="subj01"):
        """Explore sessions for a specific subject"""
        print(f"\n=== Sessions for {subject} ===")
        sessions = self.fs.glob(f"{self.bucket}/nsddata_betas/ppdata/{subject}/*.nii.gz")
        session_files = [s.split('/')[-1] for s in sessions]
        print(f"Session files: {session_files}")
        return sessions
    
    def load_stimulus_metadata(self):
        """Load stimulus metadata to understand trial structure"""
        print("\n=== Loading Stimulus Metadata ===")
        
        # Load the main stimulus info
        stim_info_path = f"{self.bucket}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
        
        with self.fs.open(stim_info_path, 'r') as f:
            stim_df = pd.read_csv(f)
        
        print(f"Stimulus info shape: {stim_df.shape}")
        print(f"Columns: {list(stim_df.columns)}")
        print("\nFirst few rows:")
        print(stim_df.head())
        
        return stim_df
    
    def load_sample_stimulus(self, nsd_id=0):
        """Load a specific stimulus image"""
        print(f"\n=== Loading Stimulus Image (NSD ID: {nsd_id}) ===")
        
        # Find the stimulus file
        stim_files = self.fs.glob(f"{self.bucket}/nsddata_stimuli/stimuli/nsd/*{nsd_id:05d}*.png")
        
        if not stim_files:
            print(f"No stimulus found for NSD ID {nsd_id}")
            return None
            
        stim_file = stim_files[0]
        print(f"Loading: {stim_file}")
        
        # Load the image
        with self.fs.open(stim_file, 'rb') as f:
            image_data = f.read()
        
        image = Image.open(io.BytesIO(image_data))
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        
        # Convert to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        return np.array(image)
    
    def load_sample_bold_data(self, subject="subj01", session=1):
        """Load and examine BOLD data for a specific subject and session"""
        print(f"\n=== Loading BOLD Data ({subject}, session {session:02d}) ===")
        
        # Find the session file
        session_file = f"{self.bucket}/nsddata_betas/ppdata/{subject}/betas_session{session:02d}.nii.gz"
        
        try:
            # Use fsspec caching to download and load the NIfTI file
            cache_dir = Path(".cache/nsd")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cached_file = f"simplecache::{session_file}"
            
            with fsspec.open(cached_file, mode="rb", anon=True, 
                           target_protocol="s3", cache_storage=str(cache_dir)) as f:
                # Get the cached local path
                local_path = f.name
                
            # Load with nibabel
            nii_img = nib.load(local_path)
            bold_data = nii_img.get_fdata()
            
            print(f"BOLD data shape: {bold_data.shape}")
            print(f"BOLD data dtype: {bold_data.dtype}")
            print(f"Value range: [{bold_data.min():.4f}, {bold_data.max():.4f}]")
            print(f"Affine matrix shape: {nii_img.affine.shape}")
            
            # The last dimension typically represents trials/volumes
            if len(bold_data.shape) == 4:
                print(f"Number of volumes/trials: {bold_data.shape[3]}")
                
                # Show statistics for first volume
                first_volume = bold_data[:, :, :, 0]
                print(f"First volume shape: {first_volume.shape}")
                print(f"First volume stats: mean={first_volume.mean():.4f}, std={first_volume.std():.4f}")
            
            return bold_data, nii_img
            
        except Exception as e:
            print(f"Error loading BOLD data: {e}")
            return None, None
    
    def explore_trial_structure(self, subject="subj01"):
        """Explore how trials are organized"""
        print(f"\n=== Trial Structure for {subject} ===")
        
        # Look for experiment files
        exp_files = self.fs.glob(f"{self.bucket}/nsddata/experiments/nsd/*.mat")
        print(f"Experiment files: {[f.split('/')[-1] for f in exp_files[:5]]}")
        
        # Look for behavior data
        behavior_files = self.fs.glob(f"{self.bucket}/nsddata/bdata/behavdata/*.csv")
        print(f"Behavior files: {[f.split('/')[-1] for f in behavior_files[:5]]}")
        
    def create_data_sample(self, subject="subj01", session=1, limit=5):
        """Create a sample of paired stimulus-BOLD data"""
        print(f"\n=== Creating Data Sample ===")
        
        # Load metadata to get trial info
        stim_df = self.load_stimulus_metadata()
        
        # Load BOLD data
        bold_data, nii_img = self.load_sample_bold_data(subject, session)
        
        if bold_data is None:
            return []
            
        samples = []
        
        # Create samples for first few trials
        for trial_idx in range(min(limit, bold_data.shape[3] if len(bold_data.shape) == 4 else 1)):
            try:
                # Get stimulus info for this trial
                if trial_idx < len(stim_df):
                    nsd_id = stim_df.iloc[trial_idx]['nsdId']
                    
                    # Load corresponding stimulus
                    stim_image = self.load_sample_stimulus(nsd_id)
                    
                    # Extract BOLD volume for this trial
                    if len(bold_data.shape) == 4:
                        bold_volume = bold_data[:, :, :, trial_idx]
                    else:
                        bold_volume = bold_data
                    
                    sample = {
                        'trial_id': trial_idx,
                        'nsd_id': nsd_id,
                        'subject': subject,
                        'session': session,
                        'stimulus_shape': stim_image.shape if stim_image is not None else None,
                        'bold_shape': bold_volume.shape,
                        'bold_mean': bold_volume.mean(),
                        'bold_std': bold_volume.std()
                    }
                    
                    samples.append(sample)
                    print(f"Sample {trial_idx}: NSD_ID={nsd_id}, "
                          f"Stim={sample['stimulus_shape']}, "
                          f"BOLD={sample['bold_shape']}")
                    
            except Exception as e:
                print(f"Error creating sample {trial_idx}: {e}")
                continue
                
        return samples

def main():
    """Main exploration function"""
    explorer = NSDDataExplorer()
    
    # 1. Explore dataset structure
    subjects = explorer.explore_subjects()
    
    # 2. Explore sessions for first subject
    sessions = explorer.explore_sessions("subj01")
    
    # 3. Load stimulus metadata
    stim_df = explorer.load_stimulus_metadata()
    
    # 4. Explore trial structure
    explorer.explore_trial_structure("subj01")
    
    # 5. Load a sample stimulus image
    sample_image = explorer.load_sample_stimulus(0)
    
    # 6. Load sample BOLD data
    bold_data, nii_img = explorer.load_sample_bold_data("subj01", 1)
    
    # 7. Create paired samples
    samples = explorer.create_data_sample("subj01", 1, limit=3)
    
    print("\n=== Summary ===")
    print(f"Found {len(subjects)} subjects")
    print(f"Stimulus metadata: {stim_df.shape[0]} entries" if 'stim_df' in locals() else "Metadata not loaded")
    print(f"Sample BOLD shape: {bold_data.shape}" if bold_data is not None else "BOLD data not loaded")
    print(f"Created {len(samples)} data samples")
    
    return explorer, samples

if __name__ == "__main__":
    explorer, samples = main()