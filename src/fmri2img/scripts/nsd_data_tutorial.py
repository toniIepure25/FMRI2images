#!/usr/bin/env python3
"""
NSD Dataset Tutorial: Understanding the structure for fMRI-to-image reconstruction

This script demonstrates how to:
1. Load stimulus images and fMRI data
2. Understand the trial-to-stimulus mapping
3. Prepare data for CLIP-based reconstruction
"""

import fsspec
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import io
import tempfile
from pathlib import Path

class NSDDataLoader:
    def __init__(self, cache_dir=".cache/nsd"):
        self.fs = fsspec.filesystem("s3", anon=True)
        self.bucket = "natural-scenes-dataset"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load stimulus metadata once
        self.stim_df = None
        self._load_stimulus_metadata()
    
    def _load_stimulus_metadata(self):
        """Load the stimulus metadata that maps trials to images"""
        print("Loading stimulus metadata...")
        stim_info_path = f"{self.bucket}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
        
        with self.fs.open(stim_info_path, 'r') as f:
            self.stim_df = pd.read_csv(f)
        
        print(f"Loaded metadata for {len(self.stim_df)} stimulus presentations")
        print(f"Columns: {list(self.stim_df.columns)}")
        
        # Show summary statistics
        print(f"Unique images: {self.stim_df['nsdId'].nunique()}")
        print(f"Subjects: {[col for col in self.stim_df.columns if col.startswith('subject')]}")
    
    def get_subject_trials(self, subject_num=1):
        """Get trials for a specific subject"""
        subject_col = f"subject{subject_num}"
        
        if subject_col not in self.stim_df.columns:
            raise ValueError(f"Subject {subject_num} not found")
        
        # Get trials where this subject saw the stimulus
        subject_trials = self.stim_df[self.stim_df[subject_col] == 1].copy()
        print(f"Subject {subject_num} saw {len(subject_trials)} stimuli")
        
        return subject_trials
    
    def load_stimulus_image(self, nsd_id):
        """Load a stimulus image by NSD ID"""
        # Find the stimulus file
        stim_file = f"{self.bucket}/nsddata_stimuli/stimuli/nsd/nsd{nsd_id:05d}.png"
        
        try:
            with self.fs.open(stim_file, 'rb') as f:
                image_data = f.read()
            
            image = Image.open(io.BytesIO(image_data))
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            
            return np.array(image)
            
        except Exception as e:
            print(f"Error loading stimulus {nsd_id}: {e}")
            return None
    
    def load_fmri_session(self, subject_num=1, session_num=1):
        """Load fMRI data for a specific subject and session"""
        subject_id = f"subj{subject_num:02d}"
        session_file = f"{self.bucket}/nsddata_betas/ppdata/{subject_id}/betas_session{session_num:02d}.nii.gz"
        
        print(f"Loading fMRI data: {subject_id}, session {session_num}")
        
        # Use fsspec caching
        cached_file = f"simplecache::s3://{session_file}"
        
        try:
            with fsspec.open(cached_file, mode="rb", anon=True, 
                           cache_storage=str(self.cache_dir)) as f:
                local_path = f.name
                
            # Load with nibabel
            nii_img = nib.load(local_path)
            fmri_data = nii_img.get_fdata()
            
            print(f"fMRI data shape: {fmri_data.shape}")
            print(f"Data type: {fmri_data.dtype}")
            print(f"Value range: [{fmri_data.min():.4f}, {fmri_data.max():.4f}]")
            
            return fmri_data, nii_img
            
        except Exception as e:
            print(f"Error loading fMRI data: {e}")
            return None, None
    
    def get_trial_mapping(self, subject_num=1, session_num=1):
        """Get the mapping between fMRI volumes and stimulus IDs for a session"""
        
        # For NSD, each session has ~750 trials
        # The trial order is stored in experiment files
        # For simplicity, we'll use the stimulus metadata directly
        
        subject_trials = self.get_subject_trials(subject_num)
        
        # Estimate trials per session (NSD typically has ~40 sessions per subject)
        trials_per_session = len(subject_trials) // 40  # Approximate
        
        start_idx = (session_num - 1) * trials_per_session
        end_idx = start_idx + trials_per_session
        
        session_trials = subject_trials.iloc[start_idx:end_idx].copy()
        session_trials['trial_in_session'] = range(len(session_trials))
        
        return session_trials
    
    def create_paired_samples(self, subject_num=1, session_num=1, limit=5):
        """Create paired stimulus-fMRI samples"""
        print(f"\n=== Creating Paired Samples ===")
        print(f"Subject: {subject_num}, Session: {session_num}, Limit: {limit}")
        
        # Load fMRI data
        fmri_data, nii_img = self.load_fmri_session(subject_num, session_num)
        if fmri_data is None:
            return []
        
        # Get trial mapping
        trial_mapping = self.get_trial_mapping(subject_num, session_num)
        
        samples = []
        
        for i in range(min(limit, len(trial_mapping), fmri_data.shape[3])):
            try:
                trial_info = trial_mapping.iloc[i]
                nsd_id = trial_info['nsdId']
                
                # Load stimulus image
                stimulus = self.load_stimulus_image(nsd_id)
                
                # Extract fMRI volume
                fmri_volume = fmri_data[:, :, :, i]
                
                sample = {
                    'trial_idx': i,
                    'nsd_id': nsd_id,
                    'subject': subject_num,
                    'session': session_num,
                    'stimulus': stimulus,
                    'fmri': fmri_volume,
                    'stimulus_shape': stimulus.shape if stimulus is not None else None,
                    'fmri_shape': fmri_volume.shape,
                    'fmri_mean': fmri_volume.mean(),
                    'fmri_std': fmri_volume.std()
                }
                
                samples.append(sample)
                
                print(f"Sample {i}: NSD_ID={nsd_id}")
                print(f"  Stimulus: {sample['stimulus_shape']}")
                print(f"  fMRI: {sample['fmri_shape']}, mean={sample['fmri_mean']:.4f}")
                
            except Exception as e:
                print(f"Error creating sample {i}: {e}")
                continue
        
        return samples

def analyze_data_characteristics(samples):
    """Analyze the characteristics of the loaded data"""
    print(f"\n=== Data Analysis ===")
    
    if not samples:
        print("No samples to analyze")
        return
    
    print(f"Number of samples: {len(samples)}")
    
    # Analyze stimulus images
    stim_shapes = [s['stimulus_shape'] for s in samples if s['stimulus'] is not None]
    if stim_shapes:
        print(f"Stimulus image shapes: {set(stim_shapes)}")
        
        # Show stimulus statistics
        stimuli = [s['stimulus'] for s in samples if s['stimulus'] is not None]
        if stimuli:
            stim_array = np.array(stimuli)
            print(f"Stimulus values range: [{stim_array.min()}, {stim_array.max()}]")
            print(f"Stimulus mean: {stim_array.mean():.2f}")
    
    # Analyze fMRI data
    fmri_shapes = [s['fmri_shape'] for s in samples]
    fmri_means = [s['fmri_mean'] for s in samples]
    fmri_stds = [s['fmri_std'] for s in samples]
    
    print(f"fMRI shapes: {set(fmri_shapes)}")
    print(f"fMRI means: {np.mean(fmri_means):.4f} ± {np.std(fmri_means):.4f}")
    print(f"fMRI stds: {np.mean(fmri_stds):.4f} ± {np.std(fmri_stds):.4f}")

def prepare_for_clip_training(samples):
    """Prepare data in a format suitable for CLIP training"""
    print(f"\n=== Preparing for CLIP Training ===")
    
    prepared_data = []
    
    for sample in samples:
        if sample['stimulus'] is not None:
            # Prepare stimulus (image)
            # CLIP typically expects 224x224 RGB images
            stimulus = sample['stimulus']
            if stimulus.shape[:2] != (224, 224):
                # Resize using PIL
                pil_img = Image.fromarray(stimulus)
                pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
                stimulus_resized = np.array(pil_img)
            else:
                stimulus_resized = stimulus
            
            # Prepare fMRI data
            # You might want to:
            # 1. Flatten the 3D volume
            # 2. Apply ROI masking
            # 3. Normalize
            fmri_flat = sample['fmri'].flatten()
            fmri_normalized = (fmri_flat - fmri_flat.mean()) / fmri_flat.std()
            
            prepared_sample = {
                'nsd_id': sample['nsd_id'],
                'image': stimulus_resized,  # (224, 224, 3)
                'fmri': fmri_normalized,    # Flattened and normalized
                'original_fmri_shape': sample['fmri_shape']
            }
            
            prepared_data.append(prepared_sample)
    
    print(f"Prepared {len(prepared_data)} samples for CLIP training")
    
    if prepared_data:
        example = prepared_data[0]
        print(f"Example prepared sample:")
        print(f"  Image shape: {example['image'].shape}")
        print(f"  fMRI shape: {example['fmri'].shape}")
        print(f"  Original fMRI shape: {example['original_fmri_shape']}")
    
    return prepared_data

def main():
    """Main tutorial function"""
    print("=== NSD Dataset Tutorial ===")
    
    # Initialize data loader
    loader = NSDDataLoader()
    
    # Create some sample pairs
    samples = loader.create_paired_samples(subject_num=1, session_num=1, limit=3)
    
    # Analyze the data
    analyze_data_characteristics(samples)
    
    # Prepare for CLIP training
    prepared_data = prepare_for_clip_training(samples)
    
    print(f"\n=== Summary ===")
    print(f"Successfully loaded {len(samples)} paired samples")
    print(f"Prepared {len(prepared_data)} samples for CLIP training")
    print(f"\nNext steps for your project:")
    print(f"1. Implement CLIP architecture with fMRI encoder")
    print(f"2. Create data loader for training")
    print(f"3. Design loss functions for reconstruction")
    print(f"4. Train the model")
    print(f"5. Evaluate reconstruction quality")
    
    return loader, samples, prepared_data

if __name__ == "__main__":
    loader, samples, prepared_data = main()