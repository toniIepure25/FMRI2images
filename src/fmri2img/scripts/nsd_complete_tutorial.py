#!/usr/bin/env python3
"""
NSD Dataset Complete Tutorial: Understanding and Loading fMRI-Image Pairs

This tutorial demonstrates:
1. The actual NSD dataset structure
2. How to load stimulus images from HDF5
3. How to load fMRI beta coefficients
4. How to create paired training data for CLIP reconstruction
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fsspec
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path
from data.nsd_stream import load_nifti_s3

class NSDDataExplorer:
    def __init__(self):
        self.fs = fsspec.filesystem("s3", anon=True)
        self.bucket = "natural-scenes-dataset"
        self.cache_dir = Path(".cache/nsd")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.stim_df = self._load_stimulus_metadata()
    
    def _load_stimulus_metadata(self):
        """Load stimulus metadata"""
        print("Loading stimulus metadata...")
        meta_path = f"{self.bucket}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
        
        with self.fs.open(meta_path, 'r') as f:
            df = pd.read_csv(f)
        
        print(f"Loaded metadata for {len(df)} stimulus presentations")
        return df
    
    def explain_dataset_structure(self):
        """Explain the complete NSD dataset structure"""
        print("=" * 60)
        print("NSD DATASET STRUCTURE FOR fMRI-TO-IMAGE RECONSTRUCTION")
        print("=" * 60)
        
        print("""
The Natural Scenes Dataset (NSD) contains:

🧠 SUBJECTS: 8 participants (subj01 through subj08)
📊 SESSIONS: ~40 sessions per subject (~30,000 trials total per subject)
🖼️  STIMULI: 73,000 unique natural scene images from COCO dataset
🧮 fMRI: Preprocessed beta coefficients (brain activation patterns)

DIRECTORY STRUCTURE:
├── nsddata_stimuli/stimuli/nsd/
│   └── nsd_stimuli.hdf5          # All stimulus images in HDF5 format
├── nsddata_betas/ppdata/subj{XX}/func1pt8mm/
│   ├── betas_fithrf_GLMdenoise_RR/
│   │   ├── betas_session01.nii.gz    # fMRI betas (~750 trials per session)
│   │   ├── betas_session02.nii.gz
│   │   └── ...
│   └── betas_fithrf/
│       ├── betas_session01.nii.gz    # Alternative preprocessing
│       └── ...
└── nsddata/experiments/nsd/
    └── nsd_stim_info_merged.csv      # Trial-to-stimulus mapping

KEY FILES FOR YOUR PROJECT:
1. Stimuli: nsd_stimuli.hdf5 contains all 73k images
2. fMRI: betas_session{XX}.nii.gz contains brain responses  
3. Mapping: nsd_stim_info_merged.csv links trials to stimuli
""")
    
    def load_stimulus_from_hdf5(self, nsd_id, show_info=True):
        """Load a stimulus image from the HDF5 file"""
        if show_info:
            print(f"\nLoading stimulus NSD ID: {nsd_id}")
        
        # Download HDF5 file to cache (this is large ~39GB, so we'll cache it)
        hdf5_path = f"{self.bucket}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
        cached_hdf5 = f"simplecache::s3://{hdf5_path}"
        
        try:
            with fsspec.open(cached_hdf5, mode="rb", anon=True, 
                           cache_storage=str(self.cache_dir)) as f:
                local_path = f.name
            
            # Open HDF5 file and extract image
            with h5py.File(local_path, 'r') as hf:
                if show_info:
                    print(f"HDF5 keys: {list(hf.keys())}")
                
                # Find the correct dataset name
                if 'imgBrick' in hf:
                    img_data = hf['imgBrick'][nsd_id]  # Shape: (H, W, 3)
                elif 'images' in hf:
                    img_data = hf['images'][nsd_id]
                else:
                    # Explore the structure
                    print(f"Available datasets: {list(hf.keys())}")
                    for key in hf.keys():
                        print(f"  {key}: {hf[key].shape if hasattr(hf[key], 'shape') else 'group'}")
                    return None
                
                if show_info:
                    print(f"Image shape: {img_data.shape}")
                    print(f"Image dtype: {img_data.dtype}")
                    print(f"Value range: [{img_data.min()}, {img_data.max()}]")
                
                # Convert to PIL Image
                if img_data.dtype != np.uint8:
                    img_data = (img_data * 255).astype(np.uint8)
                
                image = Image.fromarray(img_data)
                return image
                
        except Exception as e:
            print(f"Error loading stimulus {nsd_id}: {e}")
            return None
    
    def load_fmri_betas(self, subject_num=1, session_num=1, preprocessing="GLMdenoise_RR"):
        """Load fMRI beta coefficients for a specific subject and session"""
        print(f"\nLoading fMRI data: subj{subject_num:02d}, session {session_num}")
        
        subject_id = f"subj{subject_num:02d}"
        
        # Choose preprocessing pipeline
        if preprocessing == "GLMdenoise_RR":
            preproc_dir = "betas_fithrf_GLMdenoise_RR"
        else:
            preproc_dir = "betas_fithrf"
        
        # Construct file path
        fmri_path = (f"{self.bucket}/nsddata_betas/ppdata/{subject_id}/func1pt8mm/"
                    f"{preproc_dir}/betas_session{session_num:02d}.nii.gz")
        
        print(f"Loading: {fmri_path}")
        
        try:
            # Use the working load_nifti_s3 function
            full_url = f"s3://{fmri_path}"
            fmri_data = load_nifti_s3(full_url, cache_dir=str(self.cache_dir))
            
            print(f"fMRI data shape: {fmri_data.shape}")
            print(f"Data type: {fmri_data.dtype}")
            print(f"Value range: [{fmri_data.min():.4f}, {fmri_data.max():.4f}]")
            
            if len(fmri_data.shape) == 4:
                print(f"Number of trials in session: {fmri_data.shape[3]}")
            
            return fmri_data
            
        except Exception as e:
            print(f"Error loading fMRI data: {e}")
            return None
    
    def get_trial_mapping(self, subject_num=1, session_num=1):
        """Get the mapping between fMRI trial indices and stimulus IDs"""
        print(f"\nGetting trial mapping for subject {subject_num}, session {session_num}")
        
        # Filter metadata for this subject
        subject_col = f"subject{subject_num}"
        subject_trials = self.stim_df[self.stim_df[subject_col] == 1].copy()
        
        print(f"Subject {subject_num} viewed {len(subject_trials)} stimuli total")
        
        # For a rough approximation, divide trials across 40 sessions
        # Note: This is simplified - actual trial order would need experiment files
        trials_per_session = len(subject_trials) // 40
        
        start_idx = (session_num - 1) * trials_per_session
        end_idx = start_idx + trials_per_session
        
        session_mapping = subject_trials.iloc[start_idx:end_idx].copy()
        session_mapping['trial_in_session'] = range(len(session_mapping))
        
        print(f"Estimated {len(session_mapping)} trials in session {session_num}")
        
        return session_mapping
    
    def create_training_samples(self, subject_num=1, session_num=1, num_samples=3):
        """Create paired stimulus-fMRI samples for training"""
        print(f"\n{'='*50}")
        print(f"CREATING TRAINING SAMPLES")
        print(f"{'='*50}")
        
        # Load fMRI data
        fmri_data = self.load_fmri_betas(subject_num, session_num)
        if fmri_data is None:
            return []
        
        # Get trial mapping
        trial_mapping = self.get_trial_mapping(subject_num, session_num)
        
        samples = []
        
        # Create samples
        max_samples = min(num_samples, len(trial_mapping), 
                         fmri_data.shape[3] if len(fmri_data.shape) == 4 else 1)
        
        print(f"\nCreating {max_samples} training samples...")
        
        for i in range(max_samples):
            try:
                # Get stimulus info
                trial_info = trial_mapping.iloc[i]
                nsd_id = trial_info['nsdId']
                
                print(f"\nSample {i+1}:")
                print(f"  Trial index: {i}")
                print(f"  NSD ID: {nsd_id}")
                
                # Load stimulus (showing minimal info)
                stimulus = self.load_stimulus_from_hdf5(nsd_id, show_info=False)
                
                # Extract fMRI volume
                if len(fmri_data.shape) == 4:
                    fmri_volume = fmri_data[:, :, :, i]
                else:
                    fmri_volume = fmri_data
                
                sample = {
                    'trial_idx': i,
                    'nsd_id': nsd_id,
                    'subject': subject_num,
                    'session': session_num,
                    'stimulus': stimulus,
                    'fmri': fmri_volume,
                    'stimulus_shape': stimulus.size if stimulus else None,
                    'fmri_shape': fmri_volume.shape,
                    'fmri_mean': fmri_volume.mean(),
                    'fmri_std': fmri_volume.std()
                }
                
                samples.append(sample)
                
                print(f"  Stimulus: {sample['stimulus_shape']}")
                print(f"  fMRI: {sample['fmri_shape']}")
                print(f"  fMRI stats: mean={sample['fmri_mean']:.4f}, std={sample['fmri_std']:.4f}")
                
            except Exception as e:
                print(f"  Error creating sample {i}: {e}")
                continue
        
        return samples
    
    def prepare_for_clip_training(self, samples):
        """Prepare samples for CLIP-based training"""
        print(f"\n{'='*50}")
        print(f"PREPARING DATA FOR CLIP TRAINING")
        print(f"{'='*50}")
        
        prepared_samples = []
        
        for i, sample in enumerate(samples):
            if sample['stimulus'] is None:
                continue
                
            try:
                # Prepare image for CLIP (224x224 RGB)
                stimulus = sample['stimulus']
                if stimulus.mode != 'RGB':
                    stimulus = stimulus.convert('RGB')
                
                # Resize to CLIP input size
                stimulus_resized = stimulus.resize((224, 224), Image.Resampling.LANCZOS)
                stimulus_array = np.array(stimulus_resized)
                
                # Prepare fMRI data
                fmri_volume = sample['fmri']
                
                # Option 1: Flatten the 3D volume
                fmri_flattened = fmri_volume.flatten()
                
                # Option 2: Normalize (z-score)
                fmri_normalized = (fmri_flattened - fmri_flattened.mean()) / fmri_flattened.std()
                
                prepared_sample = {
                    'nsd_id': sample['nsd_id'],
                    'subject': sample['subject'],
                    'session': sample['session'],
                    'trial_idx': sample['trial_idx'],
                    
                    # Image data (ready for CLIP vision encoder)
                    'image': stimulus_array,  # Shape: (224, 224, 3)
                    'image_tensor_ready': stimulus_array.transpose(2, 0, 1),  # Shape: (3, 224, 224)
                    
                    # fMRI data (various formats for experimentation)
                    'fmri_raw': fmri_volume,           # Original 3D shape
                    'fmri_flattened': fmri_flattened,  # Flattened 1D
                    'fmri_normalized': fmri_normalized, # Normalized 1D
                    
                    # Metadata
                    'original_fmri_shape': fmri_volume.shape,
                    'original_image_size': stimulus.size
                }
                
                prepared_samples.append(prepared_sample)
                
                print(f"Sample {i+1} prepared:")
                print(f"  Image: {prepared_sample['image'].shape} (values: {prepared_sample['image'].min()}-{prepared_sample['image'].max()})")
                print(f"  fMRI flattened: {prepared_sample['fmri_flattened'].shape}")
                print(f"  fMRI normalized: {prepared_sample['fmri_normalized'].shape} (mean: {prepared_sample['fmri_normalized'].mean():.4f})")
                
            except Exception as e:
                print(f"Error preparing sample {i}: {e}")
                continue
        
        return prepared_samples
    
    def demonstrate_clip_architecture(self):
        """Show how to structure the CLIP model for this data"""
        print(f"\n{'='*50}")
        print(f"CLIP ARCHITECTURE FOR fMRI-TO-IMAGE RECONSTRUCTION")
        print(f"{'='*50}")
        
        print("""
ARCHITECTURE OVERVIEW:

Input: fMRI volume (81, 104, 83) → Flattened: (707,464 features)
Output: Reconstructed image (224, 224, 3)

PROPOSED ARCHITECTURE:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   fMRI Encoder  │    │  Latent Space    │    │ Image Decoder   │
│                 │    │                  │    │                 │
│ Input: 707,464  │───▶│   Embedding      │───▶│ Output: 224×224 │
│ Hidden: [4096,  │    │   Dimension      │    │                 │
│         2048,   │    │   (e.g., 512)    │    │ CNN Transpose   │
│         1024]   │    │                  │    │ or Transformer  │
│ Output: 512     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘

TRAINING STRATEGY:
1. Contrastive Learning: Match fMRI embeddings to CLIP image embeddings
2. Reconstruction Loss: Generate images similar to original stimuli
3. Multi-scale Loss: Semantic + pixel-level similarity

IMPLEMENTATION STEPS:
1. Create fMRI encoder (MLP or 3D CNN)
2. Use pre-trained CLIP vision encoder for image embeddings
3. Train alignment between fMRI and image embeddings
4. Add image generation/reconstruction head
5. Fine-tune end-to-end

DATA PREPARATION:
- Normalize fMRI data per subject (z-score)
- Augment images (rotation, scaling, color jitter)
- Create train/val/test splits
- Batch similar sessions together
""")

def main():
    """Main tutorial function"""
    print("NSD Dataset Tutorial for fMRI-to-Image Reconstruction with CLIP")
    print("=" * 70)
    
    # Initialize explorer
    explorer = NSDDataExplorer()
    
    # 1. Explain dataset structure
    explorer.explain_dataset_structure()
    
    # 2. Try to load a sample stimulus (this will likely fail due to HDF5 size)
    print(f"\n{'='*50}")
    print("ATTEMPTING TO LOAD SAMPLE DATA")
    print("=" * 50)
    print("Note: Loading from HDF5 requires downloading 39GB file first...")
    
    # For demo, we'll just show the structure without actually loading
    # In practice, you'd want to download and cache the HDF5 file locally
    
    # 3. Load fMRI data (this should work)
    fmri_data = explorer.load_fmri_betas(subject_num=1, session_num=1)
    
    # 4. Show trial mapping
    trial_mapping = explorer.get_trial_mapping(subject_num=1, session_num=1)
    
    # 5. Show CLIP architecture
    explorer.demonstrate_clip_architecture()
    
    print(f"\n{'='*70}")
    print("TUTORIAL SUMMARY")
    print("=" * 70)
    
    print("""
✅ WHAT YOU'VE LEARNED:

1. NSD Dataset Structure:
   - 8 subjects, ~40 sessions each, 73k unique images
   - Stimuli in HDF5 format (39GB file)
   - fMRI betas in NIfTI format (~500MB per session)
   - Metadata CSV maps trials to stimuli

2. Data Loading Strategy:
   - Cache large files locally for efficiency
   - Use fsspec for anonymous S3 access
   - Process data in batches to manage memory

3. CLIP Integration Plan:
   - fMRI encoder: Process brain data to embeddings
   - Image decoder: Generate images from embeddings
   - Contrastive + reconstruction training

🚀 NEXT STEPS FOR YOUR PROJECT:

1. Download key files locally:
   - nsd_stimuli.hdf5 (39GB)
   - Several sessions of beta files (~2GB each)
   - Stimulus metadata CSV

2. Implement data pipeline:
   - PyTorch Dataset class
   - Efficient data loading with caching
   - Data augmentation and normalization

3. Build CLIP model:
   - fMRI encoder architecture
   - Integration with CLIP vision encoder
   - Training loop with multiple loss functions

4. Start with small experiments:
   - Single subject, few sessions
   - Validate data loading pipeline
   - Test model components separately

The foundation is now in place - you understand the data structure
and can begin implementing your CLIP-based reconstruction model!
""")

if __name__ == "__main__":
    main()