#!/usr/bin/env python3
"""
Working NSD Data Example - Demonstrates actual data loading
"""

import fsspec
import pandas as pd
import numpy as np
from pathlib import Path

def working_data_example():
    """Demonstrate working data access patterns"""
    print("NSD Dataset - Working Data Access Example")
    print("=" * 50)
    
    fs = fsspec.filesystem("s3", anon=True)
    
    # 1. Load and examine stimulus metadata (this works)
    print("\n1. Loading stimulus metadata...")
    meta_path = "natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    
    with fs.open(meta_path, 'r') as f:
        stim_df = pd.read_csv(f)
    
    print(f"✅ Loaded metadata: {stim_df.shape}")
    print(f"Columns: {list(stim_df.columns[:10])}...")  # First 10 columns
    
    # Show sample data
    print("\nSample stimulus mapping:")
    sample_data = stim_df[['nsdId', 'cocoId', 'subject1', 'subject2', 'subject3']].head()
    print(sample_data)
    
    # 2. Analyze which subjects saw which stimuli
    print(f"\n2. Subject participation analysis...")
    subject_cols = [col for col in stim_df.columns if col.startswith('subject') and not 'rep' in col]
    
    for subj in subject_cols[:3]:  # First 3 subjects
        count = stim_df[stim_df[subj] == 1].shape[0]
        print(f"  {subj}: viewed {count} stimuli")
    
    # 3. Show file sizes and accessibility
    print(f"\n3. File accessibility check...")
    
    # Check HDF5 stimulus file
    stim_file = "natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    stim_info = fs.info(stim_file)
    stim_size_gb = stim_info['size'] / (1024**3)
    print(f"  Stimulus HDF5: {stim_size_gb:.1f} GB")
    
    # Check beta files for subj01
    beta_files = fs.glob("natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session*.nii.gz")
    print(f"  Beta files for subj01: {len(beta_files)} sessions")
    
    # Show size of a few beta files
    for i, beta_file in enumerate(beta_files[:3]):
        beta_info = fs.info(beta_file)
        beta_size_mb = beta_info['size'] / (1024**2)
        session_num = beta_file.split('session')[1].split('.')[0]
        print(f"    Session {session_num}: {beta_size_mb:.1f} MB")
    
    # 4. Create a practical data access plan
    print(f"\n4. Practical data access strategy...")
    
    return stim_df, beta_files

def create_development_plan(stim_df, beta_files):
    """Create a practical development plan"""
    print(f"\n{'='*60}")
    print("PRACTICAL DEVELOPMENT PLAN FOR YOUR CLIP PROJECT")
    print("=" * 60)
    
    print("""
PHASE 1: DATA PREPARATION (Start Here!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: Get a small working dataset to validate your pipeline

Steps:
1. Download stimulus metadata CSV (already working ✅)
2. Download 1-2 beta files for subj01 (~1GB total)
3. Create a small subset of stimulus-fMRI pairs
4. Implement basic data loading pipeline

Commands to get started:
```bash
# Create data directory
mkdir -p data/nsd

# Download metadata (small file)
wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_stim_info_merged.csv \\
     -O data/nsd/nsd_stim_info_merged.csv

# Download a few beta files for testing
wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz \\
     -O data/nsd/betas_session01.nii.gz
```

PHASE 2: STIMULUS HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Since the HDF5 file is 39GB, consider these alternatives:

Option A: Download HDF5 locally (best for full dataset)
- Download nsd_stimuli.hdf5 once
- Access images efficiently with h5py

Option B: Use COCO images directly (faster start)
- Get COCO IDs from metadata
- Download original COCO images
- May have slight preprocessing differences

Option C: Synthetic testing data
- Generate random images for initial development
- Focus on fMRI processing pipeline first
- Replace with real stimuli later

PHASE 3: MODEL DEVELOPMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture components to implement:

1. Data Loading:
   ```python
   class NSDDataset(torch.utils.data.Dataset):
       def __init__(self, metadata_path, beta_files, transform=None):
           # Load metadata and beta files
           pass
       
       def __getitem__(self, idx):
           # Return (fmri_data, stimulus_image, metadata)
           pass
   ```

2. fMRI Encoder:
   ```python
   class fMRIEncoder(nn.Module):
       def __init__(self, input_dim=707464, hidden_dims=[4096, 2048, 1024], output_dim=512):
           # MLP or 3D CNN for fMRI data
           pass
   ```

3. CLIP Integration:
   ```python
   class fMRIToImageCLIP(nn.Module):
       def __init__(self):
           self.fmri_encoder = fMRIEncoder()
           self.clip_model = clip.load("ViT-B/32")
           self.projection = nn.Linear(512, 512)
   ```

PHASE 4: TRAINING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training strategy:
1. Contrastive learning (align fMRI embeddings with CLIP image embeddings)
2. Add reconstruction loss
3. Progressive training: alignment → generation

Loss functions:
- Contrastive loss between fMRI and image embeddings
- Reconstruction loss (MSE, perceptual loss)
- Semantic similarity loss
""")
    
    # Calculate practical data sizes
    subj1_trials = stim_df[stim_df['subject1'] == 1].shape[0]
    estimated_sessions = subj1_trials // 750  # ~750 trials per session
    
    print(f"""
PRACTICAL NUMBERS FOR YOUR PROJECT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Subject 1 Statistics:
- Total trials: {subj1_trials:,}
- Estimated sessions: ~{estimated_sessions}
- Data size per session: ~500MB
- Total fMRI data: ~{estimated_sessions * 0.5:.1f}GB

Recommended starting subset:
- Subjects: 1 (start small)
- Sessions: 5-10 sessions (~2.5-5GB)
- Stimuli: ~3,750-7,500 images
- Perfect for initial development and validation!

Memory requirements:
- One fMRI session: ~500MB
- Session shape: typically (81, 104, 83, ~750)
- Flattened per trial: ~707k features
- Batch of 32 trials: ~22M features → manageable
""")

def main():
    """Main function"""
    # Load working data
    stim_df, beta_files = working_data_example()
    
    # Create development plan
    create_development_plan(stim_df, beta_files)
    
    print(f"\n{'='*60}")
    print("SUMMARY AND IMMEDIATE NEXT STEPS")
    print("=" * 60)
    
    print("""
✅ YOU NOW UNDERSTAND:
- NSD dataset structure and file organization
- How to access data via S3 (anonymously)
- Practical data sizes and computational requirements
- CLIP integration strategy for fMRI→image reconstruction

🎯 IMMEDIATE NEXT STEPS:
1. Set up local data directory and download sample files
2. Implement basic data loading (metadata + one beta file)
3. Create simple fMRI encoder (MLP to start)
4. Load pre-trained CLIP model
5. Test data pipeline with synthetic stimuli

🚀 START CODING:
Focus on getting one complete data sample loaded:
- fMRI volume (81, 104, 83) 
- Corresponding stimulus image (224, 224, 3)
- Metadata linking them together

Once this works, scaling up is straightforward!

The path forward is clear - start with small data and build up! 🧠→🖼️
""")

if __name__ == "__main__":
    main()