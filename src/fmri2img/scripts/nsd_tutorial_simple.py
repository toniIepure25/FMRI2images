#!/usr/bin/env python3
"""
Simple NSD Dataset Explorer using the working streaming functions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.nsd_stream import load_image_s3, load_nifti_s3, open_anon
import fsspec
import pandas as pd
import numpy as np
from PIL import Image

def explore_nsd_basic():
    """Basic exploration of NSD dataset structure"""
    print("=== NSD Dataset Structure ===")
    
    # Test anonymous S3 access
    fs = fsspec.filesystem("s3", anon=True)
    
    # 1. Show top-level structure
    print("\n1. Top-level directories:")
    try:
        top_dirs = fs.ls("natural-scenes-dataset")
        for d in sorted(top_dirs):
            name = d.split('/')[-1]
            print(f"   - {name}")
    except Exception as e:
        print(f"Error accessing top-level: {e}")
        return
    
    # 2. Show stimulus structure
    print("\n2. Stimulus structure:")
    try:
        stim_dirs = fs.ls("natural-scenes-dataset/nsddata_stimuli/stimuli")
        for d in sorted(stim_dirs):
            name = d.split('/')[-1]
            print(f"   - {name}")
            
            # Show some files in nsd directory
            if name == "nsd":
                nsd_files = fs.ls(d)[:5]
                for f in nsd_files:
                    fname = f.split('/')[-1]
                    fsize = fs.info(f).get('size', 'unknown')
                    print(f"     └── {fname} ({fsize} bytes)")
    except Exception as e:
        print(f"Error accessing stimuli: {e}")
    
    # 3. Show subjects
    print("\n3. Available subjects:")
    try:
        subj_dirs = fs.ls("natural-scenes-dataset/nsddata_betas/ppdata")
        subjects = [d.split('/')[-1] for d in sorted(subj_dirs)]
        print(f"   Subjects: {subjects}")
        
        # Show sessions for first subject
        if subjects:
            subj1_sessions = fs.ls(f"natural-scenes-dataset/nsddata_betas/ppdata/{subjects[0]}")
            sessions = [s.split('/')[-1] for s in sorted(subj1_sessions)[:5]]
            print(f"   {subjects[0]} sessions (first 5): {sessions}")
    except Exception as e:
        print(f"Error accessing subjects: {e}")

def load_sample_data():
    """Load sample stimulus and demonstrate the data format"""
    print("\n=== Loading Sample Data ===")
    
    # Load a sample stimulus image
    print("\n4. Loading sample stimulus:")
    try:
        # Use the working function from nsd_stream.py
        sample_url = "s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd00000.png"
        print(f"Loading: {sample_url}")
        
        image = load_image_s3(sample_url)
        print(f"   Image size: {image.size}")
        print(f"   Image mode: {image.mode}")
        
        # Convert to numpy
        img_array = np.array(image)
        print(f"   Array shape: {img_array.shape}")
        print(f"   Array dtype: {img_array.dtype}")
        print(f"   Value range: [{img_array.min()}, {img_array.max()}]")
        
    except Exception as e:
        print(f"   Error loading stimulus: {e}")
    
    # Try to load stimulus metadata
    print("\n5. Loading stimulus metadata:")
    try:
        fs = fsspec.filesystem("s3", anon=True)
        meta_url = "natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
        
        with fs.open(meta_url, 'r') as f:
            # Read just the first few lines to see structure
            lines = []
            for i, line in enumerate(f):
                lines.append(line.strip())
                if i >= 5:  # First 6 lines (header + 5 data rows)
                    break
        
        print("   CSV structure (first 6 lines):")
        for i, line in enumerate(lines):
            print(f"   {i}: {line[:100]}..." if len(line) > 100 else f"   {i}: {line}")
            
    except Exception as e:
        print(f"   Error loading metadata: {e}")

def demonstrate_data_pairing():
    """Demonstrate how stimulus and fMRI data are paired"""
    print("\n=== Data Pairing Logic ===")
    
    print("""
The NSD dataset pairs stimulus images with fMRI responses as follows:

1. **Stimulus Images**: 
   - Location: s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/
   - Format: nsd{ID:05d}.png (e.g., nsd00000.png, nsd00001.png)
   - 73,000 unique natural scene images from COCO dataset
   - Size: 722×722 pixels, RGBA format

2. **fMRI Data**:
   - Location: s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{XX}/
   - Format: betas_session{YY}.nii.gz 
   - Preprocessed beta coefficients from GLM analysis
   - Shape: typically (81, 104, 83, ~750) = (x, y, z, trials_per_session)

3. **Trial Mapping**:
   - File: nsd_stim_info_merged.csv
   - Maps each trial to its stimulus ID (nsdId)
   - Shows which subjects saw which stimuli
   - Includes repetition information

4. **For Your CLIP Project**:
   - Input: fMRI volume (3D brain activation)
   - Output: Reconstructed image
   - Training pairs: (fMRI_volume[i], stimulus_image[nsdId])
""")

def next_steps_for_clip():
    """Outline next steps for CLIP-based reconstruction"""
    print("\n=== Next Steps for Your CLIP Project ===")
    
    print("""
Step 1: Data Preprocessing
- Normalize fMRI data (z-score within subjects)
- Resize images to 224×224 for CLIP
- Create train/validation splits
- Implement efficient data loading

Step 2: Architecture Design
- fMRI Encoder: 3D CNN or flatten + MLP
- Image Encoder: Pre-trained CLIP vision encoder
- Projection heads to common embedding space
- Decoder: Generate images from fMRI embeddings

Step 3: Training Strategy
- Contrastive learning between fMRI and images
- Multi-modal alignment loss
- Reconstruction loss (if generating images)
- Progressive training (alignment → generation)

Step 4: Evaluation
- Reconstruction quality metrics
- Semantic similarity measures
- Subject-specific vs. cross-subject performance

Code Structure:
```
src/fmri2img/
├── models/
│   ├── fmri_encoder.py      # 3D CNN for fMRI
│   ├── clip_model.py        # Modified CLIP
│   └── decoder.py           # Image generation
├── data/
│   ├── nsd_dataset.py       # PyTorch Dataset
│   └── preprocessing.py     # Data normalization
├── training/
│   ├── trainer.py           # Training loop
│   └── losses.py            # Custom loss functions
└── evaluation/
    └── metrics.py           # Evaluation metrics
```
""")

def main():
    """Main function to run the exploration"""
    print("NSD Dataset Tutorial for fMRI-to-Image Reconstruction")
    print("=" * 60)
    
    # Basic exploration
    explore_nsd_basic()
    
    # Load sample data
    load_sample_data()
    
    # Explain data pairing
    demonstrate_data_pairing()
    
    # Next steps
    next_steps_for_clip()
    
    print("\n" + "=" * 60)
    print("Tutorial completed! You now understand the NSD dataset structure.")
    print("Ready to start implementing your CLIP-based reconstruction model!")

if __name__ == "__main__":
    main()