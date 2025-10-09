# .gitignore

```
# Python virtual environment
.venv

# Environment variables
.env

# Cache
.cache
```

# configs/data.yaml

```yaml
s3:
  bucket: natural-scenes-dataset
  region: us-east-2
  anon: true

cache:
  # fsspec simplecache: files are streamed & cached locally as needed
  mode: "simplecache" # or "filecache"
  cache_dir: ".cache/nsd"

nsd:
  # You’ll adjust these once you read the Data Manual:
  stimuli_root: "nsddata_stimuli" # example; verify in manual
  bold_root: "nsddata_betas" # example; per-subject/session nifti/mgz
  meta_root: "nsddata" # trial tables, mappings, etc.
  subjects: ["subj01"] # start small
  limit: 100 # for smoke tests

```

# docs/NSD_Dataset_Guide.md

```md
# Natural Scenes Dataset (NSD) - Complete Guide

## Overview

The Natural Scenes Dataset (NSD) is a large-scale fMRI dataset containing brain responses to natural scene images. It's perfect for fMRI-to-image reconstruction tasks using models like CLIP.

### Dataset Statistics

- **Subjects**: 8 participants (subj01 through subj08)
- **Sessions**: ~40 sessions per subject
- **Total Trials**: ~30,000 per subject (750 trials per session)
- **Unique Images**: 73,000 natural scene images from COCO dataset
- **Total Size**: ~300GB (including all preprocessing variants)
- **Access**: Public dataset on AWS S3 (anonymous access)

---

## Directory Structure

\`\`\`
natural-scenes-dataset/
├── nsddata/                    # Metadata and experiment information
│   ├── experiments/nsd/
│   │   ├── nsd_stim_info_merged.csv    # 🔑 KEY: Trial-to-stimulus mapping
│   │   ├── nsd_expdesign.mat           # Experiment design
│   │   └── nsd_designmatrix.csv        # Design matrix
│   ├── bdata/                  # Behavioral data
│   │   └── behavdata/          # Subject behavioral responses
│   ├── freesurfer/            # FreeSurfer anatomical data
│   └── information/           # Dataset documentation
│
├── nsddata_stimuli/           # 🖼️ Stimulus Images
│   └── stimuli/
│       ├── nsd/
│       │   └── nsd_stimuli.hdf5       # 🔑 KEY: All 73k images (36.8GB)
│       ├── nsdimagery/        # Mental imagery stimuli
│       └── nsdsynthetic/      # Synthetic stimuli
│
├── nsddata_betas/             # 🧠 fMRI Data (Preprocessed)
│   └── ppdata/                # Preprocessed data
│       └── subj{01-08}/       # Per-subject data
│           ├── func1mm/       # 1mm resolution
│           ├── func1pt8mm/    # 1.8mm resolution (most common)
│           │   ├── betas_fithrf_GLMdenoise_RR/     # 🔑 RECOMMENDED
│           │   │   ├── betas_session01.nii.gz     # ~467MB per session
│           │   │   ├── betas_session02.nii.gz
│           │   │   └── ...
│           │   └── betas_fithrf/                   # Alternative preprocessing
│           ├── MNI/           # MNI space
│           ├── fsaverage/     # FreeSurfer average
│           └── nativesurface/ # Native surface
│
├── nsddata_timeseries/        # Raw fMRI timeseries
├── nsddata_rawdata/          # Raw scanner data
└── nsddata_other/            # Additional analyses
\`\`\`

---

## Key Files Explained

### 📊 Metadata Files

#### `nsd_stim_info_merged.csv` (11MB)

**Most important file for your project!**

\`\`\`csv
,cocoId,cocoSplit,cropBox,loss,nsdId,flagged,BOLD5000,shared1000,subject1,subject2,...
0,532481,val2017,"(0, 0, 0.1671875, 0.1671875)",0.1,0,False,False,False,0,0,1,0,0,0,0,0
\`\`\`

**Columns:**

- `nsdId`: Unique stimulus ID (0-72999)
- `cocoId`: Original COCO dataset ID
- `cocoSplit`: COCO dataset split (train2017/val2017)
- `subject1-8`: Binary (1=subject saw this stimulus, 0=didn't)
- `subject{X}_rep{0-2}`: Repetition information
- `flagged`: Quality control flag
- `shared1000`: Whether stimulus is in shared set across subjects

**Usage:**

\`\`\`python
import pandas as pd
df = pd.read_csv('nsd_stim_info_merged.csv')

# Get stimuli for subject 1
subj1_stimuli = df[df['subject1'] == 1]
print(f"Subject 1 saw {len(subj1_stimuli)} stimuli")
\`\`\`

### 🖼️ Stimulus Images

#### `nsd_stimuli.hdf5` (36.8GB)

Contains all 73,000 stimulus images in HDF5 format.

**Structure:**

- Dataset: `imgBrick`
- Shape: `(73000, H, W, 3)` where H,W vary per image
- Data type: `uint8` (0-255)
- Format: RGB images

**Usage:**

\`\`\`python
import h5py
import numpy as np
from PIL import Image

with h5py.File('nsd_stimuli.hdf5', 'r') as f:
    # Load specific image
    img_data = f['imgBrick'][nsd_id]  # Shape: (H, W, 3)
    image = Image.fromarray(img_data)
\`\`\`

**Alternative: Use COCO Images Directly**
Since NSD images come from COCO, you can download original COCO images:

\`\`\`python
# Get COCO ID from metadata
coco_id = df.iloc[trial_idx]['cocoId']
coco_split = df.iloc[trial_idx]['cocoSplit']  # train2017 or val2017

# Download from COCO
url = f"http://images.cocodataset.org/{coco_split}/{coco_id:012d}.jpg"
\`\`\`

### 🧠 fMRI Data

#### `betas_session{XX}.nii.gz` (~467MB each)

Preprocessed fMRI beta coefficients from GLM analysis.

**Recommended path:**
`nsddata_betas/ppdata/subj{XX}/func1pt8mm/betas_fithrf_GLMdenoise_RR/`

**File details:**

- **Format**: NIfTI compressed (.nii.gz)
- **Shape**: `(81, 104, 83, ~750)` = (x, y, z, trials)
- **Voxel size**: 1.8mm isotropic
- **Data type**: float32
- **Content**: Beta coefficients (brain activation patterns)

**Alternative preprocessing options:**

- `betas_fithrf/`: Different preprocessing pipeline
- `func1mm/`: Higher resolution (1mm)
- `MNI/`: Normalized to MNI space

**Usage:**

\`\`\`python
import nibabel as nib

# Load session data
nii = nib.load('betas_session01.nii.gz')
data = nii.get_fdata()  # Shape: (81, 104, 83, ~750)

# Extract single trial
trial_0 = data[:, :, :, 0]  # Shape: (81, 104, 83)
\`\`\`

---

## Data Loading Workflow

### Step 1: Set Up Access

\`\`\`python
import fsspec
import pandas as pd
import nibabel as nib
import h5py

# Anonymous S3 access
fs = fsspec.filesystem("s3", anon=True)
bucket = "natural-scenes-dataset"
\`\`\`

### Step 2: Load Metadata

\`\`\`python
# Load stimulus mapping
meta_path = f"{bucket}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
with fs.open(meta_path, 'r') as f:
    stim_df = pd.read_csv(f)

print(f"Total stimulus presentations: {len(stim_df)}")
print(f"Unique images: {stim_df['nsdId'].nunique()}")
\`\`\`

### Step 3: Load fMRI Data

\`\`\`python
# Load session for subject 1
subject = "subj01"
session = 1
fmri_path = f"{bucket}/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{session:02d}.nii.gz"

# Download and cache locally
from src.fmri2img.data.nsd_stream import load_nifti_s3
fmri_data = load_nifti_s3(f"s3://{fmri_path}")
print(f"fMRI shape: {fmri_data.shape}")  # (81, 104, 83, ~750)
\`\`\`

### Step 4: Create Stimulus-fMRI Pairs

\`\`\`python
# Get subject's stimuli
subject_stimuli = stim_df[stim_df['subject1'] == 1]

# Estimate trials per session (simplified)
trials_per_session = len(subject_stimuli) // 40
start_idx = (session - 1) * trials_per_session
session_stimuli = subject_stimuli.iloc[start_idx:start_idx + trials_per_session]

# Create pairs
pairs = []
for trial_idx, (_, row) in enumerate(session_stimuli.iterrows()):
    if trial_idx < fmri_data.shape[3]:
        pairs.append({
            'nsd_id': row['nsdId'],
            'fmri': fmri_data[:, :, :, trial_idx],
            'trial_idx': trial_idx
        })
\`\`\`

---

## Practical Usage for CLIP Project

### Recommended Data Subset for Development

Start small to validate your pipeline:

\`\`\`python
# Recommended starting point
SUBJECTS = ["subj01"]           # Start with one subject
SESSIONS = [1, 2, 3]           # First 3 sessions (~2.25GB)
TOTAL_TRIALS = ~2250           # Manageable for development
\`\`\`

### Memory Requirements

\`\`\`python
# Per trial
fmri_volume = (81, 104, 83)     # = 707,464 voxels
image_size = (224, 224, 3)      # For CLIP input

# Batch processing
batch_size = 32
fmri_batch = 32 * 707464        # ~22M features
memory_per_batch = ~90MB        # Manageable
\`\`\`

### Data Preprocessing Pipeline

\`\`\`python
class NSDDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_df, fmri_sessions, transform=None):
        self.metadata = metadata_df
        self.fmri_data = fmri_sessions
        self.transform = transform

    def __getitem__(self, idx):
        # Get trial info
        trial_info = self.metadata.iloc[idx]
        nsd_id = trial_info['nsdId']

        # Load fMRI data
        session_idx = idx // 750  # Assuming 750 trials per session
        trial_in_session = idx % 750
        fmri_volume = self.fmri_data[session_idx][:, :, :, trial_in_session]

        # Load stimulus image
        stimulus = self.load_stimulus(nsd_id)

        # Apply transforms
        if self.transform:
            stimulus = self.transform(stimulus)
            fmri_volume = self.normalize_fmri(fmri_volume)

        return {
            'fmri': fmri_volume,
            'image': stimulus,
            'nsd_id': nsd_id,
            'metadata': trial_info
        }

    def normalize_fmri(self, fmri_volume):
        # Z-score normalization
        return (fmri_volume - fmri_volume.mean()) / fmri_volume.std()
\`\`\`

---

## Subject Information

| Subject | Total Trials | Sessions | Unique Stimuli | Data Size |
| ------- | ------------ | -------- | -------------- | --------- |
| subj01  | ~10,000      | ~13      | 10,000         | ~6.5GB    |
| subj02  | ~10,000      | ~13      | 10,000         | ~6.5GB    |
| subj03  | ~10,000      | ~13      | 10,000         | ~6.5GB    |
| subj04  | ~10,000      | ~13      | 10,000         | ~6.5GB    |
| subj05  | ~10,000      | ~13      | 10,000         | ~6.5GB    |
| subj06  | ~10,000      | ~13      | 10,000         | ~6.5GB    |
| subj07  | ~10,000      | ~13      | 10,000         | ~6.5GB    |
| subj08  | ~10,000      | ~13      | 10,000         | ~6.5GB    |

**Notes:**

- Each subject viewed 10,000 unique stimuli
- ~1,000 stimuli are shared across all subjects
- Subjects viewed stimuli in different orders
- Some stimuli were repeated for reliability assessment

---

## Data Quality and Preprocessing

### fMRI Preprocessing Pipeline

The beta files you'll use are already preprocessed:

1. **Motion correction**: Head motion artifacts removed
2. **Slice timing correction**: Temporal alignment
3. **GLM analysis**: Trial-by-trial beta coefficients extracted
4. **GLMdenoise**: Advanced noise reduction (recommended version)
5. **Spatial smoothing**: Optional, varies by file type

### Quality Control

- `flagged` column in metadata indicates problematic stimuli
- `R2` files contain explained variance maps
- `ncsnr` files contain noise ceiling estimates

### Coordinate Systems

- **func1pt8mm**: Native functional space (recommended)
- **MNI**: Normalized to standard brain template
- **fsaverage**: FreeSurfer average surface

---

## Getting Started: Download Commands

### Essential Files for Development

\`\`\`bash
# Create data directory
mkdir -p data/nsd

# 1. Download metadata (small file, ~11MB)
wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_stim_info_merged.csv \
     -O data/nsd/nsd_stim_info_merged.csv

# 2. Download sample beta files for testing (~1GB total)
wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz \
     -O data/nsd/betas_session01.nii.gz

wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session02.nii.gz \
     -O data/nsd/betas_session02.nii.gz

# 3. Download stimulus HDF5 (optional, large file ~37GB)
# Only download if you need all images locally
# wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
#      -O data/nsd/nsd_stimuli.hdf5
\`\`\`

---

## CLIP Integration Strategy

### Architecture Overview

\`\`\`
fMRI Volume (81×104×83) → fMRI Encoder → Embedding (512D)
                                            ↓
                                      Contrastive Loss
                                            ↓
Image (224×224×3) → CLIP Vision Encoder → Embedding (512D)
\`\`\`

### Implementation Steps

1. **Data Preparation**

   \`\`\`python
   # Normalize fMRI data
   fmri_normalized = (fmri - fmri.mean()) / fmri.std()

   # Prepare images for CLIP
   image_resized = transforms.Resize((224, 224))(image)
   image_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)
   \`\`\`

2. **Model Architecture**

   \`\`\`python
   class fMRIToImageCLIP(nn.Module):
       def __init__(self):
           super().__init__()
           self.fmri_encoder = nn.Sequential(
               nn.Linear(707464, 4096),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(4096, 2048),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(2048, 512)
           )
           self.clip_model, _ = clip.load("ViT-B/32")

       def forward(self, fmri, images):
           fmri_features = self.fmri_encoder(fmri.flatten(1))
           image_features = self.clip_model.encode_image(images)
           return fmri_features, image_features
   \`\`\`

3. **Training Loop**

   \`\`\`python
   # Contrastive loss between fMRI and image embeddings
   loss_fn = nn.CosineEmbeddingLoss()

   for batch in dataloader:
       fmri_emb, img_emb = model(batch['fmri'], batch['image'])
       loss = contrastive_loss(fmri_emb, img_emb)
       loss.backward()
   \`\`\`

---

## Tips and Best Practices

### 🚀 Start Small

- Begin with 1 subject, 2-3 sessions
- Use synthetic images initially to test pipeline
- Validate data loading before building complex models

### 💾 Memory Management

- Cache large files locally (HDF5, beta files)
- Process data in batches
- Use data loaders with num_workers for parallel loading

### 🔧 Debugging

- Check data shapes and value ranges
- Visualize fMRI volumes and images
- Start with simple baselines (linear regression)

### 📊 Evaluation

- Use shared stimuli across subjects for evaluation
- Implement semantic similarity metrics
- Compare reconstructions to original images

---

## Additional Resources

### Papers

- [Original NSD Paper](https://www.nature.com/articles/s41593-021-00962-x)
- [Dataset Documentation](https://cvnlab.slite.page/p/NKuOB0jF3y/Natural-Scenes-Dataset)

### Code Examples

- [Official NSD Code](https://github.com/cvnlab/nsd)
- [Analysis Examples](https://github.com/cvnlab/nsddatapaper)

### Contact

- For questions about the dataset: contact the Allen Institute
- For technical issues: check the GitHub repositories

---

This guide provides everything you need to start working with the NSD dataset for your fMRI-to-image reconstruction project. Begin with the recommended subset and gradually scale up as you validate your approach!

```

# Makefile

```
PY=uv run
export AWS_REGION=us-east-2

.PHONY: setup nsd-manifest sanity

setup:
\tuv sync

nsd-manifest:
\t$(PY) python scripts/nsd_build_manifest.py

sanity:
\t$(PY) python scripts/nsd_sanity_check.py --limit 3

```

# pyproject.toml

```toml

```

# README.md

```md

```

# requirements.txt

```txt
awscli
```

# src/fmri2img/__init__.py

```py

```

# src/fmri2img/io/nsd_layout.py

```py

```

# src/fmri2img/io/s3.py

```py
from __future__ import annotations
import fsspec
from typing import Iterable, List

def s3_ls(url: str, anon: bool = True) -> List[str]:
    fs = fsspec.filesystem("s3", anon=anon)
    return [f"s3://{p}" for p in fs.glob(url)]

```

# src/fmri2img/scripts/explore_nsd_streaming.py

```py
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
```

# src/fmri2img/scripts/nsd_build_manifest.py

```py
import json, os, argparse, fsspec
from tqdm import tqdm
from fmri2img.io.s3 import s3_ls

def main(prefix="s3://natural-scenes-dataset/nsddata", out="manifests/nsd.jsonl", anon=True, limit=None):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Example: list a subset (you'll refine using the Data Manual)
    # Find some stimuli (COCO/NSD selection) and matching betas for subj01:
    stim_keys = s3_ls("natural-scenes-dataset/nsddata_stimuli/**", anon=anon)
    beta_keys = s3_ls("natural-scenes-dataset/nsddata_betas/ppdata/subj01/**", anon=anon)

    # TODO: read the official trial tables from nsddata/ to align (image_id, run, trial)
    # Here we just pair first N for smoke-test:
    pairs = list(zip(sorted(beta_keys), sorted(stim_keys)))
    if limit: pairs = pairs[:limit]

    with open(out, "w") as f:
        for i, (beta, stim) in enumerate(tqdm(pairs, desc="manifest")):
            rec = {
                "trial_id": i,
                "bold": f"s3://{beta}" if not beta.startswith("s3://") else beta,
                "stim": f"s3://{stim}" if not stim.startswith("s3://") else stim,
                "subject": "subj01",
            }
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()

```

# src/fmri2img/scripts/nsd_complete_tutorial.py

```py
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
```

# src/fmri2img/scripts/nsd_data_tutorial.py

```py
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
```

# src/fmri2img/scripts/nsd_sanity_check.py

```py
import argparse, itertools
from fmri2img.data.nsd_stream import stream_samples

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="manifests/nsd.jsonl")
parser.add_argument("--limit", type=int, default=5)
args = parser.parse_args()

for sample in itertools.islice(stream_samples(args.manifest), args.limit):
    print(sample["trial_id"], sample["subject"], sample["image"].size, sample["bold"].shape)

```

# src/fmri2img/scripts/nsd_tutorial_simple.py

```py
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
\`\`\`
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
\`\`\`
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
```

# src/fmri2img/scripts/nsd_working_example.py

```py
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
\`\`\`bash
# Create data directory
mkdir -p data/nsd

# Download metadata (small file)
wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_stim_info_merged.csv \\
     -O data/nsd/nsd_stim_info_merged.csv

# Download a few beta files for testing
wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz \\
     -O data/nsd/betas_session01.nii.gz
\`\`\`

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
   \`\`\`python
   class NSDDataset(torch.utils.data.Dataset):
       def __init__(self, metadata_path, beta_files, transform=None):
           # Load metadata and beta files
           pass
       
       def __getitem__(self, idx):
           # Return (fmri_data, stimulus_image, metadata)
           pass
   \`\`\`

2. fMRI Encoder:
   \`\`\`python
   class fMRIEncoder(nn.Module):
       def __init__(self, input_dim=707464, hidden_dims=[4096, 2048, 1024], output_dim=512):
           # MLP or 3D CNN for fMRI data
           pass
   \`\`\`

3. CLIP Integration:
   \`\`\`python
   class fMRIToImageCLIP(nn.Module):
       def __init__(self):
           self.fmri_encoder = fMRIEncoder()
           self.clip_model = clip.load("ViT-B/32")
           self.projection = nn.Linear(512, 512)
   \`\`\`

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
```

# src/fmri2img/scripts/quick_check_nsd.py'

```py'
# quick_check_nsd.py
import fsspec

fs = fsspec.filesystem("s3", anon=True)
print(fs.ls("natural-scenes-dataset"))  # top-level keys
print(fs.glob("natural-scenes-dataset/nsddata_stimuli/**")[:20])  # sample

```

# src/fmri2img/utils/cache.py

```py
from __future__ import annotations
import fsspec
from typing import Optional

def make_fs(anon: bool = True) -> fsspec.AbstractFileSystem:
    return fsspec.filesystem("s3", anon=anon)

def cached_url(s3_url: str, cache_dir: str, mode: str = "simplecache") -> str:
    """
    Wrap an S3 URL so reads stream and cache locally on first access.
    - simplecache::s3://bucket/key -> saves full file once fetched
    - filecache::s3://bucket/key   -> chunked, similar behavior
    """
    prefix = f"{mode}::{s3_url}"
    # For simplecache, you can set target cache dir via fsspec.open kwarg
    return prefix

```

# src/fmri2img/utils/logging.py

```py

```

