# Natural Scenes Dataset (NSD) - Complete Guide

## Overview

The Natural Scenes Dataset (NSD) is a large-scale fMRI dataset containing brain responses to natural scene images. It's perfect for fMRI-to-image reconstruction tasks using models like CLIP.

### Dataset Statistics

> **⚠️ CRITICAL WARNING: Session Trial Counts Vary**  
> Trial counts per session are **NOT** fixed at 750! Each session has variable trial counts depending on experimental design. **Never use fixed counts for stimulus-fMRI pairing**. Always use the canonical index builder to get exact trial mappings from session design files.

- **Subjects**: 8 participants (subj01 through subj08)
- **Sessions**: ~40 sessions per subject
- **Total Trials**: ~30,000 per subject (~750 trials per session, but varies!)
- **Unique Images**: 73,000 natural scene images from COCO dataset
- **Total Size**: ~300GB (including all preprocessing variants)
- **Access**: Public dataset on AWS S3 (anonymous access)

---

## Directory Structure

```
natural-scenes-dataset/
├── nsddata/                    # Metadata and experiment information
│   ├── experiments/nsd/
│   │   ├── nsd_stim_info_merged.csv    # Stimulus catalog; join by nsdId. Trial order from per-subject session design files.
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
```

---

## Key Files Explained

### 📊 Metadata Files

#### `nsd_stim_info_merged.csv` (11MB)

**Most important file for your project!**

```csv
,cocoId,cocoSplit,cropBox,loss,nsdId,flagged,BOLD5000,shared1000,subject1,subject2,...
0,532481,val2017,"(0, 0, 0.1671875, 0.1671875)",0.1,0,False,False,False,0,0,1,0,0,0,0,0
```

**Columns:**

- `nsdId`: Unique stimulus ID (0-72999)
- `cocoId`: Original COCO dataset ID
- `cocoSplit`: COCO dataset split (train2017/val2017)
- `subject1-8`: Binary (1=subject saw this stimulus, 0=didn't)
- `subject{X}_rep{0-2}`: Repetition information
- `flagged`: Quality control flag
- `shared1000`: Whether stimulus is in shared set across subjects

**Usage:**

```python
import pandas as pd
df = pd.read_csv('nsd_stim_info_merged.csv')

# Get stimuli for subject 1
subj1_stimuli = df[df['subject1'] == 1]
print(f"Subject 1 saw {len(subj1_stimuli)} stimuli")
```

### 🖼️ Stimulus Images

#### `nsd_stimuli.hdf5` (36.8GB)

Contains all 73,000 stimulus images in HDF5 format.

**Structure:**

- Dataset: `imgBrick`
- Shape: `(73000, H, W, 3)` where H,W vary per image
- Data type: `uint8` (0-255)
- Format: RGB images

**Usage:**

```python
import h5py
import numpy as np
from PIL import Image

with h5py.File('nsd_stimuli.hdf5', 'r') as f:
    # Load specific image
    img_data = f['imgBrick'][nsd_id]  # Shape: (H, W, 3)
    image = Image.fromarray(img_data)
```

**Alternative: Use COCO Images Directly**
Since NSD images come from COCO, you can download original COCO images:

```python
# Get COCO ID from metadata
coco_id = df.iloc[trial_idx]['cocoId']
coco_split = df.iloc[trial_idx]['cocoSplit']  # train2017 or val2017

# Download from COCO
url = f"http://images.cocodataset.org/{coco_split}/{coco_id:012d}.jpg"
```

### 🧠 fMRI Data

#### `betas_session{XX}.nii.gz` (~467MB each)

Preprocessed fMRI beta coefficients from GLM analysis.

**Recommended path:**
`nsddata_betas/ppdata/subj{XX}/func1pt8mm/betas_fithrf_GLMdenoise_RR/`

**File details:**

- **Format**: NIfTI compressed (.nii.gz)
- **Shape**: `(81, 104, 83, ~750)` = (x, y, z, trials)
- **Voxel size**: 1.8mm isotropic
- **Data type**: varies (often int16). Slice a single trial and cast if your model expects floats: `img.slicer[..., beta_index].get_fdata().astype('float32')`.
- **Content**: Beta coefficients (brain activation patterns)

**Alternative preprocessing options:**

- `betas_fithrf/`: Different preprocessing pipeline
- `func1mm/`: Higher resolution (1mm)
- `MNI/`: Normalized to MNI space

**Usage:**

```python
import nibabel as nib

# Load session data
img = nib.load('betas_session01.nii.gz')
# Extract single trial efficiently (avoids loading full 4D)
vol = img.slicer[..., 0].get_fdata().astype("float32")  # Shape: (81, 104, 83)
```

---

## Data Loading Workflow

### Step 1: Set Up Access

```python
import fsspec
import pandas as pd
import nibabel as nib
import h5py

# Anonymous S3 access
fs = fsspec.filesystem("s3", anon=True)
bucket = "natural-scenes-dataset"
```

### Step 2: Load Metadata

```python
# Load stimulus mapping
meta_path = f"{bucket}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
with fs.open(meta_path, 'r') as f:
    stim_df = pd.read_csv(f)

print(f"Total stimulus presentations: {len(stim_df)}")
print(f"Unique images: {stim_df['nsdId'].nunique()}")
```

### Step 3: Load fMRI Data with Canonical Index

```python
# Load using canonical index and NIfTI loader
from fmri2img.data.nsd_index_builder import NSDIndexBuilder
from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem

# Initialize S3 filesystem and NIfTI loader
s3_fs = get_s3_filesystem()
nifti_loader = NIfTILoader(s3_fs)

# Build canonical index for proper trial mapping
builder = NSDIndexBuilder()
index_df = builder.build_index(subjects=["subj01"], max_trials_per_subject=10)

# Read only subj01 partition
from fmri2img.data.nsd_index_reader import read_subject_index, sample_trials
df = read_subject_index("data/indices/nsd_index", subject="subj01")
batch = sample_trials(df, n=4, session=1)

# Each row has (beta_path, beta_index); load 3D safely:
img = nifti_loader.load(batch.loc[0,'beta_path'])
vol = img.slicer[..., int(batch.loc[0,'beta_index'])].get_fdata().astype("float32")

# Get a sample trial from canonical index
trial = index_df.iloc[0]
print(f"Trial: subject={trial['subject']}, nsdId={trial['nsdId']}")
print(f"Beta path: {trial['beta_path']}, index: {trial['beta_index']}")

# Load using header-only access
shape = nifti_loader.get_shape(trial['beta_path'])
print(f"fMRI shape: {shape}")  # (81, 104, 83, ~750)
```

### Step 4: Use Canonical Index for Proper Trial Mapping

**⚠️ CRITICAL: Never estimate trial mapping! Use the canonical index.**

```python
# CORRECT: Use canonical index builder for proper trial mapping
from fmri2img.data.nsd_index_builder import NSDIndexBuilder

# Build canonical index with actual session design files
builder = NSDIndexBuilder()
index_df = builder.build_index(subjects=["subj01"], max_trials_per_subject=None)

# Extra columns in canonical index:
# - stimulus_repeat_count: count of repeats for that nsdId up to current trial
# - has_beta_data: boolean availability flag for mapped beta file/index
# - data_quality_flag: optional QC status if exposed by design

# Get actual trials for a session (not estimated!)
session_trials = builder.get_session_trials(index_df, "subj01", session_id=1)

# Create properly aligned pairs
pairs = []
for _, trial in session_trials.iterrows():
    pairs.append({
        'global_trial_index': trial['global_trial_index'],
        'nsdId': trial['nsdId'],
        'beta_path': trial['beta_path'],
        'beta_index': trial['beta_index'],
        'stim_locator': trial['stim_locator']
    })

# Load data using canonical mapping
from fmri2img.io.s3 import NIfTILoader
nifti_loader = NIfTILoader(s3_fs)

for pair in pairs:
    # Load exact fMRI volume
    img = nifti_loader.load(pair['beta_path'])               # header-only validate
    fmri_volume = img.slicer[..., pair['beta_index']].get_fdata().astype("float32")  # load only the 3D volume

    # Load corresponding stimulus using exact mapping
    # (stimulus loading implementation depends on your needs)
```

---

## Practical Usage for CLIP Project

### Recommended Data Subset for Development

Start small to validate your pipeline:

```python
# Recommended starting point
SUBJECTS = ["subj01"]           # Start with one subject
SESSIONS = [1, 2, 3]           # First 3 sessions (~2.25GB)
TOTAL_TRIALS = ~2250           # Manageable for development
```

### Memory Requirements

```python
# Per trial
fmri_volume = (81, 104, 83)     # = 707,464 voxels
image_size = (224, 224, 3)      # For CLIP input

# Batch processing
batch_size = 32
fmri_batch = 32 * 707464        # ~22M features
memory_per_batch = ~90MB        # Manageable
```

### Data Preprocessing Pipeline

Our preprocessing pipeline implements three transformation levels (T0/T1/T2) for production-grade fMRI processing:

```python
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.torch_dataset import NSDIterableDataset

# Fit preprocessing on training data
preprocessor = NSDPreprocessor(subject="subj01")
preprocessor.fit(train_df, loader_factory, reliability_threshold=0.1)
preprocessor.fit_pca(train_df, loader_factory, k=4096)

# Create dataset with preprocessing
dataset = NSDIterableDataset(
    index_root="data/indices/nsd_index",
    subject="subj01",
    preprocessor=preprocessor
)

class NSDDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_df, fmri_sessions, preprocessor=None):
        self.metadata = metadata_df
        self.fmri_data = fmri_sessions
        self.preprocessor = preprocessor

    def __getitem__(self, idx):
        # Get trial info from canonical index
        trial_info = self.canonical_index.iloc[idx]
        nsd_id = trial_info['nsdId']

        # Load fMRI data using canonical mapping
        beta_path = trial_info['beta_path']
        beta_index = trial_info['beta_index']
        img = self.nifti_loader.load(beta_path)

        # Load 3D volume (avoid loading full 4D file)
        vol = img.slicer[..., beta_index].get_fdata().astype('float32')

        # Apply preprocessing pipeline
        if self.preprocessor:
            vol = self.preprocessor.transform(vol)  # T0/T1/T2 transforms
        else:
            # Fallback: simple z-score normalization (T0 only)
            vol = (vol - vol.mean()) / (vol.std() + 1e-8)

        # Load stimulus image
        stimulus = self.load_stimulus(nsd_id)

        return {
            'fmri': vol,  # Either (H,W,D) or (k,) if PCA applied
            'image': stimulus,
            'nsdId': nsd_id,
            'metadata': trial_info
        }
```

#### Preprocessing Transformations

- **T0**: Per-volume z-score normalization (online, no fitting required)
- **T1**: Subject-level scaler + reliability mask
  - Fits voxel-wise mean/std from training data using Welford's algorithm
  - Computes test-retest reliability for voxels with repeat stimuli
  - Masks out unreliable voxels (r < 0.1) and low-variance voxels
- **T2**: PCA dimensionality reduction (optional)
  - Reduces masked voxels to k components (default k=4096)
  - Uses incremental PCA for memory efficiency
  - Outputs compact feature vectors instead of full volumes

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

```bash
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
```

---

## CLIP Integration Strategy

### Architecture Overview

```
fMRI Volume (81×104×83) → fMRI Encoder → Embedding (512D)
                                            ↓
                                      Contrastive Loss
                                            ↓
Image (224×224×3) → CLIP Vision Encoder → Embedding (512D)
```

### Implementation Steps

1. **Data Preparation**

   ```python
   # Normalize fMRI data
   fmri_normalized = (fmri - fmri.mean()) / fmri.std()

   # Prepare images for CLIP
   image_resized = transforms.Resize((224, 224))(image)
   image_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)
   ```

2. **Model Architecture**

   ```python
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
   ```

3. **Training Loop**

   ```python
   # Contrastive loss between fMRI and image embeddings
   loss_fn = nn.CosineEmbeddingLoss()

   for batch in dataloader:
       fmri_emb, img_emb = model(batch['fmri'], batch['image'])
       loss = contrastive_loss(fmri_emb, img_emb)
       loss.backward()
   ```

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
