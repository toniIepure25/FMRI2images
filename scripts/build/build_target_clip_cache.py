#!/usr/bin/env python3
"""
Target CLIP Cache Builder for Stable Diffusion 2.1
====================================================

Builds a 1024-D CLIP embedding cache using OpenCLIP ViT-H/14 (SD 2.1's text encoder).
Supports multiple image sources: local PNGs, S3 streaming, or HDF5 file.

Usage:
    # Use HDF5 file (auto-download ~40GB if missing)
    python scripts/build_target_clip_cache.py \
        --subject subj01 \
        --index-dir data/indices/nsd_index \
        --out outputs/clip_cache/target_clip_stabilityai_stable-diffusion-2-1.parquet \
        --source hdf5 \
        --limit 64

    # Stream from S3 (no local files needed)
    python scripts/build_target_clip_cache.py \
        --subject subj01 \
        --index-dir data/indices/nsd_index \
        --out outputs/clip_cache/target_clip_stabilityai_stable-diffusion-2-1.parquet \
        --source s3 \
        --limit 64

    # Use local PNG files only
    python scripts/build_target_clip_cache.py \
        --subject subj01 \
        --index-dir data/indices/nsd_index \
        --out outputs/clip_cache/target_clip_stabilityai_stable-diffusion-2-1.parquet \
        --source local \
        --gt-root data/stimuli/nsd \
        --limit 64

    # Auto mode (try local first, fallback to S3)
    python scripts/build_target_clip_cache.py \
        --subject subj01 \
        --index-dir data/indices/nsd_index \
        --out outputs/clip_cache/target_clip_stabilityai_stable-diffusion-2-1.parquet \
        --source auto \
        --gt-root data/stimuli/nsd \
        --limit 64

Output:
    Parquet file with columns:
        - nsdId (int32): NSD stimulus ID
        - clip1024 (fixed_size_list<float>[1024]): OpenCLIP ViT-H/14 embedding
"""

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Literal
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

try:
    import requests
except ImportError:
    print("ERROR: requests is required. Install with: pip install requests")
    sys.exit(1)

# Try importing required libraries with helpful error messages
try:
    import open_clip
except ImportError:
    print("ERROR: open_clip is required. Install with: pip install open-clip-torch")
    sys.exit(1)

# boto3 is optional - only needed for S3 source
BOTO3_AVAILABLE = False
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError, EndpointConnectionError
    BOTO3_AVAILABLE = True
except ImportError:
    pass

# h5py is optional - only needed for HDF5 source
H5PY_AVAILABLE = False
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    pass

# Setup logging with project style
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def ensure_local_hdf5(hdf5_local_path: Path) -> Path:
    """
    Ensure HDF5 file exists locally. If missing, download from S3.
    
    Args:
        hdf5_local_path: Target local path for HDF5 file
    
    Returns:
        Path to local HDF5 file
    """
    if hdf5_local_path.exists():
        logger.info(f"✅ HDF5 file found: {hdf5_local_path}")
        return hdf5_local_path
    
    # Need to download
    if not BOTO3_AVAILABLE:
        raise ImportError(
            "boto3 is required to download HDF5 file. Install with: pip install boto3"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("ONE-TIME HDF5 DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"Source: s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    logger.info(f"Target: {hdf5_local_path}")
    logger.info("Size: ~37-40 GB (this will take a while)")
    logger.info("=" * 80 + "\n")
    
    # Create parent directory
    hdf5_local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup S3 client with unsigned access
    s3 = boto3.client(
        's3',
        region_name='us-east-2',
        config=Config(signature_version=UNSIGNED)
    )
    
    bucket = "natural-scenes-dataset"
    key = "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    
    # Get file size for progress tracking
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
        total_size = response['ContentLength']
        logger.info(f"File size: {total_size / 1e9:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not get file size: {e}")
        total_size = None
    
    # Progress callback
    downloaded_bytes = [0]
    last_log_mb = [0]
    
    def progress_callback(bytes_amount):
        downloaded_bytes[0] += bytes_amount
        current_mb = downloaded_bytes[0] / 1e6
        
        # Log every 100 MB
        if current_mb - last_log_mb[0] >= 100:
            if total_size:
                pct = (downloaded_bytes[0] / total_size) * 100
                logger.info(f"Downloaded: {current_mb:.0f} MB ({pct:.1f}%)")
            else:
                logger.info(f"Downloaded: {current_mb:.0f} MB")
            last_log_mb[0] = current_mb
    
    # Download file
    try:
        logger.info("Starting download...")
        s3.download_file(
            bucket,
            key,
            str(hdf5_local_path),
            Callback=progress_callback
        )
        logger.info(f"✅ Download complete: {downloaded_bytes[0] / 1e9:.2f} GB")
        logger.info("=" * 80 + "\n")
        return hdf5_local_path
        
    except Exception as e:
        # Cleanup partial download
        if hdf5_local_path.exists():
            hdf5_local_path.unlink()
        raise RuntimeError(f"HDF5 download failed: {e}")


class HDF5Loader:
    """Load images from NSD HDF5 file with auto-detection."""
    
    def __init__(self, hdf5_path: Path):
        """
        Initialize HDF5 loader with auto-dataset detection.
        
        Args:
            hdf5_path: Path to local HDF5 file
        """
        if not H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for HDF5 source. Install with: pip install h5py"
            )
        
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        logger.info(f"Opening HDF5 file: {hdf5_path}")
        self.file = h5py.File(str(hdf5_path), 'r')
        
        # Auto-detect dataset
        candidates = []
        
        # Check root level
        for key, val in self.file.items():
            if isinstance(val, h5py.Dataset):
                if val.dtype == np.uint8 and val.ndim >= 3 and val.shape[-1] == 3:
                    candidates.append((key, val.shape[0], val))
            # Check one level down for groups
            elif isinstance(val, h5py.Group):
                for subkey, subval in val.items():
                    if isinstance(subval, h5py.Dataset):
                        if subval.dtype == np.uint8 and subval.ndim >= 3 and subval.shape[-1] == 3:
                            full_key = f"{key}/{subkey}"
                            candidates.append((full_key, subval.shape[0], subval))
        
        if not candidates:
            available_keys = list(self.file.keys())
            raise ValueError(
                f"No suitable image dataset found in HDF5 file.\n"
                f"Expected: uint8 dtype with shape [..., 3]\n"
                f"Available keys: {available_keys}"
            )
        
        # Pick dataset with largest first dimension
        candidates.sort(key=lambda x: x[1], reverse=True)
        self.dataset_name, num_images, self.dset = candidates[0]
        
        logger.info(f"✅ HDF5 loader initialized")
        logger.info(f"   Dataset: '{self.dataset_name}'")
        logger.info(f"   Shape: {self.dset.shape}")
        logger.info(f"   Dtype: {self.dset.dtype}")
        logger.info(f"   Images: {num_images}")
    
    def load_image(self, nsd_id: int) -> Optional[Image.Image]:
        """
        Load image from HDF5 dataset by index.
        
        Args:
            nsd_id: NSD stimulus ID (used as 0-based index)
        
        Returns:
            PIL Image in RGB mode, or None if index out of bounds
        """
        try:
            # Use nsd_id directly as index (0-based)
            if nsd_id < 0 or nsd_id >= self.dset.shape[0]:
                logger.warning(f"Index {nsd_id} out of bounds [0, {self.dset.shape[0]})")
                return None
            
            arr = self.dset[nsd_id]
            
            # Handle different shapes: (H, W, 3) or (3, H, W)
            if arr.shape[-1] == 3:
                # HWC format
                return Image.fromarray(arr, mode='RGB')
            elif arr.shape[0] == 3:
                # CHW format - transpose to HWC
                arr = np.transpose(arr, (1, 2, 0))
                return Image.fromarray(arr, mode='RGB')
            else:
                logger.warning(f"Unexpected array shape for nsd_id={nsd_id}: {arr.shape}")
                return None
                
        except Exception as e:
            logger.debug(f"Failed to load nsd_id={nsd_id} from HDF5: {e}")
            return None
    
    def __del__(self):
        """Close HDF5 file on cleanup."""
        if hasattr(self, 'file'):
            self.file.close()


class LocalPNGLoader:
    """Load individual PNG images from a local NSD stimuli directory."""
    
    def __init__(self, gt_root: Path):
        """
        Initialize local image loader.
        
        Args:
            gt_root: Directory containing NSD stimuli (e.g., data/stimuli/nsd)
        """
        self.gt_root = Path(gt_root)
        
        if not self.gt_root.exists():
            raise FileNotFoundError(
                f"GT root directory does not exist: {self.gt_root}\n"
                f"Please provide a valid --gt-root with NSD stimuli images."
            )
        
        logger.info(f"✅ Local PNG loader initialized: {self.gt_root}")
    
    def load_image(self, nsd_id: int) -> Optional[Image.Image]:
        """
        Load a PNG image from local filesystem.
        
        Args:
            nsd_id: NSD stimulus ID
        
        Returns:
            PIL Image in RGB mode, or None if file doesn't exist or fails to load
        """
        img_path = self.gt_root / f"nsd_{nsd_id:05d}.png"
        
        try:
            if not img_path.exists():
                return None
            
            img = Image.open(img_path).convert('RGB')
            return img
            
        except Exception as e:
            logger.debug(f"Failed to load nsd_id={nsd_id} from {img_path}: {e}")
            return None


class S3PNGLoader:
    """Stream individual PNG images from NSD AWS S3 bucket with auto-discovery."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.5):
        """
        Initialize S3 PNG loader with unsigned access and region auto-discovery.
        
        Args:
            max_retries: Maximum retry attempts per image
            retry_delay: Delay in seconds between retries
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 source. Install with: pip install boto3"
            )
        
        # NSD bucket configuration
        self.bucket = "natural-scenes-dataset"
        self.prefix = "nsddata_stimuli/stimuli/nsd"
        self.region = None
        self.endpoint_url = None
        
        # Discover bucket region using temporary client
        tmp_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        try:
            location_resp = tmp_client.get_bucket_location(Bucket=self.bucket)
            # AWS returns None for us-east-1, otherwise returns region name
            self.region = location_resp.get('LocationConstraint') or 'us-east-1'
        except Exception as e:
            logger.warning(f"Could not discover bucket region, defaulting to us-east-2: {e}")
            self.region = 'us-east-2'
        
        # Create final client bound to discovered region
        self.s3 = boto3.client(
            's3',
            region_name=self.region,
            config=Config(signature_version=UNSIGNED)
        )
        self.endpoint_url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"✅ S3 PNG loader initialized (region: {self.region}, unsigned access)")
    
    def _key(self, nsd_id: int) -> str:
        """Construct S3 key for an NSD ID."""
        return f"{self.prefix}/nsd_{nsd_id:05d}.png"
    
    def _https_url(self, key: str) -> str:
        """Build direct HTTPS URL for a key."""
        return f"{self.endpoint_url}/{key}"
    
    def _list_example_keys(self, prefix: str, limit: int = 5) -> list:
        """List a few example keys under a prefix for debugging."""
        try:
            resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=limit)
            keys = [obj['Key'] for obj in resp.get('Contents', [])]
            if not keys:
                logger.warning(f"No objects under prefix '{prefix}'.")
            else:
                logger.warning(f"Example keys under prefix '{prefix}':\n  - " + "\n  - ".join(keys))
            return keys
        except Exception as e:
            logger.warning(f"Failed to list keys for prefix '{prefix}': {e}")
            return []
    
    def load_image(self, nsd_id: int) -> Optional[Image.Image]:
        """
        Fetch a PNG image from S3 and return as PIL Image.
        Includes key verification, retries, and HTTPS fallback.
        
        Args:
            nsd_id: NSD stimulus ID
        
        Returns:
            PIL Image in RGB mode, or None if fetch fails
        """
        key = self._key(nsd_id)
        
        # First: verify key exists with HEAD request
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            code = e.response.get('Error', {}).get('Code', 'UnknownError')
            if code in ('404', 'NoSuchKey', 'NotFound'):
                logger.warning(f"S3 key missing: {key}")
                # List a few example keys for debugging on first miss
                self._list_example_keys(self.prefix, limit=5)
            else:
                logger.warning(f"S3 head_object error for {key}: {code}")
        
        # GET object with retries
        for attempt in range(self.max_retries):
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                data = resp['Body'].read()
                return Image.open(io.BytesIO(data)).convert('RGB')
                
            except (ClientError, EndpointConnectionError) as e:
                if hasattr(e, 'response'):
                    code = e.response.get('Error', {}).get('Code', str(e))
                else:
                    code = str(e)
                logger.warning(f"S3 get_object failed for {key} (attempt {attempt+1}/{self.max_retries}): {code}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            
            except Exception as e:
                logger.warning(f"Unexpected error fetching {key} (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # HTTPS fallback with discovered region
        url = self._https_url(key)
        try:
            logger.info(f"Trying HTTPS fallback: {url}")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert('RGB')
            logger.warning(f"HTTPS fallback failed {r.status_code} for {url}")
        except Exception as e:
            logger.warning(f"HTTPS fallback exception for {url}: {e}")
        
        return None


class DualSourceLoader:
    """Dual-source loader with local-first fallback to S3."""
    
    def __init__(self, gt_root: Optional[Path] = None):
        """
        Initialize dual-source loader.
        
        Args:
            gt_root: Optional local directory. If provided, tries local first.
        """
        self.local_loader = None
        self.s3_loader = None
        
        # Initialize local loader if gt_root provided and exists
        if gt_root and Path(gt_root).exists():
            self.local_loader = LocalPNGLoader(gt_root)
        
        # Initialize S3 loader
        try:
            self.s3_loader = S3PNGLoader()
        except ImportError:
            if self.local_loader is None:
                raise ImportError(
                    "Neither local files nor boto3 available. "
                    "Install boto3 with: pip install boto3"
                )
            logger.warning("boto3 not available - S3 fallback disabled")
        
        mode = []
        if self.local_loader:
            mode.append("local")
        if self.s3_loader:
            mode.append("S3")
        logger.info(f"✅ Dual-source loader initialized: {' → '.join(mode)}")
    
    def load_image(self, nsd_id: int) -> Optional[Image.Image]:
        """
        Load image with local-first, S3-fallback strategy.
        
        Args:
            nsd_id: NSD stimulus ID
        
        Returns:
            PIL Image in RGB mode, or None if all sources fail
        """
        # Try local first
        if self.local_loader:
            img = self.local_loader.load_image(nsd_id)
            if img is not None:
                return img
        
        # Fallback to S3
        if self.s3_loader:
            img = self.s3_loader.load_image(nsd_id)
            if img is not None:
                return img
        
        return None


class OpenCLIPEncoder:
    """OpenCLIP ViT-H/14 image encoder for Stable Diffusion 2.1."""
    
    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: Optional[str] = None
    ):
        """
        Initialize OpenCLIP encoder.
        
        Args:
            model_name: OpenCLIP model architecture
            pretrained: Pretrained weights identifier
            device: Device to run on ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("=" * 80)
        logger.info("INITIALIZING OPENCLIP ENCODER")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Pretrained: {pretrained}")
        logger.info(f"Device: {self.device}")
        
        # Load model and preprocessing
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        
        self.model.eval()
        
        # Get embedding dimension and validate
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_output = self.model.encode_image(dummy_input)
            self.embed_dim = dummy_output.shape[1]
        
        # Assert 1024-D for SD 2.1 compatibility
        if self.embed_dim != 1024:
            raise ValueError(
                f"Expected 1024-D embeddings for SD 2.1, got {self.embed_dim}. "
                f"Make sure you're using ViT-H-14 with laion2b_s32b_b79k weights."
            )
        
        logger.info(f"✅ Model loaded: embedding dimension = {self.embed_dim}")
        logger.info("=" * 80)
    
    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode a batch of images to CLIP embeddings.
        
        Args:
            images: List of PIL Images in RGB
        
        Returns:
            Numpy array of shape (N, embed_dim) with L2-normalized embeddings
        """
        if not images:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        
        # Preprocess images
        image_tensors = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        
        # Encode
        with torch.no_grad():
            embeddings = self.model.encode_image(image_tensors)
            
            # Normalize to unit length (standard for CLIP)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy().astype(np.float32)


def load_nsd_index(subject: str, index_dir: Path) -> pd.DataFrame:
    """
    Load NSD index for a subject and extract unique stimulus IDs.
    
    Args:
        subject: Subject ID (e.g., 'subj01')
        index_dir: Path to index directory (e.g., data/indices/nsd_index)
    
    Returns:
        DataFrame with unique nsdId values
    """
    subject_index_path = index_dir / f"subject={subject}" / "index.parquet"
    
    if not subject_index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {subject_index_path}\n"
            f"Expected structure: {index_dir}/subject={subject}/index.parquet"
        )
    
    logger.info(f"Loading index from: {subject_index_path}")
    df = pd.read_parquet(subject_index_path)
    
    # Extract unique nsdId values
    if 'nsdId' not in df.columns:
        raise ValueError(f"Index must have 'nsdId' column. Found: {df.columns.tolist()}")
    
    unique_ids = df[['nsdId']].drop_duplicates().sort_values('nsdId').reset_index(drop=True)
    
    logger.info(f"✅ Loaded {len(df)} trials with {len(unique_ids)} unique stimuli")
    
    return unique_ids


def save_clip_cache(
    nsd_ids: List[int],
    embeddings: np.ndarray,
    output_path: Path,
    embed_dim: int = 1024
):
    """
    Save CLIP embeddings to Parquet with fixed_size_list schema.
    
    Args:
        nsd_ids: List of NSD stimulus IDs
        embeddings: Array of shape (N, embed_dim)
        output_path: Output Parquet file path
        embed_dim: Embedding dimension (1024 for OpenCLIP ViT-H/14)
    """
    logger.info(f"Saving CLIP cache to: {output_path}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'nsdId': np.array(nsd_ids, dtype=np.int32),
        f'clip{embed_dim}': list(embeddings.astype(np.float32))
    })
    
    # Define schema with fixed_size_list for embeddings
    schema = pa.schema([
        pa.field('nsdId', pa.int32()),
        pa.field(f'clip{embed_dim}', pa.list_(pa.float32(), embed_dim))
    ])
    
    # Convert to PyArrow Table with schema
    table = pa.Table.from_pandas(df, schema=schema)
    
    # Write to Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression='snappy')
    
    logger.info(f"✅ Saved {len(nsd_ids)} embeddings (shape: {embeddings.shape})")


def main():
    parser = argparse.ArgumentParser(
        description="Build 1024-D CLIP cache for Stable Diffusion 2.1 with dual-source PNG loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Stream from S3 (no local files needed)
    python scripts/build_target_clip_cache.py \\
        --subject subj01 \\
        --index-dir data/indices/nsd_index \\
        --out outputs/clip_cache/target_clip_sd21.parquet \\
        --source s3 \\
        --limit 64

    # Use HDF5 file (auto-download if missing)
    python scripts/build_target_clip_cache.py \\
        --subject subj01 \\
        --index-dir data/indices/nsd_index \\
        --out outputs/clip_cache/target_clip_sd21.parquet \\
        --source hdf5 \\
        --limit 64

    # Use local PNG files only
    python scripts/build_target_clip_cache.py \\
        --subject subj01 \\
        --index-dir data/indices/nsd_index \\
        --out outputs/clip_cache/target_clip_sd21.parquet \\
        --source local \\
        --gt-root data/stimuli/nsd

    # Auto mode (try local first, fallback to S3)
    python scripts/build_target_clip_cache.py \\
        --subject subj01 \\
        --index-dir data/indices/nsd_index \\
        --out outputs/clip_cache/target_clip_sd21.parquet \\
        --source auto \\
        --gt-root data/stimuli/nsd
        """
    )
    
    parser.add_argument(
        '--subject',
        type=str,
        required=True,
        help='Subject ID (e.g., subj01)'
    )
    
    parser.add_argument(
        '--index-dir',
        type=Path,
        required=True,
        help='Path to NSD index directory (e.g., data/indices/nsd_index)'
    )
    
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output Parquet file path'
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        default='stabilityai/stable-diffusion-2-1',
        help='Model ID for logging (default: stabilityai/stable-diffusion-2-1)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of images to process (for testing)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for encoding (default: 16)'
    )
    
    parser.add_argument(
        '--gt-root',
        type=Path,
        default=None,
        help='Path to local NSD stimuli directory (required for source=local, optional for auto)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        choices=['auto', 'local', 's3', 'hdf5'],
        default='auto',
        help='Image source: auto (local→S3 fallback), local (disk only), s3 (stream only), hdf5 (HDF5 file)'
    )
    
    parser.add_argument(
        '--hdf5-path',
        type=Path,
        default=None,
        help='Path to local HDF5 file (default: cache/nsd_hdf5/nsd_stimuli.hdf5 with auto-download)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source == 'local' and not args.gt_root:
        parser.error("--gt-root is required when --source=local")
    
    if args.source == 's3' and not BOTO3_AVAILABLE:
        parser.error(
            "boto3 is required for --source=s3. Install with: pip install boto3"
        )
    
    if args.source == 'hdf5' and not H5PY_AVAILABLE:
        parser.error(
            "h5py is required for --source=hdf5. Install with: pip install h5py"
        )
    
    try:
        # Print banner
        logger.info("\n" + "=" * 80)
        logger.info("TARGET CLIP CACHE BUILDER (OpenCLIP ViT-H/14 for SD 2.1)")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Index dir: {args.index_dir}")
        logger.info(f"Output: {args.out}")
        logger.info(f"Model: OpenCLIP ViT-H/14 (laion2b_s32b_b79k)")
        logger.info(f"Target model: {args.model_id}")
        logger.info(f"Source mode: {args.source}")
        if args.gt_root:
            logger.info(f"GT root: {args.gt_root}")
        logger.info(f"Batch size: {args.batch_size}")
        if args.limit:
            logger.info(f"⚠️  Limit: {args.limit} images (testing mode)")
        logger.info("=" * 80 + "\n")
        
        # Load index
        unique_stim = load_nsd_index(args.subject, args.index_dir)
        nsd_ids = unique_stim['nsdId'].tolist()
        
        if args.limit:
            nsd_ids = nsd_ids[:args.limit]
            logger.info(f"⚠️  Limited to {len(nsd_ids)} images for testing")
        
        logger.info(f"Total images to process: {len(nsd_ids)}")
        logger.info(f"NSD ID range: {min(nsd_ids)} to {max(nsd_ids)}\n")
        
        # Initialize image loader based on source mode
        if args.source == 'hdf5':
            # HDF5 source with auto-download
            local_hdf5_path = args.hdf5_path or Path("cache/nsd_hdf5/nsd_stimuli.hdf5")
            local_hdf5_path = ensure_local_hdf5(local_hdf5_path)
            loader = HDF5Loader(local_hdf5_path)
        elif args.source == 'local':
            loader = LocalPNGLoader(gt_root=args.gt_root)
        elif args.source == 's3':
            loader = S3PNGLoader()
        else:  # auto
            loader = DualSourceLoader(gt_root=args.gt_root)
        
        # Initialize encoder
        encoder = OpenCLIPEncoder(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k"
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("LOADING AND ENCODING IMAGES")
        logger.info("=" * 80)
        
        # Process in batches
        all_embeddings = []
        successful_ids = []
        failed_ids = []
        
        pbar = tqdm(range(0, len(nsd_ids), args.batch_size), desc="Encoding batches")
        for i in pbar:
            batch_ids = nsd_ids[i:i + args.batch_size]
            batch_images = []
            batch_valid_ids = []
            
            # Load images for this batch
            for nsd_id in batch_ids:
                img = loader.load_image(nsd_id)
                if img is not None:
                    batch_images.append(img)
                    batch_valid_ids.append(nsd_id)
                else:
                    failed_ids.append(nsd_id)
            
            # Encode batch
            if batch_images:
                embeddings = encoder.encode_batch(batch_images)
                all_embeddings.append(embeddings)
                successful_ids.extend(batch_valid_ids)
            
            # Update progress bar with batch stats
            pbar.set_postfix({
                'encoded': f'{len(batch_valid_ids)}/{len(batch_ids)}',
                'total': f'{len(successful_ids)}/{len(nsd_ids)}'
            })
        
        # Combine results
        if not all_embeddings:
            logger.error("❌ No images were successfully encoded!")
            
            # Extra diagnostics for HDF5 source
            if args.source == 'hdf5':
                logger.error("\nHDF5 source diagnostics:")
                logger.error(f"  HDF5 path: {local_hdf5_path}")
                if hasattr(loader, 'dataset_name'):
                    logger.error(f"  Dataset: '{loader.dataset_name}'")
                    logger.error(f"  Shape: {loader.dset.shape}")
                logger.error(f"  Attempted IDs: {nsd_ids[:5]}...")
            
            logger.error("\nDebugging info - first few attempted URLs:")
            sample_ids = nsd_ids[:3]
            for i, nsd_id in enumerate(sample_ids, 1):
                # Try to use loader's methods if available, otherwise construct manually
                if hasattr(loader, '_key') and hasattr(loader, '_https_url'):
                    key = loader._key(nsd_id)
                    url = loader._https_url(key)
                else:
                    # Fallback for loaders without these methods
                    key = f"nsddata_stimuli/stimuli/nsd/nsd_{nsd_id:05d}.png"
                    # Use discovered region if S3 loader
                    if hasattr(loader, 'region'):
                        url = f"https://natural-scenes-dataset.s3.{loader.region}.amazonaws.com/{key}"
                    else:
                        url = f"https://natural-scenes-dataset.s3.us-east-2.amazonaws.com/{key}"
                logger.error(f"  {i}. curl -I '{url}'")
            return 1
        
        final_embeddings = np.vstack(all_embeddings)
        
        logger.info("\n" + "=" * 80)
        logger.info("ENCODING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total requested: {len(nsd_ids)}")
        logger.info(f"Successfully encoded: {len(successful_ids)}")
        logger.info(f"Failed: {len(failed_ids)}")
        if failed_ids:
            logger.warning(f"Failed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
        logger.info(f"Embedding shape: {final_embeddings.shape}")
        logger.info(f"Mean norm: {np.linalg.norm(final_embeddings, axis=1).mean():.4f}")
        logger.info("=" * 80 + "\n")
        
        # Save cache
        save_clip_cache(
            nsd_ids=successful_ids,
            embeddings=final_embeddings,
            output_path=args.out,
            embed_dim=encoder.embed_dim
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TARGET CLIP CACHE BUILD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output: {args.out}")
        logger.info(f"Shape: ({len(successful_ids)}, {encoder.embed_dim})")
        logger.info(f"Format: Parquet with fixed_size_list<float>[{encoder.embed_dim}]")
        logger.info("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ BUILD FAILED: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
