"""
NSD Image Loader
================

Load NSD stimulus images from S3 with HDF5 or HTTP fallback.

Strategy:
  1. Try HDF5 sprite (nsd_stimuli.hdf5 → dataset "imgBrick") via S3 streaming
  2. Fallback to COCO HTTP URLs when HDF5 unavailable
  3. Always return RGB PIL Images, un-resized
  4. Robust: skip missing IDs with warnings, don't crash
"""

from typing import Iterable, List, Tuple, Dict, Optional
import logging
import os
from pathlib import Path
import io

from PIL import Image

logger = logging.getLogger(__name__)


def _as_int(x) -> int:
    """Coerce to int, handling numpy/pandas types."""
    try:
        return int(x)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert {x!r} to int")


def load_nsd_images(
    nsd_ids: Iterable[int],
    *,
    layout=None,          # fmri2img.io.nsd_layout.NSDLayout or None -> auto
    s3_fs=None,           # fmri2img.io.s3.S3FileSystem or None -> auto
    prefer: str = "hdf5", # "hdf5" | "http"
    cache_dir: str = "cache/stimuli"
) -> Dict[int, Image.Image]:
    """
    Return a dict {nsdId: PIL.Image} for the requested NSD IDs.
    
    Strategy:
      1) Try HDF5 sprite (`nsd_stimuli.hdf5` → dataset "imgBrick") via S3 streaming.
         Use stim_info CSV to map nsdId → row index in the HDF5.
      2) Fallback to COCO HTTP (`layout.coco_http_url(coco_id)`) when HDF5 or h5py is unavailable.
      3) Always convert to RGB and keep images un-resized; caller can preprocess.
      4) Be robust: skip missing IDs with a warning, don't raise.
    
    Args:
        nsd_ids: Iterable of NSD stimulus IDs
        layout: NSDLayout instance (auto-created if None)
        s3_fs: S3FileSystem instance (auto-created if None)
        prefer: "hdf5" or "http" (strategy preference)
        cache_dir: Local directory for HTTP downloads
        
    Returns:
        Dictionary mapping nsdId → PIL.Image (RGB)
    """
    from fmri2img.io.nsd_layout import NSDLayout, get_nsd_layout
    from fmri2img.io.s3 import (
        get_s3_filesystem, 
        HDF5Loader, 
        CSVLoader, 
        S3LoadError
    )
    
    # Auto-initialize layout and s3_fs if needed
    if layout is None:
        layout = get_nsd_layout()
    
    if s3_fs is None:
        s3_fs = get_s3_filesystem()
    
    # Convert to list and validate
    nsd_ids_list = [_as_int(nid) for nid in nsd_ids]
    
    if not nsd_ids_list:
        return {}
    
    logger.debug(f"Loading {len(nsd_ids_list)} NSD images (prefer={prefer})")
    
    # Load stimulus metadata
    try:
        csv_loader = CSVLoader(s3_fs)
        stim_info_path = layout.stim_info_path()
        df = csv_loader.load(stim_info_path)
        df = df.reset_index(drop=True)
        
        # Build mapping: nsdId → row index
        row_by_nsd = {_as_int(row.nsdId): i for i, row in df.iterrows()}
        
        logger.debug(f"Loaded stim_info with {len(df)} stimuli")
    except Exception as e:
        logger.error(f"Failed to load stim_info: {e}")
        return {}
    
    images: Dict[int, Image.Image] = {}
    
    # Strategy 1: Try HDF5 if preferred and available
    if prefer == "hdf5":
        images = _try_load_hdf5(
            nsd_ids_list, 
            layout, 
            s3_fs, 
            row_by_nsd, 
            df
        )
    
    # Strategy 2: HTTP fallback for missing IDs (or if prefer="http")
    missing_ids = [nid for nid in nsd_ids_list if nid not in images]
    
    if missing_ids:
        logger.debug(f"Trying HTTP fallback for {len(missing_ids)} IDs")
        http_images = _try_load_http(
            missing_ids,
            layout,
            s3_fs,
            row_by_nsd,
            df,
            cache_dir
        )
        images.update(http_images)
    
    # Report final status
    loaded = len(images)
    failed = len(nsd_ids_list) - loaded
    
    if failed > 0:
        failed_ids = [nid for nid in nsd_ids_list if nid not in images]
        logger.warning(f"Failed to load {failed}/{len(nsd_ids_list)} images: {failed_ids[:5]}{'...' if len(failed_ids) > 5 else ''}")
    else:
        logger.debug(f"Successfully loaded {loaded}/{len(nsd_ids_list)} images")
    
    return images


def _try_load_hdf5(
    nsd_ids: List[int],
    layout,
    s3_fs,
    row_by_nsd: Dict[int, int],
    df
) -> Dict[int, Image.Image]:
    """Try loading images from HDF5 sprite."""
    images = {}
    
    try:
        import h5py
    except ImportError:
        logger.debug("h5py not available, skipping HDF5 strategy")
        return images
    
    try:
        # Check for local HDF5 file first
        local_hdf5 = None
        env_path = os.getenv('NSD_HDF5')
        if env_path and Path(env_path).exists():
            local_hdf5 = Path(env_path)
            logger.debug(f"Using local HDF5 from $NSD_HDF5: {local_hdf5}")
        else:
            default_path = Path("cache/nsd_hdf5/nsd_stimuli.hdf5")
            if default_path.exists():
                local_hdf5 = default_path
                logger.debug(f"Using local HDF5 from default location: {local_hdf5}")
        
        # Use local file if available, otherwise fall back to S3
        if local_hdf5:
            logger.debug(f"Opening local HDF5: {local_hdf5}")
            h5file = h5py.File(local_hdf5, 'r')
        else:
            from fmri2img.io.s3 import HDF5Loader
            hdf5_path = layout.stim_hdf5_path()
            logger.debug(f"Opening HDF5 from S3: {hdf5_path}")
            hdf5_loader = HDF5Loader(s3_fs)
            h5file = hdf5_loader.open(hdf5_path)
        
        with h5file:
            if "imgBrick" not in h5file:
                logger.warning("HDF5 file missing 'imgBrick' dataset")
                return images
            
            ds = h5file["imgBrick"]
            logger.debug(f"HDF5 imgBrick shape: {ds.shape}")
            
            for nsd_id in nsd_ids:
                if nsd_id not in row_by_nsd:
                    logger.warning(f"nsdId={nsd_id} not found in stim_info")
                    continue
                
                try:
                    idx = row_by_nsd[nsd_id]
                    
                    # Read image array (H×W×3, uint8)
                    arr = ds[idx]
                    
                    # Convert to PIL Image
                    img = Image.fromarray(arr, mode="RGB")
                    images[nsd_id] = img
                    
                except Exception as e:
                    logger.debug(f"Failed to read nsdId={nsd_id} from HDF5: {e}")
                    continue
    
    except Exception as e:
        logger.warning(f"HDF5 loading failed: {e}")
    
    return images


def _try_load_http(
    nsd_ids: List[int],
    layout,
    s3_fs,
    row_by_nsd: Dict[int, int],
    df,
    cache_dir: str
) -> Dict[int, Image.Image]:
    """Try loading images via HTTP from COCO URLs."""
    images = {}
    
    try:
        import requests
    except ImportError:
        logger.warning("requests not available, cannot use HTTP fallback")
        return images
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    for nsd_id in nsd_ids:
        if nsd_id not in row_by_nsd:
            logger.warning(f"nsdId={nsd_id} not found in stim_info")
            continue
        
        try:
            idx = row_by_nsd[nsd_id]
            row = df.iloc[idx]
            
            # Extract COCO ID (try multiple column names)
            coco_id = None
            for col in ["cocoId", "cocoIdOriginal", "coco_id"]:
                if col in row and row[col] is not None:
                    coco_id = _as_int(row[col])
                    break
            
            if coco_id is None:
                logger.warning(f"nsdId={nsd_id} missing COCO ID")
                continue
            
            # Extract COCO split (train2017, val2017, etc.)
            coco_split = "train2017"  # default
            if "cocoSplit" in row and row["cocoSplit"] is not None:
                coco_split = str(row["cocoSplit"])
            
            # Check cache first
            cache_file = cache_path / f"{coco_id}_{coco_split}.jpg"
            
            if cache_file.exists():
                img = Image.open(cache_file).convert("RGB")
                images[nsd_id] = img
                logger.debug(f"Loaded nsdId={nsd_id} from cache: {cache_file}")
                continue
            
            # Download from COCO HTTP
            url = layout.coco_http_url(coco_id, coco_split=coco_split)
            
            logger.debug(f"Downloading nsdId={nsd_id} from {url}")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_file, "wb") as f:
                f.write(response.content)
            
            # Open as PIL Image
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            images[nsd_id] = img
            
        except Exception as e:
            logger.debug(f"Failed to load nsdId={nsd_id} via HTTP: {e}")
            continue
    
    return images


if __name__ == "__main__":
    """Self-test: load a few images and save to cache."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 80)
    print("NSD Images Self-Test")
    print("=" * 80)
    
    # Test IDs (first 3 stimuli)
    test_ids = [1, 2, 3]
    
    print(f"\nLoading {len(test_ids)} test images: {test_ids}")
    print("(Using HTTP fallback to avoid large HDF5 download)")
    
    try:
        images = load_nsd_images(test_ids, prefer="http")
        
        print(f"\n✅ Loaded {len(images)}/{len(test_ids)} images")
        
        # Save to cache/stimuli/selftest_nsd<id>.png
        output_dir = Path("cache/stimuli")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for nsd_id, img in images.items():
            output_path = output_dir / f"selftest_nsd{nsd_id:05d}.png"
            img.save(output_path)
            print(f"   Saved: {output_path} ({img.size[0]}×{img.size[1]})")
        
        if len(images) < len(test_ids):
            failed = [nid for nid in test_ids if nid not in images]
            print(f"\n⚠️  Failed to load: {failed}")
            sys.exit(1)
        else:
            print("\n✅ Self-test passed!")
            sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Self-test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
