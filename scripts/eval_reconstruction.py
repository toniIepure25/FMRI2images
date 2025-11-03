#!/usr/bin/env python3
"""
You are GitHub Copilot. Enhance evaluation outputs.

GOALS
1) Per-sample NN dump:
   - In the CSV, add columns: nn_nsdId (top-1 neighbor ID), nn_sim (top-1 cosine similarity), gt_sim (sim with its own GT).
   - Also write a JSONL file {nsdId, rank, topk: [{nsdId, sim}], clipscore} at <out_csv_dir>/<stem>__nn.jsonl with top-k=10 entries per sample.

2) Distribution plots:
   - Save a histogram of CLIPScore and a histogram of ranks as:
     <out_csv_dir>/<stem>__clipscore_hist.png and <stem>__rank_hist.png.
   - Use matplotlib only, single plot per figure (no subplots), safe for empty arrays.

3) Adapter ablation (optional but automatic when --use-adapter is passed):
   - Recompute CLIPScore and retrieval with adapter OFF (using the same matched set + gallery) and report both in aggregate JSON under:
     "ablations": { "with_adapter": {...}, "without_adapter": {...} }.
   - Do not re-encode GT cache; reuse already loaded embeddings and align dimensions via existing fallback when adapter is off.

4) Logging + JSON:
   - Log top-1 neighbor nsdId and sim for the first 5 samples.
   - In aggregate JSON, add:
     - "gallery_size", "retrieval_eligible"
     - "top1_mean_sim"
     - "rank_hist": { "1": count, "2-5": count, "6-10": count, "11+": count }

ACCEPTANCE
- CSV gains nn_nsdId, nn_sim, gt_sim.
- Two extra PNGs saved (clipscore and rank hist).
- JSON contains ablations (if --use-adapter) and gallery stats.
- No changes to existing flags or default behavior.

You are GitHub Copilot. Integrate FAISS for large retrieval.

GOALS
1) Add flag --faiss (store_true). When set, use FAISS IndexFlatIP with normalized vectors for retrieval similarity.
2) Build FAISS index over gallery_embeddings_normalized. For cosine similarity, use inner product on L2-normalized vectors.
3) Replace cosine_sim matrix multiply when --faiss is on:
   - For each gen embedding batch (size 512), query top-K = max(100, max(ks)) to compute rank positions efficiently.
4) Keep existing non-FAISS path intact.

ACCEPTANCE
- With --faiss, retrieval metrics match non-FAISS within floating tolerance.
- Memory footprint is lower for very large galleries and runtime scales sublinearly.

Nice-to-have checks:
- No leakage: if you ever set --gallery all, it's fine for retrieval only (you kept CLIPScore against matched GT — good).
- Cache consistency: keep the target parquet column name embedding; the detector handles others, but consistency helps.
- Re-run with more data: once you have, say, 100–500 matched reconstructions, look at rank histogram + R@K trends and adapter ablation deltas to quantify real gains.

You are GitHub Copilot. Modify scripts/eval_reconstruction.py to support non-trivial retrieval galleries.

GOALS
1) Add an argparse option:
   --gallery {matched,test,all} with default "matched".
   - matched: current behavior (gallery = GT of matched_nsd_ids)
   - test: gallery = all GT embeddings from the TEST split (even if not reconstructed)
   - all: gallery = all GT embeddings from the full subject index (train+val+test), but ONLY for retrieval (keep metrics computed between generated and their own GT as before).

2) Efficient gallery loading:
   - Use the same parquet cache as current (target or 512-D depending on --use-adapter).
   - Build a map nsdId -> embedding once, then slice for the selected gallery.
   - If any requested nsdId is missing from the cache, skip it with a WARN and keep going.

3) Retrieval computation:
   - Keep CLIPScore between (gen_embeddings, their matched GT embeddings) exactly as now.
   - For retrieval, compute cosine similarity between gen_embeddings and GALLERY embeddings (chosen by --gallery).
   - Build gt_indices by mapping each valid_nsd_id to its index inside the gallery array (if missing, drop that sample from retrieval only and log WARN).
   - Report R@1/5/10, mean/median rank, MRR over the subset that has gallery matches.

4) Logging:
   - Log the gallery type, the gallery size, and how many generated samples were eligible for retrieval (i.e., had their GT present in the selected gallery).
   - If --gallery != matched, add a note in the JSON under "retrieval_gallery": {"type": "...", "size": N}.

5) Performance:
   - For large galleries, compute cosine similarity via normalized embeddings and matrix multiply (no Python loops). If needed, chunk the gallery to avoid OOM (chunk size 10k rows with progress logs).

ACCEPTANCE
- Running with --gallery test on a subject with many cached GT embeddings produces non-trivial R@K and ranks.
- JSON includes retrieval_gallery block and accurate counts.
- Existing behavior is unchanged when --gallery matched (default).

Reconstruction Evaluation Script
=================================

Evaluates reconstructed images using CLIPScore and retrieval metrics.

Metrics:
- CLIPScore: Per-sample cosine similarity between generated and GT image embeddings
- Retrieval@K: How often generated image retrieves correct GT from gallery
- Ranking stats: Mean/median rank, MRR

Supports both 512-D (ViT-B/32) and target-D (768/1024 for diffusion models) evaluation
using the --use-adapter flag.

Scientific Context:
- CLIPScore measures semantic similarity without pixel-level matching (Hessel et al. 2021)
- Retrieval metrics evaluate how well generated images capture semantic content
- Standard evaluation for image generation quality in neural decoding

Usage:
    # Evaluate in 512-D space (ViT-B/32)
    python scripts/eval_reconstruction.py \\
        --subject subj01 \\
        --recon-dir outputs/recon/subj01/run_001 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --out-csv outputs/reports/subj01/recon_eval.csv \\
        --out-fig outputs/reports/subj01/recon_grid.png
    
    # Evaluate in 1024-D space (SD 2.1 target CLIP)
    python scripts/eval_reconstruction.py \\
        --subject subj01 \\
        --recon-dir outputs/recon/subj01/run_001 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --use-adapter \\
        --model-id stabilityai/stable-diffusion-2-1 \\
        --out-csv outputs/reports/subj01/recon_eval_1024.csv \\
        --out-fig outputs/reports/subj01/recon_grid_1024.png
"""

import argparse
import json
import logging
import sys
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq
import h5py



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem
from fmri2img.models.train_utils import train_val_test_split, torch_seed_all
from fmri2img.eval import clip_score, retrieval_at_k, compute_ranking_metrics
from fmri2img.utils.clip_utils import load_clip_model, encode_images


def detect_embedding_col_and_dim(parquet_path: Path) -> Tuple[str, int]:
    """
    Detect embedding column name and dimension from Parquet file using PyArrow schema.
    
    Args:
        parquet_path: Path to Parquet file
    
    Returns:
        (column_name, dimension) tuple
    
    Raises:
        RuntimeError if no valid embedding column found
    """
    schema = pq.read_schema(parquet_path)
    
    candidates = []
    for field in schema:
        # Check if it's a fixed_size_list of float
        if str(field.type).startswith('fixed_size_list<'):
            # Extract inner type and size
            type_str = str(field.type)
            if 'float' in type_str or 'double' in type_str:
                # Extract list size from type string like "fixed_size_list<float>[1024]"
                import re
                match = re.search(r'\[(\d+)\]', type_str)
                if match:
                    list_size = int(match.group(1))
                    candidates.append((field.name, list_size))
    
    if not candidates:
        raise RuntimeError(
            f"No embedding column found in {parquet_path}.\n"
            f"Expected fixed_size_list<float>[N] column.\n"
            f"Schema: {schema}"
        )
    
    # Prefer known names
    preferred_names = ["embedding", "clip1024", "clip768", "clip512"]
    for name in preferred_names:
        for col_name, dim in candidates:
            if col_name == name:
                logger.info(f"✅ Detected embedding column: '{col_name}' with dimension {dim}")
                return col_name, dim
    
    # Return first candidate if no preferred name found
    col_name, dim = candidates[0]
    logger.info(f"✅ Detected embedding column: '{col_name}' with dimension {dim}")
    return col_name, dim


def _gray_placeholder(size=(256, 256)) -> Image.Image:
    """Create a neutral gray placeholder image."""
    return Image.new("RGB", size, (128, 128, 128))


def _load_nsd_from_hdf5(nsd_id: int, hdf5_path: Optional[str] = None) -> Image.Image:
    """
    Load NSD image from HDF5 file.
    
    Args:
        nsd_id: NSD ID (0-based index into HDF5)
        hdf5_path: Optional override for HDF5 path
    
    Returns:
        PIL Image or gray placeholder on failure
    """
    if hdf5_path is None:
        hdf5_path = os.environ.get("NSD_HDF5", "cache/nsd_hdf5/nsd_stimuli.hdf5")
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "imgBrick" not in f:
                logger.warning(f"'imgBrick' dataset not found in {hdf5_path}")
                return _gray_placeholder()
            ds = f["imgBrick"]
            if nsd_id < 0 or nsd_id >= ds.shape[0]:
                logger.warning(f"HDF5 index out of range for nsdId={nsd_id} in {hdf5_path}")
                return _gray_placeholder()
            arr = ds[nsd_id]  # uint8 HxWx3
        return Image.fromarray(arr, mode="RGB")
    except Exception as e:
        logger.warning(f"Failed to load HDF5 for nsd{nsd_id}: {e}")
        return _gray_placeholder()


def _load_nsd_from_png(nsd_id: int) -> Image.Image:
    """
    Load NSD image from local PNG directory.
    
    Args:
        nsd_id: NSD ID
    
    Returns:
        PIL Image or gray placeholder on failure
    """
    png_dir = os.environ.get("NSD_PNG_DIR", "cache/nsd_png/")
    png_path = Path(png_dir) / f"nsd_{nsd_id:05d}.png"
    try:
        return Image.open(png_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load PNG for nsd{nsd_id} from {png_path}: {e}")
        return _gray_placeholder()


def _load_nsd_from_s3(nsd_id: int, s3_fs) -> Image.Image:
    """
    Load NSD image from S3.
    
    Args:
        nsd_id: NSD ID
        s3_fs: S3 filesystem object
    
    Returns:
        PIL Image
    
    Raises:
        RuntimeError on failure
    """
    key = f"nsd-data/nsddata_stimuli/stimuli/nsd/nsd_{nsd_id:05d}.png"
    try:
        import io
        with s3_fs.open(key, "rb") as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Cannot open {key}: {e}")


def load_vis_image(nsd_id: int, image_source: str, s3_fs, hdf5_path: Optional[str] = None, 
                   _first_success: List[Optional[str]] = [None]) -> Image.Image:
    """
    Load visualization image from specified source with fallback.
    
    Args:
        nsd_id: NSD ID
        image_source: "auto", "s3", "png", or "hdf5"
        s3_fs: S3 filesystem object
        hdf5_path: Optional override for HDF5 path
        _first_success: Internal state tracker for logging first success
    
    Returns:
        PIL Image (never raises; returns gray placeholder on failure)
    """
    def log_first_success(source: str):
        if _first_success[0] is None:
            _first_success[0] = source
            logger.info(f"✅ First visualization image loaded from: {source}")
    
    if image_source == "png":
        img = _load_nsd_from_png(nsd_id)
        if img.size != (128, 128) or img.getpixel((0, 0)) != (128, 128, 128):  # Not placeholder
            log_first_success("PNG")
        return img
    
    if image_source == "hdf5":
        img = _load_nsd_from_hdf5(nsd_id, hdf5_path)
        if img.size != (128, 128) or img.getpixel((0, 0)) != (128, 128, 128):  # Not placeholder
            log_first_success("HDF5")
        return img
    
    if image_source == "s3":
        try:
            img = _load_nsd_from_s3(nsd_id, s3_fs)
            log_first_success("S3")
            return img
        except Exception as e:
            logger.warning(str(e))
            return _gray_placeholder()
    
    # auto: try s3, then png, then hdf5
    fallback_attempts = []
    try:
        img = _load_nsd_from_s3(nsd_id, s3_fs)
        log_first_success("S3")
        return img
    except Exception as e:
        fallback_attempts.append(f"S3: {e}")
    
    # Try PNG
    img = _load_nsd_from_png(nsd_id)
    if img.size != (128, 128) or img.getpixel((0, 0)) != (128, 128, 128):  # Not placeholder
        log_first_success("PNG")
        return img
    else:
        fallback_attempts.append("PNG: not found or failed")
    
    # Try HDF5
    img = _load_nsd_from_hdf5(nsd_id, hdf5_path)
    if img.size != (128, 128) or img.getpixel((0, 0)) != (128, 128, 128):  # Not placeholder
        log_first_success("HDF5")
        return img
    else:
        fallback_attempts.append("HDF5: not found or failed")
    
    # All fallbacks failed
    logger.warning(f"⚠️  All sources failed for nsd{nsd_id}: {'; '.join(fallback_attempts)}")
    return _gray_placeholder()



class ImageEncoder:
    """
    Unified image encoder API supporting both OpenCLIP and HuggingFace CLIP models.
    """
    
    def __init__(self, model, processor, kind: str):
        """
        Args:
            model: OpenCLIP model or HF CLIPModel
            processor: Transform function (OpenCLIP) or CLIPProcessor (HF)
            kind: "openclip" or "hf_clip"
        """
        self.model = model
        self.processor = processor
        self.kind = kind
        
    def encode(self, pil_images: List[Image.Image], device: str, batch_size: int = 32) -> np.ndarray:
        """
        Encode images to CLIP embeddings.
        
        Args:
            pil_images: List of PIL images
            device: Device to run on
            batch_size: Batch size
            
        Returns:
            L2-normalized embeddings (N, D)
        """
        all_embeddings = []
        
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            
            with torch.no_grad():
                if self.kind == "openclip":
                    # OpenCLIP: use encode_image
                    if callable(self.processor):
                        # It's a transform function
                        inputs = torch.stack([self.processor(img) for img in batch]).to(device)
                    else:
                        # Shouldn't happen, but handle gracefully
                        inputs = torch.stack([self.processor.transforms(img) for img in batch]).to(device)
                    
                    embeddings = self.model.encode_image(inputs)
                    if isinstance(embeddings, tuple):
                        embeddings = embeddings[0]
                    embeddings = embeddings.cpu().numpy()
                    
                elif self.kind == "hf_clip":
                    # HuggingFace CLIP: use get_image_features
                    inputs = self.processor(images=batch, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Use get_image_features for proper projected embeddings
                    embeddings = self.model.get_image_features(**inputs)
                    embeddings = embeddings.cpu().numpy()
                else:
                    raise ValueError(f"Unknown encoder kind: {self.kind}")
                
                # L2 normalize (with safe division to avoid NaNs)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)


def load_target_clip_encoder(model_id: str, device: str) -> Tuple[ImageEncoder, int, str]:
    """
    Load the CLIP image encoder from a diffusion model.
    
    Returns projected CLIP embeddings (1024-D for SD 2.1, 768-D for SD 1.5).
    Uses OpenCLIP or HF CLIPModel.get_image_features() to get proper projection.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load on
    
    Returns:
        (encoder: ImageEncoder, target_dim: int, clip_space_label: str)
    """
    logger.info(f"Loading target CLIP encoder from {model_id}...")
    
    # Detect model type
    if "2-1" in model_id or "2.1" in model_id:
        target_dim = 1024
        logger.info("Detected SD 2.1 → OpenCLIP ViT-H/14 (1024-D)")
        
        # Try OpenCLIP first (preferred for 1024-D)
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14', 
                pretrained='laion2b_s32b_b79k',
                device=device
            )
            model.eval()
            encoder = ImageEncoder(model, preprocess, "openclip")
            clip_space = "1024-D (OpenCLIP ViT-H/14)"
            logger.info(f"✅ Loaded OpenCLIP ViT-H/14")
            return encoder, target_dim, clip_space
            
        except ImportError:
            logger.warning("⚠️  open_clip not available, falling back to HF transformers")
        except Exception as e:
            logger.warning(f"⚠️  OpenCLIP load failed: {e}, falling back to HF transformers")
        
        # Fallback: HF transformers with get_image_features
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        model.eval()
        encoder = ImageEncoder(model, processor, "hf_clip")
        clip_space = "1024-D (HF CLIP ViT-H/14)"
        logger.info(f"✅ Loaded HF CLIP ViT-H/14 (using get_image_features)")
        return encoder, target_dim, clip_space
        
    elif "1-5" in model_id or "1.5" in model_id:
        target_dim = 768
        logger.info("Detected SD 1.5 → CLIP ViT-L/14 (768-D)")
        
        # Use HF transformers with get_image_features
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.eval()
        encoder = ImageEncoder(model, processor, "hf_clip")
        clip_space = "768-D (CLIP ViT-L/14)"
        logger.info(f"✅ Loaded CLIP ViT-L/14 (using get_image_features)")
        return encoder, target_dim, clip_space
        
    else:
        # Default to 1024-D
        target_dim = 1024
        logger.warning("Unknown model, defaulting to 1024-D (OpenCLIP ViT-H/14)")
        
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14', 
                pretrained='laion2b_s32b_b79k',
                device=device
            )
            model.eval()
            encoder = ImageEncoder(model, preprocess, "openclip")
            clip_space = "1024-D (OpenCLIP ViT-H/14)"
            logger.info(f"✅ Loaded OpenCLIP ViT-H/14")
            return encoder, target_dim, clip_space
        except:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
            processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            model.eval()
            encoder = ImageEncoder(model, processor, "hf_clip")
            clip_space = "1024-D (HF CLIP ViT-H/14)"
            logger.info(f"✅ Loaded HF CLIP ViT-H/14 (using get_image_features)")
            return encoder, target_dim, clip_space


def _downsample_embeddings(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Downsample embeddings deterministically using evenly spaced indices.
    
    Args:
        embeddings: Input embeddings (N, D_in)
        target_dim: Target dimension (D_out < D_in)
    
    Returns:
        Downsampled embeddings (N, D_out)
    """
    input_dim = embeddings.shape[1]
    if target_dim >= input_dim:
        return embeddings
    
    # Select evenly spaced indices
    indices = np.linspace(0, input_dim - 1, target_dim, dtype=int)
    return embeddings[:, indices]


def _load_adapter(subject: str, in_dim: int, out_dim: int, device: str):
    """
    Load adapter checkpoint with dimension validation.
    
    Args:
        subject: Subject ID
        in_dim: Expected input dimension
        out_dim: Expected output dimension
        device: Device to load on
    
    Returns:
        Adapter model or None if load fails
    """
    adapter_path = Path(f"checkpoints/clip_adapter/{subject}/adapter.pt")
    
    if not adapter_path.exists():
        logger.warning(f"⚠️  Adapter not found: {adapter_path}")
        return None
    
    try:
        logger.info(f"🔧 Loading adapter: {adapter_path}")
        checkpoint = torch.load(adapter_path, map_location=device)
        
        # Handle different checkpoint formats
        adapter = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Try to extract weight matrix and validate dimensions
            weight_key = None
            for key in state_dict.keys():
                if 'weight' in key.lower() and len(state_dict[key].shape) == 2:
                    weight_key = key
                    break
            
            if weight_key:
                weight = state_dict[weight_key]
                # Check if dimensions match (weight can be out×in or in×out)
                if weight.shape == (out_dim, in_dim):
                    # Standard orientation: out×in
                    adapter = {'weight': weight.to(device), 'bias': state_dict.get(weight_key.replace('weight', 'bias'), None)}
                elif weight.shape == (in_dim, out_dim):
                    # Transposed orientation: in×out
                    adapter = {'weight': weight.t().to(device), 'bias': state_dict.get(weight_key.replace('weight', 'bias'), None)}
                else:
                    logger.warning(f"⚠️  Adapter dimension mismatch: expected ({out_dim}, {in_dim}) or ({in_dim}, {out_dim}), got {weight.shape}")
                    return None
                
                if adapter['bias'] is not None:
                    adapter['bias'] = adapter['bias'].to(device)
        else:
            # It's a model object
            adapter = checkpoint
            adapter.eval()
        
        return adapter
        
    except Exception as e:
        logger.warning(f"❌ Failed to load adapter: {e}")
        return None


def align_clip_spaces(
    gen_embeddings: np.ndarray,
    gt_embeddings: np.ndarray,
    use_adapter: bool,
    subject: str,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns generated and GT embeddings by:
    - Detecting mismatched dimensions
    - Applying adapter if possible
    - Falls back to deterministic projection or zero-padding
    
    Never collapses to (1,1) or crashes on dimension mismatch.
    
    Args:
        gen_embeddings: Generated image embeddings (N, D_gen)
        gt_embeddings: Ground truth embeddings (N, D_gt)
        use_adapter: Whether adapter mode is enabled
        subject: Subject ID for adapter path
        device: Device to load adapter on
    
    Returns:
        (gen_aligned, gt_aligned): Aligned embeddings with same dimension
    """
    gen_dim = gen_embeddings.shape[1]
    gt_dim = gt_embeddings.shape[1]
    
    # If dimensions match, just normalize and return
    if gen_dim == gt_dim:
        logger.info(f"✅ CLIP dimensions match: {gen_dim}-D")
        # Normalize (with safe division to avoid NaNs)
        gen_norms = np.linalg.norm(gen_embeddings, axis=1, keepdims=True)
        gen_norms[gen_norms == 0] = 1.0
        gen_embeddings = gen_embeddings / gen_norms
        
        gt_norms = np.linalg.norm(gt_embeddings, axis=1, keepdims=True)
        gt_norms[gt_norms == 0] = 1.0
        gt_embeddings = gt_embeddings / gt_norms
        return gen_embeddings, gt_embeddings
    
    logger.info("⚙️  Aligning CLIP spaces...")
    logger.warning(f"⚠️  Mismatch detected: gen={gen_dim}, gt={gt_dim}")
    
    # Try to apply adapter if available
    adapter_applied = False
    if use_adapter:
        adapter = _load_adapter(subject, gen_dim, gt_dim, device)
        
        if adapter is not None:
            try:
                gen_tensor = torch.tensor(gen_embeddings, dtype=torch.float32, device=device)
                
                if isinstance(adapter, dict):
                    # Apply linear projection: y = xW^T + b
                    with torch.no_grad():
                        gen_embeddings = torch.mm(gen_tensor, adapter['weight'].t())
                        if adapter['bias'] is not None:
                            gen_embeddings = gen_embeddings + adapter['bias']
                        gen_embeddings = gen_embeddings.cpu().numpy()
                else:
                    # It's a model object
                    with torch.no_grad():
                        gen_embeddings = adapter(gen_tensor).cpu().numpy()
                
                logger.info(f"✅ Adapter applied: new shape={gen_embeddings.shape}")
                adapter_applied = True
                
                # Update gen_dim after adapter application
                gen_dim = gen_embeddings.shape[1]
                
            except Exception as e:
                logger.warning(f"❌ Adapter application failed: {e}")
                logger.warning(f"💡 Falling back to automatic projection")
    
    # Fallback: deterministic dimension alignment
    if not adapter_applied or gen_dim != gt_dim:
        if gen_dim != gt_dim:
            logger.info("🔧 Adapter failed or dimensions still mismatched — using automatic projection")
        
        # Always project to the GT dimension (don't modify GT embeddings)
        target_dim = gt_dim
        
        if gen_dim > target_dim:
            # Downsample using evenly spaced indices
            logger.info(f"   Downsampling gen: {gen_dim} → {target_dim}")
            gen_embeddings = _downsample_embeddings(gen_embeddings, target_dim)
        elif gen_dim < target_dim:
            # Zero-pad to match target
            logger.info(f"   Zero-padding gen: {gen_dim} → {target_dim}")
            padding = np.zeros((gen_embeddings.shape[0], target_dim - gen_dim), dtype=gen_embeddings.dtype)
            gen_embeddings = np.hstack([gen_embeddings, padding])
        
        logger.info(f"✅ Alignment complete: gen={gen_embeddings.shape}, gt={gt_embeddings.shape}")
    
    # Final normalization (critical for cosine similarity, with safe division to avoid NaNs)
    gen_norms = np.linalg.norm(gen_embeddings, axis=1, keepdims=True)
    gen_norms[gen_norms == 0] = 1.0
    gen_embeddings = gen_embeddings / gen_norms
    
    gt_norms = np.linalg.norm(gt_embeddings, axis=1, keepdims=True)
    gt_norms[gt_norms == 0] = 1.0
    gt_embeddings = gt_embeddings / gt_norms
    
    return gen_embeddings, gt_embeddings


def find_reconstructed_images(
    recon_dir: Path,
    nsd_ids: np.ndarray,
    map_csv: Optional[Path] = None
) -> Dict[int, Path]:
    """
    Find reconstructed images for given NSD IDs.
    
    Supports two modes:
    1. Pattern matching: *_nsd{nsdId}.* or *_{nsdId}.*
    2. CSV mapping: columns [nsdId, path]
    
    Args:
        recon_dir: Directory containing reconstructed images
        nsd_ids: Array of NSD IDs to find
        map_csv: Optional CSV with nsdId→path mapping
    
    Returns:
        Dictionary: {nsdId: Path}
    """
    nsd_to_path = {}
    
    if map_csv:
        # Load from CSV
        logger.info(f"Loading image paths from {map_csv}")
        df = pd.read_csv(map_csv)
        
        if "nsdId" not in df.columns or "path" not in df.columns:
            raise ValueError("CSV must have 'nsdId' and 'path' columns")
        
        for _, row in df.iterrows():
            nsd_id = int(row["nsdId"])
            if nsd_id in nsd_ids:
                path = recon_dir / row["path"]
                if path.exists():
                    nsd_to_path[nsd_id] = path
                else:
                    logger.warning(f"Image not found: {path}")
    
    else:
        # Pattern matching
        logger.info(f"Searching for images in {recon_dir}")
        
        # Find all image files
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            image_files.extend(recon_dir.glob(ext))
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Try to extract NSD ID from filename
        patterns = [
            r"nsd_?(\d+)",  # nsd12345 or nsd_12345
            r"_(\d{5,})(?:_|\.)",  # _12345_ or _12345.
        ]
        
        for img_path in image_files:
            filename = img_path.stem
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    nsd_id = int(match.group(1))
                    if nsd_id in nsd_ids:
                        nsd_to_path[nsd_id] = img_path
                    break
    
    logger.info(f"Matched {len(nsd_to_path)}/{len(nsd_ids)} images")
    
    if len(nsd_to_path) == 0:
        logger.error("No images matched! Check filename pattern or provide --map-csv")
        logger.error("Expected patterns: *_nsd{ID}.* or *_{ID}.*")
        logger.error(f"Example files in {recon_dir}:")
        for i, f in enumerate(recon_dir.glob("*")):
            if i >= 5:
                break
            logger.error(f"  {f.name}")
    
    return nsd_to_path


def build_retrieval_gallery(
    gallery_type: str,
    matched_nsd_ids: np.ndarray,
    test_nsd_ids: np.ndarray,
    all_nsd_ids: np.ndarray,
    embeddings_dict: dict,
    device: str
) -> tuple:
    """
    Build retrieval gallery embeddings based on gallery type.
    
    Args:
        gallery_type: One of "matched", "test", "all"
        matched_nsd_ids: NSD IDs of reconstructed images
        test_nsd_ids: All test split NSD IDs
        all_nsd_ids: All NSD IDs (train+val+test)
        embeddings_dict: Dict mapping nsdId -> embedding array
        device: Device for tensor operations
    
    Returns:
        gallery_embeddings: (N, D) array of gallery embeddings
        gallery_nsd_ids: (N,) array of NSD IDs in gallery
        nsd_to_gallery_idx: Dict mapping nsdId -> index in gallery
    """
    logger.info(f"Building retrieval gallery: type={gallery_type}")
    
    # Select gallery NSD IDs based on type
    if gallery_type == "matched":
        gallery_nsd_ids = matched_nsd_ids
    elif gallery_type == "test":
        gallery_nsd_ids = test_nsd_ids
    elif gallery_type == "all":
        gallery_nsd_ids = all_nsd_ids
    else:
        raise ValueError(f"Unknown gallery type: {gallery_type}")
    
    logger.info(f"Gallery candidate size: {len(gallery_nsd_ids)} NSD IDs")
    
    # Build gallery embeddings (skip missing)
    gallery_embeddings_list = []
    valid_gallery_nsd_ids = []
    
    for nsd_id in gallery_nsd_ids:
        emb = embeddings_dict.get(nsd_id)
        if emb is None:
            logger.warning(f"Missing embedding for nsd{nsd_id} in gallery, skipping")
            continue
        gallery_embeddings_list.append(emb)
        valid_gallery_nsd_ids.append(nsd_id)
    
    if len(gallery_embeddings_list) == 0:
        raise ValueError(f"No embeddings found for gallery type '{gallery_type}'")
    
    gallery_embeddings = np.vstack(gallery_embeddings_list)
    gallery_nsd_ids = np.array(valid_gallery_nsd_ids)
    
    # Build lookup map
    nsd_to_gallery_idx = {nsd_id: idx for idx, nsd_id in enumerate(gallery_nsd_ids)}
    
    logger.info(f"✅ Gallery built: {len(gallery_nsd_ids)} embeddings (dim={gallery_embeddings.shape[1]})")
    
    return gallery_embeddings, gallery_nsd_ids, nsd_to_gallery_idx


def save_histogram(values: np.ndarray, output_path: Path, title: str, xlabel: str, bins: int = 50) -> None:
    """
    Save histogram of values to PNG.
    
    Args:
        values: Array of values to plot
        output_path: Output path for histogram
        title: Plot title
        xlabel: X-axis label
        bins: Number of bins (default: 50)
    """
    if len(values) == 0:
        logger.warning(f"Empty array for histogram {output_path.name}, skipping")
        return
    
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Histogram saved to {output_path}")


def save_nn_jsonl(
    nsd_ids: List[int],
    clip_scores: np.ndarray,
    ranks: np.ndarray,
    similarity_matrix: np.ndarray,
    gallery_nsd_ids: np.ndarray,
    output_path: Path,
    top_k: int = 10
) -> None:
    """
    Save per-sample nearest neighbor information to JSONL.
    
    Args:
        nsd_ids: List of sample NSD IDs
        clip_scores: CLIPScore for each sample
        ranks: Rank matrix (samples x gallery) sorted by similarity
        similarity_matrix: Cosine similarity matrix (samples x gallery)
        gallery_nsd_ids: NSD IDs in gallery
        output_path: Output JSONL path
        top_k: Number of top neighbors to save (default: 10)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i, nsd_id in enumerate(nsd_ids):
            # Get top-k neighbors
            top_k_indices = ranks[i, :top_k]
            top_k_sims = similarity_matrix[i, top_k_indices]
            
            topk_list = []
            for idx, sim in zip(top_k_indices, top_k_sims):
                if idx < len(gallery_nsd_ids):
                    topk_list.append({
                        "nsdId": int(gallery_nsd_ids[idx]),
                        "sim": float(sim)
                    })
            
            record = {
                "nsdId": int(nsd_id),
                "clipscore": float(clip_scores[i]),
                "rank": int(ranks[i, 0]) + 1,  # Rank of GT (1-based)
                "topk": topk_list
            }
            
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"✅ NN JSONL saved to {output_path} ({len(nsd_ids)} samples, top-{top_k})")


def compute_rank_histogram(ranks: np.ndarray) -> dict:
    """
    Compute rank histogram buckets.
    
    Args:
        ranks: Array of ranks (1-based)
    
    Returns:
        Dictionary with bucket counts
    """
    if len(ranks) == 0:
        return {"1": 0, "2-5": 0, "6-10": 0, "11+": 0}
    
    valid_ranks = ranks[ranks > 0]  # Exclude invalid ranks
    
    return {
        "1": int(np.sum(valid_ranks == 1)),
        "2-5": int(np.sum((valid_ranks >= 2) & (valid_ranks <= 5))),
        "6-10": int(np.sum((valid_ranks >= 6) & (valid_ranks <= 10))),
        "11+": int(np.sum(valid_ranks > 10))
    }


def create_evaluation_grid(
    nsd_ids: List[int],
    gt_images: List[Image.Image],
    nn_images: List[Image.Image],
    gen_images: List[Image.Image],
    clip_scores: np.ndarray,
    nn_ranks: np.ndarray,
    output_path: Path,
    max_rows: int = 16
) -> None:
    """
    Create visualization grid: GT | NN | Generated
    
    Args:
        nsd_ids: List of NSD IDs
        gt_images: List of ground truth images
        nn_images: List of nearest neighbor images
        gen_images: List of generated images
        clip_scores: CLIPScore for each sample
        nn_ranks: Rank of GT in retrieval for each sample
        output_path: Output path for grid image
        max_rows: Maximum number of rows to show
    """
    n_samples = min(len(nsd_ids), max_rows)
    
    fig = plt.figure(figsize=(15, 4 * n_samples))
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, hspace=0.3, wspace=0.1)
    
    for i in range(n_samples):
        # Ground truth
        ax_gt = fig.add_subplot(gs[i, 0])
        ax_gt.imshow(gt_images[i])
        ax_gt.axis('off')
        ax_gt.set_title(f"GT (nsd{nsd_ids[i]})", fontsize=10, pad=5)
        
        # Nearest neighbor
        ax_nn = fig.add_subplot(gs[i, 1])
        ax_nn.imshow(nn_images[i])
        ax_nn.axis('off')
        rank_str = f"Rank: {nn_ranks[i]}" if nn_ranks[i] > 0 else "Rank: 1 (Perfect)"
        ax_nn.set_title(f"NN Retrieval\n{rank_str}", fontsize=10, pad=5)
        
        # Generated
        ax_gen = fig.add_subplot(gs[i, 2])
        ax_gen.imshow(gen_images[i])
        ax_gen.axis('off')
        score_color = 'green' if clip_scores[i] > 0.5 else 'orange' if clip_scores[i] > 0.3 else 'red'
        ax_gen.set_title(f"Generated\nCLIPScore: {clip_scores[i]:.3f}", 
                        fontsize=10, pad=5, color=score_color)
    
    plt.suptitle("Reconstruction Evaluation: GT | Nearest Neighbor | Generated", 
                fontsize=14, y=0.995)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Evaluation grid saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstructed images")
    
    # Data paths
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="NSD index root directory")
    parser.add_argument("--index-file", help="Path to single index file (overrides --index-root)")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    
    # Reconstruction
    parser.add_argument("--recon-dir", required=True,
                       help="Directory containing reconstructed images")
    parser.add_argument("--map-csv", help="Optional CSV mapping nsdId→path")
    
    # CLIP cache and model
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet",
                       help="Path to ViT-B/32 CLIP cache (512-D)")
    
    # Adapter mode
    parser.add_argument("--use-adapter", action="store_true",
                       help="Evaluate in target CLIP space (768/1024-D)")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1",
                       help="Diffusion model ID for target CLIP (if --use-adapter)")
    parser.add_argument("--target-clip", type=str, default=None,
                       help="Optional CLIP model override for evaluating generated images (e.g., 'openai/clip-vit-large-patch14')")
    parser.add_argument("--target-cache-dir", default="outputs/clip_cache",
                       help="Directory for target CLIP embedding cache")
    
    # Output
    parser.add_argument("--out-csv", required=True,
                       help="Output CSV path for per-sample metrics")
    parser.add_argument("--out-fig", required=True,
                       help="Output image path for visualization grid")
    parser.add_argument("--out-json", help="Optional JSON path for aggregate metrics")
    
    # System
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", default="configs/data.yaml",
                       help="Data config file")
    parser.add_argument("--image-source", choices=["auto", "s3", "png", "hdf5"], default="auto",
                       help="Source for GT/NN visualization images (auto tries s3→png→hdf5)")
    parser.add_argument("--nsd-hdf5", type=str, default=None,
                       help="Override HDF5 path (default: NSD_HDF5 env or 'cache/nsd_hdf5/nsd_stimuli.hdf5')")
    
    # Retrieval gallery
    parser.add_argument("--gallery", choices=["matched", "test", "all"], default="matched",
                       help="Retrieval gallery: 'matched' (only GT of reconstructed images), "
                            "'test' (all test split GTs), 'all' (train+val+test GTs)")
    
    # Performance
    parser.add_argument("--faiss", action="store_true",
                       help="Use FAISS IndexFlatIP for fast retrieval (recommended for large galleries)")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch_seed_all(args.seed)
    np.random.seed(args.seed)
    
    try:
        logger.info("=" * 80)
        logger.info("RECONSTRUCTION EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Recon dir: {args.recon_dir}")
        logger.info(f"Adapter mode: {'ENABLED' if args.use_adapter else 'DISABLED'}")
        if args.use_adapter:
            logger.info(f"Target model: {args.model_id}")
        logger.info(f"Device: {args.device}")
        
        # Load subject index
        if args.index_file:
            logger.info(f"Loading index from {args.index_file}")
            df = pd.read_parquet(args.index_file)
        else:
            logger.info(f"Loading index for {args.subject} from {args.index_root}")
            df = read_subject_index(args.index_root, args.subject)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Split data (same as training)
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        splits_config = config.get("preprocessing", {}).get("splits", {})
        
        _, _, test_df = train_val_test_split(
            df,
            train_ratio=splits_config.get("train_ratio", 0.8),
            val_ratio=splits_config.get("val_ratio", 0.1),
            test_ratio=splits_config.get("test_ratio", 0.1),
            random_seed=splits_config.get("random_seed", 42)
        )
        
        test_nsd_ids = test_df["nsdId"].values
        logger.info(f"Test set: {len(test_nsd_ids)} samples")
        
        # Get all NSD IDs for "all" gallery option
        all_nsd_ids = df["nsdId"].values
        logger.info(f"Full dataset: {len(all_nsd_ids)} samples")
        
        # Find reconstructed images
        recon_dir = Path(args.recon_dir)
        map_csv = Path(args.map_csv) if args.map_csv else None
        
        nsd_to_path = find_reconstructed_images(recon_dir, test_nsd_ids, map_csv)
        
        if len(nsd_to_path) == 0:
            logger.error("No reconstructed images found!")
            return 1
        
        if len(nsd_to_path) < len(test_nsd_ids):
            logger.warning(f"Only found {len(nsd_to_path)}/{len(test_nsd_ids)} images")
            logger.warning("Evaluation will be partial")
        
        # Filter test set to matched images
        matched_nsd_ids = np.array(sorted(nsd_to_path.keys()))
        logger.info(f"Evaluating {len(matched_nsd_ids)} matched samples")
        
        # Setup CLIP model
        encoder = None
        target_dim = None
        clip_space = None
        
        if args.use_adapter:
            # Check for CLIP model override
            if args.target_clip:
                from transformers import CLIPModel, CLIPProcessor
                logger.info(f"⚙️  Overriding target CLIP encoder: {args.target_clip}")
                model = CLIPModel.from_pretrained(args.target_clip).to(args.device)
                processor = CLIPProcessor.from_pretrained(args.target_clip)
                model.eval()
                encoder = ImageEncoder(model, processor, "hf_clip")
                target_dim = model.config.projection_dim
                clip_space = f"{target_dim}-D (custom: {args.target_clip})"
            else:
                # Auto-detect: check GT cache dimension
                target_cache_file = Path(args.target_cache_dir) / f"target_clip_{args.model_id.replace('/', '_')}.parquet"
                
                if target_cache_file.exists():
                    # Detect GT embedding dimension
                    try:
                        _, gt_detected_dim = detect_embedding_col_and_dim(target_cache_file)
                        logger.info(f"🎯 Detected GT dimension: {gt_detected_dim}-D")
                        
                        # If 1024-D, force OpenCLIP ViT-H/14 for generated images
                        if gt_detected_dim == 1024:
                            logger.info("🎯 Forcing OpenCLIP ViT-H/14 for generated images to match GT")
                            try:
                                import open_clip
                                model, _, preprocess = open_clip.create_model_and_transforms(
                                    'ViT-H-14', 
                                    pretrained='laion2b_s32b_b79k',
                                    device=args.device
                                )
                                model.eval()
                                encoder = ImageEncoder(model, preprocess, "openclip")
                                target_dim = 1024
                                clip_space = "1024-D (OpenCLIP ViT-H/14 - auto-matched to GT)"
                                logger.info(f"✅ Using OpenCLIP ViT-H/14 for encoding generated images")
                            except ImportError:
                                logger.warning("⚠️  open_clip not available, using HF CLIP")
                                from transformers import CLIPModel, CLIPProcessor
                                model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(args.device)
                                processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
                                model.eval()
                                encoder = ImageEncoder(model, processor, "hf_clip")
                                target_dim = 1024
                                clip_space = "1024-D (HF CLIP ViT-H/14 - auto-matched to GT)"
                        else:
                            # Load from model_id
                            encoder, target_dim, clip_space = load_target_clip_encoder(args.model_id, args.device)
                    except Exception as e:
                        logger.warning(f"⚠️  Could not auto-detect GT dimension: {e}")
                        encoder, target_dim, clip_space = load_target_clip_encoder(args.model_id, args.device)
                else:
                    # Load from model_id
                    encoder, target_dim, clip_space = load_target_clip_encoder(args.model_id, args.device)
        else:
            # Load ViT-B/32
            clip_model, preprocess, clip_dim = load_clip_model()
            clip_model = clip_model.to(args.device)
            clip_model.eval()
            clip_space = "512-D (ViT-B/32)"
            target_dim = 512
        
        logger.info(f"CLIP space: {clip_space}")
        
        # Load ALL ground truth CLIP embeddings (for gallery)
        # We'll load all embeddings and build a dict, then slice for matched/test/all as needed
        all_gt_embeddings_dict = {}
        
        if args.use_adapter:
            # Load target CLIP embeddings for GT images using PyArrow
            target_cache_file = Path(args.target_cache_dir) / f"target_clip_{args.model_id.replace('/', '_')}.parquet"
            logger.info(f"Loading ground truth embeddings from {target_cache_file}")
            
            if target_cache_file.exists():
                logger.info(f"Loading target GT embeddings from {target_cache_file}")
                
                # Detect embedding column name and dimension
                embed_col_name, embed_dim = detect_embedding_col_and_dim(target_cache_file)
                logger.info(f"📊 Reading embeddings from column '{embed_col_name}' (dim={embed_dim})")
                
                # Read only needed columns using PyArrow
                table = pq.read_table(target_cache_file, columns=["nsdId", embed_col_name])
                df_target = table.to_pandas()
                
                # Build dict with ALL embeddings (not just matched)
                for _, row in df_target.iterrows():
                    nsd_id = int(row["nsdId"])
                    # Handle both list and numpy array
                    emb = row[embed_col_name]
                    if not isinstance(emb, np.ndarray):
                        emb = np.array(emb)
                    all_gt_embeddings_dict[nsd_id] = emb
                
                logger.info(f"✅ Loaded {len(all_gt_embeddings_dict)} GT embeddings from cache")
            else:
                logger.error(f"Target CLIP cache not found: {target_cache_file}")
                logger.error("Please run train_clip_adapter.py first to generate target embeddings")
                return 1
        else:
            # Use 512-D cache - load ALL embeddings
            logger.info(f"Loading ground truth embeddings from {args.clip_cache}")
            clip_cache = CLIPCache(args.clip_cache).load()
            
            # Get all cached nsdIds
            all_cache_nsd_ids = clip_cache.list_cached_ids()
            all_gt_embeddings_dict = clip_cache.get(all_cache_nsd_ids)
            logger.info(f"✅ Loaded {len(all_gt_embeddings_dict)} GT embeddings from 512-D cache")
        
        # Extract matched GT embeddings for CLIPScore computation
        matched_gt_embeddings_dict = {nsd_id: all_gt_embeddings_dict[nsd_id] 
                                      for nsd_id in matched_nsd_ids 
                                      if nsd_id in all_gt_embeddings_dict}
        
        # Load and encode generated images
        logger.info("Loading and encoding generated images...")
        gen_images = []
        gen_embeddings_list = []
        valid_nsd_ids = []
        
        for nsd_id in matched_nsd_ids:
            try:
                # Load image
                img_path = nsd_to_path[nsd_id]
                img = Image.open(img_path).convert("RGB")
                gen_images.append(img)
                valid_nsd_ids.append(nsd_id)
                
            except Exception as e:
                logger.warning(f"Failed to load image for nsd{nsd_id}: {e}")
                continue
        
        logger.info(f"Loaded {len(gen_images)} images")
        
        # Encode generated images
        logger.info("Encoding generated images...")
        if args.use_adapter:
            gen_embeddings = encoder.encode(gen_images, args.device)
        else:
            gen_embeddings = encode_images(
                images=gen_images,
                model=clip_model,
                preprocess=preprocess,
                device=args.device,
            )
        
        logger.info(f"✅ Generated embeddings: {gen_embeddings.shape}")
        
        # Get GT embeddings in order (for CLIPScore computation)
        gt_embeddings_list = []
        for nsd_id in valid_nsd_ids:
            emb = matched_gt_embeddings_dict.get(nsd_id)
            if emb is None:
                logger.warning(f"Missing GT embedding for nsd{nsd_id}")
                continue
            gt_embeddings_list.append(emb)
        
        gt_embeddings = np.vstack(gt_embeddings_list)
        logger.info(f"✅ GT embeddings: {gt_embeddings.shape}")
        
        # Ensure same length
        min_len = min(len(gen_embeddings), len(gt_embeddings), len(valid_nsd_ids))
        gen_embeddings = gen_embeddings[:min_len]
        gt_embeddings = gt_embeddings[:min_len]
        valid_nsd_ids = valid_nsd_ids[:min_len]
        gen_images = gen_images[:min_len]
        
        # Save original embeddings for ablation
        gen_embeddings_original = gen_embeddings.copy()
        gt_embeddings_original = gt_embeddings.copy()
        
        # Align CLIP spaces (handles dimension mismatch, adapter application, normalization)
        gen_embeddings, gt_embeddings = align_clip_spaces(
            gen_embeddings, gt_embeddings, args.use_adapter, args.subject, args.device
        )
        
        # Compute CLIPScore
        logger.info("Computing CLIPScore...")
        clip_scores = clip_score(gen_embeddings, gt_embeddings)
        
        logger.info(f"CLIPScore: {clip_scores.mean():.3f} ± {clip_scores.std():.3f}")
        logger.info(f"Min: {clip_scores.min():.3f}, Max: {clip_scores.max():.3f}")
        
        # Build retrieval gallery
        logger.info("=" * 80)
        logger.info(f"Building retrieval gallery: --gallery {args.gallery}")
        
        gallery_embeddings, gallery_nsd_ids, nsd_to_gallery_idx = build_retrieval_gallery(
            gallery_type=args.gallery,
            matched_nsd_ids=matched_nsd_ids,
            test_nsd_ids=test_nsd_ids,
            all_nsd_ids=all_nsd_ids,
            embeddings_dict=all_gt_embeddings_dict,
            device=args.device
        )
        
        # Normalize gallery embeddings
        gallery_norms = np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
        gallery_norms[gallery_norms == 0] = 1.0  # Prevent division by zero
        gallery_embeddings_normalized = gallery_embeddings / gallery_norms
        
        # Build gt_indices: for each valid_nsd_id, find its index in the gallery
        gt_indices = []
        retrieval_valid_mask = []
        
        for nsd_id in valid_nsd_ids:
            if nsd_id in nsd_to_gallery_idx:
                gt_indices.append(nsd_to_gallery_idx[nsd_id])
                retrieval_valid_mask.append(True)
            else:
                logger.warning(f"nsd{nsd_id} not in gallery, excluding from retrieval metrics")
                gt_indices.append(-1)  # Placeholder
                retrieval_valid_mask.append(False)
        
        gt_indices = np.array(gt_indices)
        retrieval_valid_mask = np.array(retrieval_valid_mask)
        
        n_retrieval_eligible = retrieval_valid_mask.sum()
        logger.info(f"Retrieval eligible: {n_retrieval_eligible}/{len(valid_nsd_ids)} samples have GT in gallery")
        
        # Compute retrieval metrics (only for samples with GT in gallery)
        if n_retrieval_eligible > 0:
            logger.info("Computing retrieval metrics...")
            
            gen_embeddings_for_retrieval = gen_embeddings[retrieval_valid_mask]
            gt_indices_for_retrieval = gt_indices[retrieval_valid_mask]
            
            retrieval_metrics = retrieval_at_k(
                gen_embeddings_for_retrieval, 
                gallery_embeddings_normalized, 
                gt_indices_for_retrieval, 
                ks=(1, 5, 10)
            )
            
            ranking_metrics = compute_ranking_metrics(
                gen_embeddings_for_retrieval, 
                gallery_embeddings_normalized, 
                gt_indices_for_retrieval
            )
            
            for k, v in retrieval_metrics.items():
                logger.info(f"{k}: {v:.4f} ({v*100:.2f}%)")
            
            logger.info(f"Mean rank: {ranking_metrics['mean_rank']:.2f}")
            logger.info(f"Median rank: {ranking_metrics['median_rank']:.2f}")
            logger.info(f"MRR: {ranking_metrics['mrr']:.4f}")
        else:
            logger.warning("No samples eligible for retrieval metrics!")
            retrieval_metrics = {}
            ranking_metrics = {}
        
        # Find NN ranks for visualization (use full gallery)
        logger.info("Computing similarity for visualization...")
        from fmri2img.eval import cosine_sim
        
        # Choose retrieval method: FAISS or numpy
        if args.faiss:
            try:
                import faiss
                logger.info(f"Using FAISS IndexFlatIP for retrieval (gallery size: {len(gallery_embeddings_normalized)})")
                
                # Build FAISS index
                d = gallery_embeddings_normalized.shape[1]
                index = faiss.IndexFlatIP(d)  # Inner product for normalized vectors = cosine similarity
                index.add(gallery_embeddings_normalized.astype(np.float32))
                
                # Query all gen embeddings
                k = min(len(gallery_embeddings_normalized), 100)  # Top-100 or gallery size
                similarities, indices = index.search(gen_embeddings.astype(np.float32), k)
                
                # Build full similarity matrix for compatibility
                sim = np.zeros((len(gen_embeddings), len(gallery_embeddings_normalized)), dtype=np.float32)
                for i in range(len(gen_embeddings)):
                    sim[i, indices[i]] = similarities[i]
                
                ranks = indices  # Already sorted by similarity
                logger.info("✅ FAISS retrieval complete")
                
            except ImportError:
                logger.warning("⚠️  FAISS not available, falling back to numpy")
                args.faiss = False
        
        if not args.faiss:
            # Standard numpy path
            # Compute similarity in chunks to avoid OOM for large galleries
            chunk_size = 10000
            n_chunks = (len(gallery_embeddings_normalized) + chunk_size - 1) // chunk_size
            
            if n_chunks > 1:
                logger.info(f"Computing similarity in {n_chunks} chunks (gallery size: {len(gallery_embeddings_normalized)})")
            
            sim_chunks = []
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(gallery_embeddings_normalized))
                
                gallery_chunk = gallery_embeddings_normalized[start_idx:end_idx]
                sim_chunk = cosine_sim(gen_embeddings, gallery_chunk)
                sim_chunks.append(sim_chunk)
                
                if n_chunks > 1:
                    logger.info(f"  Chunk {chunk_idx+1}/{n_chunks}: [{start_idx}:{end_idx}]")
            
            sim = np.concatenate(sim_chunks, axis=1)
            ranks = np.argsort(-sim, axis=1)
        
        # Compute NN ranks: find where each sample's GT appears in the ranked list
        nn_ranks = []
        for i in range(len(gen_embeddings)):
            if gt_indices[i] >= 0:  # Valid GT in gallery
                gt_pos = np.where(ranks[i] == gt_indices[i])[0][0]
                nn_ranks.append(gt_pos + 1)  # 1-based rank
            else:
                nn_ranks.append(-1)  # Not in gallery
        nn_ranks = np.array(nn_ranks)
        
        # Load images for visualization (GT and NN)
        logger.info("Loading images for visualization...")
        hdf5_path = args.nsd_hdf5 or os.environ.get('NSD_HDF5', 'cache/nsd_hdf5/nsd_stimuli.hdf5')
        png_dir = os.environ.get('NSD_PNG_DIR', 'cache/nsd_png/')
        logger.info(f"Visualization source: {args.image_source}")
        if args.image_source in ['auto', 'hdf5'] or (args.image_source == 'auto'):
            logger.info(f"  HDF5 path: {hdf5_path}")
        if args.image_source in ['auto', 'png']:
            logger.info(f"  PNG dir: {png_dir}")
        
        s3_fs = get_s3_filesystem()
        gt_images_vis = []
        nn_images_vis = []
        
        for i, nsd_id in enumerate(valid_nsd_ids[:16]):  # Limit to 16 for grid
            # Load GT image
            gt_img = load_vis_image(nsd_id, args.image_source, s3_fs, hdf5_path)
            gt_images_vis.append(gt_img)
            
            # Load NN image
            nn_idx = ranks[i, 0]  # Top-1 retrieval from gallery
            if nn_idx < len(gallery_nsd_ids):
                nn_nsd_id = gallery_nsd_ids[nn_idx]
                nn_img = load_vis_image(nn_nsd_id, args.image_source, s3_fs, hdf5_path)
            else:
                # Fallback to gray placeholder if out of bounds
                nn_img = _gray_placeholder()
            nn_images_vis.append(nn_img)
        
        # Create visualization grid
        logger.info("Creating visualization grid...")
        create_evaluation_grid(
            valid_nsd_ids[:16],
            gt_images_vis,
            nn_images_vis,
            gen_images[:16],
            clip_scores[:16],
            nn_ranks[:16],
            Path(args.out_fig)
        )
        
        # Save per-sample CSV
        logger.info("Saving per-sample metrics...")
        
        # Compute per-sample retrieval indicators and NN info
        r_at_1 = []
        r_at_5 = []
        r_at_10 = []
        nn_nsd_ids = []
        nn_sims = []
        gt_sims = []
        
        for i, rank in enumerate(nn_ranks):
            if rank > 0:  # Valid rank
                r_at_1.append(1 if rank == 1 else 0)
                r_at_5.append(1 if rank <= 5 else 0)
                r_at_10.append(1 if rank <= 10 else 0)
            else:
                # Not in gallery
                r_at_1.append(-1)
                r_at_5.append(-1)
                r_at_10.append(-1)
            
            # Top-1 NN info
            top1_idx = ranks[i, 0]
            if top1_idx < len(gallery_nsd_ids):
                nn_nsd_ids.append(int(gallery_nsd_ids[top1_idx]))
                nn_sims.append(float(sim[i, top1_idx]))
            else:
                nn_nsd_ids.append(-1)
                nn_sims.append(0.0)
            
            # GT similarity (diagonal of gen vs matched GT)
            gt_sims.append(float(clip_scores[i]))  # CLIPScore is already GT similarity
        
        # Log first 5 samples' NN info
        logger.info("Top-1 neighbors for first 5 samples:")
        for i in range(min(5, len(valid_nsd_ids))):
            logger.info(f"  nsd{valid_nsd_ids[i]}: NN=nsd{nn_nsd_ids[i]}, sim={nn_sims[i]:.4f}, gt_sim={gt_sims[i]:.4f}, rank={nn_ranks[i]}")
        
        results_df = pd.DataFrame({
            "nsdId": valid_nsd_ids,
            "clipscore": clip_scores,
            "rank": nn_ranks,
            "r@1": r_at_1,
            "r@5": r_at_5,
            "r@10": r_at_10,
            "in_gallery": [1 if r > 0 else 0 for r in nn_ranks],
            "nn_nsdId": nn_nsd_ids,
            "nn_sim": nn_sims,
            "gt_sim": gt_sims,
        })
        
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.out_csv, index=False)
        logger.info(f"✅ Per-sample metrics saved to {args.out_csv}")
        
        # Save NN JSONL
        nn_jsonl_path = Path(args.out_csv).parent / (Path(args.out_csv).stem + "__nn.jsonl")
        save_nn_jsonl(
            valid_nsd_ids,
            clip_scores,
            ranks,
            sim,
            gallery_nsd_ids,
            nn_jsonl_path,
            top_k=10
        )
        
        # Save histograms
        logger.info("Saving distribution plots...")
        out_stem = Path(args.out_csv).stem
        out_dir = Path(args.out_csv).parent
        
        clipscore_hist_path = out_dir / f"{out_stem}__clipscore_hist.png"
        save_histogram(clip_scores, clipscore_hist_path, 
                      "CLIPScore Distribution", "CLIPScore", bins=50)
        
        rank_hist_path = out_dir / f"{out_stem}__rank_hist.png"
        valid_ranks_for_hist = nn_ranks[nn_ranks > 0]
        save_histogram(valid_ranks_for_hist, rank_hist_path,
                      "Retrieval Rank Distribution", "Rank (1-based)", bins=50)
        
        # Save aggregate JSON
        aggregate_metrics = {
            "subject": args.subject,
            "recon_dir": str(args.recon_dir),
            "clip_space": clip_space,
            "clip_dim": target_dim,
            "use_adapter": args.use_adapter,
            "model_id": args.model_id if args.use_adapter else "ViT-B/32",
            "n_samples": len(valid_nsd_ids),
            "n_test_total": len(test_nsd_ids),
            "clipscore": {
                "mean": float(clip_scores.mean()),
                "std": float(clip_scores.std()),
                "min": float(clip_scores.min()),
                "max": float(clip_scores.max()),
            },
            "retrieval": retrieval_metrics,
            "ranking": ranking_metrics,
            "gallery_size": len(gallery_nsd_ids),
            "retrieval_eligible": int(n_retrieval_eligible),
            "top1_mean_sim": float(np.mean(nn_sims)) if len(nn_sims) > 0 else 0.0,
            "rank_hist": compute_rank_histogram(nn_ranks),
        }
        
        # Add retrieval gallery info
        aggregate_metrics["retrieval_gallery"] = {
            "type": args.gallery,
            "size": len(gallery_nsd_ids),
            "n_eligible": int(n_retrieval_eligible),
            "n_total": len(valid_nsd_ids),
        }
        
        # Adapter ablation (if --use-adapter is on)
        if args.use_adapter:
            logger.info("=" * 80)
            logger.info("Running adapter ablation (without adapter)...")
            
            # Recompute without adapter using fallback alignment
            gen_embeddings_no_adapter, gt_embeddings_no_adapter = align_clip_spaces(
                gen_embeddings_original.copy(),
                gt_embeddings_original.copy(),
                use_adapter=False,  # Force no adapter
                subject=args.subject,
                device=args.device
            )
            
            # CLIPScore without adapter
            clip_scores_no_adapter = clip_score(gen_embeddings_no_adapter, gt_embeddings_no_adapter)
            
            # Rebuild gallery without adapter (align all GT embeddings)
            gallery_embeddings_no_adapter_list = []
            for nsd_id in gallery_nsd_ids:
                emb = all_gt_embeddings_dict.get(nsd_id)
                if emb is not None:
                    gallery_embeddings_no_adapter_list.append(emb)
            
            if len(gallery_embeddings_no_adapter_list) > 0:
                gallery_embeddings_no_adapter = np.vstack(gallery_embeddings_no_adapter_list)
                
                # Align gallery without adapter
                dummy_gen = np.zeros((1, gallery_embeddings_no_adapter.shape[1]))
                _, gallery_aligned_no_adapter = align_clip_spaces(
                    dummy_gen,
                    gallery_embeddings_no_adapter,
                    use_adapter=False,
                    subject=args.subject,
                    device=args.device
                )
                
                # Normalize
                gallery_norms = np.linalg.norm(gallery_aligned_no_adapter, axis=1, keepdims=True)
                gallery_norms[gallery_norms == 0] = 1.0
                gallery_aligned_no_adapter = gallery_aligned_no_adapter / gallery_norms
                
                # Compute retrieval metrics without adapter
                from fmri2img.eval import cosine_sim
                sim_no_adapter = cosine_sim(gen_embeddings_no_adapter, gallery_aligned_no_adapter)
                
                retrieval_metrics_no_adapter = retrieval_at_k(
                    gen_embeddings_no_adapter[retrieval_valid_mask],
                    gallery_aligned_no_adapter,
                    gt_indices[retrieval_valid_mask],
                    ks=(1, 5, 10)
                )
                
                ranking_metrics_no_adapter = compute_ranking_metrics(
                    gen_embeddings_no_adapter[retrieval_valid_mask],
                    gallery_aligned_no_adapter,
                    gt_indices[retrieval_valid_mask]
                )
                
                logger.info(f"Ablation (no adapter) - CLIPScore: {clip_scores_no_adapter.mean():.3f}")
                logger.info(f"Ablation (no adapter) - R@1: {retrieval_metrics_no_adapter.get('R@1', 0):.4f}")
                
                # Add to aggregate JSON
                aggregate_metrics["ablations"] = {
                    "with_adapter": {
                        "clipscore": {
                            "mean": float(clip_scores.mean()),
                            "std": float(clip_scores.std()),
                        },
                        "retrieval": retrieval_metrics,
                        "ranking": ranking_metrics,
                    },
                    "without_adapter": {
                        "clipscore": {
                            "mean": float(clip_scores_no_adapter.mean()),
                            "std": float(clip_scores_no_adapter.std()),
                        },
                        "retrieval": retrieval_metrics_no_adapter,
                        "ranking": ranking_metrics_no_adapter,
                    }
                }
            else:
                logger.warning("⚠️  Could not run adapter ablation (no gallery embeddings)")
        
        if args.out_json:
            json_path = Path(args.out_json)
        else:
            json_path = Path(args.out_csv).parent / Path(args.out_csv).stem + ".json"
        
        with open(json_path, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        logger.info(f"✅ Aggregate metrics saved to {json_path}")
        
        logger.info("=" * 80)
        logger.info("✅ Evaluation complete!")
        logger.info(f"CSV: {args.out_csv}")
        logger.info(f"JSON: {json_path}")
        logger.info(f"Grid: {args.out_fig}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
