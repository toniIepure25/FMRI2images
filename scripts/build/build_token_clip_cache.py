#!/usr/bin/env python3
"""
Build Token-Level CLIP Cache (HDF5)
====================================

Extracts full token-level CLIP features (CLS + 256 spatial patches = 257 tokens)
from NSD stimulus images and stores them in HDF5 format.

MindEye (Scotti et al., 2024) demonstrates that predicting the full 257×768
token set dramatically outperforms CLS-only prediction for retrieval (93.2%
vs ~60% R@1 on NSD shared1000).

Output format (HDF5):
    /tokens       — (N, 257, D) float32, L2-normalized per token
    /nsd_ids      — (N,) int32, NSD stimulus IDs
    /metadata     — attributes: model_name, pretrained, hidden_dim, proj_dim,
                    num_tokens, projected (bool), num_images, build_date

Usage:
    # ViT-L/14 hidden-layer tokens (257 × 1024)
    python scripts/build/build_token_clip_cache.py --mode hidden

    # ViT-L/14 projected tokens (257 × 768) — same space as CLS embedding
    python scripts/build/build_token_clip_cache.py --mode projected

    # ViT-bigG/14 hidden-layer tokens (257 × 1664)
    python scripts/build/build_token_clip_cache.py \
        --clip-config configs/system/clip_bigg.yaml --mode hidden

References:
    - Scotti et al. (2024) "Reconstructing the Mind's Eye" — MindEye
    - Scotti et al. (2024) "MindEye2" — token-level CLIP with bigG
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fmri2img.utils.clip_utils import (
    load_clip_model,
    encode_images_tokens,
    encode_images_tokens_projected,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_nsd_images(hdf5_path: str, nsd_ids: list[int]) -> dict:
    """Load NSD stimulus images from HDF5 by nsdId.

    Args:
        hdf5_path: Path to nsd_stimuli.hdf5
        nsd_ids: List of nsdIds to load (0-indexed into the HDF5 dataset)

    Returns:
        Dict mapping nsdId → PIL Image
    """
    from PIL import Image

    images = {}
    with h5py.File(hdf5_path, "r") as f:
        stim_data = f["imgBrick"]  # (73000, 425, 425, 3) uint8
        for nsd_id in tqdm(nsd_ids, desc="Loading images"):
            # NSD nsdId is 1-indexed; HDF5 is 0-indexed
            idx = nsd_id  # nsd_stim_info_merged uses 0-based indexing
            img_arr = stim_data[idx]
            images[nsd_id] = Image.fromarray(img_arr)
    return images


def get_nsd_ids_for_subject(
    subject: str = "subj01",
    index_root: str = "data/indices/nsd_index",
) -> list[int]:
    """Get all unique nsdIds for a subject from the index."""
    import pandas as pd

    index_path = Path(index_root) / f"subject={subject}" / "index.parquet"
    if not index_path.exists():
        # Fallback: load from CSV
        csv_path = Path("cache/nsd_stim_info_merged.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Return all unique nsdIds (for the full corpus)
            return sorted(df["nsdId"].unique().tolist())
        raise FileNotFoundError(f"No index at {index_path} or {csv_path}")

    df = pd.read_parquet(index_path)
    return sorted(df["nsdId"].unique().tolist())


def get_all_nsd_ids(csv_path: str = "cache/nsd_stim_info_merged.csv") -> list[int]:
    """Get all unique nsdIds from the NSD stimulus info CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    return sorted(df["nsdId"].unique().tolist())


def build_token_cache(
    clip_config: str,
    output_path: str,
    hdf5_stim_path: str,
    mode: str = "projected",
    batch_size: int = 32,
    nsd_ids: list[int] | None = None,
    device: str = "cuda",
    subject: str | None = None,
):
    """Build the token-level CLIP cache.

    Args:
        clip_config: Path to CLIP YAML config (e.g. configs/system/clip.yaml)
        output_path: Output HDF5 path
        hdf5_stim_path: Path to nsd_stimuli.hdf5
        mode: "hidden" (raw last-layer tokens) or "projected" (ln_post + proj)
        batch_size: Images per CLIP forward pass
        nsd_ids: Specific nsdIds to encode (None = all in CSV)
        device: CUDA device
        subject: If provided, only encode images for this subject
    """
    from PIL import Image

    logger.info("=" * 70)
    logger.info("TOKEN-LEVEL CLIP CACHE BUILDER")
    logger.info("=" * 70)
    logger.info(f"CLIP config:  {clip_config}")
    logger.info(f"Output:       {output_path}")
    logger.info(f"Mode:         {mode}")
    logger.info(f"Stimuli HDF5: {hdf5_stim_path}")

    # Load CLIP model
    model, preprocess, cfg = load_clip_model(clip_config, device=device)
    model_name = cfg["model_name"]
    pretrained = cfg.get("pretrained", "openai")
    logger.info(f"Loaded CLIP model: {model_name} (pretrained={pretrained})")

    # Determine token dimensions by running a dummy image
    dummy_img = Image.new("RGB", (224, 224))
    if mode == "projected":
        dummy_out = encode_images_tokens_projected(
            model, preprocess, [dummy_img], device=device, normalize=False
        )
    else:
        dummy_out = encode_images_tokens(
            model, preprocess, [dummy_img], device=device, normalize=False
        )
    _, num_tokens, token_dim = dummy_out.shape
    logger.info(f"Token shape: ({num_tokens}, {token_dim})  mode={mode}")

    # Get nsdIds
    if nsd_ids is None:
        if subject:
            nsd_ids = get_nsd_ids_for_subject(subject)
            logger.info(f"Encoding {len(nsd_ids)} unique images for {subject}")
        else:
            nsd_ids = get_all_nsd_ids()
            logger.info(f"Encoding {len(nsd_ids)} unique images (full NSD corpus)")

    # Check for existing cache (resume support)
    output_path = Path(output_path)
    existing_ids = set()
    if output_path.exists():
        with h5py.File(output_path, "r") as f:
            if "nsd_ids" in f:
                existing_ids = set(f["nsd_ids"][:].tolist())
        logger.info(f"Resuming: {len(existing_ids)} images already cached")
        nsd_ids = [nid for nid in nsd_ids if nid not in existing_ids]
        if not nsd_ids:
            logger.info("All images already cached. Nothing to do.")
            return

    logger.info(f"Will encode {len(nsd_ids)} images in batches of {batch_size}")

    # Open HDF5 stimuli
    stim_file = h5py.File(hdf5_stim_path, "r")
    stim_data = stim_file["imgBrick"]

    # Collect all tokens in memory then write at once
    # (for 10K images × 257 × 768 ≈ 7.4 GB — fits in RAM)
    all_tokens = []
    all_ids = []

    encode_fn = (
        encode_images_tokens_projected if mode == "projected"
        else encode_images_tokens
    )

    for batch_start in tqdm(range(0, len(nsd_ids), batch_size), desc="Encoding"):
        batch_ids = nsd_ids[batch_start : batch_start + batch_size]
        batch_images = []
        for nid in batch_ids:
            img_arr = stim_data[nid]
            batch_images.append(Image.fromarray(img_arr))

        tokens = encode_fn(
            model, preprocess, batch_images,
            device=device, normalize=True,
        )  # (B, T, D)
        all_tokens.append(tokens)
        all_ids.extend(batch_ids)

    stim_file.close()

    # Stack results
    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, T, D)
    all_ids = np.array(all_ids, dtype=np.int32)
    logger.info(f"Encoded {len(all_ids)} images → shape {all_tokens.shape}")

    # Merge with existing cache if resuming
    if existing_ids and output_path.exists():
        with h5py.File(output_path, "r") as f:
            prev_tokens = f["tokens"][:]
            prev_ids = f["nsd_ids"][:]
        all_tokens = np.concatenate([prev_tokens, all_tokens], axis=0)
        all_ids = np.concatenate([prev_ids, all_ids], axis=0)
        logger.info(f"Merged with existing: total {len(all_ids)} images")

    # Sort by nsdId for consistent ordering
    sort_idx = np.argsort(all_ids)
    all_tokens = all_tokens[sort_idx]
    all_ids = all_ids[sort_idx]

    # Write HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "tokens", data=all_tokens, dtype="float32",
            chunks=(1, num_tokens, token_dim),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset("nsd_ids", data=all_ids, dtype="int32")

        # Metadata
        f.attrs["model_name"] = model_name
        f.attrs["pretrained"] = pretrained
        f.attrs["hidden_dim"] = int(token_dim) if mode == "hidden" else int(cfg["embedding_dim"])
        f.attrs["proj_dim"] = int(cfg["embedding_dim"])
        f.attrs["num_tokens"] = int(num_tokens)
        f.attrs["token_dim"] = int(token_dim)
        f.attrs["projected"] = mode == "projected"
        f.attrs["mode"] = mode
        f.attrs["num_images"] = len(all_ids)
        f.attrs["build_date"] = datetime.datetime.now().isoformat()
        f.attrs["normalized"] = True

    file_size_gb = output_path.stat().st_size / 1e9
    logger.info(f"Wrote {output_path} ({file_size_gb:.2f} GB)")
    logger.info(f"  Shape: ({len(all_ids)}, {num_tokens}, {token_dim})")
    logger.info(f"  Model: {model_name}, mode={mode}")
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Build token-level CLIP cache (HDF5) for MindEye-style training"
    )
    parser.add_argument(
        "--clip-config", default="configs/system/clip.yaml",
        help="Path to CLIP config YAML (model_name, pretrained, etc.)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output HDF5 path (default: outputs/clip_cache/tokens_{model}_{mode}.h5)",
    )
    parser.add_argument(
        "--stimuli", default="cache/nsd_hdf5/nsd_stimuli.hdf5",
        help="Path to NSD stimuli HDF5",
    )
    parser.add_argument(
        "--mode", choices=["hidden", "projected"], default="projected",
        help="Token extraction mode: 'hidden' (last hidden layer) or "
             "'projected' (through ln_post + visual.proj)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--subject", default=None,
        help="Only encode images for this subject (default: all NSD images)",
    )
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        from fmri2img.utils.clip_utils import load_clip_config
        cfg = load_clip_config(args.clip_config)
        model_tag = cfg["model_name"].replace("/", "-").replace(" ", "_")
        args.output = f"outputs/clip_cache/tokens_{model_tag}_{args.mode}.h5"

    build_token_cache(
        clip_config=args.clip_config,
        output_path=args.output,
        hdf5_stim_path=args.stimuli,
        mode=args.mode,
        batch_size=args.batch_size,
        device=args.device,
        subject=args.subject,
    )


if __name__ == "__main__":
    main()
