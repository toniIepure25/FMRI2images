#!/usr/bin/env python3
"""
Build embedding preprocessors (center_pcr or center_whiten) from training embeddings.

Usage:
    python scripts/build_embedding_preproc.py --mode center_pcr --k_components 8
    python scripts/build_embedding_preproc.py --mode center_whiten
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.embedding_preproc import EmbeddingPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_train_embeddings(cache_dir: Path) -> np.ndarray:
    """Load CLIP embeddings for training split from cache."""
    # Try different possible paths for training embeddings
    possible_paths = [
        cache_dir / "clip_embeddings" / "nsd_train_clip_embeddings.npy",
        cache_dir / "clip_embeddings" / "nsd_train_clipvit_embeddings.npy",
        cache_dir / "clip_embeddings" / "train_embeddings.npy",
    ]
    
    clip_emb_path = None
    for path in possible_paths:
        if path.exists():
            clip_emb_path = path
            break
    
    if clip_emb_path is None:
        raise FileNotFoundError(
            f"Training embeddings not found. Searched:\n" +
            "\n".join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease ensure CLIP embeddings are extracted to cache/clip_embeddings/"
        )
    
    embeddings = np.load(clip_emb_path)
    logger.info(f"Loaded {len(embeddings)} training embeddings from {clip_emb_path}")
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    return embeddings


def build_center_pcr(embeddings: np.ndarray, k: int, output_path: Path):
    """Build and save center_pcr preprocessor."""
    logger.info(f"Building center_pcr with k={k}...")
    logger.info("This performs PCA to remove top-k principal components.")
    
    preprocessor = EmbeddingPreprocessor(mode="center_pcr", k_components=k)
    preprocessor.fit(embeddings)
    
    # Save preprocessor
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(output_path)
    
    logger.info(f"✓ Saved preprocessor to {output_path}")
    logger.info(f"  - Original dim: {embeddings.shape[1]}")
    logger.info(f"  - Mode: center_pcr (k={k})")
    if preprocessor.artifacts and preprocessor.artifacts.pca_explained_variance is not None:
        total_var = preprocessor.artifacts.pca_explained_variance[:k].sum()
        logger.info(f"  - Top-{k} PC variance: {total_var:.4f}")
    
    # Verify by transforming a sample
    sample_input = embeddings[:5]
    sample_output = preprocessor.transform(sample_input)
    logger.info(f"  - Sample output shape: {sample_output.shape}")
    logger.info(f"  - Sample output mean: {sample_output.mean():.6f} (should be ~zero)")


def build_center_whiten(embeddings: np.ndarray, output_path: Path):
    """Build and save center_whiten preprocessor."""
    logger.info("Building center_whiten...")
    logger.info("This performs mean-centering followed by whitening transformation.")
    
    preprocessor = EmbeddingPreprocessor(mode="center_whiten")
    preprocessor.fit(embeddings)
    
    # Save preprocessor
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(output_path)
    
    logger.info(f"✓ Saved preprocessor to {output_path}")
    logger.info(f"  - Embedding dim: {embeddings.shape[1]}")
    logger.info(f"  - Mode: center_whiten")
    
    # Verify by transforming a sample
    sample_input = embeddings[:5]
    sample_output = preprocessor.transform(sample_input)
    logger.info(f"  - Sample output shape: {sample_output.shape}")
    logger.info(f"  - Sample output mean: {sample_output.mean():.6f} (should be ~zero)")
    logger.info(f"  - Sample output std: {sample_output.std():.6f} (should be ~1)")


def main():
    parser = argparse.ArgumentParser(description="Build embedding preprocessor")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["center_pcr", "center_whiten"],
        help="Preprocessing method to use"
    )
    parser.add_argument(
        "--k_components",
        type=int,
        default=8,
        help="Number of PCA components for center_pcr (default: 8)"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("cache"),
        help="Path to cache directory containing CLIP embeddings (default: cache)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for preprocessor pickle (default: cache/<mode>.pkl)"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        if args.mode == "center_pcr":
            args.output = args.cache_dir / f"center_pcr_k{args.k_components}.pkl"
        else:
            args.output = args.cache_dir / f"{args.mode}.pkl"
    
    logger.info("="*60)
    logger.info(f"Building Embedding Preprocessor: {args.mode}")
    logger.info("="*60)
    
    # Load training embeddings
    train_embeddings = load_train_embeddings(args.cache_dir)
    
    # Build and save preprocessor
    if args.mode == "center_pcr":
        build_center_pcr(train_embeddings, args.k_components, args.output)
    elif args.mode == "center_whiten":
        build_center_whiten(train_embeddings, args.output)
    
    logger.info("\n" + "="*60)
    logger.info(f"✓ Preprocessor ready at {args.output}")
    logger.info("="*60)
    logger.info("\nYou can now use this preprocessor in your experiment configs:")
    logger.info(f"  preprocess_embeddings: true")
    logger.info(f"  embedding_preproc_path: '{args.output}'")


if __name__ == "__main__":
    main()
