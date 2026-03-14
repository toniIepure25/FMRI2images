#!/usr/bin/env python3
"""Build PCA-compressed rerank target cache for V30d.

Fits PCA on train-split-only token embeddings, then projects ALL images
(train + val + shared1000) into the compressed space. Outputs are
L2-normalised for cosine retrieval.

The train/val split is reproduced exactly from the experiment config
(same seed, same exclude_shared1000 logic, same val_ratio).

Usage:
    python scripts/preprocessing/build_rerank_cache.py \
        --config configs/experiments/N1v30d_rerank_head_1024.yaml \
        --rerank-dim 1024

    # Or specify split params directly:
    python scripts/preprocessing/build_rerank_cache.py \
        --token-cache outputs/clip_cache/tokens_ViT-L-14_projected.h5 \
        --index-root outputs/preproc/subj01 \
        --rerank-dim 1024 \
        --seed 42 \
        --val-ratio 0.10 \
        --subject subj01

Output:
    outputs/rerank_cache/compressed_targets_seed42_trainonly_dim1024.npz
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _reproduce_train_split(
    all_nsd_ids: np.ndarray,
    seed: int,
    val_ratio: float,
    exclude_nsd_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the exact train/val image split from the dataset.

    Mirrors ``MultiSubjectDataset._split_by_image``:
    sorted nsdIds → PCG64 shuffle → first n_val are val, rest are train.

    Parameters
    ----------
    all_nsd_ids : all unique NSD IDs in the dataset
    seed : split seed
    val_ratio : fraction for validation
    exclude_nsd_ids : NSD IDs to exclude before splitting (e.g. shared1000)

    Returns
    -------
    train_nsd_ids, val_nsd_ids : sorted arrays
    """
    unique_nsd = np.sort(np.unique(all_nsd_ids))

    if exclude_nsd_ids:
        unique_nsd = np.array(
            [nid for nid in unique_nsd if int(nid) not in exclude_nsd_ids],
            dtype=unique_nsd.dtype,
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_nsd)

    n_val = max(1, int(len(unique_nsd) * val_ratio))
    val_ids = np.sort(unique_nsd[:n_val])
    train_ids = np.sort(unique_nsd[n_val:])

    return train_ids, val_ids


def _load_shared1000_nsd_ids(index_root: Path, subject: str) -> set[int]:
    """Load shared1000 NSD IDs from the index."""
    import pandas as pd

    subj_dir = index_root
    if not (subj_dir / "index.parquet").exists():
        subj_dir = index_root / subject
    idx_path = subj_dir / "index.parquet"
    if not idx_path.exists():
        logger.warning("No index.parquet found at %s, cannot exclude shared1000", idx_path)
        return set()

    df = pd.read_parquet(idx_path)
    if "shared1000" in df.columns:
        return set(int(nid) for nid in df[df["shared1000"]]["nsdId"].unique())
    return set()


def build_cache(
    token_cache_path: str,
    index_root: Path,
    subject: str,
    rerank_dim: int,
    seed: int,
    val_ratio: float,
    exclude_shared1000: bool,
    output_dir: Path,
):
    from fmri2img.data.token_clip_cache import TokenCLIPCache

    # Load token cache
    logger.info("Loading token cache from %s ...", token_cache_path)
    tc = TokenCLIPCache(token_cache_path)
    tc.load()
    all_nsd_ids = np.array(tc._nsd_ids, dtype=np.int32)
    n_total = len(all_nsd_ids)
    logger.info("Token cache: %d images, tokens shape per image: %s",
                n_total, tc._tokens.shape[1:])

    # Determine shared1000 exclusion
    shared1000_ids: set[int] = set()
    if exclude_shared1000:
        shared1000_ids = _load_shared1000_nsd_ids(index_root, subject)
        logger.info("Shared1000: %d images to exclude from PCA fit", len(shared1000_ids))

    # Reproduce train/val split
    train_ids, val_ids = _reproduce_train_split(
        all_nsd_ids, seed=seed, val_ratio=val_ratio,
        exclude_nsd_ids=shared1000_ids if exclude_shared1000 else None,
    )
    logger.info("Split: %d train images, %d val images (seed=%d, val_ratio=%.2f)",
                len(train_ids), len(val_ids), seed, val_ratio)

    # Build index mapping
    id_to_idx = {int(nid): i for i, nid in enumerate(all_nsd_ids)}
    train_indices = np.array([id_to_idx[int(nid)] for nid in train_ids
                              if int(nid) in id_to_idx])
    flat_dim = tc._tokens.shape[1] * tc._tokens.shape[2]
    logger.info("PCA training data: %d images × %d dims (chunked, never fully in RAM)",
                len(train_indices), flat_dim)

    # Fit IncrementalPCA in chunks to avoid OOM
    from sklearn.decomposition import IncrementalPCA

    chunk_size = 512  # ~512 × 197376 × 4 bytes ≈ 380 MB per chunk
    pca = IncrementalPCA(n_components=rerank_dim)

    n_train = len(train_indices)
    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        chunk_idx = train_indices[start:end]
        chunk = np.array(
            [tc._tokens[i].reshape(-1) for i in chunk_idx], dtype=np.float32)
        pca.partial_fit(chunk)
        logger.info("  PCA partial_fit: %d/%d train images", end, n_train)
        del chunk

    cumulative_var = float(pca.explained_variance_ratio_.sum())
    logger.info("PCA fit complete. Cumulative explained variance: %.4f (%.1f%%)",
                cumulative_var, cumulative_var * 100)

    # Project ALL images in chunks
    logger.info("Projecting all %d images (chunked) ...", n_total)
    compressed = np.zeros((n_total, rerank_dim), dtype=np.float32)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk = np.array(
            [tc._tokens[i].reshape(-1) for i in range(start, end)], dtype=np.float32)
        compressed[start:end] = pca.transform(chunk)
        del chunk
        if end % 2000 < chunk_size:
            logger.info("  Projected %d/%d images", end, n_total)

    # L2-normalise
    norms = np.linalg.norm(compressed, axis=-1, keepdims=True)
    compressed = compressed / np.maximum(norms, 1e-8)
    logger.info("Compressed targets: %s, L2-normalised", compressed.shape)

    # Build metadata
    train_ids_hash = hashlib.sha256(
        np.sort(train_ids).tobytes()
    ).hexdigest()[:16]

    metadata = {
        "split_seed": seed,
        "rerank_dim": rerank_dim,
        "val_ratio": val_ratio,
        "n_train_images": len(train_ids),
        "n_val_images": len(val_ids),
        "n_total_images": n_total,
        "train_nsd_ids_hash": f"sha256:{train_ids_hash}",
        "shared1000_excluded_from_fit": exclude_shared1000,
        "n_shared1000_excluded": len(shared1000_ids),
        "token_cache_source": str(token_cache_path),
        "token_shape_per_image": list(tc._tokens.shape[1:]),
        "flat_dim": flat_dim,
        "cumulative_variance": cumulative_var,
        "subject": subject,
        "created": datetime.now().isoformat(),
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"compressed_targets_seed{seed}_trainonly_dim{rerank_dim}.npz"
    out_path = output_dir / out_name

    # Save targets (compact) — PCA components are large (~760 MB for 1024×197376)
    # so we save them in a separate file only if needed for re-projection
    np.savez_compressed(
        out_path,
        targets=compressed.astype(np.float32),
        nsd_ids=all_nsd_ids.astype(np.int32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        metadata=json.dumps(metadata),
    )

    # Save PCA basis separately (for re-projection of new images if needed)
    pca_path = output_dir / f"pca_basis_seed{seed}_dim{rerank_dim}.npz"
    np.savez_compressed(
        pca_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
    )
    logger.info("Saved compressed target cache to %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    logger.info("Saved PCA basis to %s (%.1f MB)", pca_path, pca_path.stat().st_size / 1e6)

    # Print summary
    print("\n" + "=" * 60)
    print("RERANK CACHE SUMMARY")
    print("=" * 60)
    print(f"  Output:             {out_path}")
    print(f"  Rerank dim:         {rerank_dim}")
    print(f"  Train images:       {len(train_ids)}")
    print(f"  Total images:       {n_total}")
    print(f"  Cumulative var:     {cumulative_var:.4f} ({cumulative_var*100:.1f}%)")
    print(f"  Split seed:         {seed}")
    print(f"  Shared1000 excl:    {exclude_shared1000} ({len(shared1000_ids)} images)")
    print(f"  Train IDs hash:     sha256:{train_ids_hash}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build PCA-compressed rerank target cache for V30d")

    parser.add_argument("--config", type=str, default=None,
                        help="Experiment config YAML (extracts all params automatically)")
    parser.add_argument("--token-cache", type=str, default=None,
                        help="Path to token CLIP cache HDF5")
    parser.add_argument("--index-root", type=str, default=None,
                        help="Path to preprocessed index root (for shared1000 detection)")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--rerank-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--exclude-shared1000", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="outputs/rerank_cache")

    args = parser.parse_args()

    # If config provided, extract params from it
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get("data", {})
        decoder_cfg = cfg.get("model", {}).get("decoder", {})

        args.token_cache = args.token_cache or data_cfg.get("token_cache_path")
        args.subject = data_cfg.get("subject", args.subject)
        args.seed = data_cfg.get("seed", args.seed)
        args.val_ratio = data_cfg.get("val_split", args.val_ratio)
        args.exclude_shared1000 = data_cfg.get("exclude_shared1000", args.exclude_shared1000)
        args.rerank_dim = decoder_cfg.get("rerank_dim", args.rerank_dim)
        if args.index_root is None:
            args.index_root = f"outputs/preproc/{args.subject}"

    if not args.token_cache:
        parser.error("--token-cache is required (or provide --config)")
    if not args.index_root:
        args.index_root = f"outputs/preproc/{args.subject}"

    build_cache(
        token_cache_path=args.token_cache,
        index_root=Path(args.index_root),
        subject=args.subject,
        rerank_dim=args.rerank_dim,
        seed=args.seed,
        val_ratio=args.val_ratio,
        exclude_shared1000=args.exclude_shared1000,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
