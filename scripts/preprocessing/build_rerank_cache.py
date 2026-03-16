#!/usr/bin/env python3
"""Build compressed rerank target caches for V30/V32 experiments.

Supported methods:
    - ``random_projection``: deterministic SRHT-style projection (fit-free)
    - ``pca``: train-only IncrementalPCA fit on experiment train image IDs

Examples
--------
Random projection from config:
    python scripts/preprocessing/build_rerank_cache.py \
        --config configs/experiments/V30e_rerank_head_2048.yaml

Train-only PCA from config:
    python scripts/preprocessing/build_rerank_cache.py \
        --config configs/experiments/V32_pca_rerank_2048.yaml \
        --method pca
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

METADATA_VERSION = 3


def _next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError(f"input_dim must be positive, got {n}")
    return 1 << (n - 1).bit_length()


def _fwht_inplace(x: np.ndarray) -> np.ndarray:
    """In-place fast Walsh-Hadamard transform over the last dimension."""
    n = x.shape[1]
    h = 1
    while h < n:
        x_view = x.reshape(x.shape[0], -1, 2, h)
        a = x_view[:, :, 0, :].copy()
        b = x_view[:, :, 1, :].copy()
        x_view[:, :, 0, :] = a + b
        x_view[:, :, 1, :] = a - b
        h *= 2
    return x


def _iter_chunk_bounds(n_items: int, batch_size: int, min_batch_size: int = 1):
    """Yield chunk bounds, merging the last chunk if it would be too small."""
    if n_items <= 0:
        return
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    start = 0
    while start < n_items:
        end = min(start + batch_size, n_items)
        remaining = n_items - end
        if remaining and remaining < min_batch_size:
            end = n_items
        yield start, end
        start = end


def _build_projection_params(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    padded_dim = _next_power_of_two(input_dim)
    if output_dim < 1:
        raise ValueError(f"output_dim must be positive, got {output_dim}")
    if output_dim > padded_dim:
        raise ValueError(
            f"output_dim={output_dim} exceeds padded_dim={padded_dim} "
            f"for input_dim={input_dim}"
        )

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=padded_dim)
    output_indices = np.sort(
        rng.choice(padded_dim, size=output_dim, replace=False).astype(np.int32)
    )
    return padded_dim, signs, output_indices


def _project_batch_srht(
    flat_batch: np.ndarray,
    padded_dim: int,
    signs: np.ndarray,
    output_indices: np.ndarray,
) -> np.ndarray:
    batch_size, input_dim = flat_batch.shape
    work = np.zeros((batch_size, padded_dim), dtype=np.float32)
    work[:, :input_dim] = flat_batch
    work *= signs[None, :]
    _fwht_inplace(work)
    projected = work[:, output_indices] / math.sqrt(output_indices.shape[0])
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    return projected / np.maximum(norms, 1e-8)


def _resolve_output_path(
    output_dir: Path,
    output_path: str | None,
    method: str,
    output_dim: int,
    seed: int,
) -> Path:
    if output_path:
        return Path(output_path)
    if method == "random_projection":
        return output_dir / f"randomproj_dim{output_dim}_seed{seed}.npz"
    if method == "pca":
        return output_dir / f"pca_trainonly_dim{output_dim}_seed{seed}.npz"
    raise ValueError(f"Unknown method: {method}")


def _resolve_train_image_ids(
    subject: str,
    token_nsd_ids: np.ndarray,
    val_ratio: float,
    seed: int,
    exclude_shared1000: bool,
) -> np.ndarray:
    import pandas as pd

    index_path = Path(f"data/indices/nsd_index/subject={subject}/index.parquet")
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found for subject={subject}: {index_path}"
        )

    index_df = pd.read_parquet(index_path)
    if exclude_shared1000 and "shared1000" in index_df.columns:
        index_df = index_df[~index_df["shared1000"].fillna(False).astype(bool)].reset_index(drop=True)

    unique_nsd = np.sort(index_df["nsdId"].unique().astype(np.int32))
    token_set = set(int(x) for x in token_nsd_ids)
    unique_nsd = np.array([nid for nid in unique_nsd if int(nid) in token_set], dtype=np.int32)

    if len(unique_nsd) == 0:
        raise ValueError(f"No overlap between token cache and subject {subject} image IDs")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_nsd)

    n_val = max(1, int(len(unique_nsd) * val_ratio))
    train_ids = unique_nsd[n_val:]
    if len(train_ids) == 0:
        raise ValueError(
            f"Train split is empty for subject={subject} with val_ratio={val_ratio}"
        )

    logger.info(
        "PCA fit split: subject=%s, train_images=%d, val_images=%d, exclude_shared1000=%s, seed=%d",
        subject,
        len(train_ids),
        n_val,
        exclude_shared1000,
        seed,
    )
    return np.sort(train_ids)


def _flat_tokens_from_indices(token_cache, indices: np.ndarray) -> np.ndarray:
    batch = np.asarray(token_cache._tokens[indices], dtype=np.float32)
    return batch.reshape(len(indices), -1)


def _resolve_transform_device(device: str) -> str:
    """Pick a practical device for PCA transform matmuls."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested device=cuda but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return device


def _transform_batch_pca(
    flat_batch: np.ndarray,
    mean: np.ndarray,
    components_t: np.ndarray,
    device: str,
    mean_t: torch.Tensor | None = None,
    components_t_t: torch.Tensor | None = None,
) -> np.ndarray:
    """Apply PCA transform on CPU or GPU and return float32 CPU array."""
    if device == "cpu":
        centered = flat_batch - mean[None, :]
        return centered @ components_t

    batch_t = torch.from_numpy(flat_batch).to(device=device, dtype=torch.float32, non_blocking=True)
    centered_t = batch_t - mean_t.unsqueeze(0)
    projected_t = centered_t @ components_t_t
    return projected_t.cpu().numpy()


def _build_random_projection_cache(
    token_cache,
    token_cache_path: str,
    output_dim: int,
    seed: int,
    batch_size: int,
    out_path: Path,
) -> dict[str, Any]:
    all_nsd_ids = np.array(token_cache._nsd_ids, dtype=np.int32)
    n_total = len(all_nsd_ids)
    token_shape = tuple(int(x) for x in token_cache._tokens.shape[1:])
    input_dim = int(np.prod(token_shape))

    padded_dim, signs, output_indices = _build_projection_params(
        input_dim=input_dim,
        output_dim=output_dim,
        seed=seed,
    )
    logger.info(
        "Projection method=random_projection (SRHT), seed=%d, padded_dim=%d, output_dim=%d",
        seed, padded_dim, output_dim,
    )

    compressed = np.zeros((n_total, output_dim), dtype=np.float32)
    for start, end in _iter_chunk_bounds(n_total, batch_size):
        flat_batch = np.asarray(token_cache._tokens[start:end], dtype=np.float32).reshape(end - start, -1)
        compressed[start:end] = _project_batch_srht(
            flat_batch=flat_batch,
            padded_dim=padded_dim,
            signs=signs,
            output_indices=output_indices,
        )
        if end % 1000 < batch_size or end == n_total:
            logger.info("  Projected %d/%d images", end, n_total)

    metadata = {
        "method": "random_projection",
        "seed": int(seed),
        "input_dim": input_dim,
        "output_dim": int(output_dim),
        "token_cache_source": str(token_cache_path),
        "created": datetime.now().isoformat(),
        "metadata_version": METADATA_VERSION,
    }

    np.savez_compressed(
        out_path,
        targets=compressed.astype(np.float32),
        nsd_ids=all_nsd_ids.astype(np.int32),
        metadata=json.dumps(metadata),
    )
    return metadata


def _build_pca_cache(
    token_cache,
    token_cache_path: str,
    output_dim: int,
    seed: int,
    batch_size: int,
    out_path: Path,
    subject: str,
    val_ratio: float,
    exclude_shared1000: bool,
    device: str,
) -> dict[str, Any]:
    from sklearn.decomposition import IncrementalPCA

    all_nsd_ids = np.array(token_cache._nsd_ids, dtype=np.int32)
    n_total = len(all_nsd_ids)
    token_shape = tuple(int(x) for x in token_cache._tokens.shape[1:])
    input_dim = int(np.prod(token_shape))

    train_ids = _resolve_train_image_ids(
        subject=subject,
        token_nsd_ids=all_nsd_ids,
        val_ratio=val_ratio,
        seed=seed,
        exclude_shared1000=exclude_shared1000,
    )
    id_to_idx = {int(nid): i for i, nid in enumerate(all_nsd_ids)}
    train_indices = np.array([id_to_idx[int(nid)] for nid in train_ids if int(nid) in id_to_idx], dtype=np.int64)
    train_indices = np.sort(train_indices)
    if len(train_indices) == 0:
        raise ValueError("Resolved zero train-only image IDs for PCA fit")

    k_eff = min(output_dim, len(train_indices), input_dim)
    if k_eff < output_dim:
        logger.warning(
            "Requested output_dim=%d but only %d train images are available; using k_eff=%d",
            output_dim, len(train_indices), k_eff,
        )
    batch_size_eff = max(batch_size, k_eff)
    transform_batch_size = max(batch_size, min(512, k_eff))
    transform_device = _resolve_transform_device(device)
    logger.info(
        "PCA method=train_only IncrementalPCA, subject=%s, train_images=%d, "
        "input_dim=%d, output_dim=%d, fit_batch_size=%d, transform_batch_size=%d, transform_device=%s",
        subject, len(train_indices), input_dim, k_eff, batch_size_eff, transform_batch_size, transform_device,
    )

    pca = IncrementalPCA(n_components=k_eff, batch_size=batch_size_eff)
    for start, end in _iter_chunk_bounds(len(train_indices), batch_size_eff, min_batch_size=k_eff):
        batch_indices = train_indices[start:end]
        flat_batch = _flat_tokens_from_indices(token_cache, batch_indices)
        pca.partial_fit(flat_batch)
        logger.info("  PCA partial_fit: %d/%d train images", end, len(train_indices))

    pca_mean = np.asarray(pca.mean_, dtype=np.float32)
    pca_components_t = np.asarray(pca.components_.T, dtype=np.float32)
    mean_t = None
    components_t_t = None
    if transform_device == "cuda":
        mean_t = torch.from_numpy(pca_mean).to(device="cuda", dtype=torch.float32)
        components_t_t = torch.from_numpy(pca_components_t).to(device="cuda", dtype=torch.float32)
        logger.info(
            "PCA transform: loaded components to GPU (%s), starting full-cache projection...",
            torch.cuda.get_device_name(0),
        )
    else:
        logger.info("PCA transform: running on CPU, starting full-cache projection...")

    compressed = np.zeros((n_total, k_eff), dtype=np.float32)
    for start, end in _iter_chunk_bounds(n_total, transform_batch_size):
        flat_batch = np.asarray(token_cache._tokens[start:end], dtype=np.float32).reshape(end - start, -1)
        projected = _transform_batch_pca(
            flat_batch=flat_batch,
            mean=pca_mean,
            components_t=pca_components_t,
            device=transform_device,
            mean_t=mean_t,
            components_t_t=components_t_t,
        )
        norms = np.linalg.norm(projected, axis=-1, keepdims=True)
        compressed[start:end] = projected / np.maximum(norms, 1e-8)
        if end % 1000 < transform_batch_size or end == n_total:
            logger.info("  PCA transformed %d/%d images", end, n_total)

    explained = float(getattr(pca, "explained_variance_ratio_", np.array([], dtype=np.float32)).sum())
    metadata = {
        "method": "pca",
        "seed": int(seed),
        "input_dim": input_dim,
        "output_dim": int(k_eff),
        "rerank_dim": int(k_eff),
        "fit_split_description": (
            f"subject={subject}, split_by_image=True, exclude_shared1000={exclude_shared1000}, "
            f"val_ratio={val_ratio:.2f}, train_only_unique_images={len(train_indices)}"
        ),
        "train_image_count": int(len(train_indices)),
        "token_cache_source": str(token_cache_path),
        "created": datetime.now().isoformat(),
        "metadata_version": METADATA_VERSION,
        "cumulative_variance": explained,
        "pca_batch_size": int(batch_size_eff),
        "pca_transform_batch_size": int(transform_batch_size),
        "pca_transform_device": transform_device,
    }

    np.savez_compressed(
        out_path,
        targets=compressed.astype(np.float32),
        nsd_ids=all_nsd_ids.astype(np.int32),
        metadata=json.dumps(metadata),
    )
    return metadata


def build_cache(
    token_cache_path: str,
    output_dim: int,
    seed: int,
    output_dir: Path,
    batch_size: int,
    method: str,
    output_path: str | None = None,
    subject: str | None = None,
    val_ratio: float = 0.10,
    exclude_shared1000: bool = True,
    device: str = "auto",
):
    from fmri2img.data.token_clip_cache import TokenCLIPCache

    logger.info("Loading token cache from %s ...", token_cache_path)
    tc = TokenCLIPCache(token_cache_path)
    tc.load(mmap=True)

    try:
        out_path = _resolve_output_path(output_dir, output_path, method, output_dim, seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Token cache: %d images, tokens shape per image: %s",
            len(tc._nsd_ids),
            tuple(int(x) for x in tc._tokens.shape[1:]),
        )

        if method == "random_projection":
            metadata = _build_random_projection_cache(
                token_cache=tc,
                token_cache_path=token_cache_path,
                output_dim=output_dim,
                seed=seed,
                batch_size=batch_size,
                out_path=out_path,
            )
        elif method == "pca":
            if not subject:
                raise ValueError("PCA mode requires a subject so the train-only split can be reconstructed")
            metadata = _build_pca_cache(
                token_cache=tc,
                token_cache_path=token_cache_path,
                output_dim=output_dim,
                seed=seed,
                batch_size=batch_size,
                out_path=out_path,
                subject=subject,
                val_ratio=val_ratio,
                exclude_shared1000=exclude_shared1000,
                device=device,
            )
        else:
            raise ValueError(f"Unknown cache method: {method}")

        logger.info(
            "Saved rerank cache to %s (%.1f MB)",
            out_path,
            out_path.stat().st_size / 1e6,
        )

        print("\n" + "=" * 60)
        print("RERANK CACHE SUMMARY")
        print("=" * 60)
        print(f"  Output:             {out_path}")
        print(f"  Method:             {metadata['method']}")
        print(f"  Seed:               {metadata['seed']}")
        print(f"  Input dim:          {metadata['input_dim']}")
        print(f"  Output dim:         {metadata['output_dim']}")
        if metadata["method"] == "pca":
            print(f"  Train images:       {metadata['train_image_count']}")
            print(f"  Fit split:          {metadata['fit_split_description']}")
            print(f"  Cum. variance:      {metadata.get('cumulative_variance', -1):.4f}")
        print(f"  L2-normalised:      True")
        print("=" * 60)
    finally:
        tc.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build compressed rerank target cache for V30/V32 experiments"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment config YAML (extracts token cache path, seed, dims, subject)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="random_projection",
        choices=["random_projection", "pca"],
        help="Compression method for rerank targets",
    )
    parser.add_argument(
        "--token-cache",
        type=str,
        default=None,
        help="Path to token CLIP cache HDF5",
    )
    parser.add_argument(
        "--output-dim",
        "--rerank-dim",
        dest="output_dim",
        type=int,
        default=1024,
        help="Compressed rerank target dimension",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject used to reconstruct the exact train-only fit split for PCA")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--exclude-shared1000", action="store_true", default=True)
    parser.add_argument("--include-shared1000", action="store_true",
                        help="Disable shared1000 exclusion when reconstructing the PCA fit split")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Images per chunk for projection/transform",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rerank_cache",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Explicit output .npz path (overrides method-based naming)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for PCA transform stage (fit remains on CPU)",
    )

    args = parser.parse_args()

    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get("data", {})
        decoder_cfg = cfg.get("model", {}).get("decoder", {})

        args.token_cache = args.token_cache or data_cfg.get("token_cache_path")
        args.seed = data_cfg.get("seed", args.seed)
        args.subject = args.subject or data_cfg.get("subject")
        args.val_ratio = data_cfg.get("val_split", args.val_ratio)
        args.exclude_shared1000 = data_cfg.get("exclude_shared1000", args.exclude_shared1000)
        args.output_dim = decoder_cfg.get("rerank_dim", args.output_dim)
        args.method = data_cfg.get("rerank_cache_method", args.method)
        args.output_path = args.output_path or data_cfg.get("rerank_cache_path")

    if args.include_shared1000:
        args.exclude_shared1000 = False

    if not args.token_cache:
        parser.error("--token-cache is required (or provide --config)")

    build_cache(
        token_cache_path=args.token_cache,
        output_dim=args.output_dim,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        method=args.method,
        output_path=args.output_path,
        subject=args.subject,
        val_ratio=float(args.val_ratio),
        exclude_shared1000=bool(args.exclude_shared1000),
        device=args.device,
    )


if __name__ == "__main__":
    main()
