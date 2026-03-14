#!/usr/bin/env python3
"""Build deterministic random-projection rerank target cache for V30d.

This version intentionally avoids any train-fit compression stage. It applies
the same fixed random projection to every flattened token target, then
L2-normalizes the result for cosine-based reranking.

Usage:
    python scripts/preprocessing/build_rerank_cache.py \
        --config configs/experiments/N1v30d_rerank_head_1024.yaml

    python scripts/preprocessing/build_rerank_cache.py \
        --token-cache outputs/clip_cache/tokens_ViT-L-14_projected.h5 \
        --output-dim 1024 \
        --seed 42

Output:
    outputs/rerank_cache/randomproj_dim1024_seed42.npz
"""

import argparse
import json
import logging
import math
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

METADATA_VERSION = 2


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

    # For unnormalized Hadamard H, PHD / sqrt(output_dim) gives the desired
    # scale up to a constant factor, and we L2-normalize afterwards anyway.
    projected = work[:, output_indices] / math.sqrt(output_indices.shape[0])
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    return projected / np.maximum(norms, 1e-8)


def build_cache(
    token_cache_path: str,
    output_dim: int,
    seed: int,
    output_dir: Path,
    batch_size: int,
):
    from fmri2img.data.token_clip_cache import TokenCLIPCache

    logger.info("Loading token cache from %s ...", token_cache_path)
    tc = TokenCLIPCache(token_cache_path)
    tc.load(mmap=True)

    try:
        all_nsd_ids = np.array(tc._nsd_ids, dtype=np.int32)
        n_total = len(all_nsd_ids)
        token_shape = tuple(int(x) for x in tc._tokens.shape[1:])
        input_dim = int(np.prod(token_shape))
        logger.info(
            "Token cache: %d images, tokens shape per image: %s, flattened_dim=%d",
            n_total,
            token_shape,
            input_dim,
        )

        padded_dim, signs, output_indices = _build_projection_params(
            input_dim=input_dim,
            output_dim=output_dim,
            seed=seed,
        )
        logger.info(
            "Projection method=random_projection (SRHT), seed=%d, padded_dim=%d, output_dim=%d",
            seed,
            padded_dim,
            output_dim,
        )

        compressed = np.zeros((n_total, output_dim), dtype=np.float32)
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            flat_batch = np.array(tc._tokens[start:end], dtype=np.float32).reshape(end - start, -1)
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

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"randomproj_dim{output_dim}_seed{seed}.npz"
        np.savez_compressed(
            out_path,
            targets=compressed.astype(np.float32),
            nsd_ids=all_nsd_ids.astype(np.int32),
            metadata=json.dumps(metadata),
        )

        logger.info(
            "Saved random-projection rerank cache to %s (%.1f MB)",
            out_path,
            out_path.stat().st_size / 1e6,
        )

        print("\n" + "=" * 60)
        print("RERANK CACHE SUMMARY")
        print("=" * 60)
        print(f"  Output:             {out_path}")
        print(f"  Method:             random_projection")
        print(f"  Seed:               {seed}")
        print(f"  Input dim:          {input_dim}")
        print(f"  Output dim:         {output_dim}")
        print(f"  Total images:       {n_total}")
        print(f"  L2-normalised:      True")
        print("=" * 60)
    finally:
        tc.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build deterministic random-projection rerank target cache for V30d"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment config YAML (extracts token-cache path, seed, and output dim)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="random_projection",
        choices=["random_projection"],
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Images per projection batch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rerank_cache",
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
        args.output_dim = decoder_cfg.get("rerank_dim", args.output_dim)

    if not args.token_cache:
        parser.error("--token-cache is required (or provide --config)")

    build_cache(
        token_cache_path=args.token_cache,
        output_dim=args.output_dim,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
