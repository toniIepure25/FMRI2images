#!/usr/bin/env python3
"""
Research-grade reconstruction evaluation.

Guarantees:
- Robust ID mapping between recon filenames, manifest samples, clip cache, and NSD index.
- Hard failures on empty matches or silent zero metrics.
- Consistency checks for CLIP model vs cache metadata.
- Rich diagnostics and debug artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageDraw, ImageFont

from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics, clip_score
from fmri2img.eval.stimulus import (
    StimulusKey,
    parse_stimulus_id_from_filename,
    resolve_from_cache,
    resolve_from_index,
)

logger = logging.getLogger("eval_reconstruction")


# ----------------------------- Dataclasses & helpers -----------------------------


@dataclass
class ReconSample:
    path: Path
    parsed_id: Optional[int]
    stimulus: Optional[StimulusKey]
    matched_field: Optional[str]


def _list_images(recon_dir: Path, limit: int) -> List[Path]:
    img_dir = recon_dir / "images"
    if not img_dir.exists():
        return []
    imgs = sorted(img_dir.glob("*.png"))
    return imgs[:limit] if limit else imgs


def _load_manifest_samples(manifest_path: Optional[Path]) -> Dict[str, Dict]:
    if not manifest_path or not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text())
    except Exception as exc:  # pragma: no cover - guardrails
        logger.warning("Failed to read manifest %s: %s", manifest_path, exc)
        return {}
    samples = data.get("samples", []) or []
    by_name = {}
    for s in samples:
        fname = Path(s.get("recon_filename", s.get("recon_path", ""))).name
        by_name[fname] = s
    return by_name


def _resolve_stimulus(
    img_path: Path,
    cache_df: pd.DataFrame,
    index_df: Optional[pd.DataFrame],
    manifest_samples: Dict[str, Dict],
) -> ReconSample:
    parsed_id = parse_stimulus_id_from_filename(img_path)
    manifest_entry = manifest_samples.get(img_path.name)
    if manifest_entry:
        stim = StimulusKey(
            nsd_id=manifest_entry.get("nsd_id"),
            nsdId=manifest_entry.get("nsdId"),
        )
        return ReconSample(img_path, parsed_id, stim, manifest_entry.get("matched_field"))

    if parsed_id is not None:
        cache_match = resolve_from_cache(parsed_id, cache_df)
        if cache_match:
            stim, field = cache_match
            return ReconSample(img_path, parsed_id, stim, field)

    if parsed_id is not None and index_df is not None:
        stim = resolve_from_index(parsed_id, index_df)
        if stim:
            return ReconSample(img_path, parsed_id, stim, "index")

    return ReconSample(img_path, parsed_id, None, None)


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _encode_images(paths: List[Path], model, preprocess, device: str = "cuda") -> np.ndarray:
    batch = []
    embeddings = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        batch.append(preprocess(img))
        if len(batch) == 32:
            emb = _forward_clip(model, batch, device)
            embeddings.append(emb)
            batch = []
    if batch:
        embeddings.append(_forward_clip(model, batch, device))
    return np.concatenate(embeddings, axis=0)


def _forward_clip(model, batch, device: str) -> np.ndarray:
    import torch
    from contextlib import nullcontext

    x = torch.stack(batch).to(device)
    ctx = torch.amp.autocast("cuda") if device.startswith("cuda") and torch.cuda.is_available() else nullcontext()
    with torch.no_grad(), ctx:
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def _save_grid(imgs: List[Path], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    if not imgs:
        Image.new("RGB", (256, 256), color=(220, 220, 220)).save(out_fig)
        return
    tiles = []
    for p in imgs[:4]:
        try:
            tiles.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    if not tiles:
        Image.new("RGB", (256, 256), color=(220, 220, 220)).save(out_fig)
        return
    w, h = tiles[0].size
    grid_w = w * len(tiles)
    grid_h = h
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    x = 0
    for t in tiles:
        grid.paste(t.resize((w, h)), (x, 0))
        x += w
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = None
    draw.text((5, 5), f"n={len(imgs)}", fill=(0, 0, 0), font=font)
    grid.save(out_fig)


_EMB_COL_CANDIDATES = ["clip_embedding", "embedding", "final", "clip512"]


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(f"CLIP cache not found: {cache_path}")
    df = pd.read_parquet(cache_path)
    resolved = None
    for col in _EMB_COL_CANDIDATES:
        if col in df.columns:
            resolved = col
            break
    if resolved is None:
        raise ValueError(
            f"CLIP cache missing embedding column. Tried: {_EMB_COL_CANDIDATES}. "
            f"Available columns: {list(df.columns)}"
        )
    if resolved != "clip512":
        df["clip512"] = df[resolved]
    return df


def _guess_model_from_dim(dim: int) -> Optional[str]:
    return {512: "ViT-B/32", 768: "ViT-L/14"}.get(dim)


def _load_cache_meta(cache_path: Path, cache_df: pd.DataFrame) -> Dict:
    cache = CLIPCache(str(cache_path))
    try:
        meta = cache.load_metadata(raise_on_missing=True)
    except FileNotFoundError:
        emb_len = len(cache_df.iloc[0]["clip512"]) if len(cache_df) else None
        model_guess = _guess_model_from_dim(emb_len) if emb_len else None
        meta = {
            "clip_model_name": model_guess or "unknown",
            "pretrained": "openai",
            "embedding_dim": emb_len,
            "normalize": True,
            "created": datetime.now().isoformat(),
            "inferred": True,
            "note": "Metadata inferred automatically; please rebuild cache with explicit metadata for strict checks.",
        }
        cache.write_metadata(meta, overwrite=False)
        logger.warning("clip_cache_meta.json missing; wrote inferred metadata to %s", cache.meta_path)
    if "embedding_dim" not in meta or "clip_model_name" not in meta:
        raise ValueError("clip_cache_meta.json missing required fields: embedding_dim, clip_model_name")
    return meta


def _load_clip_from_meta(meta: Dict, device: str):
    model_name = meta.get("clip_model_name")
    pretrained = meta.get("pretrained", "openai")
    normalize = meta.get("normalize", True)
    if not model_name:
        raise ValueError("clip_cache_meta.json must include clip_model_name")
    if not meta.get("embedding_dim"):
        raise ValueError("clip_cache_meta.json must include embedding_dim")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    # quick dimension sanity check
    dummy_img = Image.new("RGB", (224, 224))
    dummy_tensor = preprocess(dummy_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.encode_image(dummy_tensor)
    dim = out.shape[-1]
    if meta.get("embedding_dim") and int(meta["embedding_dim"]) != dim:
        raise ValueError(
            f"CLIP cache embedding_dim={meta['embedding_dim']} but model outputs {dim}. "
            "Rebuild cache or update metadata.")
    return model, preprocess, normalize


def _select_id_field(df: pd.DataFrame, pref: str) -> str:
    if pref != "auto":
        if pref not in df.columns:
            raise ValueError(f"Requested id field {pref} not in cache columns {list(df.columns)}")
        return pref
    if "nsdId" in df.columns:
        return "nsdId"
    if "nsd_id" in df.columns:
        return "nsd_id"
    raise ValueError("CLIP cache missing nsdId/nsd_id columns")


def _write_debug(debug_path: Path, payload: Dict) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(payload, indent=2))
    logger.error("Wrote debug diagnostics to %s", debug_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate reconstructed images against CLIP cache")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--recon-dir", type=Path, required=True)
    ap.add_argument("--clip-cache", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-fig", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gallery", default="matched")
    ap.add_argument("--image-source", default="hdf5")
    ap.add_argument("--index-root")
    ap.add_argument("--index-file")
    ap.add_argument("--id-field", choices=["nsdId", "nsd_id", "auto"], default="auto")
    ap.add_argument("--manifest", type=Path, help="Override manifest path")
    args = ap.parse_args()

    # Images
    imgs = _list_images(args.recon_dir, args.limit)
    if not imgs:
        raise SystemExit("No reconstructed images found; aborting")

    # Cache and metadata
    cache_df = _load_cache(args.clip_cache)
    cache_meta = _load_cache_meta(args.clip_cache, cache_df)
    id_field = _select_id_field(cache_df, args.id_field)
    logger.info("Using id field: %s", id_field)

    embed_len = len(cache_df.iloc[0]["clip512"])
    if cache_meta.get("embedding_dim") and int(cache_meta["embedding_dim"]) != embed_len:
        raise SystemExit(
            f"Cache embedding dim {embed_len} != metadata {cache_meta.get('embedding_dim')}" )

    # Index
    index_df = None
    if args.index_file:
        index_df = read_subject_index(args.index_file, args.subject, allow_fallback_index=True)
    elif args.index_root:
        index_df = read_subject_index(args.index_root, args.subject, allow_fallback_index=True)

    # Manifest samples
    manifest_path = args.manifest if args.manifest else args.recon_dir / "manifest.json"
    manifest_samples = _load_manifest_samples(manifest_path if manifest_path.exists() else None)

    resolved: List[ReconSample] = []
    for p in imgs:
        resolved.append(_resolve_stimulus(p, cache_df, index_df, manifest_samples))

    unmatched = [r for r in resolved if r.stimulus is None]
    if unmatched:
        payload = {
            "recon_files": [str(r.path) for r in resolved],
            "parsed_ids": [r.parsed_id for r in resolved],
            "unmatched": [str(r.path) for r in unmatched],
            "cache_id_ranges": {
                "nsdId_min": int(cache_df.get("nsdId", pd.Series([np.nan])).min()) if "nsdId" in cache_df else None,
                "nsdId_max": int(cache_df.get("nsdId", pd.Series([np.nan])).max()) if "nsdId" in cache_df else None,
                "nsd_id_min": int(cache_df.get("nsd_id", pd.Series([np.nan])).min()) if "nsd_id" in cache_df else None,
                "nsd_id_max": int(cache_df.get("nsd_id", pd.Series([np.nan])).max()) if "nsd_id" in cache_df else None,
            },
        }
        debug_path = args.out_json.parent / "debug_eval_matching.json"
        _write_debug(debug_path, payload)
        raise SystemExit("Failed to match all recon images to cache; see debug_eval_matching.json")

    # Encode recon images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess, _normalize_flag = _load_clip_from_meta(cache_meta, device)
    recon_embs = _encode_images([r.path for r in resolved], clip_model, clip_preprocess, device=device)
    recon_embs = _normalize(recon_embs)

    # Gallery embeddings
    emb_col = "clip512" if "clip512" in cache_df.columns else "embedding"
    gallery_ids = cache_df[id_field].astype(int).to_numpy()
    gallery_embs = np.vstack([np.array(v, dtype=np.float32) for v in cache_df[emb_col].to_list()])
    if gallery_embs.size == 0:
        raise SystemExit("Gallery embeddings are empty; cannot evaluate")
    gallery_embs = _normalize(gallery_embs)
    if gallery_embs.shape[1] != recon_embs.shape[1]:
        raise SystemExit(
            f"Embedding dim mismatch: recon {recon_embs.shape[1]} vs cache {gallery_embs.shape[1]}" )

    # Ground truth indices and embeddings
    gt_indices = []
    gt_embs = []
    for sample in resolved:
        key_field, key_val = sample.stimulus.best_id()
        if key_field != id_field:
            # try alternate field mapping using cache columns
            if key_field in cache_df.columns:
                mask = cache_df[key_field] == key_val
                if not mask.any():
                    raise SystemExit(f"Ground truth id {key_val} not found in cache ({key_field})")
                # remap to selected id_field
                key_val = int(cache_df.loc[mask, id_field].iloc[0])
            else:
                raise SystemExit(f"Cache missing required field {key_field}")
        matches = np.where(gallery_ids == key_val)[0]
        if len(matches) == 0:
            raise SystemExit(f"No gallery entry for id {key_val}")
        gt_indices.append(matches[0])
        gt_embs.append(gallery_embs[matches[0]])

    gt_indices_arr = np.array(gt_indices)
    gt_embs_arr = np.vstack(gt_embs)

    # Metrics
    per_sample_clip = clip_score(recon_embs, gt_embs_arr)
    retrieval = retrieval_at_k(recon_embs, gallery_embs, gt_indices_arr, ks=(1, 5, 10))
    ranking = compute_ranking_metrics(recon_embs, gallery_embs, gt_indices_arr)
    if any(r <= 0 for r in [ranking["mean_rank"], ranking["median_rank"]]):
        raise SystemExit("Computed non-positive ranks; aborting")

    sim_matrix = recon_embs @ gallery_embs.T
    per_sample_rank = []
    for i, gt_idx in enumerate(gt_indices_arr):
        order = np.argsort(-sim_matrix[i])
        rank_pos = int(np.where(order == gt_idx)[0][0]) + 1
        if rank_pos <= 0:
            raise SystemExit(f"Invalid rank computed for sample {i}: {rank_pos}")
        per_sample_rank.append(rank_pos)

    # Records
    records = []
    for rec, clip_score_val, rank_val in zip(resolved, per_sample_clip, per_sample_rank):
        key_field, key_val = rec.stimulus.best_id()
        records.append({
            "subject": args.subject,
            "gallery": args.gallery,
            "recon_path": str(rec.path),
            "recon_filename": rec.path.name,
            "parsed_id": rec.parsed_id,
            "matched_cache_field": key_field,
            "matched_cache_value": key_val,
            "gt_nsd_id": rec.stimulus.nsd_id,
            "gt_nsdId": rec.stimulus.nsdId,
            "clip_score": float(clip_score_val),
            "rank": int(rank_val),
            "reciprocal_rank": float(1.0 / rank_val),
        })

    # Persist
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out_csv, index=False)

    summary = {
        "subject": args.subject,
        "gallery": args.gallery,
        "n_images": len(resolved),
        "n_gallery": len(gallery_embs),
        "id_field": id_field,
        "retrieval": retrieval,
        "ranking": ranking,
        "clipscore": {
            "mean": float(per_sample_clip.mean()),
            "std": float(per_sample_clip.std()),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))

    _save_grid(imgs, args.out_fig)

    logger.info("Images evaluated: %d", len(resolved))
    logger.info("Matched pairs: %d", len(gt_indices))
    logger.info("Gallery size: %d", len(gallery_embs))
    logger.info("CLIPScore mean: %.4f", summary["clipscore"]["mean"])
    logger.info("R@1: %.3f, R@5: %.3f, R@10: %.3f", retrieval.get("R@1", 0), retrieval.get("R@5", 0), retrieval.get("R@10", 0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
