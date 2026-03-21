#!/usr/bin/env python3
"""Export a thesis-grade true diffusion reconstruction addon.

This script keeps the frozen best retrieval system unchanged and adds an
inference-only reconstruction stage:

    fMRI -> frozen tri-fusion retrieval -> weighted top-k prior -> true diffusion img2img

It must never silently fall back to a heuristic when true diffusion dependencies
are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from export_final_qualitatives import StimulusStore, resolve_stimuli_hdf5
from final_best_system import (
    DEFAULT_LEGACY_RESULTS_DIR,
    DEFAULT_TRI_RESULTS_DIR,
    EXPECTED_FROZEN_TRI_CONFIG,
    EXPECTED_SHARED1000_R1,
    load_final_best_system_analysis,
)
from sweep_tri_fusion_retrieval import _gt_rank_from_scores, _normalize_shortlist_scores, _rank_within_shortlist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("final_outputs/true_diffusion_addon")
DEFAULT_MODEL_ID = "sd2-community/stable-diffusion-2-1"
BUCKETS = ["perfect", "good", "near_good", "hard"]


@dataclass
class DependencyReport:
    versions: dict[str, str]
    model_id: str
    diffusion_model_path: str
    true_diffusion_run: bool


@dataclass
class ClipContext:
    model: Any
    preprocess: Any
    device: str


@dataclass
class ExampleArtifacts:
    prior_image: Image.Image
    final_image: Image.Image
    selected_seed: int
    candidate_scores: list[dict[str, Any]]


REQUIRED_MODULES = [
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("open_clip", "open_clip_torch"),
    ("lpips", "lpips"),
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _bucket_from_rank(rank: int) -> str:
    if rank == 1:
        return "perfect"
    if rank <= 5:
        return "good"
    if rank <= 20:
        return "near_good"
    return "hard"


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    temp = max(float(temperature), 1e-6)
    shifted = values / temp
    shifted = shifted - np.max(shifted)
    probs = np.exp(shifted)
    probs /= np.maximum(np.sum(probs), 1e-12)
    return probs


def ensure_true_diffusion_dependencies() -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name, package_name in REQUIRED_MODULES:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            missing.append(f"- {package_name}: {type(exc).__name__}: {exc}")
    try:
        import skimage  # noqa: F401
        versions["skimage"] = getattr(skimage, "__version__", "unknown")
    except Exception as exc:
        missing.append(f"- scikit-image: {type(exc).__name__}: {exc}")
    if missing:
        raise RuntimeError(
            "True diffusion addon cannot run because required dependencies are missing.\n"
            "Install them first; no heuristic fallback is allowed in this script.\n"
            + "\n".join(missing)
        )
    return versions


def _resolve_torch_device(requested: str) -> str:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    return "cpu"


def load_clip_context(device: str, cache_dir: Path | None) -> ClipContext:
    import open_clip

    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", **kwargs)
    device = _resolve_torch_device(device)
    model = model.to(device)
    model.eval()
    return ClipContext(model=model, preprocess=preprocess, device=device)


def load_diffusion_pipe(model_id: str, device: str, allow_download: bool, cache_dir: Path | None):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

    torch_dtype = torch.float16 if _resolve_torch_device(device).startswith("cuda") else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
        "local_files_only": not allow_download,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(_resolve_torch_device(device))
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass
    return pipe


def _tile(img: Image.Image, title: str, lines: list[str], tile_size: int = 256, footer_h: int = 96) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size + footer_h), "white")
    thumb = ImageOps.contain(img.convert("RGB"), (tile_size, tile_size))
    canvas.paste(thumb, ((tile_size - thumb.width) // 2, (tile_size - thumb.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, tile_size, tile_size, tile_size + footer_h), fill=(248, 248, 248))
    draw.text((8, tile_size + 6), title, fill=(0, 0, 0))
    for i, line in enumerate(lines[:5]):
        draw.text((8, tile_size + 24 + 14 * i), line, fill=(20, 20, 20))
    return canvas


def _topk_strip(store: StimulusStore, nsd_ids: list[int], weights: list[float], tile_size: int = 96) -> Image.Image:
    width = max(1, len(nsd_ids)) * tile_size
    strip = Image.new("RGB", (width, tile_size + 20), "white")
    for i, (nid, weight) in enumerate(zip(nsd_ids, weights)):
        img = store.get_image(int(nid)).convert("RGB")
        thumb = ImageOps.contain(img, (tile_size, tile_size))
        tile = Image.new("RGB", (tile_size, tile_size + 20), "white")
        tile.paste(thumb, ((tile_size - thumb.width) // 2, (tile_size - thumb.height) // 2))
        ImageDraw.Draw(tile).text((4, tile_size + 2), f"{int(nid)} | {weight:.2f}", fill=(0, 0, 0))
        strip.paste(tile, (i * tile_size, 0))
    return strip


def _contact_sheet(panel_paths: list[Path], out_path: Path, cols: int = 3) -> None:
    if not panel_paths:
        return
    images = [Image.open(p).convert("RGB") for p in panel_paths]
    tile_w = max(img.width for img in images)
    tile_h = max(img.height for img in images)
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        sheet.paste(img, (c * tile_w, r * tile_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _fused_shortlist_data(system, analysis, topk: int) -> list[dict[str, Any]]:
    compact_scores = analysis.scores["compact_csls"]
    legacy_scores = analysis.scores["legacy_csls"]
    rerank_scores = analysis.scores["rerank"]

    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_scores)
    legacy_order, legacy_gt_rank = _gt_rank_from_scores(legacy_scores)
    rerank_order, rerank_gt_rank = _gt_rank_from_scores(rerank_scores)

    n = compact_scores.shape[0]
    shortlist_k = min(int(system.frozen_config["shortlist_k"]), n)
    shortlist = compact_order[:, :shortlist_k]
    row_idx = np.arange(n)[:, None]

    compact_sl = compact_scores[row_idx, shortlist]
    legacy_sl = legacy_scores[row_idx, shortlist]
    rerank_sl = rerank_scores[row_idx, shortlist]

    alpha = float(system.frozen_config["alpha"])
    beta = float(system.frozen_config["beta"])
    gamma = float(system.frozen_config["gamma"])
    family = str(system.frozen_config["family"])
    normalization = str(system.frozen_config["normalization"])

    if family == "normalized_weighted":
        fused_sl = (
            alpha * _normalize_shortlist_scores(compact_sl, normalization)
            + beta * _normalize_shortlist_scores(rerank_sl, normalization)
            + gamma * _normalize_shortlist_scores(legacy_sl, normalization)
        )
    elif family == "weighted":
        fused_sl = alpha * compact_sl + beta * rerank_sl + gamma * legacy_sl
    elif family == "rrf":
        fused_sl = (
            alpha / (60.0 + _rank_within_shortlist(compact_sl))
            + beta / (60.0 + _rank_within_shortlist(rerank_sl))
            + gamma / (60.0 + _rank_within_shortlist(legacy_sl))
        )
    elif family == "rank_average":
        fused_sl = -(
            alpha * _rank_within_shortlist(compact_sl)
            + beta * _rank_within_shortlist(rerank_sl)
            + gamma * _rank_within_shortlist(legacy_sl)
        )
    else:
        raise ValueError(f"Unsupported family: {family}")

    local_order = np.argsort(-fused_sl, axis=1)
    fused_shortlist = shortlist[row_idx, local_order]
    fused_sorted = fused_sl[row_idx, local_order]

    rows: list[dict[str, Any]] = []
    for i, base in enumerate(analysis.per_query):
        top_idx = fused_shortlist[i, :topk].astype(np.int32)
        top_nsd = analysis.aligned["nsd_ids"][top_idx].astype(np.int32).tolist()
        compact_top = compact_scores[i, top_idx].astype(np.float64).tolist()
        legacy_top = legacy_scores[i, top_idx].astype(np.float64).tolist()
        rerank_top = rerank_scores[i, top_idx].astype(np.float64).tolist()
        fused_top = fused_sorted[i, :topk].astype(np.float64)
        fused_weights = _softmax(fused_top)
        margin = None if fused_sorted.shape[1] < 2 else float(fused_sorted[i, 0] - fused_sorted[i, 1])

        top1_idx = int(base["fused_top1_index"])
        diversity_embedding = analysis.aligned["compact_gts"][top1_idx].astype(np.float32)
        prior_embedding = np.average(
            analysis.aligned["compact_gts"][top_idx].astype(np.float32),
            axis=0,
            weights=fused_weights,
        ).astype(np.float32)

        rows.append(
            {
                "split": analysis.split,
                "row_index": int(base["row_index"]),
                "query_nsd_id": int(base["nsd_id"]),
                "gt_nsd_id": int(base["nsd_id"]),
                "compact_top1_nsd_id": int(base["compact_top1_nsd_id"]),
                "legacy_top1_nsd_id": int(base["legacy_top1_nsd_id"]),
                "rerank_top1_nsd_id": int(base["rerank_top1_nsd_id"]),
                "fused_top1_nsd_id": int(base["fused_top1_nsd_id"]),
                "compact_gt_rank": int(compact_gt_rank[i]),
                "legacy_gt_rank": int(legacy_gt_rank[i]),
                "rerank_gt_rank": int(rerank_gt_rank[i]),
                "fused_gt_rank": int(base["fused_gt_rank"]),
                "bucket": _bucket_from_rank(int(base["fused_gt_rank"])),
                "fused_fixed_case": bool(base["fused_fixed_case"]),
                "fused_fixed_to_top1": bool(base["fused_fixed_to_top1"]),
                "fused_margin_top1_top2": margin,
                "topk": top_nsd,
                "topk_compact_scores": compact_top,
                "topk_legacy_scores": legacy_top,
                "topk_rerank_scores": rerank_top,
                "topk_fused_scores": fused_top.tolist(),
                "topk_weights": fused_weights.astype(np.float64).tolist(),
                "_diversity_embedding": diversity_embedding,
                "_prior_embedding": prior_embedding,
            }
        )
    return rows


def _priority(row: dict[str, Any]) -> float:
    margin = float(row["fused_margin_top1_top2"] or 0.0)
    rank = int(row["fused_gt_rank"])
    bucket = row["bucket"]
    if bucket == "perfect":
        return 5.0 + margin + 0.15 * (row["compact_gt_rank"] + row["legacy_gt_rank"])
    if bucket == "good":
        return 4.0 + (6 - rank) + 0.5 * margin
    if bucket == "near_good":
        return 3.0 + (21 - rank) + 0.25 * margin
    return 2.0 + rank - 0.25 * margin


def _diverse_pick(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(chosen) < count:
        best_idx = 0
        best_score = None
        for idx, row in enumerate(remaining):
            base = _priority(row)
            if chosen:
                diversity = min(
                    1.0 - _cosine(row["_prior_embedding"], prev["_prior_embedding"])
                    for prev in chosen
                )
            else:
                diversity = 1.0
            score = base + 0.75 * diversity
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        chosen.append(remaining.pop(best_idx))
    return chosen


def select_examples(rows: list[dict[str, Any]], per_bucket_total: int = 4, per_split_target: int = 2) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        bucket_rows = [r for r in rows if r["bucket"] == bucket]
        split_selected: list[dict[str, Any]] = []
        for split in ["val", "shared1000"]:
            split_pool = [r for r in bucket_rows if r["split"] == split]
            split_pool.sort(key=_priority, reverse=True)
            split_selected.extend(_diverse_pick(split_pool, per_split_target))
        used_ids = {id(r) for r in split_selected}
        if len(split_selected) < per_bucket_total:
            remainder = [r for r in bucket_rows if id(r) not in used_ids]
            remainder.sort(key=_priority, reverse=True)
            split_selected.extend(_diverse_pick(remainder, per_bucket_total - len(split_selected)))
        for row in split_selected[:per_bucket_total]:
            clean = dict(row)
            clean.pop("_diversity_embedding", None)
            clean.pop("_prior_embedding", None)
            selected.append(clean)
    for idx, row in enumerate(selected):
        row["selection_order"] = idx
    return selected


def encode_images_clip(images: list[Image.Image], clip_ctx: ClipContext) -> np.ndarray:
    tensors = torch.stack([clip_ctx.preprocess(img.convert("RGB")) for img in images]).to(clip_ctx.device)
    with torch.no_grad():
        emb = clip_ctx.model.encode_image(tensors)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return emb.detach().cpu().numpy().astype(np.float32)


def build_weighted_latent_prior(pipe, images: list[Image.Image], weights: list[float]) -> tuple[Image.Image, torch.Tensor]:
    vae = pipe.vae
    device = pipe.device
    vae_dtype = next(vae.parameters()).dtype
    latent_list = []
    with torch.no_grad():
        for img in images:
            tensor = pipe.image_processor.preprocess(img.convert("RGB")).to(device=device, dtype=vae_dtype)
            latent = vae.encode(tensor).latent_dist.mean * vae.config.scaling_factor
            latent_list.append(latent)
        latents = torch.cat(latent_list, dim=0)
        weight_t = torch.tensor(weights, device=device, dtype=latents.dtype).view(len(weights), 1, 1, 1)
        prior_latent = (latents * weight_t).sum(dim=0, keepdim=True) / weight_t.sum().clamp_min(1e-8)
        decoded = vae.decode(prior_latent / vae.config.scaling_factor).sample
        decoded = decoded.to(dtype=torch.float32)
        prior_image = pipe.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGB")
    return prior_image, prior_latent


def generate_candidates(
    pipe,
    prior_image: Image.Image,
    *,
    num_candidates: int,
    guidance_scale: float,
    strength: float,
    num_inference_steps: int,
    seed: int,
) -> list[tuple[int, Image.Image]]:
    candidates: list[tuple[int, Image.Image]] = []
    for offset in range(num_candidates):
        cand_seed = int(seed + offset)
        generator = torch.Generator(device=pipe.device).manual_seed(cand_seed)
        result = pipe(
            prompt="",
            negative_prompt="text, watermark, blurry, low quality, distorted, oversaturated",
            image=prior_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        candidates.append((cand_seed, result.images[0].convert("RGB")))
    return candidates


def choose_candidate(
    candidates: list[tuple[int, Image.Image]],
    retrieved_images: list[Image.Image],
    retrieved_weights: list[float],
    clip_ctx: ClipContext,
) -> tuple[int, Image.Image, list[dict[str, Any]]]:
    retrieved_embs = encode_images_clip(retrieved_images, clip_ctx)
    weighted_proto = np.average(retrieved_embs, axis=0, weights=np.asarray(retrieved_weights, dtype=np.float32))
    weighted_proto = weighted_proto / np.maximum(np.linalg.norm(weighted_proto), 1e-8)

    candidate_imgs = [img for _, img in candidates]
    candidate_embs = encode_images_clip(candidate_imgs, clip_ctx)

    score_rows: list[dict[str, Any]] = []
    best_idx = 0
    best_score = None
    for idx, (seed, img) in enumerate(candidates):
        cand_emb = candidate_embs[idx]
        proto_sim = _cosine(cand_emb, weighted_proto)
        retrieved_sim = float(np.sum(np.asarray(retrieved_weights, dtype=np.float32) * (retrieved_embs @ cand_emb)))
        score = 0.5 * proto_sim + 0.5 * retrieved_sim
        score_rows.append(
            {
                "seed": int(seed),
                "prior_proto_similarity": float(proto_sim),
                "retrieved_weighted_similarity": float(retrieved_sim),
                "selection_score": float(score),
            }
        )
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
    chosen_seed, chosen_img = candidates[best_idx]
    return chosen_seed, chosen_img, score_rows


def compute_metrics(pred: Image.Image, gt: Image.Image, clip_ctx: ClipContext, lpips_eval: Any) -> dict[str, Any]:
    from fmri2img.eval.image_metrics import clip_score
    from fmri2img.eval.recon_eval import compute_pixcorr, compute_psnr, compute_ssim

    pred_arr = np.asarray(ImageOps.contain(pred.convert("RGB"), (256, 256)), dtype=np.float32) / 255.0
    gt_arr = np.asarray(ImageOps.contain(gt.convert("RGB"), (256, 256)), dtype=np.float32) / 255.0
    mse = float(np.mean((pred_arr - gt_arr) ** 2))
    pixel_l2 = float(np.sqrt(np.sum((pred_arr - gt_arr) ** 2)))
    pixcorr = float(compute_pixcorr(pred_arr, gt_arr))
    ssim = float(compute_ssim(pred_arr, gt_arr))
    psnr = float(compute_psnr(pred_arr, gt_arr))
    lpips_val = float(lpips_eval.compute(pred_arr, gt_arr))
    clip_sim = float(clip_score(pred, gt, clip_ctx.model, device=clip_ctx.device))
    return {
        "mse": mse,
        "pixel_l2": pixel_l2,
        "pixcorr": pixcorr,
        "ssim": ssim,
        "psnr": psnr,
        "lpips": lpips_val,
        "clip_image_similarity": clip_sim,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_examples": len(rows)}
    names = [
        "retrieval_mse", "retrieval_pixel_l2", "retrieval_pixcorr", "retrieval_ssim", "retrieval_psnr", "retrieval_lpips", "retrieval_clip_image_similarity",
        "diffusion_mse", "diffusion_pixel_l2", "diffusion_pixcorr", "diffusion_ssim", "diffusion_psnr", "diffusion_lpips", "diffusion_clip_image_similarity",
        "delta_mse", "delta_pixel_l2", "delta_pixcorr", "delta_ssim", "delta_psnr", "delta_lpips", "delta_clip_image_similarity",
    ]
    for name in names:
        vals = [_safe_float(r.get(name)) for r in rows]
        vals = [v for v in vals if v is not None]
        out[f"{name}_mean"] = None if not vals else float(np.mean(vals))
        out[f"{name}_median"] = None if not vals else float(np.median(vals))
    return out


def _jsonify_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple, dict)):
                clean[key] = json.dumps(value)
            else:
                clean[key] = value
        out.append(clean)
    return out


def save_example_outputs(
    output_dir: Path,
    bucket: str,
    row: dict[str, Any],
    store: StimulusStore,
    gt: Image.Image,
    retrieval_only: Image.Image,
    prior_image: Image.Image,
    final_image: Image.Image,
) -> Path:
    query = int(row["query_nsd_id"])
    recon_dir = output_dir / "reconstructions" / bucket
    fig_dir = output_dir / "figures" / "examples" / bucket
    recon_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    gt_path = recon_dir / f"query_{query}_ground_truth.png"
    ret_path = recon_dir / f"query_{query}_retrieval_top1.png"
    prior_path = recon_dir / f"query_{query}_diffusion_prior.png"
    final_path = recon_dir / f"query_{query}_diffusion_final.png"
    strip_path = recon_dir / f"query_{query}_topk_strip.png"

    gt.save(gt_path)
    retrieval_only.save(ret_path)
    prior_image.save(prior_path)
    final_image.save(final_path)
    strip = _topk_strip(store, [int(x) for x in row["topk"]], [float(x) for x in row["topk_weights"]])
    strip.save(strip_path)

    panel = Image.new("RGB", (4 * 256, 256 + 96 + strip.height + 12), "white")
    tiles = [
        _tile(gt, "Ground Truth", [f"split={row['split']}", f"query={query}", f"bucket={bucket}"]),
        _tile(retrieval_only, "Retrieval Top-1", [f"retrieved={row['fused_top1_nsd_id']}", f"fused GT rank={row['fused_gt_rank']}", f"ssim={row['retrieval_ssim']:.4f}"]),
        _tile(prior_image, "Diffusion Prior", [f"top-k={len(row['topk'])}", f"top1 weight={float(row['topk_weights'][0]):.3f}", f"margin={float(row['fused_margin_top1_top2'] or 0.0):.4f}"]),
        _tile(final_image, "Final Diffusion", [f"seed={row['selected_seed']}", f"ssim={row['diffusion_ssim']:.4f}", f"clip={row['diffusion_clip_image_similarity']:.4f}"]),
    ]
    for idx, tile in enumerate(tiles):
        panel.paste(tile, (idx * tile.width, 0))
    panel.paste(strip, (0, tiles[0].height + 8))
    panel_path = fig_dir / f"query_{query}_panel.png"
    panel.save(panel_path)
    return panel_path


def write_readme(output_dir: Path, dep: DependencyReport) -> None:
    text = f"""# True Diffusion Add-on

This directory contains the final thesis-grade qualitative reconstruction addon built on top of the frozen best retrieval system.

## Frozen Retrieval Backbone

- tri/main results dir: `experimental_results/V35_legacy_teacher_distill/subj01`
- legacy results dir: `experimental_results/N1v28a_dual_head/subj01`
- frozen tri-fusion config: `{EXPECTED_FROZEN_TRI_CONFIG}`
- frozen SHARED1000 R@1: `{EXPECTED_SHARED1000_R1:.1%}`

## Method

This addon performs:

1. frozen tri-fusion shortlist retrieval
2. score-weighted top-k prior construction
3. score-weighted latent prior formation in the diffusion VAE space
4. true Stable Diffusion img2img generation
5. multi-sample candidate selection using retrieval-prior consistency only

## Integrity constraints

- no retrieval retraining
- no fusion changes
- no GT during generation or candidate selection
- no heuristic fallback in this script

## Runtime status

- true diffusion run: `{dep.true_diffusion_run}`
- diffusion model: `{dep.model_id}`
- diffusion path: `{dep.diffusion_model_path}`
- dependency versions: `{dep.versions}`

## Outputs

- `DIFFUSION_ADDON_DESIGN.md`
- `selected_examples.json`
- `selected_examples.csv`
- `metrics/`
- `figures/`
- `reconstructions/`
"""
    (output_dir / "README.md").write_text(text)


def write_design_note(output_dir: Path, dep: DependencyReport) -> None:
    text = f"""# DIFFUSION ADDON DESIGN

## Audit of previous code

### `scripts/reconstruction/decode_two_stage.py`

This path is **not thesis-grade** for the frozen final system.
It creates a fresh random `Linear(512 -> 1024)` projection at inference time and injects that into Stable Diffusion prompt embeddings.
That projection is never trained, never justified, and is not aligned with the validated retrieval architecture.
It is scientifically unsuitable for final reporting.

### `scripts/reconstruction/generate_images.py`

This is largely legacy scaffolding for embedding export and visualization.
It is not an end-to-end retrieval-aware reconstruction system.
It does not use the frozen tri-fusion benchmark outputs.

### `scripts/reconstruction/decode_diffusion.py`

This path is stronger than the legacy scripts, but still not sufficient as the final thesis addon.
It performs generic CLIP-to-diffusion conditioning and optional pooling tricks, but it does **not** make principled use of the frozen tri-expert retrieval system.
It ignores the validated tri-fusion shortlist/top-k structure and therefore is only loosely aligned with the reportable architecture.

### Reusable components

The following pieces are reused:

- frozen best-system loader from `scripts/evaluation/final_best_system.py`
- NSD stimulus loading from `scripts/evaluation/export_final_qualitatives.py`
- image metrics from `src/fmri2img/eval/image_metrics.py`
- reconstruction metrics from `src/fmri2img/eval/recon_eval.py`

## Final algorithm

The final addon is architecture-aligned because it uses the actual frozen retrieval outputs as the semantic and visual prior:

1. compute the frozen fixed tri-fusion shortlist using the approved production recipe
2. take fused top-k retrieved images as the visual prior
3. turn fused local scores into score weights
4. construct a score-weighted latent prior in the diffusion VAE space
5. run true Stable Diffusion img2img from that prior
6. generate multiple seeded candidates deterministically
7. select the final sample using retrieval-prior consistency only, never ground truth

## Why this is more professional

- it keeps the frozen 77.2% retrieval benchmark untouched
- it visibly exploits the real compact/legacy/fused architecture instead of bypassing it
- it uses retrieval top-k as the reconstruction prior rather than arbitrary text prompting
- it avoids random untrained projections at inference time
- it forbids heuristic fallback under the label of diffusion
- it records dependency availability and the exact diffusion path used

## Runtime result for this export

- true diffusion run: `{dep.true_diffusion_run}`
- diffusion model: `{dep.model_id}`
- diffusion model path: `{dep.diffusion_model_path}`
- dependency versions: `{dep.versions}`
"""
    (output_dir / "DIFFUSION_ADDON_DESIGN.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tri-results-dir", type=Path, default=DEFAULT_TRI_RESULTS_DIR)
    parser.add_argument("--legacy-results-dir", type=Path, default=DEFAULT_LEGACY_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stimuli-hdf5", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--samples-per-query", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--allow-config-drift", action="store_true")
    args = parser.parse_args()

    versions = ensure_true_diffusion_dependencies()

    system, val_analysis, shared_analysis = load_final_best_system_analysis(
        tri_results_dir=args.tri_results_dir,
        legacy_results_dir=args.legacy_results_dir,
        strict=not args.allow_config_drift,
    )
    stimuli_hdf5 = resolve_stimuli_hdf5(args.stimuli_hdf5)
    logger.info("Using stimuli HDF5: %s", stimuli_hdf5)

    cache_dir = args.hf_cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))

    clip_ctx = load_clip_context(args.device, cache_dir)
    from fmri2img.eval.recon_eval import LPIPSEvaluator

    lpips_eval = LPIPSEvaluator(device=_resolve_torch_device(args.device))
    pipe = load_diffusion_pipe(args.model_id, args.device, args.allow_model_download, cache_dir)
    dep = DependencyReport(
        versions=versions,
        model_id=args.model_id,
        diffusion_model_path=str(args.model_id),
        true_diffusion_run=True,
    )

    output_dir = args.output_dir
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "examples").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "contact_sheets").mkdir(parents=True, exist_ok=True)
    for bucket in BUCKETS:
        (output_dir / "reconstructions" / bucket).mkdir(parents=True, exist_ok=True)

    all_rows = _fused_shortlist_data(system, val_analysis, args.topk) + _fused_shortlist_data(system, shared_analysis, args.topk)
    selected = select_examples(all_rows, per_bucket_total=4, per_split_target=2)

    store = StimulusStore(stimuli_hdf5)
    panel_paths_by_bucket: dict[str, list[Path]] = {bucket: [] for bucket in BUCKETS}
    metric_rows: list[dict[str, Any]] = []
    selected_payload_rows: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(selected):
            gt = store.get_image(int(row["gt_nsd_id"]))
            retrieval_only = store.get_image(int(row["fused_top1_nsd_id"]))
            retrieved_images = [store.get_image(int(nid)) for nid in row["topk"]]
            retrieved_weights = [float(x) for x in row["topk_weights"]]

            prior_image, _ = build_weighted_latent_prior(pipe, retrieved_images, retrieved_weights)
            candidates = generate_candidates(
                pipe,
                prior_image,
                num_candidates=args.samples_per_query,
                guidance_scale=args.guidance_scale,
                strength=args.strength,
                num_inference_steps=args.steps,
                seed=args.seed + idx * args.samples_per_query,
            )
            selected_seed, final_image, candidate_scores = choose_candidate(
                candidates,
                retrieved_images,
                retrieved_weights,
                clip_ctx,
            )

            retrieval_metrics = compute_metrics(retrieval_only, gt, clip_ctx, lpips_eval)
            diffusion_metrics = compute_metrics(final_image, gt, clip_ctx, lpips_eval)

            metric_row = {
                "split": row["split"],
                "bucket": row["bucket"],
                "query_nsd_id": row["query_nsd_id"],
                "gt_nsd_id": row["gt_nsd_id"],
                "fused_top1_nsd_id": row["fused_top1_nsd_id"],
                "selected_seed": int(selected_seed),
                "retrieval_mse": retrieval_metrics["mse"],
                "retrieval_pixel_l2": retrieval_metrics["pixel_l2"],
                "retrieval_pixcorr": retrieval_metrics["pixcorr"],
                "retrieval_ssim": retrieval_metrics["ssim"],
                "retrieval_psnr": retrieval_metrics["psnr"],
                "retrieval_lpips": retrieval_metrics["lpips"],
                "retrieval_clip_image_similarity": retrieval_metrics["clip_image_similarity"],
                "diffusion_mse": diffusion_metrics["mse"],
                "diffusion_pixel_l2": diffusion_metrics["pixel_l2"],
                "diffusion_pixcorr": diffusion_metrics["pixcorr"],
                "diffusion_ssim": diffusion_metrics["ssim"],
                "diffusion_psnr": diffusion_metrics["psnr"],
                "diffusion_lpips": diffusion_metrics["lpips"],
                "diffusion_clip_image_similarity": diffusion_metrics["clip_image_similarity"],
                "delta_mse": diffusion_metrics["mse"] - retrieval_metrics["mse"],
                "delta_pixel_l2": diffusion_metrics["pixel_l2"] - retrieval_metrics["pixel_l2"],
                "delta_pixcorr": diffusion_metrics["pixcorr"] - retrieval_metrics["pixcorr"],
                "delta_ssim": diffusion_metrics["ssim"] - retrieval_metrics["ssim"],
                "delta_psnr": diffusion_metrics["psnr"] - retrieval_metrics["psnr"],
                "delta_lpips": diffusion_metrics["lpips"] - retrieval_metrics["lpips"],
                "delta_clip_image_similarity": diffusion_metrics["clip_image_similarity"] - retrieval_metrics["clip_image_similarity"],
            }
            row.update(metric_row)
            row["selected_seed"] = int(selected_seed)
            row["candidate_scores"] = candidate_scores
            metric_rows.append(metric_row)
            selected_payload_rows.append(row)

            panel_path = save_example_outputs(output_dir, row["bucket"], row, store, gt, retrieval_only, prior_image, final_image)
            panel_paths_by_bucket[row["bucket"]].append(panel_path)
    finally:
        store.close()

    selected_json = {
        "system": {
            "tri_results_dir": str(system.tri_results_dir),
            "legacy_results_dir": str(system.legacy_results_dir),
            "frozen_tri_fusion_config": system.frozen_config,
            "expected_shared1000_r1": EXPECTED_SHARED1000_R1,
            "saved_shared1000_r1": system.frozen_shared1000_r1,
        },
        "generation": {
            "true_diffusion_run": True,
            "model_id": args.model_id,
            "topk": args.topk,
            "samples_per_query": args.samples_per_query,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "steps": args.steps,
            "seed": args.seed,
        },
        "selected_examples": selected_payload_rows,
    }
    _write_json(output_dir / "selected_examples.json", selected_json)
    _write_csv(output_dir / "selected_examples.csv", _jsonify_rows(selected_payload_rows))

    summary = {
        "true_diffusion_run": True,
        "method": "score_weighted_latent_prior_img2img",
        "model_id": args.model_id,
        "overall": summarize_rows(metric_rows),
        "by_bucket": {bucket: summarize_rows([r for r in metric_rows if r["bucket"] == bucket]) for bucket in BUCKETS},
    }
    _write_json(output_dir / "metrics" / "summary_metrics.json", summary)
    _write_csv(output_dir / "metrics" / "per_example_metrics.csv", metric_rows)
    summary_rows = [{"bucket": "overall", **summary["overall"]}] + [
        {"bucket": bucket, **summary["by_bucket"][bucket]} for bucket in BUCKETS
    ]
    _write_csv(output_dir / "metrics" / "summary_metrics.csv", summary_rows)
    _write_csv(output_dir / "metrics" / "bucket_summary.csv", summary_rows[1:])

    for bucket, paths in panel_paths_by_bucket.items():
        _contact_sheet(paths, output_dir / "figures" / "contact_sheets" / f"{bucket}_sheet.png")
    mixed_paths = []
    for bucket in BUCKETS:
        mixed_paths.extend(panel_paths_by_bucket[bucket][:2])
    _contact_sheet(mixed_paths, output_dir / "figures" / "contact_sheets" / "mixed_sheet.png")

    write_readme(output_dir, dep)
    write_design_note(output_dir, dep)

    logger.info("Saved true diffusion addon to %s", output_dir)
    logger.info("Frozen SHARED1000 R@1 remains %.1f%%", EXPECTED_SHARED1000_R1 * 100)
    logger.info("True diffusion run: yes")


if __name__ == "__main__":
    main()
