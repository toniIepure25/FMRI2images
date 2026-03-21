#!/usr/bin/env python3
"""Export V40 confidence-aware retrieval-guided diffusion reconstructions.

This script keeps the frozen tri-expert retrieval benchmark unchanged and adds a
retrieval-preserving, inference-only diffusion refinement stage for a small set
of thesis-quality qualitative examples.

Core design:
    fMRI -> frozen tri-fusion retrieval -> confidence-aware route controller
         -> anchor-preserving latent prior -> SD img2img refinement
         -> prior-consistency final selection (never GT)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
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
from sweep_tri_fusion_retrieval import (
    _gt_rank_from_scores,
    _normalize_shortlist_scores,
    _rank_within_shortlist,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("final_outputs/v40_confidence_aware_diffusion")
DEFAULT_MODEL_ID = "sd2-community/stable-diffusion-2-1"
BUCKETS = ["perfect", "good", "near_good", "hard"]

REQUIRED_MODULES = [
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("open_clip", "open_clip_torch"),
    ("lpips", "lpips"),
]


@dataclass(frozen=True)
class RouteConfig:
    name: str
    strength: float
    steps: int
    guidance_scale: float
    num_candidates: int
    support_gain: float
    select_anchor_weight: float
    select_support_weight: float
    select_prior_weight: float
    select_layout_weight: float
    select_edge_weight: float
    select_color_penalty: float


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
class ControllerDecision:
    route: str
    route_score: float
    route_reason: str
    anchor_nsd_id: int
    anchor_source: str
    anchor_index_within_topk: int
    top1_weight: float
    normalized_entropy: float
    expert_vote_fraction: float
    anchor_consensus: float
    anchor_margin: float
    support_spread: float


ROUTE_CONFIGS: dict[str, RouteConfig] = {
    "identity_pass": RouteConfig(
        name="identity_pass",
        strength=0.0,
        steps=0,
        guidance_scale=0.0,
        num_candidates=0,
        support_gain=0.0,
        select_anchor_weight=0.55,
        select_support_weight=0.20,
        select_prior_weight=0.10,
        select_layout_weight=0.10,
        select_edge_weight=0.05,
        select_color_penalty=0.05,
    ),
    "low_strength_refine": RouteConfig(
        name="low_strength_refine",
        strength=0.05,
        steps=20,
        guidance_scale=3.5,
        num_candidates=2,
        support_gain=0.04,
        select_anchor_weight=0.40,
        select_support_weight=0.18,
        select_prior_weight=0.08,
        select_layout_weight=0.24,
        select_edge_weight=0.10,
        select_color_penalty=0.06,
    ),
    "guided_refine": RouteConfig(
        name="guided_refine",
        strength=0.14,
        steps=30,
        guidance_scale=4.5,
        num_candidates=4,
        support_gain=0.10,
        select_anchor_weight=0.32,
        select_support_weight=0.23,
        select_prior_weight=0.12,
        select_layout_weight=0.20,
        select_edge_weight=0.13,
        select_color_penalty=0.06,
    ),
    "exploratory_refine": RouteConfig(
        name="exploratory_refine",
        strength=0.24,
        steps=40,
        guidance_scale=5.5,
        num_candidates=6,
        support_gain=0.16,
        select_anchor_weight=0.24,
        select_support_weight=0.30,
        select_prior_weight=0.16,
        select_layout_weight=0.16,
        select_edge_weight=0.14,
        select_color_penalty=0.05,
    ),
}


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


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec)
    if float(denom) <= 1e-8:
        return vec.astype(np.float32)
    return (vec / denom).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    shifted = values / max(float(temperature), 1e-6)
    shifted = shifted - np.max(shifted)
    probs = np.exp(shifted)
    probs /= np.maximum(np.sum(probs), 1e-12)
    return probs.astype(np.float64)


def _bucket_from_rank(rank: int) -> str:
    if rank == 1:
        return "perfect"
    if rank <= 5:
        return "good"
    if rank <= 20:
        return "near_good"
    return "hard"


def _normalized_entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size <= 1:
        return 0.0
    safe = np.clip(weights, 1e-12, 1.0)
    entropy = -float(np.sum(safe * np.log(safe)))
    return float(entropy / np.log(len(weights)))


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
            "V40 cannot run because required true-diffusion dependencies are missing.\n"
            "Install them first; no heuristic fallback is permitted.\n"
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
    model = model.to(_resolve_torch_device(device))
    model.eval()
    return ClipContext(model=model, preprocess=preprocess, device=_resolve_torch_device(device))


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


def _to_np01(img: Image.Image, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    return np.asarray(ImageOps.contain(img.convert("RGB"), size), dtype=np.float32) / 255.0


def _rgb_mean_drift(img_a: Image.Image, img_b: Image.Image) -> float:
    arr_a = _to_np01(img_a)
    arr_b = _to_np01(img_b)
    return float(np.mean(np.abs(arr_a.mean(axis=(0, 1)) - arr_b.mean(axis=(0, 1)))))


def _edge_map(arr: np.ndarray) -> np.ndarray:
    gray = arr.mean(axis=2)
    dx = np.diff(gray, axis=1, prepend=gray[:, :1])
    dy = np.diff(gray, axis=0, prepend=gray[:1, :])
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _edge_corr(img_a: Image.Image, img_b: Image.Image) -> float:
    ea = _edge_map(_to_np01(img_a)).reshape(-1)
    eb = _edge_map(_to_np01(img_b)).reshape(-1)
    if float(np.std(ea)) <= 1e-8 or float(np.std(eb)) <= 1e-8:
        return 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


def _tile(img: Image.Image, title: str, lines: list[str], tile_size: int = 256, footer_h: int = 118) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size + footer_h), "white")
    thumb = ImageOps.contain(img.convert("RGB"), (tile_size, tile_size))
    canvas.paste(thumb, ((tile_size - thumb.width) // 2, (tile_size - thumb.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, tile_size, tile_size, tile_size + footer_h), fill=(248, 248, 248))
    draw.text((8, tile_size + 6), title, fill=(0, 0, 0))
    for i, line in enumerate(lines[:7]):
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


def encode_images_clip(images: list[Image.Image], clip_ctx: ClipContext) -> np.ndarray:
    tensors = torch.stack([clip_ctx.preprocess(img.convert("RGB")) for img in images]).to(clip_ctx.device)
    with torch.no_grad():
        emb = clip_ctx.model.encode_image(tensors)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return emb.detach().cpu().numpy().astype(np.float32)


def _combined_consensus_matrix(compact_embs: np.ndarray, legacy_embs: np.ndarray) -> np.ndarray:
    compact_cos = compact_embs @ compact_embs.T
    legacy_cos = legacy_embs @ legacy_embs.T
    combined = 0.3 * compact_cos + 0.7 * legacy_cos
    np.fill_diagonal(combined, 0.0)
    return combined.astype(np.float32)


def _choose_anchor(
    topk_ids: list[int],
    weights: np.ndarray,
    compact_embs: np.ndarray,
    legacy_embs: np.ndarray,
    expert_top1_ids: list[int],
) -> tuple[int, str, float, float, float, float]:
    consensus = _combined_consensus_matrix(compact_embs, legacy_embs)
    weighted_consensus = (consensus * weights[None, :]).sum(axis=1)
    support_spread = float((consensus * weights[None, :]).sum(axis=1).mean())
    top1_id = int(topk_ids[0])
    top1_idx = 0
    top1_score = float(weighted_consensus[top1_idx])

    expert_votes = sum(1 for nid in expert_top1_ids if int(nid) == top1_id)
    top1_vote_fraction = expert_votes / max(len(expert_top1_ids), 1)

    best_idx = int(np.argmax(weighted_consensus))
    best_score = float(weighted_consensus[best_idx])

    ranking = np.sort(weighted_consensus)[::-1]
    second_score = float(ranking[1]) if ranking.size > 1 else best_score

    use_medoid = (
        best_idx != top1_idx
        and top1_vote_fraction < 0.34
        and best_score - top1_score > 0.03
    )
    chosen_idx = best_idx if use_medoid else top1_idx
    chosen_source = "consensus_medoid" if use_medoid else "fused_top1"
    chosen_score = float(weighted_consensus[chosen_idx])
    chosen_margin = chosen_score - second_score
    chosen_agreement = float((consensus[chosen_idx] * weights).sum())
    return chosen_idx, chosen_source, top1_vote_fraction, chosen_agreement, float(chosen_margin), support_spread


def _controller_route(
    *,
    top1_weight: float,
    normalized_entropy: float,
    expert_vote_fraction: float,
    anchor_consensus: float,
    anchor_margin: float,
    anchor_source: str,
) -> tuple[str, float, str]:
    route_score = (
        0.30 * top1_weight
        + 0.22 * anchor_consensus
        + 0.18 * (1.0 - normalized_entropy)
        + 0.18 * expert_vote_fraction
        + 0.12 * max(anchor_margin, 0.0)
    )
    if (
        anchor_source == "fused_top1"
        and expert_vote_fraction >= 0.34
        and anchor_consensus >= 0.78
        and route_score >= 0.44
    ):
        return "identity_pass", float(route_score), "stable anchor and concentrated support"
    if anchor_consensus >= 0.66 and route_score >= 0.34:
        return "low_strength_refine", float(route_score), "strong anchor but mild ambiguity"
    if anchor_consensus >= 0.54 or route_score >= 0.26:
        return "guided_refine", float(route_score), "meaningful support but refinement justified"
    return "exploratory_refine", float(route_score), "diffuse shortlist or unstable anchor"


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
        raise ValueError(f"Unsupported fusion family: {family}")

    local_order = np.argsort(-fused_sl, axis=1)
    fused_shortlist = shortlist[row_idx, local_order]
    fused_sorted = fused_sl[row_idx, local_order]

    rows: list[dict[str, Any]] = []
    for i, base in enumerate(analysis.per_query):
        top_idx = fused_shortlist[i, :topk].astype(np.int32)
        top_nsd = analysis.aligned["nsd_ids"][top_idx].astype(np.int32).tolist()
        fused_top = fused_sorted[i, :topk].astype(np.float64)
        topk_weights = _softmax(fused_top)
        entropy = _normalized_entropy(topk_weights)
        compact_top_embs = analysis.aligned["compact_gts"][top_idx].astype(np.float32)
        legacy_top_embs = analysis.aligned["legacy_gts"][top_idx].astype(np.float32)

        anchor_idx, anchor_source, vote_fraction, anchor_consensus, anchor_margin, support_spread = _choose_anchor(
            topk_ids=top_nsd,
            weights=topk_weights.astype(np.float32),
            compact_embs=compact_top_embs,
            legacy_embs=legacy_top_embs,
            expert_top1_ids=[
                int(base["compact_top1_nsd_id"]),
                int(base["legacy_top1_nsd_id"]),
                int(base["rerank_top1_nsd_id"]),
            ],
        )
        route_name, route_score, route_reason = _controller_route(
            top1_weight=float(topk_weights[0]),
            normalized_entropy=entropy,
            expert_vote_fraction=vote_fraction,
            anchor_consensus=anchor_consensus,
            anchor_margin=anchor_margin,
            anchor_source=anchor_source,
        )

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
                "fused_margin_top1_top2": None if fused_sorted.shape[1] < 2 else float(fused_sorted[i, 0] - fused_sorted[i, 1]),
                "topk": top_nsd,
                "topk_compact_scores": compact_scores[i, top_idx].astype(np.float64).tolist(),
                "topk_legacy_scores": legacy_scores[i, top_idx].astype(np.float64).tolist(),
                "topk_rerank_scores": rerank_scores[i, top_idx].astype(np.float64).tolist(),
                "topk_fused_scores": fused_top.tolist(),
                "topk_weights": topk_weights.astype(np.float64).tolist(),
                "controller_route": route_name,
                "controller_score": route_score,
                "controller_reason": route_reason,
                "controller_top1_weight": float(topk_weights[0]),
                "controller_entropy": float(entropy),
                "controller_expert_vote_fraction": float(vote_fraction),
                "controller_anchor_consensus": float(anchor_consensus),
                "controller_anchor_margin": float(anchor_margin),
                "controller_support_spread": float(support_spread),
                "anchor_nsd_id": int(top_nsd[anchor_idx]),
                "anchor_source": anchor_source,
                "anchor_index_within_topk": int(anchor_idx),
                "_compact_top_embs": compact_top_embs,
                "_legacy_top_embs": legacy_top_embs,
            }
        )
    return rows


def _priority(row: dict[str, Any]) -> float:
    bucket = row["bucket"]
    rank = int(row["fused_gt_rank"])
    confidence = float(row["controller_score"])
    spread = float(row["controller_support_spread"])
    if bucket == "perfect":
        return 5.0 + confidence + spread
    if bucket == "good":
        return 4.0 + (6 - rank) + 0.5 * confidence + 0.25 * spread
    if bucket == "near_good":
        return 3.0 + (21 - rank) + 0.4 * confidence + 0.3 * spread
    return 2.0 + rank + 0.25 * spread


def _diverse_pick(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(chosen) < count:
        best_idx = 0
        best_score = None
        for idx, row in enumerate(remaining):
            base = _priority(row)
            row_proto = np.average(
                row["_compact_top_embs"],
                axis=0,
                weights=np.asarray(row["topk_weights"], dtype=np.float32),
            ).astype(np.float32)
            if chosen:
                diversity = min(
                    1.0
                    - _cosine(
                        row_proto,
                        np.average(
                            prev["_compact_top_embs"],
                            axis=0,
                            weights=np.asarray(prev["topk_weights"], dtype=np.float32),
                        ).astype(np.float32),
                    )
                    for prev in chosen
                )
            else:
                diversity = 1.0
            score = base + 0.75 * diversity
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
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
            clean.pop("_compact_top_embs", None)
            clean.pop("_legacy_top_embs", None)
            selected.append(clean)
    for idx, row in enumerate(selected):
        row["selection_order"] = idx
    return selected


def _encode_latent(pipe, img: Image.Image) -> torch.Tensor:
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype
    device = pipe.device
    with torch.no_grad():
        tensor = pipe.image_processor.preprocess(img.convert("RGB")).to(device=device, dtype=vae_dtype)
        latent = vae.encode(tensor).latent_dist.mean * vae.config.scaling_factor
    return latent


def _decode_latent(pipe, latent: torch.Tensor) -> Image.Image:
    vae = pipe.vae
    with torch.no_grad():
        decoded = vae.decode(latent / vae.config.scaling_factor).sample
        decoded = decoded.to(dtype=torch.float32)
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGB")


def build_anchor_residual_prior(
    pipe,
    anchor_image: Image.Image,
    support_images: list[Image.Image],
    support_weights: list[float],
    route_cfg: RouteConfig,
) -> tuple[Image.Image, torch.Tensor]:
    anchor_latent = _encode_latent(pipe, anchor_image)
    if not support_images or route_cfg.support_gain <= 0.0:
        return anchor_image.copy(), anchor_latent

    support_latents = [_encode_latent(pipe, img) for img in support_images]
    stacked = torch.cat(support_latents, dim=0)
    weight_t = torch.tensor(support_weights, device=stacked.device, dtype=stacked.dtype).view(len(support_weights), 1, 1, 1)
    residual = ((stacked - anchor_latent) * weight_t).sum(dim=0, keepdim=True) / weight_t.sum().clamp_min(1e-8)
    prior_latent = anchor_latent + float(route_cfg.support_gain) * residual
    prior_image = _decode_latent(pipe, prior_latent)
    return prior_image, prior_latent


def generate_candidates(
    pipe,
    init_image: Image.Image,
    *,
    route_cfg: RouteConfig,
    seed: int,
) -> list[dict[str, Any]]:
    if route_cfg.num_candidates <= 0 or route_cfg.strength <= 0.0:
        return []
    out: list[dict[str, Any]] = []
    for offset in range(route_cfg.num_candidates):
        cand_seed = int(seed + offset)
        generator = torch.Generator(device=pipe.device).manual_seed(cand_seed)
        result = pipe(
            prompt="",
            negative_prompt="text, watermark, blurry, low quality, distorted, oversaturated, painting, illustration",
            image=init_image,
            strength=route_cfg.strength,
            guidance_scale=route_cfg.guidance_scale,
            num_inference_steps=route_cfg.steps,
            generator=generator,
        )
        out.append({"kind": "diffusion_sample", "seed": cand_seed, "image": result.images[0].convert("RGB")})
    return out


def _candidate_features(
    img: Image.Image,
    *,
    candidate_emb: np.ndarray,
    anchor_img: Image.Image,
    anchor_emb: np.ndarray,
    prior_img: Image.Image,
    prior_emb: np.ndarray,
    support_embs: np.ndarray,
    support_weights: np.ndarray,
) -> dict[str, float]:
    from fmri2img.eval.recon_eval import compute_ssim

    anchor_clip = _cosine(candidate_emb, anchor_emb)
    prior_clip = _cosine(candidate_emb, prior_emb)
    support_clip = float(np.sum(support_weights * (support_embs @ candidate_emb)))
    layout_ssim = float(compute_ssim(_to_np01(img), _to_np01(anchor_img)))
    edge_corr = _edge_corr(img, anchor_img)
    color_drift = _rgb_mean_drift(img, anchor_img)
    return {
        "anchor_clip_similarity": float(anchor_clip),
        "prior_clip_similarity": float(prior_clip),
        "shortlist_consistency_score": float(support_clip),
        "anchor_layout_ssim": float(layout_ssim),
        "anchor_edge_corr": float(edge_corr),
        "anchor_color_drift": float(color_drift),
    }


def choose_candidate(
    candidate_rows: list[dict[str, Any]],
    *,
    anchor_image: Image.Image,
    prior_image: Image.Image,
    support_images: list[Image.Image],
    support_weights: list[float],
    clip_ctx: ClipContext,
    route_cfg: RouteConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = list(candidate_rows)
    if not any(row["kind"] == "anchor_baseline" for row in pool):
        pool.insert(0, {"kind": "anchor_baseline", "seed": None, "image": anchor_image.copy()})
    if not any(row["kind"] == "prior_image" for row in pool):
        pool.insert(1, {"kind": "prior_image", "seed": None, "image": prior_image.copy()})

    clip_images = [anchor_image, prior_image] + support_images + [row["image"] for row in pool]
    clip_embs = encode_images_clip(clip_images, clip_ctx)
    anchor_emb = clip_embs[0]
    prior_emb = clip_embs[1]
    support_embs = clip_embs[2 : 2 + len(support_images)]
    candidate_embs = clip_embs[2 + len(support_images) :]
    support_w = np.asarray(support_weights, dtype=np.float32)

    scored_rows: list[dict[str, Any]] = []
    best_idx = 0
    best_score = None
    for idx, row in enumerate(pool):
        feats = _candidate_features(
            row["image"],
            candidate_emb=candidate_embs[idx],
            anchor_img=anchor_image,
            anchor_emb=anchor_emb,
            prior_img=prior_image,
            prior_emb=prior_emb,
            support_embs=support_embs,
            support_weights=support_w,
        )
        selection_score = (
            route_cfg.select_anchor_weight * feats["anchor_clip_similarity"]
            + route_cfg.select_support_weight * feats["shortlist_consistency_score"]
            + route_cfg.select_prior_weight * feats["prior_clip_similarity"]
            + route_cfg.select_layout_weight * feats["anchor_layout_ssim"]
            + route_cfg.select_edge_weight * feats["anchor_edge_corr"]
            - route_cfg.select_color_penalty * feats["anchor_color_drift"]
        )
        scored = {
            "kind": row["kind"],
            "seed": row.get("seed"),
            **feats,
            "selection_score": float(selection_score),
        }
        scored_rows.append(scored)
        if best_score is None or selection_score > best_score:
            best_score = float(selection_score)
            best_idx = idx
    return pool[best_idx], scored_rows


def compute_metrics(pred: Image.Image, gt: Image.Image, clip_ctx: ClipContext, lpips_eval: Any) -> dict[str, Any]:
    from fmri2img.eval.image_metrics import clip_score
    from fmri2img.eval.recon_eval import compute_pixcorr, compute_psnr, compute_ssim

    pred_arr = _to_np01(pred)
    gt_arr = _to_np01(gt)
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


def compute_consistency_metrics(
    img: Image.Image,
    *,
    anchor_img: Image.Image,
    prior_img: Image.Image,
    support_images: list[Image.Image],
    support_weights: list[float],
    clip_ctx: ClipContext,
) -> dict[str, Any]:
    clip_images = [anchor_img, prior_img] + support_images + [img]
    clip_embs = encode_images_clip(clip_images, clip_ctx)
    anchor_emb = clip_embs[0]
    prior_emb = clip_embs[1]
    support_embs = clip_embs[2 : 2 + len(support_images)]
    img_emb = clip_embs[-1]
    support_w = np.asarray(support_weights, dtype=np.float32)
    from fmri2img.eval.recon_eval import compute_ssim

    return {
        "anchor_similarity": float(_cosine(img_emb, anchor_emb)),
        "prior_similarity": float(_cosine(img_emb, prior_emb)),
        "shortlist_consistency_score": float(np.sum(support_w * (support_embs @ img_emb))),
        "anchor_layout_ssim": float(compute_ssim(_to_np01(img), _to_np01(anchor_img))),
        "anchor_edge_corr": float(_edge_corr(img, anchor_img)),
        "anchor_color_drift": float(_rgb_mean_drift(img, anchor_img)),
    }


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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_examples": len(rows)}
    metric_names = [
        "retrieval_mse", "retrieval_pixel_l2", "retrieval_pixcorr", "retrieval_ssim", "retrieval_psnr", "retrieval_lpips", "retrieval_clip_image_similarity",
        "prior_mse", "prior_pixel_l2", "prior_pixcorr", "prior_ssim", "prior_psnr", "prior_lpips", "prior_clip_image_similarity",
        "diffusion_mse", "diffusion_pixel_l2", "diffusion_pixcorr", "diffusion_ssim", "diffusion_psnr", "diffusion_lpips", "diffusion_clip_image_similarity",
        "retrieval_anchor_similarity", "retrieval_shortlist_consistency_score",
        "prior_anchor_similarity", "prior_shortlist_consistency_score",
        "diffusion_anchor_similarity", "diffusion_shortlist_consistency_score",
        "diffusion_anchor_layout_ssim", "diffusion_anchor_edge_corr",
        "delta_mse", "delta_pixel_l2", "delta_pixcorr", "delta_ssim", "delta_psnr", "delta_lpips", "delta_clip_image_similarity",
    ]
    for name in metric_names:
        vals = [_safe_float(r.get(name)) for r in rows]
        vals = [v for v in vals if v is not None]
        out[f"{name}_mean"] = None if not vals else float(np.mean(vals))
        out[f"{name}_median"] = None if not vals else float(np.median(vals))
    return out


def save_example_outputs(
    output_dir: Path,
    bucket: str,
    row: dict[str, Any],
    gt: Image.Image,
    retrieval_only: Image.Image,
    prior_image: Image.Image,
    final_image: Image.Image,
    strip: Image.Image,
) -> Path:
    query = int(row["query_nsd_id"])
    recon_dir = output_dir / "reconstructions" / bucket
    fig_dir = output_dir / "figures" / "examples" / bucket
    recon_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    gt.save(recon_dir / f"query_{query}_ground_truth.png")
    retrieval_only.save(recon_dir / f"query_{query}_retrieval_top1.png")
    prior_image.save(recon_dir / f"query_{query}_diffusion_prior.png")
    final_image.save(recon_dir / f"query_{query}_diffusion_final.png")
    strip.save(recon_dir / f"query_{query}_topk_strip.png")

    tiles = [
        _tile(gt, "Ground Truth", [f"split={row['split']}", f"query={query}", f"bucket={bucket}"]),
        _tile(
            retrieval_only,
            "Retrieval Top-1",
            [
                f"retrieved={row['fused_top1_nsd_id']}",
                f"route={row['controller_route']}",
                f"anchor={row['anchor_nsd_id']} ({row['anchor_source']})",
                f"retrieval ssim={row['retrieval_ssim']:.4f}",
            ],
        ),
        _tile(
            prior_image,
            "Diffusion Prior",
            [
                f"anchor={row['anchor_nsd_id']}",
                f"topk={','.join(str(int(x)) for x in row['topk'])}",
                f"top1_w={row['controller_top1_weight']:.3f}",
                f"entropy={row['controller_entropy']:.3f}",
            ],
        ),
        _tile(
            final_image,
            "Final Diffusion",
            [
                f"kind={row['selected_kind']}",
                f"seed={row['selected_seed']}",
                f"sel={row['selected_selection_score']:.4f}",
                f"diff ssim={row['diffusion_ssim']:.4f}",
            ],
        ),
    ]
    strip_pad = 12
    panel = Image.new("RGB", (4 * tiles[0].width, tiles[0].height + strip.height + strip_pad), "white")
    for idx, tile in enumerate(tiles):
        panel.paste(tile, (idx * tile.width, 0))
    panel.paste(strip, (0, tiles[0].height + 8))
    panel_path = fig_dir / f"query_{query}_panel.png"
    panel.save(panel_path)
    return panel_path


def write_readme(output_dir: Path, dep: DependencyReport) -> None:
    text = f"""# V40 Confidence-Aware Retrieval-Guided Diffusion

This directory contains the qualitative V40 diffusion addon built on top of the frozen final retrieval system.

## Frozen Retrieval Backbone

- tri/main results dir: `experimental_results/V35_legacy_teacher_distill/subj01`
- legacy results dir: `experimental_results/N1v28a_dual_head/subj01`
- frozen tri-fusion config: `{EXPECTED_FROZEN_TRI_CONFIG}`
- frozen SHARED1000 R@1: `{EXPECTED_SHARED1000_R1:.1%}`

## What V40 Changes

- keeps the retrieval benchmark frozen
- routes each query through a confidence-aware controller
- preserves exact / near-exact retrieval cases with `identity_pass`
- uses an anchor-preserving latent prior instead of unconditional top-k latent averaging
- selects the final output using anchor similarity, shortlist consistency, and layout preservation only
- never uses ground truth for generation or candidate selection

## True Diffusion

- true diffusion run: `{dep.true_diffusion_run}`
- method: `score_weighted_anchor_residual_img2img`
- model: `{dep.model_id}`

## Main Outputs

- `selected_examples.json`, `selected_examples.csv`
- `metrics/per_example_metrics.csv`
- `metrics/summary_metrics.json`
- `metrics/summary_metrics.csv`
- `metrics/bucket_summary.csv`
- `figures/examples/`
- `figures/contact_sheets/`
- `reconstructions/`
"""
    (output_dir / "README.md").write_text(text)


def write_design_note(output_dir: Path) -> None:
    text = """# DIFFUSION ADDON DESIGN

## Audit of the Previous Addon

The previous true-diffusion addon had three main failure modes:

1. It applied one global diffusion recipe to every case, so even perfect retrievals were denoised and damaged.
2. It built the prior by directly averaging top-k latents, which washed out structure and encouraged blur / semantic averaging.
3. It used a soft semantic selector that was too permissive, so decorative or abstract samples could beat more faithful anchor-preserving outputs.

This made the old addon visually destructive and only weakly aligned with the frozen retrieval system.

## What V40 Reuses

- the frozen tri-expert retrieval system and its exact approved fusion recipe
- the frozen shortlist / top-k retrieval outputs
- the NSD stimulus image loading path
- the existing reconstruction metrics (MSE, PSNR, SSIM, pixel correlation, LPIPS, CLIP similarity)

## What V40 Replaces

- replaces naive top-k latent averaging with an anchor-first residual prior
- adds a confidence-aware controller with four routes:
  - identity_pass
  - low_strength_refine
  - guided_refine
  - exploratory_refine
- adds stronger candidate selection using:
  - anchor similarity
  - shortlist consistency
  - prior similarity
  - layout preservation
  - edge preservation
  - color-drift penalty
- always keeps the anchor and prior themselves as selectable candidates, so diffusion cannot force a worse output

## Why This Is More Scientifically Defensible

V40 is architecture-aligned because it treats the frozen tri-fusion shortlist as the semantic backbone and diffusion only as a refinement stage. It respects the retrieval evidence, preserves strong retrievals, and only broadens exploration when the shortlist itself is uncertain.

Ground truth is used for evaluation and reporting only. It is never used to build the prior, choose the route, or select the final candidate.

## Honest Limitations

- If retrieval is already exact, diffusion is usually unnecessary and often worse; V40 therefore explicitly allows pass-through behavior.
- The controller is still heuristic because it is inference-only and deliberately avoids new training.
- Hard cases remain limited by the semantic content of the frozen shortlist itself.
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
    parser.add_argument("--per-bucket-total", type=int, default=4)
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
    selected = select_examples(all_rows, per_bucket_total=args.per_bucket_total, per_split_target=max(1, args.per_bucket_total // 2))
    selected_clean = []
    for row in selected:
        clean = dict(row)
        clean.pop("_compact_top_embs", None)
        clean.pop("_legacy_top_embs", None)
        selected_clean.append(clean)
    _write_json(output_dir / "selected_examples.json", {"examples": selected_clean})
    _write_csv(output_dir / "selected_examples.csv", _jsonify_rows(selected_clean))

    store = StimulusStore(stimuli_hdf5)
    metric_rows: list[dict[str, Any]] = []
    panel_paths_by_bucket: dict[str, list[Path]] = {bucket: [] for bucket in BUCKETS}
    try:
        for idx, row in enumerate(selected):
            route_cfg = ROUTE_CONFIGS[row["controller_route"]]
            gt = store.get_image(int(row["gt_nsd_id"]))
            retrieval_only = store.get_image(int(row["fused_top1_nsd_id"]))
            topk_images = [store.get_image(int(nid)) for nid in row["topk"]]
            topk_weights = [float(x) for x in row["topk_weights"]]
            anchor_idx = int(row["anchor_index_within_topk"])
            anchor_image = topk_images[anchor_idx]
            support_images = [img for j, img in enumerate(topk_images) if j != anchor_idx]
            support_weights = [w for j, w in enumerate(topk_weights) if j != anchor_idx]

            prior_image, _ = build_anchor_residual_prior(
                pipe,
                anchor_image=anchor_image,
                support_images=support_images,
                support_weights=support_weights,
                route_cfg=route_cfg,
            )

            if route_cfg.name == "identity_pass":
                chosen = {"kind": "anchor_baseline", "seed": None, "image": anchor_image.copy()}
                candidate_scores = []
            else:
                generated = generate_candidates(
                    pipe,
                    init_image=prior_image,
                    route_cfg=route_cfg,
                    seed=args.seed + idx * 100,
                )
                chosen, candidate_scores = choose_candidate(
                    generated,
                    anchor_image=anchor_image,
                    prior_image=prior_image,
                    support_images=topk_images,
                    support_weights=topk_weights,
                    clip_ctx=clip_ctx,
                    route_cfg=route_cfg,
                )
            final_image = chosen["image"].copy()

            retrieval_metrics = compute_metrics(retrieval_only, gt, clip_ctx, lpips_eval)
            prior_metrics = compute_metrics(prior_image, gt, clip_ctx, lpips_eval)
            diffusion_metrics = compute_metrics(final_image, gt, clip_ctx, lpips_eval)

            retrieval_consistency = compute_consistency_metrics(
                retrieval_only,
                anchor_img=anchor_image,
                prior_img=prior_image,
                support_images=topk_images,
                support_weights=topk_weights,
                clip_ctx=clip_ctx,
            )
            prior_consistency = compute_consistency_metrics(
                prior_image,
                anchor_img=anchor_image,
                prior_img=prior_image,
                support_images=topk_images,
                support_weights=topk_weights,
                clip_ctx=clip_ctx,
            )
            diffusion_consistency = compute_consistency_metrics(
                final_image,
                anchor_img=anchor_image,
                prior_img=prior_image,
                support_images=topk_images,
                support_weights=topk_weights,
                clip_ctx=clip_ctx,
            )

            selected_score = None
            for cand in candidate_scores:
                if cand.get("kind") == chosen["kind"] and cand.get("seed") == chosen.get("seed"):
                    selected_score = cand.get("selection_score")
                    break
            if selected_score is None:
                selected_score = diffusion_consistency["shortlist_consistency_score"]

            metric_row = {
                **{k: v for k, v in row.items() if not k.startswith("_")},
                "selected_kind": chosen["kind"],
                "selected_seed": chosen.get("seed"),
                "selected_selection_score": float(selected_score),
                "candidate_scores": candidate_scores,
                "retrieval_mse": retrieval_metrics["mse"],
                "retrieval_pixel_l2": retrieval_metrics["pixel_l2"],
                "retrieval_pixcorr": retrieval_metrics["pixcorr"],
                "retrieval_ssim": retrieval_metrics["ssim"],
                "retrieval_psnr": retrieval_metrics["psnr"],
                "retrieval_lpips": retrieval_metrics["lpips"],
                "retrieval_clip_image_similarity": retrieval_metrics["clip_image_similarity"],
                "retrieval_anchor_similarity": retrieval_consistency["anchor_similarity"],
                "retrieval_prior_similarity": retrieval_consistency["prior_similarity"],
                "retrieval_shortlist_consistency_score": retrieval_consistency["shortlist_consistency_score"],
                "prior_mse": prior_metrics["mse"],
                "prior_pixel_l2": prior_metrics["pixel_l2"],
                "prior_pixcorr": prior_metrics["pixcorr"],
                "prior_ssim": prior_metrics["ssim"],
                "prior_psnr": prior_metrics["psnr"],
                "prior_lpips": prior_metrics["lpips"],
                "prior_clip_image_similarity": prior_metrics["clip_image_similarity"],
                "prior_anchor_similarity": prior_consistency["anchor_similarity"],
                "prior_prior_similarity": prior_consistency["prior_similarity"],
                "prior_shortlist_consistency_score": prior_consistency["shortlist_consistency_score"],
                "diffusion_mse": diffusion_metrics["mse"],
                "diffusion_pixel_l2": diffusion_metrics["pixel_l2"],
                "diffusion_pixcorr": diffusion_metrics["pixcorr"],
                "diffusion_ssim": diffusion_metrics["ssim"],
                "diffusion_psnr": diffusion_metrics["psnr"],
                "diffusion_lpips": diffusion_metrics["lpips"],
                "diffusion_clip_image_similarity": diffusion_metrics["clip_image_similarity"],
                "diffusion_anchor_similarity": diffusion_consistency["anchor_similarity"],
                "diffusion_prior_similarity": diffusion_consistency["prior_similarity"],
                "diffusion_shortlist_consistency_score": diffusion_consistency["shortlist_consistency_score"],
                "diffusion_anchor_layout_ssim": diffusion_consistency["anchor_layout_ssim"],
                "diffusion_anchor_edge_corr": diffusion_consistency["anchor_edge_corr"],
                "delta_mse": diffusion_metrics["mse"] - retrieval_metrics["mse"],
                "delta_pixel_l2": diffusion_metrics["pixel_l2"] - retrieval_metrics["pixel_l2"],
                "delta_pixcorr": diffusion_metrics["pixcorr"] - retrieval_metrics["pixcorr"],
                "delta_ssim": diffusion_metrics["ssim"] - retrieval_metrics["ssim"],
                "delta_psnr": diffusion_metrics["psnr"] - retrieval_metrics["psnr"],
                "delta_lpips": diffusion_metrics["lpips"] - retrieval_metrics["lpips"],
                "delta_clip_image_similarity": diffusion_metrics["clip_image_similarity"] - retrieval_metrics["clip_image_similarity"],
            }
            metric_rows.append(metric_row)
            strip = _topk_strip(store, [int(x) for x in row["topk"]], [float(x) for x in row["topk_weights"]])
            panel_path = save_example_outputs(
                output_dir=output_dir,
                bucket=row["bucket"],
                row=metric_row,
                gt=gt,
                retrieval_only=retrieval_only,
                prior_image=prior_image,
                final_image=final_image,
                strip=strip,
            )
            panel_paths_by_bucket[row["bucket"]].append(panel_path)
    finally:
        store.close()

    summary_payload = {
        "true_diffusion_run": True,
        "method": "score_weighted_anchor_residual_img2img",
        "model_id": args.model_id,
        "route_configs": {name: asdict(cfg) for name, cfg in ROUTE_CONFIGS.items()},
        "overall": summarize_rows(metric_rows),
        "by_bucket": {bucket: summarize_rows([r for r in metric_rows if r["bucket"] == bucket]) for bucket in BUCKETS},
    }

    _write_csv(output_dir / "metrics" / "per_example_metrics.csv", _jsonify_rows(metric_rows))
    _write_json(output_dir / "metrics" / "summary_metrics.json", summary_payload)
    _write_csv(output_dir / "metrics" / "summary_metrics.csv", _jsonify_rows([summary_payload["overall"]]))
    bucket_rows = []
    for bucket in BUCKETS:
        row = {"bucket": bucket}
        row.update(summary_payload["by_bucket"][bucket])
        bucket_rows.append(row)
    _write_csv(output_dir / "metrics" / "bucket_summary.csv", _jsonify_rows(bucket_rows))

    mixed = []
    for bucket in BUCKETS:
        _contact_sheet(panel_paths_by_bucket[bucket], output_dir / "figures" / "contact_sheets" / f"{bucket}_sheet.png")
        mixed.extend(panel_paths_by_bucket[bucket][:1])
    _contact_sheet(mixed, output_dir / "figures" / "contact_sheets" / "mixed_sheet.png", cols=2)

    write_readme(output_dir, dep)
    write_design_note(output_dir)

    logger.info("Saved V40 confidence-aware diffusion addon to %s", output_dir)
    logger.info("Frozen SHARED1000 R@1 remains %.1f%%", EXPECTED_SHARED1000_R1 * 100.0)
    logger.info("True diffusion run: yes")


if __name__ == "__main__":
    main()
