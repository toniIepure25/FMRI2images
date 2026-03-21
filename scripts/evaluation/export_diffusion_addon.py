#!/usr/bin/env python3
"""Export a small-scale retrieval-guided diffusion add-on package.

This is an inference-only qualitative add-on built on top of the frozen
best retrieval system. It never retrains the retrieval backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from export_final_qualitatives import StimulusStore, resolve_stimuli_hdf5
from final_best_system import (
    DEFAULT_LEGACY_RESULTS_DIR,
    DEFAULT_TRI_RESULTS_DIR,
    load_final_best_system_analysis,
)
from sweep_tri_fusion_retrieval import (
    _gt_rank_from_scores,
    _normalize_shortlist_scores,
    _rank_within_shortlist,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("final_outputs/diffusion_addon")
DEFAULT_MODEL_ID = "sd2-community/stable-diffusion-2-1"


@dataclass
class DependencyStatus:
    diffusion_available: bool
    diffusion_reason: str | None
    clip_available: bool
    clip_reason: str | None
    lpips_available: bool
    lpips_reason: str | None
    skimage_available: bool


@dataclass
class OptionalEvaluators:
    clip_model: Any | None
    clip_device: str | None
    clip_reason: str | None
    lpips_evaluator: Any | None
    lpips_reason: str | None


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


def detect_dependencies(model_id: str) -> DependencyStatus:
    diffusion_available = False
    diffusion_reason = None
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        diffusion_available = True
    except Exception as exc:
        diffusion_reason = f"{type(exc).__name__}: {exc}"

    clip_available = False
    clip_reason = None
    try:
        import open_clip  # noqa: F401
        clip_available = True
    except Exception as exc:
        clip_reason = f"{type(exc).__name__}: {exc}"

    lpips_available = False
    lpips_reason = None
    try:
        import lpips  # noqa: F401
        lpips_available = True
    except Exception as exc:
        lpips_reason = f"{type(exc).__name__}: {exc}"

    try:
        from skimage.metrics import structural_similarity  # noqa: F401
        from skimage.metrics import peak_signal_noise_ratio  # noqa: F401
        skimage_available = True
    except Exception:
        skimage_available = False

    return DependencyStatus(
        diffusion_available=diffusion_available,
        diffusion_reason=diffusion_reason,
        clip_available=clip_available,
        clip_reason=clip_reason,
        lpips_available=lpips_available,
        lpips_reason=lpips_reason,
        skimage_available=skimage_available,
    )


def _compute_split_selection_rows(system, analysis, split: str, topk: int) -> list[dict[str, Any]]:
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

    family = str(system.frozen_config["family"])
    normalization = str(system.frozen_config["normalization"])
    alpha = float(system.frozen_config["alpha"])
    beta = float(system.frozen_config["beta"])
    gamma = float(system.frozen_config["gamma"])

    if family == "normalized_weighted":
        compact_f = _normalize_shortlist_scores(compact_sl, normalization)
        legacy_f = _normalize_shortlist_scores(legacy_sl, normalization)
        rerank_f = _normalize_shortlist_scores(rerank_sl, normalization)
        fused_sl = alpha * compact_f + beta * rerank_f + gamma * legacy_f
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
        raise ValueError(f"Unsupported frozen fusion family: {family}")

    local_order = np.argsort(-fused_sl, axis=1)
    fused_shortlist = shortlist[row_idx, local_order]
    fused_scores_sorted = fused_sl[row_idx, local_order]
    fused_prob = np.exp(fused_scores_sorted - fused_scores_sorted.max(axis=1, keepdims=True))
    fused_prob = fused_prob / np.maximum(fused_prob.sum(axis=1, keepdims=True), 1e-8)

    rows: list[dict[str, Any]] = []
    for i, base_row in enumerate(analysis.per_query):
        local_ids = analysis.aligned["nsd_ids"][fused_shortlist[i, :topk]].astype(np.int32).tolist()
        local_scores = fused_scores_sorted[i, :topk].astype(np.float64).tolist()
        local_probs = fused_prob[i, :topk].astype(np.float64).tolist()
        margin_top1_top2 = None
        if fused_scores_sorted.shape[1] > 1:
            margin_top1_top2 = float(fused_scores_sorted[i, 0] - fused_scores_sorted[i, 1])

        rows.append(
            {
                "split": split,
                "row_index": int(base_row["row_index"]),
                "query_nsd_id": int(base_row["nsd_id"]),
                "gt_nsd_id": int(base_row["nsd_id"]),
                "fused_top1_nsd_id": int(base_row["fused_top1_nsd_id"]),
                "compact_top1_nsd_id": int(base_row["compact_top1_nsd_id"]),
                "legacy_top1_nsd_id": int(base_row["legacy_top1_nsd_id"]),
                "compact_gt_rank": int(compact_gt_rank[i]),
                "legacy_gt_rank": int(legacy_gt_rank[i]),
                "rerank_gt_rank": int(rerank_gt_rank[i]),
                "fused_gt_rank": int(base_row["fused_gt_rank"]),
                "fused_topk_nsd_ids": local_ids,
                "fused_topk_scores": local_scores,
                "fused_topk_probs": local_probs,
                "fused_margin_top1_top2": margin_top1_top2,
                "fused_fixed_case": bool(base_row["fused_fixed_case"]),
                "fused_fixed_to_top1": bool(base_row["fused_fixed_to_top1"]),
                "fused_top1_correct": bool(base_row["fused_top1_nsd_id"] == base_row["nsd_id"]),
                "selection_bucket": "",
                "selection_reason": "",
            }
        )
    return rows


def select_examples(
    val_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
    *,
    best_per_split: int,
    medium_per_split: int,
) -> list[dict[str, Any]]:
    def best_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pool = [r for r in rows if r["fused_top1_correct"]]
        pool.sort(
            key=lambda r: (
                1 if r["fused_fixed_to_top1"] else 0,
                r["fused_margin_top1_top2"] if r["fused_margin_top1_top2"] is not None else -1e9,
                -r["compact_gt_rank"],
                -r["legacy_gt_rank"],
            ),
            reverse=True,
        )
        return pool

    def medium_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pool = [r for r in rows if 2 <= r["fused_gt_rank"] <= 5]
        pool.sort(
            key=lambda r: (
                -r["fused_gt_rank"],
                r["fused_margin_top1_top2"] if r["fused_margin_top1_top2"] is not None else -1e9,
            ),
            reverse=True,
        )
        return pool

    selected: list[dict[str, Any]] = []
    for rows in [val_rows, shared_rows]:
        chosen_best = best_pool(rows)[:best_per_split]
        for row in chosen_best:
            row = dict(row)
            row["selection_bucket"] = "best"
            row["selection_reason"] = "fused_top1_correct_high_confidence"
            selected.append(row)

        chosen_medium = medium_pool(rows)[:medium_per_split]
        for row in chosen_medium:
            row = dict(row)
            row["selection_bucket"] = "medium"
            row["selection_reason"] = "fused_gt_rank_2_to_5_for_contrast"
            selected.append(row)

    return selected


def _resize_like(img: Image.Image, ref: Image.Image) -> Image.Image:
    return img.convert("RGB").resize(ref.size, Image.Resampling.LANCZOS)


def _resolve_metric_device(requested: str) -> str:
    try:
        import torch
        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
    except Exception:
        pass
    return "cpu"


def load_optional_evaluators(dep: DependencyStatus, device: str) -> OptionalEvaluators:
    clip_model = None
    clip_device = None
    clip_reason = dep.clip_reason
    lpips_evaluator = None
    lpips_reason = dep.lpips_reason

    metric_device = _resolve_metric_device(device)

    if dep.clip_available:
        try:
            import open_clip
            clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            clip_model = clip_model.to(metric_device)
            clip_model.eval()
            clip_device = metric_device
            clip_reason = None
        except Exception as exc:
            clip_model = None
            clip_device = None
            clip_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("CLIP metric unavailable; continuing without it: %s", clip_reason)

    if dep.lpips_available:
        try:
            from fmri2img.eval.recon_eval import LPIPSEvaluator
            lpips_evaluator = LPIPSEvaluator(device=metric_device)
            lpips_reason = None
        except Exception as exc:
            lpips_evaluator = None
            lpips_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("LPIPS metric unavailable; continuing without it: %s", lpips_reason)

    return OptionalEvaluators(
        clip_model=clip_model,
        clip_device=clip_device,
        clip_reason=clip_reason,
        lpips_evaluator=lpips_evaluator,
        lpips_reason=lpips_reason,
    )


def build_weighted_prototype(store: StimulusStore, nsd_ids: list[int], weights: list[float], size: tuple[int, int]) -> Image.Image:
    arr = np.zeros((size[1], size[0], 3), dtype=np.float32)
    denom = max(sum(weights), 1e-8)
    ref = Image.new("RGB", size)
    for nid, weight in zip(nsd_ids, weights):
        img = _resize_like(store.get_image(int(nid)), ref)
        arr += weight * np.asarray(img, dtype=np.float32)
    arr /= denom
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def heuristic_refinement(top1: Image.Image, prototype: Image.Image) -> Image.Image:
    top1 = top1.convert("RGB")
    prototype = _resize_like(prototype, top1)
    blended = Image.blend(top1, prototype, alpha=0.35)
    blended = ImageOps.autocontrast(blended)
    blended = blended.filter(ImageFilter.UnsharpMask(radius=1.6, percent=110, threshold=3))
    return blended


def maybe_load_diffusion_img2img(model_id: str, device: str, allow_download: bool) -> tuple[Any | None, str | None]:
    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
    except Exception as exc:
        return None, f"diffusion dependencies unavailable: {type(exc).__name__}: {exc}"

    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=not allow_download,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe, None
    except Exception as exc:
        return None, f"failed to load img2img pipeline: {type(exc).__name__}: {exc}"


def diffusion_refinement(
    pipe: Any,
    init_image: Image.Image,
    *,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> Image.Image:
    import torch

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    return image.convert("RGB")


def _to_float01(img: Image.Image, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    arr = np.asarray(img.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    return arr / 255.0


def compute_basic_metrics(
    pred: Image.Image,
    gt: Image.Image,
    *,
    dependency_status: DependencyStatus,
    evaluators: OptionalEvaluators,
) -> dict[str, Any]:
    pred_arr = _to_float01(pred)
    gt_arr = _to_float01(gt)
    mse = float(np.mean((pred_arr - gt_arr) ** 2))
    pixel_l2 = float(np.sqrt(np.sum((pred_arr - gt_arr) ** 2)))

    pixcorr_value = None
    try:
        from fmri2img.eval.recon_eval import compute_pixcorr
        pixcorr_value = float(compute_pixcorr(pred_arr, gt_arr))
    except Exception:
        pixcorr_value = None

    ssim_value = None
    psnr_value = None
    if dependency_status.skimage_available:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        ssim_value = float(structural_similarity(gt_arr, pred_arr, data_range=1.0, channel_axis=2))
        psnr_value = float(peak_signal_noise_ratio(gt_arr, pred_arr, data_range=1.0))

    clip_similarity = None
    if evaluators.clip_model is not None and evaluators.clip_device is not None:
        try:
            from fmri2img.eval.image_metrics import clip_score
            clip_similarity = float(clip_score(pred, gt, evaluators.clip_model, device=evaluators.clip_device))
        except Exception as exc:
            logger.warning("CLIP image similarity failed for one example: %s", exc)
            clip_similarity = None

    lpips_value = None
    if evaluators.lpips_evaluator is not None:
        try:
            lpips_value = float(evaluators.lpips_evaluator.compute(pred_arr, gt_arr))
        except Exception as exc:
            logger.warning("LPIPS failed for one example: %s", exc)
            lpips_value = None

    return {
        "mse": mse,
        "pixel_l2": pixel_l2,
        "pixcorr": pixcorr_value,
        "ssim": ssim_value,
        "psnr": psnr_value,
        "clip_image_similarity_to_gt": clip_similarity,
        "lpips": lpips_value,
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_examples": len(rows)}
    metric_names = [
        "retrieval_mse",
        "retrieval_pixel_l2",
        "retrieval_pixcorr",
        "retrieval_ssim",
        "retrieval_psnr",
        "retrieval_clip_image_similarity_to_gt",
        "retrieval_lpips",
        "addon_mse",
        "addon_pixel_l2",
        "addon_pixcorr",
        "addon_ssim",
        "addon_psnr",
        "addon_clip_image_similarity_to_gt",
        "addon_lpips",
        "delta_mse",
        "delta_pixel_l2",
        "delta_pixcorr",
        "delta_ssim",
        "delta_psnr",
        "delta_clip_image_similarity_to_gt",
        "delta_lpips",
    ]
    for name in metric_names:
        values = [
            float(r[name])
            for r in rows
            if r.get(name) is not None and not math.isnan(float(r[name])) and not math.isinf(float(r[name]))
        ]
        if values:
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_median"] = float(np.median(values))
        else:
            summary[f"{name}_mean"] = None
            summary[f"{name}_median"] = None
    return summary


def _draw_lines(draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[str]) -> None:
    offset = 0
    for line in lines:
        draw.text((x, y + offset), line, fill=(20, 20, 20))
        offset += 14


def _metric_tile(img: Image.Image, title: str, lines: list[str], tile_size: int = 256, footer_h: int = 96) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size + footer_h), "white")
    thumb = ImageOps.contain(img.convert("RGB"), (tile_size, tile_size))
    canvas.paste(thumb, ((tile_size - thumb.width) // 2, (tile_size - thumb.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, tile_size, tile_size, tile_size + footer_h), fill=(248, 248, 248))
    draw.text((8, tile_size + 6), title, fill=(0, 0, 0))
    _draw_lines(draw, 8, tile_size + 24, lines[:5])
    return canvas


def _topk_strip(store: StimulusStore, nsd_ids: list[int], tile_size: int = 96) -> Image.Image:
    width = max(1, len(nsd_ids)) * tile_size
    strip = Image.new("RGB", (width, tile_size + 18), "white")
    for i, nid in enumerate(nsd_ids):
        img = store.get_image(int(nid)).convert("RGB")
        thumb = ImageOps.contain(img, (tile_size, tile_size))
        tile = Image.new("RGB", (tile_size, tile_size + 18), "white")
        tile.paste(thumb, ((tile_size - thumb.width) // 2, (tile_size - thumb.height) // 2))
        ImageDraw.Draw(tile).text((4, tile_size + 2), str(int(nid)), fill=(0, 0, 0))
        strip.paste(tile, (i * tile_size, 0))
    return strip


def _save_example_figure(
    row: dict[str, Any],
    gt: Image.Image,
    retrieval_only: Image.Image,
    addon_img: Image.Image,
    store: StimulusStore,
    output_path: Path,
) -> None:
    strip = _topk_strip(store, [int(x) for x in row["fused_topk_nsd_ids"][:5]])
    tiles = [
        _metric_tile(gt, "Ground Truth", [f"split={row['split']}", f"query={row['query_nsd_id']}", f"bucket={row['selection_bucket']}"]),
        _metric_tile(
            retrieval_only,
            "Retrieval-Only",
            [
                f"top1={row['fused_top1_nsd_id']}",
                f"gt_rank={row['fused_gt_rank']}",
                f"ssim={row['retrieval_ssim']:.4f}" if row["retrieval_ssim"] is not None else "ssim=n/a",
            ],
        ),
        _metric_tile(
            addon_img,
            row["addon_label"],
            [
                f"method={row['addon_method']}",
                f"mse={row['addon_mse']:.5f}",
                f"ssim={row['addon_ssim']:.4f}" if row["addon_ssim"] is not None else "ssim=n/a",
            ],
        ),
    ]
    tile_w = tiles[0].width
    tile_h = tiles[0].height
    panel = Image.new("RGB", (tile_w * 3, tile_h + strip.height + 20), "white")
    for i, tile in enumerate(tiles):
        panel.paste(tile, (i * tile_w, 0))
    panel.paste(strip, (0, tile_h + 8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def _contact_sheet(panel_paths: list[Path], out_path: Path, cols: int = 4) -> None:
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


def _write_readme(output_dir: Path, method_used: str, dep: DependencyStatus, evaluators: OptionalEvaluators, selected_count: int) -> None:
    text = f"""# Diffusion Add-on

This directory contains a small-scale qualitative reconstruction add-on built on top of the frozen best retrieval system.

## Frozen Retrieval Backbone

- main results dir: `experimental_results/V35_legacy_teacher_distill/subj01`
- legacy results dir: `experimental_results/N1v28a_dual_head/subj01`
- frozen tri-fusion:
  - compact score: `csls`
  - legacy score: `csls`
  - family: `normalized_weighted`
  - normalization: `zscore`
  - shortlist_k: `150`
  - alpha / beta / gamma: `0.3 / 0.0 / 0.7`
- frozen result:
  - VAL R@1 = `77.6%`
  - SHARED1000 R@1 = `77.2%`

## Scope

This add-on is **not** a new benchmark system and does **not** retrain the retrieval backbone.
It is a small-scale reconstruction appendix for thesis/presentation use on `{selected_count}` selected examples.

## Selection Policy

- prefer high-confidence fused examples first
- include a small number of medium examples for contrast
- save selection tables in:
  - `selected_examples.json`
  - `selected_examples.csv`

## Reconstruction Path Used

- requested mode: retrieval-guided diffusion / refinement
- actual method used: `{method_used}`

Dependency availability at export time:

- diffusion available: `{dep.diffusion_available}` ({dep.diffusion_reason or 'ok'})
- CLIP image encoder available: `{dep.clip_available}` ({evaluators.clip_reason or 'ok'})
- LPIPS available: `{dep.lpips_available}` ({evaluators.lpips_reason or 'ok'})
- scikit-image available: `{dep.skimage_available}`

If diffusion dependencies are missing, the script falls back to a clearly labeled
heuristic retrieval-guided refinement:

1. build a weighted prototype from the fused top-k retrieved images
2. blend prototype with the fused top-1 retrieval
3. apply deterministic contrast + unsharp refinement

This fallback is a qualitative retrieval-guided enhancement, **not a true diffusion model**.

## Metrics

Saved under `metrics/`:

- `diffusion_metrics_per_example.csv`
- `diffusion_metrics_summary.json`
- `diffusion_metrics_summary.csv`

These are **small-scale qualitative metrics**, not the main benchmark.

## Figures

- `figures/examples/`
- `figures/contact_sheets/`

Each example panel includes:

1. ground truth
2. retrieval-only reconstruction
3. diffusion-guided / fallback refinement
4. top-5 retrieval strip

## Regeneration

```bash
PYTHONPATH=src /opt/conda/bin/python3 scripts/evaluation/export_diffusion_addon.py   --tri-results-dir experimental_results/V35_legacy_teacher_distill/subj01   --legacy-results-dir experimental_results/N1v28a_dual_head/subj01   --output-dir final_outputs/diffusion_addon   --stimuli-hdf5 /path/to/nsd_stimuli.hdf5
```
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tri-results-dir", type=Path, default=DEFAULT_TRI_RESULTS_DIR)
    parser.add_argument("--legacy-results-dir", type=Path, default=DEFAULT_LEGACY_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stimuli-hdf5", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--mode", choices=["auto", "diffusion", "heuristic"], default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--best-per-split", type=int, default=6)
    parser.add_argument("--medium-per-split", type=int, default=2)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-config-drift", action="store_true")
    args = parser.parse_args()

    system, val_analysis, shared_analysis = load_final_best_system_analysis(
        tri_results_dir=args.tri_results_dir,
        legacy_results_dir=args.legacy_results_dir,
        strict=not args.allow_config_drift,
    )
    dep = detect_dependencies(args.model_id)
    evaluators = load_optional_evaluators(dep, args.device)
    stimuli_hdf5 = resolve_stimuli_hdf5(args.stimuli_hdf5)
    logger.info("Using stimuli HDF5: %s", stimuli_hdf5)

    val_rows = _compute_split_selection_rows(system, val_analysis, "val", args.topk)
    shared_rows = _compute_split_selection_rows(system, shared_analysis, "shared1000", args.topk)
    selected = select_examples(
        val_rows,
        shared_rows,
        best_per_split=args.best_per_split,
        medium_per_split=args.medium_per_split,
    )
    if not selected:
        raise RuntimeError("No examples were selected for the diffusion add-on.")

    output_dir = args.output_dir
    figures_examples_dir = output_dir / "figures" / "examples"
    figures_contact_dir = output_dir / "figures" / "contact_sheets"
    metrics_dir = output_dir / "metrics"
    for path in [figures_examples_dir, figures_contact_dir, metrics_dir]:
        path.mkdir(parents=True, exist_ok=True)

    selected_json_rows = []
    selected_csv_rows = []

    diffusion_pipe = None
    method_used = "heuristic_retrieval_guided_refinement"
    if args.mode in {"auto", "diffusion"}:
        diffusion_pipe, diffusion_error = maybe_load_diffusion_img2img(
            model_id=args.model_id,
            device=args.device,
            allow_download=args.allow_download,
        )
        if diffusion_pipe is not None:
            method_used = "stable_diffusion_img2img"
            logger.info("Loaded diffusion img2img pipeline for add-on export.")
        elif args.mode == "diffusion":
            raise RuntimeError(f"Requested diffusion mode, but no stable pipeline was available: {diffusion_error}")
        else:
            logger.warning("Falling back to heuristic refinement: %s", diffusion_error)

    metric_rows: list[dict[str, Any]] = []
    panel_paths_by_bucket: dict[str, list[Path]] = {"best": [], "medium": []}

    store = StimulusStore(stimuli_hdf5)
    try:
        for idx, row in enumerate(selected):
            gt = store.get_image(int(row["gt_nsd_id"]))
            retrieval_only = store.get_image(int(row["fused_top1_nsd_id"]))
            topk_ids = [int(x) for x in row["fused_topk_nsd_ids"]]
            topk_probs = [float(x) for x in row["fused_topk_probs"]]
            prototype = build_weighted_prototype(store, topk_ids, topk_probs, gt.size)

            if diffusion_pipe is not None:
                addon_img = diffusion_refinement(
                    diffusion_pipe,
                    prototype,
                    prompt="a natural high-quality photograph",
                    negative_prompt="blurry, low quality, artifacts, text, watermark",
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.steps,
                    seed=args.seed + idx,
                )
                addon_method = "stable_diffusion_img2img"
                addon_label = "Diffusion-Guided"
            else:
                addon_img = heuristic_refinement(retrieval_only, prototype)
                addon_method = "heuristic_retrieval_guided_refinement"
                addon_label = "Heuristic Refinement"

            retrieval_metrics = compute_basic_metrics(
                retrieval_only,
                gt,
                dependency_status=dep,
                evaluators=evaluators,
            )
            addon_metrics = compute_basic_metrics(
                addon_img,
                gt,
                dependency_status=dep,
                evaluators=evaluators,
            )

            metric_row = {
                "split": row["split"],
                "selection_bucket": row["selection_bucket"],
                "query_nsd_id": row["query_nsd_id"],
                "gt_nsd_id": row["gt_nsd_id"],
                "fused_top1_nsd_id": row["fused_top1_nsd_id"],
                "addon_method": addon_method,
                "retrieval_mse": retrieval_metrics["mse"],
                "retrieval_pixel_l2": retrieval_metrics["pixel_l2"],
                "retrieval_pixcorr": retrieval_metrics["pixcorr"],
                "retrieval_ssim": retrieval_metrics["ssim"],
                "retrieval_psnr": retrieval_metrics["psnr"],
                "addon_mse": addon_metrics["mse"],
                "addon_pixel_l2": addon_metrics["pixel_l2"],
                "addon_pixcorr": addon_metrics["pixcorr"],
                "addon_ssim": addon_metrics["ssim"],
                "addon_psnr": addon_metrics["psnr"],
                "delta_mse": None if addon_metrics["mse"] is None else addon_metrics["mse"] - retrieval_metrics["mse"],
                "delta_pixel_l2": None if addon_metrics["pixel_l2"] is None else addon_metrics["pixel_l2"] - retrieval_metrics["pixel_l2"],
                "delta_pixcorr": None if addon_metrics["pixcorr"] is None or retrieval_metrics["pixcorr"] is None else addon_metrics["pixcorr"] - retrieval_metrics["pixcorr"],
                "delta_ssim": None if addon_metrics["ssim"] is None or retrieval_metrics["ssim"] is None else addon_metrics["ssim"] - retrieval_metrics["ssim"],
                "delta_psnr": None if addon_metrics["psnr"] is None or retrieval_metrics["psnr"] is None else addon_metrics["psnr"] - retrieval_metrics["psnr"],
                "retrieval_clip_image_similarity_to_gt": retrieval_metrics["clip_image_similarity_to_gt"],
                "addon_clip_image_similarity_to_gt": addon_metrics["clip_image_similarity_to_gt"],
                "delta_clip_image_similarity_to_gt": None if addon_metrics["clip_image_similarity_to_gt"] is None or retrieval_metrics["clip_image_similarity_to_gt"] is None else addon_metrics["clip_image_similarity_to_gt"] - retrieval_metrics["clip_image_similarity_to_gt"],
                "retrieval_lpips": retrieval_metrics["lpips"],
                "addon_lpips": addon_metrics["lpips"],
                "delta_lpips": None if addon_metrics["lpips"] is None or retrieval_metrics["lpips"] is None else addon_metrics["lpips"] - retrieval_metrics["lpips"],
            }
            row.update(metric_row)
            row["addon_label"] = addon_label
            row["addon_method"] = addon_method
            metric_rows.append(metric_row)

            panel_path = figures_examples_dir / row["split"] / row["selection_bucket"] / f"query_{int(row['query_nsd_id'])}.png"
            _save_example_figure(row, gt, retrieval_only, addon_img, store, panel_path)
            panel_paths_by_bucket.setdefault(row["selection_bucket"], []).append(panel_path)

            selected_json_rows.append(row)
            csv_row = dict(row)
            csv_row["fused_topk_nsd_ids"] = json.dumps(csv_row["fused_topk_nsd_ids"])
            csv_row["fused_topk_scores"] = json.dumps(csv_row["fused_topk_scores"])
            csv_row["fused_topk_probs"] = json.dumps(csv_row["fused_topk_probs"])
            selected_csv_rows.append(csv_row)

        _contact_sheet(panel_paths_by_bucket.get("best", []), figures_contact_dir / "best_examples.png", cols=3)
        mixed = panel_paths_by_bucket.get("best", [])[:4] + panel_paths_by_bucket.get("medium", [])[:4]
        _contact_sheet(mixed, figures_contact_dir / "mixed_examples.png", cols=3)
        _contact_sheet(panel_paths_by_bucket.get("medium", []), figures_contact_dir / "medium_examples.png", cols=3)
    finally:
        store.close()

    selected_payload = {
        "system": {
            "tri_results_dir": str(system.tri_results_dir),
            "legacy_results_dir": str(system.legacy_results_dir),
            "frozen_tri_fusion_config": system.frozen_config,
            "saved_shared1000_r@1": system.frozen_shared1000_r1,
        },
        "method_used": method_used,
        "dependency_status": {
            "diffusion_available": dep.diffusion_available,
            "diffusion_reason": dep.diffusion_reason,
            "clip_available": dep.clip_available,
            "clip_reason": evaluators.clip_reason,
            "lpips_available": dep.lpips_available,
            "lpips_reason": evaluators.lpips_reason,
            "skimage_available": dep.skimage_available,
        },
        "selected_examples": selected_json_rows,
    }

    _write_json(output_dir / "selected_examples.json", selected_payload)
    _write_csv(output_dir / "selected_examples.csv", selected_csv_rows)

    summary = {
        "method_used": method_used,
        "n_examples": len(metric_rows),
        "notes": [
            "Small-scale qualitative metrics only; not the main benchmark.",
            "Core retrieval system remained frozen.",
            "If diffusion dependencies were unavailable, heuristic retrieval-guided refinement was used instead.",
        ],
        "dependency_status": {
            "diffusion_available": dep.diffusion_available,
            "diffusion_reason": dep.diffusion_reason,
            "clip_available": dep.clip_available,
            "clip_reason": evaluators.clip_reason,
            "lpips_available": dep.lpips_available,
            "lpips_reason": evaluators.lpips_reason,
            "skimage_available": dep.skimage_available,
        },
        "overall": summarize_metric_rows(metric_rows),
        "best_subset": summarize_metric_rows([r for r in metric_rows if r["selection_bucket"] == "best"]),
        "medium_subset": summarize_metric_rows([r for r in metric_rows if r["selection_bucket"] == "medium"]),
    }
    _write_json(metrics_dir / "diffusion_metrics_summary.json", summary)
    _write_csv(metrics_dir / "diffusion_metrics_per_example.csv", metric_rows)
    summary_rows = []
    for subset_name, subset_payload in [
        ("overall", summary["overall"]),
        ("best_subset", summary["best_subset"]),
        ("medium_subset", summary["medium_subset"]),
    ]:
        row = {"subset": subset_name}
        row.update(subset_payload)
        summary_rows.append(row)
    _write_csv(metrics_dir / "diffusion_metrics_summary.csv", summary_rows)
    _write_readme(output_dir, method_used, dep, evaluators, len(metric_rows))

    logger.info("Saved diffusion add-on bundle to %s", output_dir)
    logger.info("Method used: %s", method_used)
    logger.info("Selected %d examples", len(metric_rows))


if __name__ == "__main__":
    main()
