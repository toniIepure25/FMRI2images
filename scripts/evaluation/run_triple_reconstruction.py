#!/usr/bin/env python3
"""Triple-fusion-anchored reconstruction: swap V35+N1v28a anchors with triple-fusion top-1.

Reads the original V40 selected_examples.csv for route distribution reference,
replaces all val-split examples with shared1000-only examples, then runs SD 2.1
img2img with the triple-fusion top-1 as anchor image.

Does NOT touch V35, N1v28a, or V40 controller code. Uses the same
StableDiffusionImg2ImgPipeline + DPMSolverMultistepScheduler as V40.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUTE_PARAMS = {
    "identity_pass": {"strength": 0.0, "steps": 0, "guidance_scale": 0.0},
    "low_strength_refine": {"strength": 0.05, "steps": 20, "guidance_scale": 3.5},
    "guided_refine": {"strength": 0.14, "steps": 30, "guidance_scale": 4.5},
    "exploratory_refine": {"strength": 0.24, "steps": 40, "guidance_scale": 5.5},
}

BUCKETS = ["perfect", "good", "near_good", "hard"]
IMG_SIZE = (256, 256)


def bucket_from_rank(rank: int) -> str:
    if rank == 1:
        return "perfect"
    elif rank <= 5:
        return "good"
    elif rank <= 15:
        return "near_good"
    return "hard"


def route_for_bucket(bucket: str) -> str:
    return {
        "perfect": "identity_pass",
        "good": "low_strength_refine",
        "near_good": "guided_refine",
        "hard": "exploratory_refine",
    }[bucket]


# ---------------------------------------------------------------------------
# Triple fusion computation (reuses logic from triple_fusion_shared1000_fixed.py)
# ---------------------------------------------------------------------------

def compute_sim_matrix_gpu(preds: np.ndarray, gts: np.ndarray, device: str) -> np.ndarray:
    p = torch.from_numpy(preds).to(device).float()
    g = torch.from_numpy(gts).to(device).float()
    p = p / p.norm(dim=1, keepdim=True).clamp(min=1e-8)
    g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (p @ g.T).cpu().numpy()


def compute_csls(sims: np.ndarray, k: int = 3) -> np.ndarray:
    topk_p = np.partition(-sims, k, axis=1)[:, :k]
    hub_s = -topk_p.mean(axis=1)
    topk_g = np.partition(-(sims.T), k, axis=1)[:, :k]
    hub_t = -topk_g.mean(axis=1)
    return sims - hub_s[:, None] / 2 - hub_t[None, :] / 2


def zscore_normalize(sims: np.ndarray) -> np.ndarray:
    mu, std = sims.mean(), sims.std()
    return (sims - mu) / max(std, 1e-8)


def compute_triple_fusion(repo_root: str, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (csls_fused_1000x1000, gt_ranks_1000, shared1000_nsd_ids_1000)."""
    er = os.path.join(repo_root, "experimental_results")
    streams = {
        "V61a": (
            os.path.join(er, "V61a_finetune_difflr", "subj01", "metrics", "shared1000_predictions_mctta16.npy"),
            os.path.join(er, "V61a_finetune_difflr", "subj01", "metrics", "shared1000_ground_truth.npy"),
            0.7,
        ),
        "V62a": (
            os.path.join(er, "V62a_cls_retrieval_768d", "subj01", "metrics", "shared1000_predictions.npy"),
            os.path.join(er, "V62a_cls_retrieval_768d", "subj01", "metrics", "shared1000_ground_truth.npy"),
            0.1,
        ),
        "V66a": (
            os.path.join(er, "V66a_roi_pretrain", "subj01", "metrics", "shared1000_predictions.npy"),
            os.path.join(er, "V66a_roi_pretrain", "subj01", "metrics", "shared1000_ground_truth.npy"),
            0.2,
        ),
    }
    fused = None
    for name, (pred_p, gt_p, w) in streams.items():
        preds = np.load(pred_p)
        gts = np.load(gt_p)
        raw = compute_sim_matrix_gpu(preds, gts, device)
        z = zscore_normalize(raw)
        fused = z * w if fused is None else fused + z * w
        logger.info(f"  {name}: preds {preds.shape}, weight {w}")

    csls = compute_csls(fused, k=3)
    n = csls.shape[0]
    diag = np.array([csls[i, i] for i in range(n)])
    gt_ranks = np.array([(csls[i] > diag[i]).sum() + 1 for i in range(n)])
    top1_ids = np.argmax(csls, axis=1).astype(np.int64)
    r1 = float((gt_ranks == 1).mean())
    logger.info(f"  Triple fusion CSLS R@1 = {r1:.4f} ({(gt_ranks == 1).sum()}/{n})")

    nsd_ids_path = os.path.join(
        er, "V61a_finetune_difflr", "subj01", "metrics", "shared1000_nsd_ids.npy"
    )
    nsd_ids = np.load(nsd_ids_path)
    return csls, gt_ranks, nsd_ids


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------

def select_16_shared1000(gt_ranks: np.ndarray, nsd_ids: np.ndarray, seed: int = 42) -> list[dict]:
    """Select 4 examples per bucket, all from shared1000."""
    rng = np.random.RandomState(seed)
    pool_by_bucket: dict[str, list[int]] = {b: [] for b in BUCKETS}
    for i in range(len(gt_ranks)):
        b = bucket_from_rank(int(gt_ranks[i]))
        pool_by_bucket[b].append(i)

    for b in BUCKETS:
        logger.info(f"  Bucket {b}: {len(pool_by_bucket[b])} candidates")

    selected = []
    for bucket in BUCKETS:
        pool = pool_by_bucket[bucket]
        if len(pool) < 4:
            raise RuntimeError(f"Only {len(pool)} shared1000 examples in bucket '{bucket}', need 4")
        indices = sorted(rng.choice(pool, size=4, replace=False))
        route = route_for_bucket(bucket)
        for row_idx in indices:
            selected.append({
                "split": "shared1000",
                "row_index": int(row_idx),
                "query_nsd_id": int(nsd_ids[row_idx]),
                "gt_rank": int(gt_ranks[row_idx]),
                "bucket": bucket,
                "controller_route": route,
            })
    return selected


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

class StimulusStore:
    def __init__(self, hdf5_path: str):
        self._path = hdf5_path
        self._f = h5py.File(hdf5_path, "r")
        self._dset = self._f["imgBrick"]

    def get_image(self, nsd_id: int) -> Image.Image:
        arr = self._dset[nsd_id]
        return Image.fromarray(arr).convert("RGB")

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# SD img2img
# ---------------------------------------------------------------------------

def load_sd_pipeline(model_id: str, device_str: str, cache_dir: str | None = None):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

    dtype = torch.float16 if device_str.startswith("cuda") else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device_str)
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass
    return pipe


def run_img2img(pipe, anchor_img: Image.Image, route: str, seed: int) -> Image.Image:
    params = ROUTE_PARAMS[route]
    if route == "identity_pass" or params["strength"] <= 0.0:
        return anchor_img.copy()

    init = anchor_img.resize((768, 768), Image.LANCZOS)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt="",
        negative_prompt="text, watermark, blurry, low quality, distorted, oversaturated, painting, illustration",
        image=init,
        strength=params["strength"],
        guidance_scale=params["guidance_scale"],
        num_inference_steps=params["steps"],
        generator=generator,
    )
    return result.images[0].convert("RGB")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def to_np01(img: Image.Image, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    return np.asarray(ImageOps.contain(img.convert("RGB"), size), dtype=np.float32) / 255.0


def compute_all_metrics(
    pred: Image.Image, gt: Image.Image, clip_model, clip_preprocess, device: str
) -> dict[str, float]:
    from fmri2img.eval.recon_eval import compute_pixcorr, compute_psnr, compute_ssim
    from fmri2img.eval.image_metrics import clip_score

    pred_arr = to_np01(pred)
    gt_arr = to_np01(gt)
    pixcorr = float(compute_pixcorr(pred_arr, gt_arr))
    ssim_val = float(compute_ssim(pred_arr, gt_arr))
    psnr_val = float(compute_psnr(pred_arr, gt_arr))

    import lpips as lpips_mod
    lpips_net = lpips_mod.LPIPS(net="alex").to(device)
    pred_t = torch.from_numpy(pred_arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    gt_t = torch.from_numpy(gt_arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        lpips_val = float(lpips_net(pred_t, gt_t).item())

    clip_sim = float(clip_score(pred, gt, clip_model, device=device))

    return {
        "pixcorr": pixcorr,
        "ssim": ssim_val,
        "psnr": psnr_val,
        "lpips": lpips_val,
        "clip_image_similarity": clip_sim,
    }


# ---------------------------------------------------------------------------
# Composite image generation
# ---------------------------------------------------------------------------

def make_composite(
    rows: list[dict], store: StimulusStore, output_path: str, tile_size: int = 200
):
    """4 rows (buckets) x N columns: GT | Anchor | Diffusion per example."""
    cols_per_bucket = 4
    n_img_cols = 3
    total_cols = cols_per_bucket * n_img_cols
    label_w = 100
    gap = 4
    header_h = 30

    w = label_w + total_cols * tile_size + (total_cols - 1) * gap + 20
    h = header_h + len(BUCKETS) * (tile_size + gap + 20) + 20
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    col_labels = []
    for i in range(cols_per_bucket):
        col_labels.extend(["GT", "Anchor", "Diffusion"])
    for ci, lbl in enumerate(col_labels):
        x = label_w + ci * (tile_size + gap) + tile_size // 2
        draw.text((x, 5), lbl, fill="black", font=font_sm, anchor="mt")

    by_bucket = {b: [] for b in BUCKETS}
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    for bi, bucket in enumerate(BUCKETS):
        y0 = header_h + bi * (tile_size + gap + 20)
        draw.text((5, y0 + tile_size // 2), bucket.replace("_", " ").title(), fill="black", font=font, anchor="lm")
        examples = by_bucket[bucket][:cols_per_bucket]
        for ei, ex in enumerate(examples):
            gt_img = store.get_image(ex["query_nsd_id"]).resize((tile_size, tile_size), Image.LANCZOS)
            anchor_img = ex["_anchor_img"].resize((tile_size, tile_size), Image.LANCZOS)
            diff_img = ex["_diffusion_img"].resize((tile_size, tile_size), Image.LANCZOS)
            base_col = ei * n_img_cols
            for ci, img in enumerate([gt_img, anchor_img, diff_img]):
                x = label_w + (base_col + ci) * (tile_size + gap)
                canvas.paste(img, (x, y0))

    canvas.save(output_path, quality=95)
    logger.info(f"Composite saved to {output_path}")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_metrics(rows: list[dict]) -> dict[str, Any]:
    metric_keys = ["pixcorr", "ssim", "psnr", "lpips", "clip_image_similarity"]
    overall: dict[str, Any] = {"n_examples": len(rows)}
    for prefix in ["retrieval", "diffusion"]:
        for k in metric_keys:
            vals = [r[f"{prefix}_{k}"] for r in rows if r.get(f"{prefix}_{k}") is not None]
            if vals:
                overall[f"{prefix}_{k}_mean"] = float(np.mean(vals))
                overall[f"{prefix}_{k}_median"] = float(np.median(vals))
    for k in metric_keys:
        rk = f"retrieval_{k}_mean"
        dk = f"diffusion_{k}_mean"
        if rk in overall and dk in overall:
            overall[f"delta_{k}_mean"] = overall[dk] - overall[rk]

    by_bucket: dict[str, Any] = {}
    for bucket in BUCKETS:
        bucket_rows = [r for r in rows if r["bucket"] == bucket]
        if bucket_rows:
            bm: dict[str, Any] = {"n_examples": len(bucket_rows)}
            for prefix in ["retrieval", "diffusion"]:
                for k in metric_keys:
                    vals = [r[f"{prefix}_{k}"] for r in bucket_rows if r.get(f"{prefix}_{k}") is not None]
                    if vals:
                        bm[f"{prefix}_{k}_mean"] = float(np.mean(vals))
            for k in metric_keys:
                rk = f"retrieval_{k}_mean"
                dk = f"diffusion_{k}_mean"
                if rk in bm and dk in bm:
                    bm[f"delta_{k}_mean"] = bm[dk] - bm[rk]
            by_bucket[bucket] = bm

    return {"overall": overall, "by_bucket": by_bucket}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Triple-fusion-anchored reconstruction")
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument("--stimuli-hdf5", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="sd2-community/stable-diffusion-2-1")
    ap.add_argument("--hf-cache", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    logger.info("Step 1: Computing triple fusion...")
    csls, gt_ranks, nsd_ids = compute_triple_fusion(args.repo_root, device)
    top1_gallery_idx = np.argmax(csls, axis=1).astype(np.int64)

    logger.info("Step 2: Selecting 16 shared1000 examples...")
    selected = select_16_shared1000(gt_ranks, nsd_ids, args.seed)

    csv_path = out_dir / "selected_examples_triple.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "row_index", "query_nsd_id", "gt_rank", "bucket",
                                          "controller_route", "anchor_nsd_id"])
        w.writeheader()
        for ex in selected:
            anchor_gallery_idx = top1_gallery_idx[ex["row_index"]]
            ex["anchor_nsd_id"] = int(nsd_ids[anchor_gallery_idx])
            w.writerow({k: v for k, v in ex.items() if not k.startswith("_")})
    logger.info(f"  Saved selection to {csv_path}")

    logger.info("Step 3: Loading CLIP (ViT-L/14) for metrics...")
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(device).eval()

    logger.info("Step 4: Loading SD 2.1 pipeline...")
    pipe = load_sd_pipeline(args.model_id, device, cache_dir=args.hf_cache)

    logger.info("Step 5: Loading stimulus store...")
    store = StimulusStore(args.stimuli_hdf5)

    logger.info("Step 6: Running reconstruction for 16 examples...")
    import lpips as lpips_mod
    lpips_net = lpips_mod.LPIPS(net="alex").to(device)

    metric_rows = []
    for i, ex in enumerate(selected):
        qid = ex["query_nsd_id"]
        anchor_gallery_idx = top1_gallery_idx[ex["row_index"]]
        anchor_nsd_id = int(nsd_ids[anchor_gallery_idx])
        route = ex["controller_route"]

        gt_img = store.get_image(qid)
        anchor_img = store.get_image(anchor_nsd_id)

        logger.info(f"  [{i+1}/16] NSD {qid} | bucket={ex['bucket']} | route={route} | "
                     f"anchor_nsd={anchor_nsd_id} | gt_rank={ex['gt_rank']}")

        diffusion_img = run_img2img(pipe, anchor_img, route, args.seed)

        ex["_anchor_img"] = anchor_img
        ex["_diffusion_img"] = diffusion_img

        retrieval_metrics = _compute_metrics_pair(anchor_img, gt_img, clip_model, lpips_net, device)
        diffusion_metrics = _compute_metrics_pair(diffusion_img, gt_img, clip_model, lpips_net, device)

        row = {
            "query_nsd_id": qid,
            "anchor_nsd_id": anchor_nsd_id,
            "gt_rank": ex["gt_rank"],
            "bucket": ex["bucket"],
            "controller_route": route,
        }
        for k, v in retrieval_metrics.items():
            row[f"retrieval_{k}"] = v
        for k, v in diffusion_metrics.items():
            row[f"diffusion_{k}"] = v
        for k in retrieval_metrics:
            row[f"delta_{k}"] = diffusion_metrics[k] - retrieval_metrics[k]
        metric_rows.append(row)

    logger.info("Step 7: Computing summary metrics...")
    summary = summarize_metrics(metric_rows)
    summary["method"] = "triple_fusion_anchor_img2img"
    summary["model_id"] = args.model_id
    summary["seed"] = args.seed
    summary["route_params"] = ROUTE_PARAMS

    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved {metrics_path}")

    per_example_path = out_dir / "per_example.csv"
    if metric_rows:
        with open(per_example_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=metric_rows[0].keys())
            w.writeheader()
            w.writerows(metric_rows)
        logger.info(f"  Saved {per_example_path}")

    logger.info("Step 8: Generating composite image...")
    composite_path = out_dir / "reconstruction_examples_composite.png"
    make_composite(selected, store, str(composite_path))

    store.close()
    logger.info("Done.")
    return 0


def _compute_metrics_pair(
    pred: Image.Image, gt: Image.Image, clip_model, lpips_net, device: str
) -> dict[str, float]:
    from fmri2img.eval.recon_eval import compute_pixcorr, compute_psnr, compute_ssim
    from fmri2img.eval.image_metrics import clip_score

    pred_arr = to_np01(pred)
    gt_arr = to_np01(gt)

    pixcorr = float(compute_pixcorr(pred_arr, gt_arr))
    ssim_val = float(compute_ssim(pred_arr, gt_arr))
    psnr_val = float(compute_psnr(pred_arr, gt_arr))

    pred_t = torch.from_numpy(pred_arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    gt_t = torch.from_numpy(gt_arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        lpips_val = float(lpips_net(pred_t, gt_t).item())

    clip_sim = float(clip_score(pred, gt, clip_model, device=device))

    return {
        "pixcorr": pixcorr,
        "ssim": ssim_val,
        "psnr": psnr_val,
        "lpips": lpips_val,
        "clip_image_similarity": clip_sim,
    }


if __name__ == "__main__":
    sys.exit(main())
