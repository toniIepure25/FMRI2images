#!/usr/bin/env python3
"""Improved triple-fusion-anchored reconstruction with CLIP-conditioned img2img,
best-of-N candidate selection, and uncertainty-aware strength.

Tiers implemented:
  1. CLIP-conditioned img2img — predicted CLIP embedding as prompt_embeds
  2. Best-of-N (N=16) — select candidate with highest cosine to predicted CLIP
  3. Uncertainty-aware strength — vMF kappa maps to img2img strength
  5. AlexNet(2), AlexNet(5) metrics added to evaluation

Does NOT touch V35, N1v28a, or V40 controller code.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BUCKETS = ["perfect", "good", "near_good", "hard"]
IMG_SIZE = (256, 256)

NEG_PROMPT = "text, watermark, blurry, low quality, distorted, oversaturated, painting, illustration"

STRENGTH_MAX = 0.30
STRENGTH_MIN = 0.0
GUIDANCE_MIN = 3.0
GUIDANCE_MAX = 7.5
STEPS_MIN = 15
STEPS_MAX = 40
N_CANDIDATES = 16


def bucket_from_rank(rank: int) -> str:
    if rank == 1:
        return "perfect"
    elif rank <= 5:
        return "good"
    elif rank <= 15:
        return "near_good"
    return "hard"


# ---------------------------------------------------------------------------
# Triple fusion (unchanged from previous version)
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


def compute_triple_fusion(repo_root: str, device: str):
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
    r1 = float((gt_ranks == 1).mean())
    logger.info(f"  Triple fusion CSLS R@1 = {r1:.4f} ({(gt_ranks == 1).sum()}/{n})")

    nsd_ids_path = os.path.join(er, "V61a_finetune_difflr", "subj01", "metrics", "shared1000_nsd_ids.npy")
    nsd_ids = np.load(nsd_ids_path)
    return csls, gt_ranks, nsd_ids


# ---------------------------------------------------------------------------
# Example selection (unchanged)
# ---------------------------------------------------------------------------

def select_16_shared1000(gt_ranks: np.ndarray, nsd_ids: np.ndarray, seed: int = 42) -> list[dict]:
    rng = np.random.RandomState(seed)
    pool_by_bucket: dict[str, list[int]] = {b: [] for b in BUCKETS}
    for i in range(len(gt_ranks)):
        pool_by_bucket[bucket_from_rank(int(gt_ranks[i]))].append(i)

    for b in BUCKETS:
        logger.info(f"  Bucket {b}: {len(pool_by_bucket[b])} candidates")

    selected = []
    for bucket in BUCKETS:
        pool = pool_by_bucket[bucket]
        if len(pool) < 4:
            raise RuntimeError(f"Only {len(pool)} shared1000 examples in bucket '{bucket}', need 4")
        indices = sorted(rng.choice(pool, size=4, replace=False))
        for row_idx in indices:
            selected.append({
                "split": "shared1000",
                "row_index": int(row_idx),
                "query_nsd_id": int(nsd_ids[row_idx]),
                "gt_rank": int(gt_ranks[row_idx]),
                "bucket": bucket,
            })
    return selected


# ---------------------------------------------------------------------------
# HDF5 store
# ---------------------------------------------------------------------------

class StimulusStore:
    def __init__(self, hdf5_path: str):
        self._f = h5py.File(hdf5_path, "r")
        self._dset = self._f["imgBrick"]

    def get_image(self, nsd_id: int) -> Image.Image:
        return Image.fromarray(self._dset[nsd_id]).convert("RGB")

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Tier 3: Uncertainty-aware strength from V62a predicted CLIP embeddings
# ---------------------------------------------------------------------------

def load_predicted_clip_embeddings(repo_root: str) -> np.ndarray:
    """Load V62a's predicted 768-D CLIP embeddings for shared1000."""
    path = os.path.join(
        repo_root, "experimental_results",
        "V62a_cls_retrieval_768d", "subj01", "metrics",
        "shared1000_predictions.npy",
    )
    preds = np.load(path)
    norms = np.linalg.norm(preds, axis=1, keepdims=True)
    return preds / np.clip(norms, 1e-8, None)


def compute_per_example_confidence(csls_row: np.ndarray) -> float:
    """Derive confidence from CSLS score distribution: margin between top-1 and top-2."""
    sorted_scores = np.sort(csls_row)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]
    return float(np.clip(margin / 2.0, 0.0, 1.0))


def strength_from_confidence(confidence: float) -> float:
    return STRENGTH_MAX * (1.0 - confidence)


def guidance_from_confidence(confidence: float) -> float:
    return GUIDANCE_MIN + confidence * (GUIDANCE_MAX - GUIDANCE_MIN)


def steps_from_confidence(confidence: float) -> int:
    return int(STEPS_MAX - confidence * (STEPS_MAX - STEPS_MIN))


# ---------------------------------------------------------------------------
# SD pipeline loading
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


# ---------------------------------------------------------------------------
# Tier 1: CLIP-conditioned img2img
# ---------------------------------------------------------------------------

def build_prompt_embeds(pipe, pred_clip_768: np.ndarray) -> torch.Tensor:
    """Build prompt_embeds for SD-2.1 img2img from a 768-D predicted CLIP vector.

    SD-2.1 uses OpenCLIP ViT-H/14 with 1024-D embeddings internally, but the
    text encoder produces (B, 77, 1024) sequence embeddings. We inject the
    predicted 768-D vector by projecting it to 1024-D (zero-padded) and blending
    it into the pooled embedding position, then letting the text encoder's
    sequence structure carry it through.

    For img2img, we can pass prompt_embeds directly. The simplest robust approach:
    get base (empty-prompt) embeddings and replace the pooled component.
    """
    unet_dtype = pipe.unet.dtype
    device = pipe.device

    pred = torch.from_numpy(pred_clip_768).to(device=device, dtype=unet_dtype)
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    pred = F.normalize(pred, dim=-1)

    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        base_embeds = pipe.text_encoder(text_inputs.input_ids)[0]

    seq_dim = base_embeds.shape[-1]
    if pred.shape[-1] < seq_dim:
        pad = torch.zeros(pred.shape[0], seq_dim - pred.shape[-1], device=device, dtype=unet_dtype)
        pred_proj = torch.cat([pred.to(unet_dtype), pad], dim=-1)
    else:
        pred_proj = pred[:, :seq_dim].to(unet_dtype)
    pred_proj = F.normalize(pred_proj, dim=-1)

    prompt_embeds = base_embeds.clone()
    prompt_embeds[:, 0, :] = pred_proj
    scale = 1.5
    for t in range(1, min(4, prompt_embeds.shape[1])):
        prompt_embeds[:, t, :] = prompt_embeds[:, t, :] + scale * pred_proj
        scale *= 0.5

    return prompt_embeds


# ---------------------------------------------------------------------------
# Tier 2: Best-of-N generation + selection
# ---------------------------------------------------------------------------

def run_clip_conditioned_best_of_n(
    pipe,
    anchor_img: Image.Image,
    pred_clip_768: np.ndarray,
    clip_model,
    *,
    strength: float,
    guidance_scale: float,
    num_steps: int,
    n_candidates: int,
    base_seed: int,
    device: str,
) -> Image.Image:
    """CLIP-conditioned img2img with best-of-N selection by predicted CLIP cosine."""
    if strength <= 0.0:
        return anchor_img.copy()

    from fmri2img.eval.image_metrics import preprocess_image_for_clip

    prompt_embeds = build_prompt_embeds(pipe, pred_clip_768)
    init = anchor_img.resize((768, 768), Image.LANCZOS)

    candidates: list[Image.Image] = []
    for offset in range(n_candidates):
        seed = base_seed + offset
        gen = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt=NEG_PROMPT,
            image=init,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=gen,
        )
        candidates.append(result.images[0].convert("RGB"))

    if len(candidates) == 1:
        return candidates[0]

    pred_t = torch.from_numpy(pred_clip_768).unsqueeze(0).to(device).float()
    pred_t = F.normalize(pred_t, dim=-1)

    best_score = -1.0
    best_img = candidates[0]
    for cand in candidates:
        cand_tensor = preprocess_image_for_clip(cand).unsqueeze(0).to(device)
        with torch.no_grad():
            cand_emb = clip_model.encode_image(cand_tensor).float()
            cand_emb = F.normalize(cand_emb, dim=-1)
            score = (pred_t * cand_emb).sum().item()
        if score > best_score:
            best_score = score
            best_img = cand

    return best_img


# ---------------------------------------------------------------------------
# Metrics (Tier 5: adds AlexNet-2, AlexNet-5)
# ---------------------------------------------------------------------------

def to_np01(img: Image.Image, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    return np.asarray(ImageOps.contain(img.convert("RGB"), size), dtype=np.float32) / 255.0


def _load_alexnet_features(device: str):
    """Load AlexNet and return feature extractors for layers 2 and 5."""
    import torchvision.models as models

    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device).eval()
    features = alexnet.features

    layer2 = torch.nn.Sequential(*list(features.children())[:5]).to(device)
    layer5 = torch.nn.Sequential(*list(features.children())[:12]).to(device)

    normalize = torch.nn.Sequential(
        torch.nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
    )
    return layer2, layer5, normalize


def compute_alexnet_scores(
    pred_img: Image.Image, gt_img: Image.Image,
    layer2, layer5, normalize, device: str,
) -> tuple[float, float]:
    """Compute AlexNet(2) and AlexNet(5) cosine similarity."""
    pred_arr = to_np01(pred_img, (224, 224))
    gt_arr = to_np01(gt_img, (224, 224))

    pred_t = torch.from_numpy(pred_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        p2 = layer2(pred_t).flatten(1)
        g2 = layer2(gt_t).flatten(1)
        alex2 = F.cosine_similarity(p2, g2).item()

        p5 = layer5(pred_t).flatten(1)
        g5 = layer5(gt_t).flatten(1)
        alex5 = F.cosine_similarity(p5, g5).item()

    return alex2, alex5


def compute_all_metrics(
    pred: Image.Image, gt: Image.Image,
    clip_model, lpips_net,
    alex_layer2, alex_layer5, alex_norm,
    device: str,
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

    alex2, alex5 = compute_alexnet_scores(pred, gt, alex_layer2, alex_layer5, alex_norm, device)

    return {
        "pixcorr": pixcorr,
        "ssim": ssim_val,
        "psnr": psnr_val,
        "lpips": lpips_val,
        "clip_image_similarity": clip_sim,
        "alexnet2": alex2,
        "alexnet5": alex5,
    }


# ---------------------------------------------------------------------------
# Composite image generation
# ---------------------------------------------------------------------------

def make_composite(rows: list[dict], store: StimulusStore, output_path: str, tile_size: int = 320):
    """Lay out the qualitative reconstruction composite.

    Two examples per bucket (instead of four) at 320 px tiles, with column
    headers per example group, a soft separator strip between buckets, and a
    bold-coloured side-label for each bucket. The result is much more legible
    at thesis print size than the previous 4-example-per-row grid.
    """
    cols_per_bucket = 2
    n_img_cols = 3       # GT | Anchor | Diffusion
    total_cols = cols_per_bucket * n_img_cols
    label_w = 200        # wider so the bucket label has breathing room
    gap = 8
    group_gap = 40       # extra space between the two example groups
    header_h = 64        # taller header so column titles are clearly readable
    row_pad = 22         # padding between buckets

    # Width accounts for the extra group-gap between the two example triplets.
    w = (label_w
         + total_cols * tile_size
         + (n_img_cols - 1) * gap * cols_per_bucket
         + group_gap
         + 28)
    h = header_h + len(BUCKETS) * (tile_size + row_pad + 6) + 32
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_bucket = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font_grp = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font_bucket = ImageFont.load_default()
        font_header = font_bucket
        font_grp = font_bucket

    # Compute x positions for each of the 6 image tiles
    def tile_x(group_idx: int, col_in_group: int) -> int:
        base = label_w + group_idx * (n_img_cols * tile_size + (n_img_cols - 1) * gap + group_gap)
        return base + col_in_group * (tile_size + gap)

    # ---- column header row ----
    for g in range(cols_per_bucket):
        for ci, lbl in enumerate(["Ground truth", "Anchor", "Diffusion"]):
            x = tile_x(g, ci) + tile_size // 2
            draw.text((x, 14), lbl, fill=(40, 40, 40), font=font_header, anchor="mt")
        # group label "Example 1" / "Example 2" above each triplet
        ex_x = (tile_x(g, 0) + tile_x(g, n_img_cols - 1) + tile_size) // 2
        draw.text((ex_x, header_h - 22), f"Example {g + 1}",
                  fill=(110, 110, 110), font=font_grp, anchor="mb")
        # thin underline below the per-group header
        ux0 = tile_x(g, 0)
        ux1 = tile_x(g, n_img_cols - 1) + tile_size
        draw.line([(ux0, header_h - 4), (ux1, header_h - 4)],
                  fill=(180, 180, 180), width=1)

    # ---- bucket rows ----
    by_bucket = {b: [] for b in BUCKETS}
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    BUCKET_COLOR = {
        "perfect": (40, 110, 60),
        "good": (60, 100, 160),
        "near_good": (150, 105, 40),
        "hard": (150, 50, 50),
    }

    for bi, bucket in enumerate(BUCKETS):
        y0 = header_h + bi * (tile_size + row_pad + 6)
        bucket_label = bucket.replace("_", " ").title()
        draw.text(
            (10, y0 + tile_size // 2),
            bucket_label,
            fill=BUCKET_COLOR.get(bucket, (40, 40, 40)),
            font=font_bucket, anchor="lm",
        )
        examples = by_bucket[bucket][:cols_per_bucket]
        for ei, ex in enumerate(examples):
            gt_img = store.get_image(ex["query_nsd_id"]).resize((tile_size, tile_size), Image.LANCZOS)
            anchor_img = ex["_anchor_img"].resize((tile_size, tile_size), Image.LANCZOS)
            diff_img = ex["_diffusion_img"].resize((tile_size, tile_size), Image.LANCZOS)
            for ci, img in enumerate([gt_img, anchor_img, diff_img]):
                x = tile_x(ei, ci)
                canvas.paste(img, (x, y0))
        # soft separator at the bottom of each bucket row (skip the last)
        if bi < len(BUCKETS) - 1:
            sy = y0 + tile_size + row_pad // 2
            draw.line([(label_w, sy), (w - 12, sy)], fill=(225, 225, 225), width=1)

    canvas.save(output_path, dpi=(300, 300), quality=95)
    logger.info(f"Composite saved to {output_path} ({w}x{h}, 300 dpi)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_metrics(rows: list[dict]) -> dict[str, Any]:
    metric_keys = ["pixcorr", "ssim", "psnr", "lpips", "clip_image_similarity", "alexnet2", "alexnet5"]
    overall: dict[str, Any] = {"n_examples": len(rows)}
    for prefix in ["retrieval", "diffusion"]:
        for k in metric_keys:
            vals = [r[f"{prefix}_{k}"] for r in rows if r.get(f"{prefix}_{k}") is not None]
            finite = [v for v in vals if np.isfinite(v)]
            if finite:
                overall[f"{prefix}_{k}_mean"] = float(np.mean(finite))
                overall[f"{prefix}_{k}_median"] = float(np.median(finite))
    for k in metric_keys:
        rk = f"retrieval_{k}_mean"
        dk = f"diffusion_{k}_mean"
        if rk in overall and dk in overall:
            overall[f"delta_{k}_mean"] = overall[dk] - overall[rk]

    by_bucket: dict[str, Any] = {}
    for bucket in BUCKETS:
        bucket_rows = [r for r in rows if r["bucket"] == bucket]
        if not bucket_rows:
            continue
        bm: dict[str, Any] = {"n_examples": len(bucket_rows)}
        for prefix in ["retrieval", "diffusion"]:
            for k in metric_keys:
                vals = [r[f"{prefix}_{k}"] for r in bucket_rows if r.get(f"{prefix}_{k}") is not None]
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    bm[f"{prefix}_{k}_mean"] = float(np.mean(finite))
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
    ap = argparse.ArgumentParser(description="Improved triple-fusion reconstruction")
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument("--stimuli-hdf5", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="sd2-community/stable-diffusion-2-1")
    ap.add_argument("--hf-cache", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    logger.info("Step 1: Computing triple fusion...")
    csls, gt_ranks, nsd_ids = compute_triple_fusion(args.repo_root, device)
    top1_gallery_idx = np.argmax(csls, axis=1).astype(np.int64)

    logger.info("Step 2: Selecting 16 shared1000 examples...")
    selected = select_16_shared1000(gt_ranks, nsd_ids, args.seed)

    logger.info("Step 3: Loading predicted CLIP embeddings (V62a)...")
    pred_clips = load_predicted_clip_embeddings(args.repo_root)
    logger.info(f"  Loaded predicted CLIP: {pred_clips.shape}")

    logger.info("Step 4: Loading CLIP ViT-L/14...")
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(device).eval()

    logger.info("Step 5: Loading SD 2.1 img2img pipeline...")
    pipe = load_sd_pipeline(args.model_id, device, cache_dir=args.hf_cache)

    logger.info("Step 6: Loading stimulus store + metric evaluators...")
    store = StimulusStore(args.stimuli_hdf5)

    import lpips as lpips_mod
    lpips_net = lpips_mod.LPIPS(net="alex").to(device)
    alex_l2, alex_l5, alex_norm = _load_alexnet_features(device)

    csv_path = out_dir / "selected_examples_triple.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "split", "row_index", "query_nsd_id", "gt_rank", "bucket",
            "anchor_nsd_id", "confidence", "strength", "guidance_scale", "steps",
        ])
        w.writeheader()
        for ex in selected:
            anchor_gallery_idx = top1_gallery_idx[ex["row_index"]]
            ex["anchor_nsd_id"] = int(nsd_ids[anchor_gallery_idx])

            conf = compute_per_example_confidence(csls[ex["row_index"]])
            ex["confidence"] = conf

            if ex["bucket"] == "perfect":
                ex["strength"] = 0.0
                ex["guidance_scale"] = 0.0
                ex["steps"] = 0
            else:
                ex["strength"] = round(strength_from_confidence(conf), 4)
                ex["guidance_scale"] = round(guidance_from_confidence(conf), 2)
                ex["steps"] = steps_from_confidence(conf)

            w.writerow({k: v for k, v in ex.items() if not k.startswith("_")})
    logger.info(f"  Saved selection to {csv_path}")

    logger.info("Step 7: Running CLIP-conditioned best-of-N reconstruction...")
    metric_rows = []
    for i, ex in enumerate(selected):
        qid = ex["query_nsd_id"]
        row_idx = ex["row_index"]
        anchor_gallery_idx = top1_gallery_idx[row_idx]
        anchor_nsd_id = int(nsd_ids[anchor_gallery_idx])

        gt_img = store.get_image(qid)
        anchor_img = store.get_image(anchor_nsd_id)

        pred_clip_768 = pred_clips[row_idx]
        strength = ex["strength"]
        guidance = ex["guidance_scale"]
        steps = ex["steps"]

        logger.info(
            f"  [{i+1}/16] NSD {qid} | bucket={ex['bucket']} | "
            f"conf={ex['confidence']:.3f} | str={strength:.3f} | "
            f"guid={guidance:.1f} | steps={steps} | "
            f"anchor_nsd={anchor_nsd_id} | gt_rank={ex['gt_rank']}"
        )

        diffusion_img = run_clip_conditioned_best_of_n(
            pipe, anchor_img, pred_clip_768, clip_model,
            strength=strength,
            guidance_scale=guidance,
            num_steps=steps,
            n_candidates=args.n_candidates,
            base_seed=args.seed + i * 100,
            device=device,
        )

        ex["_anchor_img"] = anchor_img
        ex["_diffusion_img"] = diffusion_img

        retrieval_m = compute_all_metrics(anchor_img, gt_img, clip_model, lpips_net, alex_l2, alex_l5, alex_norm, device)
        diffusion_m = compute_all_metrics(diffusion_img, gt_img, clip_model, lpips_net, alex_l2, alex_l5, alex_norm, device)

        row = {
            "query_nsd_id": qid,
            "anchor_nsd_id": anchor_nsd_id,
            "gt_rank": ex["gt_rank"],
            "bucket": ex["bucket"],
            "confidence": ex["confidence"],
            "strength": strength,
            "guidance_scale": guidance,
            "steps": steps,
            "n_candidates": args.n_candidates if strength > 0 else 0,
        }
        for k, v in retrieval_m.items():
            row[f"retrieval_{k}"] = v
        for k, v in diffusion_m.items():
            row[f"diffusion_{k}"] = v
        for k in retrieval_m:
            row[f"delta_{k}"] = diffusion_m[k] - retrieval_m[k]
        metric_rows.append(row)

    logger.info("Step 8: Computing summary metrics...")
    summary = summarize_metrics(metric_rows)
    summary["method"] = "clip_conditioned_best_of_n_ua_strength"
    summary["model_id"] = args.model_id
    summary["seed"] = args.seed
    summary["n_candidates"] = args.n_candidates
    summary["strength_range"] = [STRENGTH_MIN, STRENGTH_MAX]
    summary["guidance_range"] = [GUIDANCE_MIN, GUIDANCE_MAX]
    summary["steps_range"] = [STEPS_MIN, STEPS_MAX]

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

    logger.info("Step 9: Generating composite image...")
    composite_path = out_dir / "reconstruction_examples_composite.png"
    make_composite(selected, store, str(composite_path))

    store.close()
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
