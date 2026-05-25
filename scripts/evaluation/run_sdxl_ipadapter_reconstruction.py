#!/usr/bin/env python3
"""SDXL + IP-Adapter reconstruction with n-way identification.

Key improvements:
  - SDXL 1.0 replaces SD 2.1 (stronger prior, native 1024x1024, dual text encoder)
  - IP-Adapter visual conditioning from the anchor image (dedicated cross-attention)
  - Predicted CLIP (768-D ViT-L/14) as prompt_embeds — same space as SDXL's first
    text encoder, so injection is clean rather than hacked
  - Best-of-N candidate selection (N configurable, default 16)
  - Uncertainty-aware strength from CSLS confidence margin
  - N-way identification accuracy (AlexNet-2, AlexNet-5) for MindEye2 comparison
  - Scalable to 100+ examples for statistical defensibility
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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
STEPS_MIN = 20
STEPS_MAX = 50
N_CANDIDATES = 16
IP_ADAPTER_SCALE = 0.5


def bucket_from_rank(rank: int) -> str:
    if rank == 1:
        return "perfect"
    elif rank <= 5:
        return "good"
    elif rank <= 15:
        return "near_good"
    return "hard"


# ---------------------------------------------------------------------------
# Triple fusion (unchanged)
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
    logger.info(f"  Triple fusion CSLS R@1 = {(gt_ranks == 1).mean():.4f}")

    nsd_ids_path = os.path.join(er, "V61a_finetune_difflr", "subj01", "metrics", "shared1000_nsd_ids.npy")
    nsd_ids = np.load(nsd_ids_path)
    return csls, gt_ranks, nsd_ids


# ---------------------------------------------------------------------------
# Example selection (scalable)
# ---------------------------------------------------------------------------

def select_examples(
    gt_ranks: np.ndarray,
    nsd_ids: np.ndarray,
    n_perfect: int = 4,
    max_per_bucket: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Select examples. Uses all available non-perfect examples by default."""
    rng = np.random.RandomState(seed)
    pool_by_bucket: dict[str, list[int]] = {b: [] for b in BUCKETS}
    for i in range(len(gt_ranks)):
        pool_by_bucket[bucket_from_rank(int(gt_ranks[i]))].append(i)

    for b in BUCKETS:
        logger.info(f"  Bucket {b}: {len(pool_by_bucket[b])} candidates")

    selected = []
    for bucket in BUCKETS:
        pool = pool_by_bucket[bucket]
        if bucket == "perfect":
            n = min(n_perfect, len(pool))
        elif max_per_bucket is not None:
            n = min(max_per_bucket, len(pool))
        else:
            n = len(pool)

        if n == 0:
            continue
        indices = sorted(rng.choice(pool, size=n, replace=False))
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
# Predicted CLIP + confidence
# ---------------------------------------------------------------------------

def load_predicted_clip_embeddings(repo_root: str) -> np.ndarray:
    path = os.path.join(
        repo_root, "experimental_results",
        "V62a_cls_retrieval_768d", "subj01", "metrics",
        "shared1000_predictions.npy",
    )
    preds = np.load(path)
    norms = np.linalg.norm(preds, axis=1, keepdims=True)
    return preds / np.clip(norms, 1e-8, None)


def compute_per_example_confidence(csls_row: np.ndarray) -> float:
    sorted_scores = np.sort(csls_row)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]
    return float(np.clip(margin / 2.0, 0.0, 1.0))


def strength_from_confidence(c: float) -> float:
    return STRENGTH_MAX * (1.0 - c)


def guidance_from_confidence(c: float) -> float:
    return GUIDANCE_MIN + c * (GUIDANCE_MAX - GUIDANCE_MIN)


def steps_from_confidence(c: float) -> int:
    return int(STEPS_MAX - c * (STEPS_MAX - STEPS_MIN))


# ---------------------------------------------------------------------------
# SDXL + IP-Adapter pipeline
# ---------------------------------------------------------------------------

def load_sdxl_pipeline(model_id: str, device_str: str, cache_dir: str | None = None):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline

    dtype = torch.float16 if device_str.startswith("cuda") else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "use_safetensors": True,
        "variant": "fp16",
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, **kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device_str)
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass
    return pipe


def load_ip_adapter(pipe, cache_dir: str | None = None):
    """Load IP-Adapter for SDXL (ViT-H variant, 1024-D) with matching image encoder."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    cache_kw = {"cache_dir": cache_dir} if cache_dir else {}
    dtype = pipe.unet.dtype

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=dtype,
        **cache_kw,
    ).to(pipe.device)
    pipe.image_encoder = image_encoder
    pipe.feature_extractor = CLIPImageProcessor()
    logger.info("  ViT-H/14 image encoder loaded for IP-Adapter")

    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl_vit-h.safetensors",
        **cache_kw,
    )
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    logger.info(f"  IP-Adapter loaded (scale={IP_ADAPTER_SCALE})")


def build_sdxl_prompt_embeds(pipe, pred_clip_768: np.ndarray):
    """Build SDXL prompt_embeds from predicted 768-D CLIP ViT-L/14 embedding.

    SDXL's first text encoder is CLIP ViT-L/14 — exactly our predicted space.
    We inject the predicted embedding into the first token positions of the
    (B, 77, 768) sequence, giving the UNet a semantic signal from the brain.

    Returns (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds).
    """
    unet_dtype = pipe.unet.dtype
    device = pipe.device

    pred = torch.from_numpy(pred_clip_768).to(device=device, dtype=unet_dtype)
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    pred = F.normalize(pred, dim=-1)

    with torch.no_grad():
        tok1 = pipe.tokenizer(
            [""], padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        base_embeds_1 = pipe.text_encoder(tok1.input_ids, output_hidden_states=True)
        prompt_embeds = base_embeds_1.hidden_states[-2]

        tok2 = pipe.tokenizer_2(
            [""], padding="max_length",
            max_length=pipe.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        base_embeds_2 = pipe.text_encoder_2(tok2.input_ids, output_hidden_states=True)
        pooled = base_embeds_2[0]
        prompt_embeds_2 = base_embeds_2.hidden_states[-2]

    prompt_embeds = prompt_embeds.to(dtype=unet_dtype)
    prompt_embeds_2 = prompt_embeds_2.to(dtype=unet_dtype)
    pooled = pooled.to(dtype=unet_dtype)

    scale = 1.5
    for t in range(min(4, prompt_embeds.shape[1])):
        prompt_embeds[:, t, :] = prompt_embeds[:, t, :] + scale * pred.to(unet_dtype)
        scale *= 0.5

    prompt_embeds_cat = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

    neg_tok1 = pipe.tokenizer(
        [NEG_PROMPT], padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    neg_embeds_1 = pipe.text_encoder(neg_tok1.input_ids, output_hidden_states=True)
    neg_prompt_embeds = neg_embeds_1.hidden_states[-2].to(dtype=unet_dtype)

    neg_tok2 = pipe.tokenizer_2(
        [NEG_PROMPT], padding="max_length",
        max_length=pipe.tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    neg_embeds_2 = pipe.text_encoder_2(neg_tok2.input_ids, output_hidden_states=True)
    neg_pooled = neg_embeds_2[0].to(dtype=unet_dtype)
    neg_prompt_embeds_2 = neg_embeds_2.hidden_states[-2].to(dtype=unet_dtype)

    neg_prompt_embeds_cat = torch.cat([neg_prompt_embeds, neg_prompt_embeds_2], dim=-1)

    return prompt_embeds_cat, neg_prompt_embeds_cat, pooled, neg_pooled


# ---------------------------------------------------------------------------
# Generation: CLIP-conditioned + IP-Adapter + best-of-N
# ---------------------------------------------------------------------------

def run_sdxl_best_of_n(
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
    use_ip_adapter: bool = True,
) -> Image.Image:
    if strength <= 0.0:
        return anchor_img.copy()

    from fmri2img.eval.image_metrics import preprocess_image_for_clip

    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = build_sdxl_prompt_embeds(
        pipe, pred_clip_768,
    )

    init = anchor_img.resize((1024, 1024), Image.LANCZOS)

    pipe_kwargs: dict[str, Any] = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": neg_prompt_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_pooled_prompt_embeds": neg_pooled,
        "image": init,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
    }
    if use_ip_adapter:
        pipe_kwargs["ip_adapter_image"] = init

    candidates: list[Image.Image] = []
    for offset in range(n_candidates):
        gen = torch.Generator(device=pipe.device).manual_seed(base_seed + offset)
        pipe_kwargs["generator"] = gen
        result = pipe(**pipe_kwargs)
        candidates.append(result.images[0].convert("RGB"))

    if len(candidates) == 1:
        return candidates[0]

    pred_t = torch.from_numpy(pred_clip_768).unsqueeze(0).to(device).float()
    pred_t = F.normalize(pred_t, dim=-1)

    best_score, best_img = -1.0, candidates[0]
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
# Metrics (including n-way identification)
# ---------------------------------------------------------------------------

def to_np01(img: Image.Image, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    return np.asarray(ImageOps.contain(img.convert("RGB"), size), dtype=np.float32) / 255.0


def _load_alexnet_features(device: str):
    import torchvision.models as models
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device).eval()
    features = alexnet.features
    layer2 = torch.nn.Sequential(*list(features.children())[:5]).to(device)
    layer5 = torch.nn.Sequential(*list(features.children())[:12]).to(device)
    return layer2, layer5


def _alexnet_embed(img: Image.Image, layer, device: str) -> torch.Tensor:
    arr = to_np01(img, (224, 224))
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return layer(t).flatten(1)


def compute_all_metrics(
    pred: Image.Image, gt: Image.Image,
    clip_model, lpips_net,
    alex_l2, alex_l5,
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

    p2 = _alexnet_embed(pred, alex_l2, device)
    g2 = _alexnet_embed(gt, alex_l2, device)
    alex2 = float(F.cosine_similarity(p2, g2).item())

    p5 = _alexnet_embed(pred, alex_l5, device)
    g5 = _alexnet_embed(gt, alex_l5, device)
    alex5 = float(F.cosine_similarity(p5, g5).item())

    return {
        "pixcorr": pixcorr, "ssim": ssim_val, "psnr": psnr_val,
        "lpips": lpips_val, "clip_image_similarity": clip_sim,
        "alexnet2": alex2, "alexnet5": alex5,
    }


def compute_nway_identification(
    recon_imgs: list[Image.Image],
    gt_imgs: list[Image.Image],
    alex_layer,
    device: str,
) -> float:
    """N-way identification: does argmax cosine(recon_i, gallery) == i?"""
    n = len(recon_imgs)
    if n < 2:
        return 1.0
    feats_r = torch.cat([_alexnet_embed(r, alex_layer, device) for r in recon_imgs], dim=0)
    feats_g = torch.cat([_alexnet_embed(g, alex_layer, device) for g in gt_imgs], dim=0)
    feats_r = F.normalize(feats_r, dim=-1)
    feats_g = F.normalize(feats_g, dim=-1)
    sim = feats_r @ feats_g.T
    preds = sim.argmax(dim=1)
    correct = (preds == torch.arange(n, device=device)).float()
    return float(correct.mean().item())


def compute_2way_identification(
    recon_imgs: list[Image.Image],
    gt_imgs: list[Image.Image],
    alex_layer,
    device: str,
    n_trials: int = 1000,
    seed: int = 42,
) -> float:
    """2-way forced-choice: given recon, is correct GT closer than a random distractor?"""
    rng = np.random.RandomState(seed)
    n = len(recon_imgs)
    if n < 2:
        return 1.0
    feats_r = torch.cat([_alexnet_embed(r, alex_layer, device) for r in recon_imgs], dim=0)
    feats_g = torch.cat([_alexnet_embed(g, alex_layer, device) for g in gt_imgs], dim=0)
    feats_r = F.normalize(feats_r, dim=-1)
    feats_g = F.normalize(feats_g, dim=-1)

    correct = 0
    for _ in range(n_trials):
        i = rng.randint(n)
        j = rng.randint(n - 1)
        if j >= i:
            j += 1
        sim_correct = (feats_r[i] * feats_g[i]).sum().item()
        sim_distractor = (feats_r[i] * feats_g[j]).sum().item()
        if sim_correct > sim_distractor:
            correct += 1
    return correct / n_trials


# ---------------------------------------------------------------------------
# Composite image
# ---------------------------------------------------------------------------

def make_composite(rows: list[dict], store: StimulusStore, output_path: str,
                   tile_size: int = 200, max_cols: int = 4):
    n_img_cols = 3
    label_w, gap, header_h = 100, 4, 30

    by_bucket = {b: [r for r in rows if r["bucket"] == b] for b in BUCKETS}
    active_buckets = [b for b in BUCKETS if by_bucket[b]]
    cols_per_bucket = min(max_cols, max(len(by_bucket[b]) for b in active_buckets))

    total_cols = cols_per_bucket * n_img_cols
    w = label_w + total_cols * tile_size + (total_cols - 1) * gap + 20
    h = header_h + len(active_buckets) * (tile_size + gap + 20) + 20
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    col_labels = []
    for _ in range(cols_per_bucket):
        col_labels.extend(["GT", "Anchor", "Diffusion"])
    for ci, lbl in enumerate(col_labels):
        x = label_w + ci * (tile_size + gap) + tile_size // 2
        draw.text((x, 5), lbl, fill="black", font=font_sm, anchor="mt")

    for bi, bucket in enumerate(active_buckets):
        y0 = header_h + bi * (tile_size + gap + 20)
        label = bucket.replace("_", " ").title()
        draw.text((5, y0 + tile_size // 2), label, fill="black", font=font, anchor="lm")
        examples = by_bucket[bucket][:cols_per_bucket]
        for ei, ex in enumerate(examples):
            gt_img = store.get_image(ex["query_nsd_id"]).resize((tile_size, tile_size), Image.LANCZOS)
            a = ex["_anchor_img"].resize((tile_size, tile_size), Image.LANCZOS)
            d = ex["_diffusion_img"].resize((tile_size, tile_size), Image.LANCZOS)
            base_col = ei * n_img_cols
            for ci, img in enumerate([gt_img, a, d]):
                x = label_w + (base_col + ci) * (tile_size + gap)
                canvas.paste(img, (x, y0))

    canvas.save(output_path, quality=95)
    logger.info(f"Composite saved to {output_path}")


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
                overall[f"{prefix}_{k}_std"] = float(np.std(finite))
                overall[f"{prefix}_{k}_q25"] = float(np.percentile(finite, 25))
                overall[f"{prefix}_{k}_q75"] = float(np.percentile(finite, 75))
    for k in metric_keys:
        rk, dk = f"retrieval_{k}_mean", f"diffusion_{k}_mean"
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
                    bm[f"{prefix}_{k}_std"] = float(np.std(finite))
        for k in metric_keys:
            rk, dk = f"retrieval_{k}_mean", f"diffusion_{k}_mean"
            if rk in bm and dk in bm:
                bm[f"delta_{k}_mean"] = bm[dk] - bm[rk]
        by_bucket[bucket] = bm

    return {"overall": overall, "by_bucket": by_bucket}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SDXL + IP-Adapter reconstruction")
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument("--stimuli-hdf5", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--hf-cache", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    ap.add_argument("--n-perfect", type=int, default=4,
                    help="Number of perfect-bucket examples to include")
    ap.add_argument("--max-per-bucket", type=int, default=None,
                    help="Max examples per non-perfect bucket (None=all available)")
    ap.add_argument("--no-ip-adapter", action="store_true",
                    help="Disable IP-Adapter (ablation)")
    ap.add_argument("--ip-adapter-scale", type=float, default=IP_ADAPTER_SCALE)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    ip_scale = args.ip_adapter_scale

    logger.info("Step 1: Computing triple fusion...")
    csls, gt_ranks, nsd_ids = compute_triple_fusion(args.repo_root, device)
    top1_gallery_idx = np.argmax(csls, axis=1).astype(np.int64)

    logger.info("Step 2: Selecting examples...")
    selected = select_examples(
        gt_ranks, nsd_ids,
        n_perfect=args.n_perfect,
        max_per_bucket=args.max_per_bucket,
        seed=args.seed,
    )
    n_total = len(selected)
    n_nonperfect = sum(1 for s in selected if s["bucket"] != "perfect")
    logger.info(f"  Selected {n_total} examples ({n_nonperfect} non-perfect)")

    logger.info("Step 3: Loading predicted CLIP embeddings (V62a)...")
    pred_clips = load_predicted_clip_embeddings(args.repo_root)

    logger.info("Step 4: Loading CLIP ViT-L/14...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(device).eval()

    logger.info("Step 5: Loading SDXL pipeline...")
    pipe = load_sdxl_pipeline(args.model_id, device, cache_dir=args.hf_cache)

    use_ip = not args.no_ip_adapter
    if use_ip:
        logger.info("Step 5b: Loading IP-Adapter...")
        load_ip_adapter(pipe, cache_dir=args.hf_cache)
        pipe.set_ip_adapter_scale(ip_scale)

    logger.info("Step 6: Loading stimulus store + metric evaluators...")
    store = StimulusStore(args.stimuli_hdf5)

    import lpips as lpips_mod
    lpips_net = lpips_mod.LPIPS(net="alex").to(device)
    alex_l2, alex_l5 = _load_alexnet_features(device)

    csv_rows: list[dict] = []
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
        csv_rows.append({k: v for k, v in ex.items() if not k.startswith("_")})

    csv_path = out_dir / "selected_examples.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)

    logger.info(f"Step 7: Running SDXL reconstruction ({n_total} examples, N={args.n_candidates})...")
    metric_rows = []
    recon_imgs_nonperfect: list[Image.Image] = []
    gt_imgs_nonperfect: list[Image.Image] = []

    for i, ex in enumerate(selected):
        qid = ex["query_nsd_id"]
        row_idx = ex["row_index"]
        anchor_nsd_id = ex["anchor_nsd_id"]

        gt_img = store.get_image(qid)
        anchor_img = store.get_image(anchor_nsd_id)
        pred_clip_768 = pred_clips[row_idx]

        strength = ex["strength"]
        guidance = ex["guidance_scale"]
        steps = ex["steps"]

        logger.info(
            f"  [{i+1}/{n_total}] NSD {qid} | bucket={ex['bucket']} | "
            f"conf={ex['confidence']:.3f} | str={strength:.3f} | "
            f"guid={guidance:.1f} | steps={steps} | rank={ex['gt_rank']}"
        )

        diffusion_img = run_sdxl_best_of_n(
            pipe, anchor_img, pred_clip_768, clip_model,
            strength=strength, guidance_scale=guidance,
            num_steps=steps, n_candidates=args.n_candidates,
            base_seed=args.seed + i * 100, device=device,
            use_ip_adapter=use_ip,
        )

        ex["_anchor_img"] = anchor_img
        ex["_diffusion_img"] = diffusion_img

        if ex["bucket"] != "perfect":
            recon_imgs_nonperfect.append(diffusion_img)
            gt_imgs_nonperfect.append(gt_img)

        ret_m = compute_all_metrics(anchor_img, gt_img, clip_model, lpips_net, alex_l2, alex_l5, device)
        dif_m = compute_all_metrics(diffusion_img, gt_img, clip_model, lpips_net, alex_l2, alex_l5, device)

        row: dict[str, Any] = {
            "query_nsd_id": qid, "anchor_nsd_id": anchor_nsd_id,
            "gt_rank": ex["gt_rank"], "bucket": ex["bucket"],
            "confidence": ex["confidence"], "strength": strength,
            "guidance_scale": guidance, "steps": steps,
            "n_candidates": args.n_candidates if strength > 0 else 0,
        }
        for k, v in ret_m.items():
            row[f"retrieval_{k}"] = v
        for k, v in dif_m.items():
            row[f"diffusion_{k}"] = v
        for k in ret_m:
            row[f"delta_{k}"] = dif_m[k] - ret_m[k]
        metric_rows.append(row)

    logger.info("Step 8: Computing n-way identification accuracy...")
    ident = {}
    for layer_name, layer in [("alexnet2", alex_l2), ("alexnet5", alex_l5)]:
        if recon_imgs_nonperfect:
            nway = compute_nway_identification(recon_imgs_nonperfect, gt_imgs_nonperfect, layer, device)
            twoway = compute_2way_identification(recon_imgs_nonperfect, gt_imgs_nonperfect, layer, device)
            ident[f"{layer_name}_nway_acc"] = nway
            ident[f"{layer_name}_2way_acc"] = twoway
            logger.info(f"  {layer_name}: {len(recon_imgs_nonperfect)}-way={nway:.4f}, 2-way={twoway:.4f}")

    logger.info("Step 9: Computing summary metrics...")
    summary = summarize_metrics(metric_rows)
    summary["method"] = "sdxl_ip_adapter_clip_conditioned_best_of_n"
    summary["model_id"] = args.model_id
    summary["seed"] = args.seed
    summary["n_candidates"] = args.n_candidates
    summary["ip_adapter_scale"] = ip_scale
    summary["ip_adapter_enabled"] = use_ip
    summary["n_way_identification"] = ident

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved metrics_summary.json")

    if metric_rows:
        with open(out_dir / "per_example.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=metric_rows[0].keys())
            w.writeheader()
            w.writerows(metric_rows)
        logger.info(f"  Saved per_example.csv")

    logger.info("Step 10: Generating composite image...")
    make_composite(selected, store, str(out_dir / "reconstruction_examples_composite.png"))

    store.close()
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
