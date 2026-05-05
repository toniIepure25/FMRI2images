#!/usr/bin/env python3
"""Extract assets from qualitative comparison panels into public/assets/cases.

Reads ``data/demo_cases.json``, finds each case's panel under ``best_system`` or
copies pre-rendered tiles from ``true_diffusion_addon``, writes standardized
filenames, updates JSON fields, and mirrors JSON to ``public/data/``.

Layout detection assumes mosaic panels (~1024×460) with four top quarters and a
bottom top-5 row. Main crops (target, reconstruction) aim for ≥100px edges;
thumbnail row cells are often ~96px tall, so validation uses a lower threshold.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "final_outputs"
BEST = RESULTS / "best_system"
QUAL = BEST / "qualitatives"
RECON_SHARED = BEST / "reconstructions" / "shared1000"
ADDON_ROOT = RESULTS / "true_diffusion_addon" / "reconstructions"
ADDON_BUCKETS = ("perfect", "good", "near_good", "hard")
CASE_BUCKETS = ("best_cases", "medium_cases", "hard_cases")
QUAL_LOCS = ("shared1000", "val")
DEMO_CASES_SRC = ROOT / "data" / "demo_cases.json"
DEMO_CASES_PUBLIC = ROOT / "public" / "data" / "demo_cases.json"

MIN_MAIN_DIM = 100
MIN_THUMB_DIM = 80  # bottom row is short (~95px tall) and may be <100px high


def find_qualitative_panel(case_id: str) -> Path | None:
    for loc in QUAL_LOCS:
        for sub in CASE_BUCKETS:
            p = QUAL / loc / sub / f"{case_id}.png"
            if p.is_file():
                return p
    return None


def find_addon_dir(case_id: str) -> Path | None:
    stem = f"{case_id}_ground_truth.png"
    for bucket in ADDON_BUCKETS:
        if (ADDON_ROOT / bucket / stem).is_file():
            return ADDON_ROOT / bucket
    return None


def _to_rgb_array(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float32)


def _row_std_per_band(gray: np.ndarray, x0: int, x1: int) -> np.ndarray:
    band = gray[:, x0:x1]
    h = gray.shape[0]
    return band.reshape(h, -1).std(axis=1)


def detect_main_image_bottom(gray: np.ndarray, w: int, h: int) -> int:
    """Last row (exclusive upper bound) where main photo pixels still have texture."""
    y_cap = min(h - 1, int(0.58 * h))
    last_y = 0
    for col in range(4):
        x0 = col * w // 4
        x1 = (col + 1) * w // 4
        std = _row_std_per_band(gray, x0, x1)
        for y in range(y_cap, -1, -1):
            if std[y] > 20.0:
                last_y = max(last_y, y)
                break
    fallback = int(0.55 * h)
    return max(32, min(last_y + 1, int(0.62 * h))) if last_y > 0 else fallback


def detect_thumb_vertical_range(gray: np.ndarray, h: int, w: int) -> tuple[int, int]:
    """Estimate top and bottom (exclusive) of the bottom thumbnail row.

    The strongest horizontal edge in the lower panel (skipping the top mosaic)
    separates the gutter from the top-K thumbnails; ``y0`` is the first thumb row.
    """
    gy = gray.mean(axis=1)
    row_diff = np.abs(np.diff(gy.astype(np.float32)))

    search_lo = max(1, int(0.58 * h))
    search_hi = min(h - 2, int(0.92 * h))
    if search_lo >= search_hi:
        search_lo, search_hi = int(0.55 * h), min(h - 2, int(0.95 * h))

    edge = search_lo + int(np.argmax(row_diff[search_lo:search_hi]))
    y0 = min(h - 2, edge + 1)

    # Bottom: last row before a sustained white/footer run.
    y1 = min(h, y0 + max(60, int(0.32 * h)))
    white_run = 0
    for y in range(y0, h):
        row = gray[y, :]
        if row.mean() > 252.0 and row.std() < 3.0:
            white_run += 1
            if white_run >= 4:
                y1 = y - white_run + 1
                break
        else:
            white_run = 0
            y1 = y + 1

    # Ratio fallback if detection collapsed (very tall/short panels).
    if y1 - y0 < 40:
        y0_fb = int(0.72 * h)
        y1_fb = int(min(h, 0.96 * h))
        if y1_fb > y0_fb + 40:
            y0, y1 = y0_fb, y1_fb

    y1 = max(y0 + MIN_MAIN_DIM // 2, min(y1, h))
    return y0, y1


def trim_margin_rgba(im: Image.Image, border: int = 2) -> Image.Image:
    """Crop near-white margins from RGB/RGBA tiles."""
    arr = np.asarray(im.convert("RGB"))
    mask = (arr < 245).any(axis=2)
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    b = border
    y0 = max(0, y0 - b)
    x0 = max(0, x0 - b)
    y1 = min(im.height, y1 + b)
    x1 = min(im.width, x1 + b)
    return im.crop((x0, y0, x1, y1))


def crop_panel_tiles(panel: Image.Image) -> dict[str, Image.Image]:
    """Split a comparison panel into target, compact col, fused, strip, and 5 thumbs."""
    gray = _to_rgb_array(panel).mean(axis=2)
    h, w = gray.shape
    y_main_end = detect_main_image_bottom(gray, w, h)
    y_thumb0, y_thumb1 = detect_thumb_vertical_range(gray, h, w)

    out: dict[str, Image.Image] = {}
    qu = w // 4

    # Row 1 quarters: trim label bands via texture-based y_main_end.
    out["target"] = trim_margin_rgba(panel.crop((0 * qu, 0, 1 * qu, y_main_end)))
    out["fused_top1"] = trim_margin_rgba(panel.crop((3 * qu, 0, 4 * qu, y_main_end)))

    strip = panel.crop((0, y_thumb0, w, y_thumb1))
    out["topk_strip"] = strip.copy()

    tw = max(1, w // 5)
    for i in range(5):
        x0, x1 = i * tw, (i + 1) * tw if i < 4 else w
        # Do not trim thumbs: internal whitespace can erase narrow cells.
        out[f"thumb_{i + 1}"] = strip.crop((x0, 0, x1, strip.height))

    return out


def split_addon_topk_strip(strip_path: Path) -> list[Image.Image]:
    im = Image.open(strip_path).convert("RGB")
    w, h = im.size
    tw = max(1, w // 5)
    thumbs: list[Image.Image] = []
    for i in range(5):
        x0, x1 = i * tw, (i + 1) * tw if i < 4 else w
        thumbs.append(im.crop((x0, 0, x1, h)))
    return thumbs


def record_size(
    log: list[str],
    label: str,
    path: Path,
    im: Image.Image | None,
    *,
    min_dim: int = MIN_MAIN_DIM,
) -> None:
    if im is None:
        log.append(f"    [{label}] (missing)")
        return
    w, h = im.size
    status = "ok" if min(w, h) >= min_dim else "WARN small"
    log.append(f"    [{label}] {w}x{h}  {status}  -> {path.relative_to(ROOT)}")


def process_addon_cases(case_id: str, dest_dir: Path, log: list[str]) -> bool:
    addon = find_addon_dir(case_id)
    if addon is None:
        return False

    log.append(f"  mode: true_diffusion_addon ({addon.name})")

    def cp(name: str, src_name: str) -> None:
        src = addon / src_name
        dst = dest_dir / name
        if not src.is_file():
            log.append(f"  skip missing {src_name}")
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        with Image.open(dst) as im:
            mdim = MIN_THUMB_DIM if name.startswith("topk_strip") else MIN_MAIN_DIM
            record_size(log, name, dst, im, min_dim=mdim)

    cp("target.png", f"{case_id}_ground_truth.png")
    cp("retrieved_1.png", f"{case_id}_retrieval_top1.png")
    cp("reconstruction.png", f"{case_id}_diffusion_final.png")
    cp("diffusion_final.png", f"{case_id}_diffusion_final.png")
    cp("diffusion_prior.png", f"{case_id}_diffusion_prior.png")
    cp("topk_strip.png", f"{case_id}_topk_strip.png")

    strip_path = addon / f"{case_id}_topk_strip.png"
    if strip_path.is_file():
        thumbs = split_addon_topk_strip(strip_path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for rank in range(2, 6):
            thumb = thumbs[rank - 1]
            dst = dest_dir / f"retrieved_{rank}.png"
            thumb.save(dst)
            record_size(log, f"retrieved_{rank}.png", dst, thumb, min_dim=MIN_THUMB_DIM)

    return True


def process_panel_case(case_id: str, dest_dir: Path, log: list[str]) -> bool:
    panel_path = find_qualitative_panel(case_id)
    if panel_path is None:
        return False

    log.append(f"  panel: {panel_path.relative_to(ROOT)}")
    panel = Image.open(panel_path).convert("RGB")
    tiles = crop_panel_tiles(panel)
    dest_dir.mkdir(parents=True, exist_ok=True)

    tiles["target"].save(dest_dir / "target.png")
    record_size(log, "target.png", dest_dir / "target.png", tiles["target"])

    top1_file = RECON_SHARED / f"{case_id}_top1_reconstruction.png"
    if top1_file.is_file():
        shutil.copy2(top1_file, dest_dir / "reconstruction.png")
        with Image.open(dest_dir / "reconstruction.png") as recon_im:
            record_size(
                log,
                "reconstruction.png (clean top1 file)",
                dest_dir / "reconstruction.png",
                recon_im,
            )
    else:
        tiles["fused_top1"].save(dest_dir / "reconstruction.png")
        record_size(
            log,
            "reconstruction.png (panel fused col)",
            dest_dir / "reconstruction.png",
            tiles["fused_top1"],
        )

    tiles["topk_strip"].save(dest_dir / "topk_strip.png")
    record_size(
        log, "topk_strip.png", dest_dir / "topk_strip.png", tiles["topk_strip"], min_dim=MIN_THUMB_DIM
    )

    # retrieved_1..5 from bottom row (rank order); matches demo retrievedImages.
    for k in range(1, 6):
        key = f"thumb_{k}"
        dst = dest_dir / f"retrieved_{k}.png"
        tiles[key].save(dst)
        record_size(log, f"retrieved_{k}.png", dst, tiles[key], min_dim=MIN_THUMB_DIM)

    panel.close()
    return True


def update_case_json(case: dict, case_id: str) -> None:
    base = f"/assets/cases/{case_id}"
    case["targetImage"] = f"{base}/target.png"
    case["reconstructionImage"] = f"{base}/reconstruction.png"
    case["conservativeRecon"] = f"{base}/reconstruction.png"
    case["creativeRecon"] = f"{base}/reconstruction.png"
    if case.get("retrievedImages"):
        for item in case["retrievedImages"]:
            r = item.get("rank")
            if r is not None:
                item["image"] = f"{base}/retrieved_{int(r)}.png"
    for key in ("topkStrip", "diffusionPrior", "diffusionFinal"):
        if key in case:
            fname = {
                "topkStrip": "topk_strip.png",
                "diffusionPrior": "diffusion_prior.png",
                "diffusionFinal": "diffusion_final.png",
            }[key]
            case[key] = f"{base}/{fname}"


def main() -> int:
    if not DEMO_CASES_SRC.is_file():
        print(f"Missing {DEMO_CASES_SRC}", file=sys.stderr)
        return 1

    with open(DEMO_CASES_SRC, encoding="utf-8") as f:
        cases: list[dict] = json.load(f)

    for case in cases:
        case_id = case["id"]
        dest_dir = ROOT / "public" / "assets" / "cases" / case_id
        log: list[str] = [f"{case_id}:"]

        done = process_addon_cases(case_id, dest_dir, log)
        if not done:
            if not process_panel_case(case_id, dest_dir, log):
                log.append("  (no addon dir and no qualitative panel found)")

        update_case_json(case, case_id)
        print("\n".join(log))

    DEMO_CASES_PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_CASES_SRC, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
        f.write("\n")
    shutil.copy2(DEMO_CASES_SRC, DEMO_CASES_PUBLIC)

    print("\nWrote", DEMO_CASES_SRC, "and", DEMO_CASES_PUBLIC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
