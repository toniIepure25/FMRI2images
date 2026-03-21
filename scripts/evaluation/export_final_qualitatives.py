#!/usr/bin/env python3
"""Export final qualitative panels and retrieval-based reconstructions."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from final_best_system import (
    DEFAULT_LEGACY_RESULTS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRI_RESULTS_DIR,
    load_final_best_system_analysis,
)
from fmri2img.io.nsd_layout import NSDLayout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class StimulusStore:
    def __init__(self, hdf5_path: Path):
        self.path = Path(hdf5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Stimuli HDF5 not found: {self.path}")
        self._h5 = h5py.File(self.path, "r")
        self._dataset = self._resolve_dataset()

    def _resolve_dataset(self):
        preferred = ["imgBrick", "stimuli", "images"]
        for key in preferred:
            if key in self._h5:
                return self._h5[key]
        candidates = []
        self._h5.visititems(lambda name, obj: candidates.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
        for name, ds in candidates:
            if ds.ndim in (3, 4):
                return ds
        raise KeyError(f"Could not locate image dataset in {self.path}; tried {preferred}")

    def get_image(self, nsd_id: int) -> Image.Image:
        arr = self._dataset[int(nsd_id)]
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] != 3 and arr.shape[0] == 3:
            arr = np.moveaxis(arr, 0, -1)
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass


FALLBACK_STIMULI_CANDIDATES = [
    Path(os.environ.get("NSD_STIMULI_HDF5", "")) if os.environ.get("NSD_STIMULI_HDF5") else None,
    Path(os.environ.get("NSD_HDF5", "")) if os.environ.get("NSD_HDF5") else None,
    Path("cache/nsd_hdf5/nsd_stimuli.hdf5"),
    Path("data/nsd/nsd_stimuli.hdf5"),
    Path("/home/jovyan/work/data/nsd/nsd_stimuli.hdf5"),
    Path("/home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"),
]


def resolve_stimuli_hdf5(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    try:
        layout = NSDLayout()
        candidates.append(Path(layout.stim_hdf5_path(full_url=False)))
    except Exception:
        pass
    for candidate in FALLBACK_STIMULI_CANDIDATES:
        if candidate is not None:
            candidates.append(candidate)
    for path in candidates:
        if path and path.exists():
            return path
    tried = "\n".join(f"  - {p}" for p in candidates if p)
    raise FileNotFoundError(
        "Could not resolve nsd_stimuli.hdf5. Provide --stimuli-hdf5 or set NSD_STIMULI_HDF5.\n"
        f"Tried:\n{tried}"
    )


def _draw_text_block(draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[str]) -> None:
    offset = 0
    for line in lines:
        draw.text((x, y + offset), line, fill=(20, 20, 20))
        offset += 14


def _labeled_tile(img: Image.Image, title: str, lines: list[str], tile_size: int = 256) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size + 74), "white")
    thumb = ImageOps.contain(img.convert("RGB"), (tile_size, tile_size))
    paste_x = (tile_size - thumb.width) // 2
    paste_y = (tile_size - thumb.height) // 2
    canvas.paste(thumb, (paste_x, paste_y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, tile_size, tile_size, tile_size + 74), fill=(248, 248, 248))
    draw.text((8, tile_size + 6), title, fill=(0, 0, 0))
    _draw_text_block(draw, 8, tile_size + 22, lines)
    return canvas


def _topk_strip(store: StimulusStore, nsd_ids: list[int], tile_size: int = 96) -> Image.Image:
    images = []
    for nid in nsd_ids:
        img = ImageOps.contain(store.get_image(int(nid)), (tile_size, tile_size))
        tile = Image.new("RGB", (tile_size, tile_size + 18), "white")
        tile.paste(img, ((tile_size - img.width) // 2, (tile_size - img.height) // 2))
        ImageDraw.Draw(tile).text((4, tile_size + 2), str(int(nid)), fill=(0, 0, 0))
        images.append(tile)
    strip = Image.new("RGB", (len(images) * tile_size, tile_size + 18), "white")
    for i, tile in enumerate(images):
        strip.paste(tile, (i * tile_size, 0))
    return strip


def _case_dirs(base: Path, split: str) -> dict[str, Path]:
    dirs = {
        "best": base / "qualitatives" / split / "best_cases",
        "medium": base / "qualitatives" / split / "medium_cases",
        "hard": base / "qualitatives" / split / "hard_cases",
        "recon": base / "reconstructions" / split,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    (base / "qualitatives" / "contact_sheets").mkdir(parents=True, exist_ok=True)
    return dirs


def _select_examples(rows: list[dict[str, Any]], total: int) -> dict[str, list[dict[str, Any]]]:
    best = [r for r in rows if r["fused_fixed_to_top1"]]
    best.sort(key=lambda r: (r["compact_gt_rank"] + r["legacy_gt_rank"] - 2 * r["fused_gt_rank"], r["nsd_id"]), reverse=True)

    medium = [r for r in rows if 2 <= r["fused_gt_rank"] <= 10 and not r["fused_fixed_to_top1"]]
    medium.sort(key=lambda r: (abs(r["fused_gt_rank"] - 5), r["nsd_id"]))

    hard = [r for r in rows if r["fused_gt_rank"] > 10]
    hard.sort(key=lambda r: (r["fused_gt_rank"], r["compact_gt_rank"], r["legacy_gt_rank"]), reverse=True)

    target_best = min(8, total)
    target_medium = min(6, max(total - target_best, 0))
    target_hard = min(6, max(total - target_best - target_medium, 0))

    selected_best = best[:target_best]
    selected_medium = medium[:target_medium]
    selected_hard = hard[:target_hard]

    used = {r["nsd_id"] for r in selected_best + selected_medium + selected_hard}
    if len(used) < total:
        remaining = [r for r in rows if r["nsd_id"] not in used]
        remaining.sort(key=lambda r: (r["fused_gt_rank"], r["compact_gt_rank"], r["legacy_gt_rank"]))
        for row in remaining[: total - len(used)]:
            selected_medium.append(row)
    return {"best": selected_best, "medium": selected_medium, "hard": selected_hard}


def _contact_sheet(panel_paths: list[Path], output_path: Path, cols: int = 4) -> None:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _render_split(split: str, rows: list[dict[str, Any]], store: StimulusStore, output_dir: Path, total: int) -> None:
    dirs = _case_dirs(output_dir, split)
    selected = _select_examples(rows, total)
    sheet_inputs: list[Path] = []

    for category, items in selected.items():
        category_dir = dirs[category]
        for row in items:
            gt = store.get_image(int(row["nsd_id"]))
            compact = store.get_image(int(row["compact_top1_nsd_id"]))
            legacy = store.get_image(int(row["legacy_top1_nsd_id"]))
            fused = store.get_image(int(row["fused_top1_nsd_id"]))
            strip = _topk_strip(store, [int(x) for x in row["fused_top5_nsd_ids"]])

            tiles = [
                _labeled_tile(gt, "Ground Truth", [f"query nsd_id={row['nsd_id']}", f"fused fixed={row['fused_fixed_case']}"]),
                _labeled_tile(compact, "Compact Top-1", [f"GT rank={row['compact_gt_rank']}", f"top1={row['compact_top1_nsd_id']}"]),
                _labeled_tile(legacy, "Legacy Top-1", [f"GT rank={row['legacy_gt_rank']}", f"top1={row['legacy_top1_nsd_id']}"]),
                _labeled_tile(fused, "Fused Top-1", [f"GT rank={row['fused_gt_rank']}", f"top1={row['fused_top1_nsd_id']}"]),
            ]
            panel = Image.new("RGB", (4 * tiles[0].width, tiles[0].height + strip.height + 16), "white")
            for i, tile in enumerate(tiles):
                panel.paste(tile, (i * tile.width, 0))
            panel.paste(strip, (0, tiles[0].height + 8))
            panel_path = category_dir / f"query_{int(row['nsd_id'])}.png"
            panel.save(panel_path)
            sheet_inputs.append(panel_path)

            recon_dir = dirs["recon"]
            fused.save(recon_dir / f"query_{int(row['nsd_id'])}_top1_reconstruction.png")
            recon_panel = Image.new("RGB", (2 * tiles[0].width, tiles[0].height + strip.height + 16), "white")
            recon_panel.paste(_labeled_tile(gt, "Ground Truth", [f"nsd_id={row['nsd_id']}"]), (0, 0))
            recon_panel.paste(_labeled_tile(fused, "Top-1 Retrieval Reconstruction", [f"fused rank={row['fused_gt_rank']}", f"retrieved={row['fused_top1_nsd_id']}"]), (tiles[0].width, 0))
            recon_panel.paste(strip, (0, tiles[0].height + 8))
            recon_panel.save(recon_dir / f"query_{int(row['nsd_id'])}_comparison.png")

    _contact_sheet(sheet_inputs[: min(len(sheet_inputs), total)], output_dir / "qualitatives" / "contact_sheets" / f"{split}_overview.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tri-results-dir", type=Path, default=DEFAULT_TRI_RESULTS_DIR)
    parser.add_argument("--legacy-results-dir", type=Path, default=DEFAULT_LEGACY_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stimuli-hdf5", type=Path, default=None)
    parser.add_argument("--examples-per-split", type=int, default=20)
    parser.add_argument("--allow-config-drift", action="store_true")
    args = parser.parse_args()

    system, val_analysis, shared_analysis = load_final_best_system_analysis(
        tri_results_dir=args.tri_results_dir,
        legacy_results_dir=args.legacy_results_dir,
        strict=not args.allow_config_drift,
    )
    stimuli_hdf5 = resolve_stimuli_hdf5(args.stimuli_hdf5)
    logger.info("Using stimuli HDF5: %s", stimuli_hdf5)

    store = StimulusStore(stimuli_hdf5)
    try:
        _render_split("val", val_analysis.per_query, store, args.output_dir, args.examples_per_split)
        _render_split("shared1000", shared_analysis.per_query, store, args.output_dir, args.examples_per_split)
    finally:
        store.close()

    manifest = {
        "tri_results_dir": str(system.tri_results_dir),
        "legacy_results_dir": str(system.legacy_results_dir),
        "stimuli_hdf5": str(stimuli_hdf5),
        "examples_per_split": args.examples_per_split,
    }
    manifest_path = args.output_dir / "qualitatives" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved final qualitative exports to %s", args.output_dir)


if __name__ == "__main__":
    main()
