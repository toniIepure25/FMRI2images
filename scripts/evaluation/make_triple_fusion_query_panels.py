#!/usr/bin/env python3
"""Render 5-column qualitative retrieval panels for the triple-fusion system.

For each target NSD stimulus, produces a panel:
  Ground Truth | V61a-MCTTA16 top-1 | V62a top-1 | V66a top-1 | Triple Fusion top-1

Inputs (all produced by ``triple_fusion_shared1000_fixed.py --save-top1-dir``):
  - ``shared1000_nsd_ids.npy`` — sorted unique NSD IDs (row-order of prediction matrices)
  - ``{triple,v61a,v62a,v66a}_top1_ids.npy`` — gallery row indices of CSLS top-1
  - ``{triple,v61a,v62a,v66a}_gt_ranks.npy`` — GT CSLS rank per query
  - ``nsd_stimuli.hdf5`` — NSD stimulus images

Output: one ``query_<nsd_id>.png`` per target, plus a ``panel_metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

QUERY_NSD_IDS = [8262, 21279, 55649, 8509]

EXPERT_LABELS = {
    "v61a": "V61a-MCTTA16",
    "v62a": "V62a",
    "v66a": "V66a",
    "triple": "Triple Fusion",
}

TILE_SIZE = 480
LABEL_HEIGHT = 56
GUTTER = 12
GT_BORDER_COLOR = (66, 113, 174)       # muted blue for the ground-truth column
FUSION_BORDER_COLOR = (28, 28, 28)     # near-black for the headline triple-fusion column
EXPERT_BORDER_COLOR = (210, 210, 210)  # soft grey for the individual experts
BORDER_PX = 3


class StimulusStore:
    """Minimal HDF5 reader matching export_final_qualitatives.py."""

    def __init__(self, hdf5_path: str):
        self._h5 = h5py.File(hdf5_path, "r")
        self._dataset = self._resolve_dataset()

    def _resolve_dataset(self):
        for key in ("imgBrick", "stimuli", "images"):
            if key in self._h5:
                return self._h5[key]
        candidates = []
        self._h5.visititems(
            lambda name, obj: candidates.append(obj)
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 3
            else None
        )
        if candidates:
            return candidates[0]
        raise KeyError("Could not locate image dataset in HDF5")

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


def _try_load_font(bold: bool = False, size: int = 13) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _labeled_tile(
    img: Image.Image,
    title: str,
    annotation: str,
    *,
    is_correct: bool | None = None,
    column_role: str = "expert",
    tile_size: int = TILE_SIZE,
) -> Image.Image:
    """Render a single tile of the qualitative panel.

    column_role :
        ``"gt"``     for the ground-truth column (muted-blue border, no rank line)
        ``"fusion"`` for the headline triple-fusion column (thicker dark border)
        ``"expert"`` for the individual experts (soft grey border)
    annotation :
        Short one-line tag rendered under the title, e.g.
        ``"rank 1"`` / ``"rank 5"`` / ``"target stimulus"``.
    is_correct :
        Append a small check / cross glyph after the annotation. Pass ``None``
        for tiles where correctness is undefined (e.g. ground truth).
    """
    canvas = Image.new("RGB", (tile_size, tile_size + LABEL_HEIGHT), "white")
    thumb = ImageOps.contain(img.convert("RGB"), (tile_size, tile_size))
    paste_x = (tile_size - thumb.width) // 2
    paste_y = (tile_size - thumb.height) // 2
    canvas.paste(thumb, (paste_x, paste_y))

    draw = ImageDraw.Draw(canvas)

    # Border styled by column role
    if column_role == "gt":
        border_color = GT_BORDER_COLOR
        border_w = BORDER_PX + 1
    elif column_role == "fusion":
        border_color = FUSION_BORDER_COLOR
        border_w = BORDER_PX + 1
    else:
        border_color = EXPERT_BORDER_COLOR
        border_w = BORDER_PX
    for k in range(border_w):
        draw.rectangle((k, k, tile_size - 1 - k, tile_size - 1 - k), outline=border_color)

    # Label strip
    label_top = tile_size + 2
    title_font = _try_load_font(bold=True, size=20)
    detail_font = _try_load_font(bold=False, size=16)
    title_color = (0, 0, 0) if column_role != "expert" else (40, 40, 40)
    draw.text((8, label_top + 4), title, fill=title_color, font=title_font)

    if annotation:
        full_text = annotation
        if is_correct is True:
            full_text = f"{annotation}  ✓"
        elif is_correct is False:
            full_text = f"{annotation}  ✗"
        annot_color = (40, 110, 60) if is_correct is True else (
            (150, 50, 50) if is_correct is False else (60, 60, 60)
        )
        draw.text((8, label_top + 30), full_text, fill=annot_color, font=detail_font)

    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--top1-dir",
        type=str,
        required=True,
        help="Directory containing *_top1_ids.npy and *_gt_ranks.npy",
    )
    ap.add_argument(
        "--nsd-ids-npy",
        type=str,
        default=None,
        help="Path to shared1000_nsd_ids.npy (auto-detected from repo metrics if omitted)",
    )
    ap.add_argument(
        "--stimuli-hdf5",
        type=str,
        default=None,
        help="Path to nsd_stimuli.hdf5",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write query_<nsd_id>.png panels",
    )
    ap.add_argument(
        "--queries",
        type=int,
        nargs="+",
        default=QUERY_NSD_IDS,
        help="NSD IDs to render panels for (default: 8262 21279 55649 8509)",
    )
    args = ap.parse_args()

    nsd_ids_path = args.nsd_ids_npy
    if nsd_ids_path is None:
        candidates = [
            os.path.join(args.top1_dir, "shared1000_nsd_ids.npy"),
            os.path.join(args.top1_dir, "..", "shared1000_nsd_ids.npy"),
        ]
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates.append(
            os.path.join(repo_root, "experimental_results", "V61a_finetune_difflr", "subj01", "metrics", "shared1000_nsd_ids.npy")
        )
        for c in candidates:
            if os.path.isfile(c):
                nsd_ids_path = c
                break
        if nsd_ids_path is None:
            print("ERROR: Cannot find shared1000_nsd_ids.npy. Provide --nsd-ids-npy.", file=sys.stderr)
            return 1

    stimuli_path = args.stimuli_hdf5
    if stimuli_path is None:
        for c in [
            os.environ.get("NSD_HDF5", ""),
            "/home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
            "/home/jovyan/work/data/nsd/nsd_stimuli.hdf5",
        ]:
            if c and os.path.isfile(c):
                stimuli_path = c
                break
        if stimuli_path is None:
            print("ERROR: Cannot find nsd_stimuli.hdf5. Provide --stimuli-hdf5.", file=sys.stderr)
            return 1

    print(f"NSD IDs array: {nsd_ids_path}")
    print(f"Stimuli HDF5:  {stimuli_path}")
    print(f"Top-1 dir:     {args.top1_dir}")
    print(f"Output dir:    {args.output_dir}")

    nsd_ids = np.load(nsd_ids_path)
    print(f"SHARED1000 nsd_ids: {nsd_ids.shape} (first 5: {nsd_ids[:5]})")

    top1 = {}
    gt_ranks = {}
    for tag in ("triple", "v61a", "v62a", "v66a"):
        t1_path = os.path.join(args.top1_dir, f"{tag}_top1_ids.npy")
        rk_path = os.path.join(args.top1_dir, f"{tag}_gt_ranks.npy")
        if not os.path.isfile(t1_path):
            print(f"ERROR: Missing {t1_path}", file=sys.stderr)
            return 1
        top1[tag] = np.load(t1_path)
        if os.path.isfile(rk_path):
            gt_ranks[tag] = np.load(rk_path)
        print(f"  {tag}_top1_ids: shape={top1[tag].shape}")

    store = StimulusStore(stimuli_path)
    os.makedirs(args.output_dir, exist_ok=True)
    metadata: dict[str, dict] = {}

    for qid in args.queries:
        row_matches = np.where(nsd_ids == qid)[0]
        if len(row_matches) == 0:
            print(f"WARNING: NSD ID {qid} not found in shared1000_nsd_ids.npy, skipping.")
            continue
        row_idx = int(row_matches[0])

        gt_img = store.get_image(qid)
        tiles = [
            _labeled_tile(
                gt_img,
                "Ground truth",
                "target stimulus",
                is_correct=None,
                column_role="gt",
            )
        ]
        query_meta: dict[str, dict] = {"query_nsd_id": qid, "row_idx": row_idx, "experts": {}}

        for tag in ("v61a", "v62a", "v66a", "triple"):
            retrieved_gallery_idx = int(top1[tag][row_idx])
            retrieved_nsd_id = int(nsd_ids[retrieved_gallery_idx])
            rank = int(gt_ranks[tag][row_idx]) if tag in gt_ranks else -1
            is_correct = (retrieved_nsd_id == qid)

            expert_img = store.get_image(retrieved_nsd_id)
            label = EXPERT_LABELS[tag]
            annotation = f"rank {rank}" if rank > 0 else "rank n/a"
            column_role = "fusion" if tag == "triple" else "expert"

            tiles.append(
                _labeled_tile(
                    expert_img,
                    label,
                    annotation,
                    is_correct=is_correct,
                    column_role=column_role,
                )
            )
            query_meta["experts"][tag] = {
                "retrieved_nsd_id": retrieved_nsd_id,
                "gt_rank": rank,
                "correct": is_correct,
            }

        # Compose with a small gutter between tiles so the column borders read clearly
        n_tiles = len(tiles)
        panel_w = n_tiles * TILE_SIZE + (n_tiles - 1) * GUTTER
        panel_h = TILE_SIZE + LABEL_HEIGHT
        panel = Image.new("RGB", (panel_w, panel_h), "white")
        for i, tile in enumerate(tiles):
            panel.paste(tile, (i * (TILE_SIZE + GUTTER), 0))

        out_path = os.path.join(args.output_dir, f"query_{qid}.png")
        # dpi=300 lets LaTeX scale the panel cleanly to thesis width without resampling artefacts
        panel.save(out_path, dpi=(300, 300), optimize=True)
        print(f"  Wrote {out_path}  ({panel_w}x{panel_h}, 300 dpi)")
        metadata[str(qid)] = query_meta

    store.close()

    meta_path = os.path.join(args.output_dir, "panel_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
