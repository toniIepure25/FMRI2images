#!/usr/bin/env python3
"""
Prepare demo data for Cortex2Canvas.

Searches the repository for existing reconstruction results,
copies them into the demo's public/assets directory, and generates
placeholder images for any missing assets.

Usage:
    python prepare_demo_data.py [--repo-root /path/to/repo]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Attempt to import PIL for placeholder generation
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = ImageDraw = ImageFilter = ImageFont = None  # type: ignore[misc, assignment]


def _demo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_repo_root() -> Path:
    return _demo_root().parent


DEMO_ROOT = _demo_root()
REPO_ROOT = _default_repo_root()
PUBLIC_ASSETS = DEMO_ROOT / "public" / "assets" / "cases"
DATA_DIR = DEMO_ROOT / "data"

DIFFICULTY_BY_SUBDIR = {
    "best_cases": "best",
    "medium_cases": "medium",
    "hard_cases": "hard",
}


def _build_result_dirs(repo_root: Path) -> dict[str, Path]:
    """Canonical paths under reconstruction_results/."""
    base = repo_root / "reconstruction_results"
    return {
        "best": base / "best_cases",
        "medium": base / "medium_cases",
        "hard": base / "hard_cases",
    }


RESULT_DIRS: dict[str, Path] = _build_result_dirs(REPO_ROOT)


def discover_result_dirs(repo_root: Path, max_depth: int = 6) -> dict[str, Path]:
    """
    Resolve result folders: prefer reconstruction_results/*, else first match under repo.

    Limits directory walking depth to keep the search cheap on large trees.
    """
    canonical = _build_result_dirs(repo_root)
    if all(p.exists() and p.is_dir() for p in canonical.values()):
        return canonical

    found: dict[str, Path] = {}
    repo_root = repo_root.resolve()
    for root, dirnames, _ in _walk_limited(repo_root, max_depth):
        for d in list(dirnames):
            key = DIFFICULTY_BY_SUBDIR.get(d)
            if key and key not in found:
                candidate = root / d
                if candidate.is_dir():
                    found[key] = candidate.resolve()
        if len(found) == 3:
            break

    out: dict[str, Path] = {}
    for key in ("best", "medium", "hard"):
        c = canonical[key]
        out[key] = c if c.is_dir() else found.get(key, c)
    return out


def _walk_limited(start: Path, max_depth: int):
    """Fallback depth-limited walk for Python < 3.12."""
    start = start.resolve()
    stack = [(start, 0)]
    while stack:
        root, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        dirnames = [e.name for e in entries if e.is_dir()]
        yield root, dirnames, [e.name for e in entries if e.is_file()]
        for e in entries:
            if e.is_dir():
                stack.append((e, depth + 1))


def load_demo_cases() -> list[dict[str, Any]]:
    cases_path = DATA_DIR / "demo_cases.json"
    if not cases_path.exists():
        print(f"Error: {cases_path} not found", file=sys.stderr)
        sys.exit(1)
    try:
        with open(cases_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {cases_path}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"Error: {cases_path} must contain a JSON array of cases", file=sys.stderr)
        sys.exit(1)
    return data


def _subtle_line_color(bg: tuple[int, int, int], accent: tuple[int, int, int]) -> tuple[int, int, int]:
    """RGB grid line color (no alpha — RGB mode)."""
    return (
        min(255, (bg[0] * 3 + accent[0]) // 4),
        min(255, (bg[1] * 3 + accent[1]) // 4),
        min(255, (bg[2] * 3 + accent[2]) // 4),
    )


def create_placeholder_image(
    path: Path,
    label: str,
    img_type: str = "target",
    size: tuple[int, int] = (256, 256),
) -> None:
    """Create a professional-looking placeholder image."""
    if not HAS_PIL:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    colors: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
        "target": ((20, 60, 40), (0, 180, 120)),
        "reconstruction": ((40, 20, 60), (140, 80, 220)),
        "retrieved": ((20, 30, 60), (60, 120, 220)),
        "ensemble": ((60, 40, 20), (220, 160, 60)),
        "distractor": ((40, 40, 40), (120, 120, 140)),
    }
    bg, accent = colors.get(img_type, colors["target"])

    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    for y in range(size[1]):
        t = y / max(size[1], 1)
        r = int(bg[0] * (1 - t) + accent[0] * t * 0.3)
        g = int(bg[1] * (1 - t) + accent[1] * t * 0.3)
        b = int(bg[2] * (1 - t) + accent[2] * t * 0.3)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))

    grid = _subtle_line_color(bg, accent)
    for x in range(0, size[0], 16):
        draw.line([(x, 0), (x, size[1])], fill=grid, width=1)
    for yy in range(0, size[1], 16):
        draw.line([(0, yy), (size[0], yy)], fill=grid, width=1)

    cx, cy = size[0] // 2, size[1] // 2
    r = min(size) // 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=accent, width=2)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    type_label = img_type.upper()
    bbox = draw.textbbox((0, 0), type_label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((size[0] - tw) // 2, 10), type_label, fill=accent, font=font)

    short_label = label[:30]
    bbox = draw.textbbox((0, 0), short_label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(
        ((size[0] - tw) // 2, size[1] - 30),
        short_label,
        fill=(180, 180, 180),
        font=font,
    )

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img.save(path, format="PNG")


def copy_or_generate_case_assets(case_data: dict[str, Any], result_dirs: dict[str, Path]) -> None:
    """For a case, try to copy real images or generate placeholders."""
    case_id = case_data["id"]
    difficulty = case_data.get("difficulty", "")
    case_dir = PUBLIC_ASSETS / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    if difficulty in result_dirs:
        panel_path = result_dirs[difficulty] / f"{case_id}.png"
        if panel_path.is_file():
            dest = case_dir / "target.png"
            try:
                if not dest.exists():
                    shutil.copy2(panel_path, dest)
                    print(f"  Copied real panel: {panel_path.name}")
            except OSError as e:
                print(f"  Warning: could not copy {panel_path}: {e}", file=sys.stderr)

    assets: list[tuple[str, str, str]] = [
        ("target.png", "target", f"Target: {case_id}"),
        ("reconstruction.png", "reconstruction", f"Recon: {case_id}"),
        ("recon_conservative.png", "reconstruction", "Conservative"),
        ("recon_creative.png", "reconstruction", "Creative"),
    ]

    if not HAS_PIL:
        for filename, _, _ in assets:
            path = case_dir / filename
            if not path.exists():
                print(f"  Skip placeholder (no Pillow): {filename}")
        for i in range(1, 6):
            path = case_dir / f"retrieved_{i}.png"
            if not path.exists():
                print(f"  Skip placeholder (no Pillow): retrieved_{i}.png")
        for i in range(1, 4):
            path = case_dir / f"ensemble_{i}.png"
            if not path.exists():
                print(f"  Skip placeholder (no Pillow): ensemble_{i}.png")
        return

    for filename, img_type, label in assets:
        path = case_dir / filename
        if not path.exists():
            try:
                create_placeholder_image(path, label, img_type)
                print(f"  Generated placeholder: {filename}")
            except OSError as e:
                print(f"  Warning: could not write {path}: {e}", file=sys.stderr)

    for i in range(1, 6):
        path = case_dir / f"retrieved_{i}.png"
        if not path.exists():
            try:
                create_placeholder_image(path, f"Retrieved #{i}", "retrieved")
            except OSError as e:
                print(f"  Warning: could not write {path}: {e}", file=sys.stderr)

    for i in range(1, 4):
        path = case_dir / f"ensemble_{i}.png"
        if not path.exists():
            try:
                create_placeholder_image(path, f"Ensemble #{i}", "ensemble")
            except OSError as e:
                print(f"  Warning: could not write {path}: {e}", file=sys.stderr)


def generate_distractors() -> None:
    """Generate shared distractor images."""
    dist_dir = PUBLIC_ASSETS / "distractors"
    dist_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        "distractor_A",
        "distractor_B",
        "distractor_C",
        "distractor_D",
        "distractor_E",
        "distractor_F",
    ]
    for label in labels:
        path = dist_dir / f"{label}.png"
        if path.exists():
            continue
        if not HAS_PIL:
            print(f"  Skip distractor (no Pillow): {label}.png")
            continue
        try:
            create_placeholder_image(path, label, "distractor")
            print(f"  Generated distractor: {label}.png")
        except OSError as e:
            print(f"  Warning: could not write {path}: {e}", file=sys.stderr)


def validate_data_files() -> bool:
    """Check that all required JSON data files exist."""
    required = [
        "demo_cases.json",
        "roi_layout.json",
        "clip_projection.json",
        "metrics_summary.json",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        print(f"Warning: Missing data files: {', '.join(missing)}")
    else:
        print("All data files present.")
    return len(missing) == 0


def copy_data_to_public() -> None:
    """Copy data JSON files to public/ so Vite serves them."""
    public_data = DEMO_ROOT / "public" / "data"
    public_data.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.is_dir():
        print(f"  Warning: data directory missing: {DATA_DIR}", file=sys.stderr)
        return
    copied = 0
    for f in sorted(DATA_DIR.glob("*.json")):
        dest = public_data / f.name
        try:
            shutil.copy2(f, dest)
            print(f"  Copied {f.name} -> public/data/")
            copied += 1
        except OSError as e:
            print(f"  Warning: could not copy {f} -> {dest}: {e}", file=sys.stderr)
    if copied == 0:
        print("  Warning: no JSON files copied (data/ empty or missing)", file=sys.stderr)


def _preflight_pillow() -> None:
    if not HAS_PIL:
        print(
            "Warning: Pillow not installed. Placeholder images will be skipped.\n"
            "Install with: pip install Pillow",
            file=sys.stderr,
        )


def main() -> None:
    global REPO_ROOT, RESULT_DIRS

    parser = argparse.ArgumentParser(description="Prepare Cortex2Canvas demo data")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of demo_thesis)",
    )
    args = parser.parse_args()

    repo_root = (args.repo_root if args.repo_root is not None else _default_repo_root()).resolve()
    REPO_ROOT = repo_root
    RESULT_DIRS = discover_result_dirs(REPO_ROOT)

    print("=" * 60)
    print("Cortex2Canvas — Demo Data Preparation")
    print("=" * 60)
    print(f"Demo root:  {DEMO_ROOT}")
    print(f"Repo root:  {REPO_ROOT}")
    print("Result dirs:")
    for k, p in RESULT_DIRS.items():
        status = "ok" if p.is_dir() else "missing"
        print(f"  [{status}] {k}: {p}")
    print()

    _preflight_pillow()

    print("[1/4] Validating data files...")
    validate_data_files()
    print()

    print("[2/4] Copying data to public/...")
    copy_data_to_public()
    print()

    print("[3/4] Processing case assets...")
    cases = load_demo_cases()
    for case in cases:
        print(f"\n  Case: {case['id']} ({case.get('difficulty', '?')})")
        copy_or_generate_case_assets(case, RESULT_DIRS)
    print()

    print("[4/4] Generating distractor images...")
    generate_distractors()
    print()

    print("=" * 60)
    print("Done! Demo data is ready.")
    print(f"Assets: {PUBLIC_ASSETS}")
    print(f"Data:   {DEMO_ROOT / 'public' / 'data'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
