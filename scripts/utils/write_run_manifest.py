#!/usr/bin/env python3
"""Create a reproducibility manifest for an output directory.

Why this exists
--------------
Several scripts already write manifests (e.g. `run_reconstruct_and_eval`), but
for paper/thesis workflows you sometimes need to *backfill* a `manifest.json`
for older runs or for ad-hoc notebooks.

This tool writes a manifest using the library implementation:

- `fmri2img.utils.manifest.gather_env_info`
- `fmri2img.utils.manifest.write_manifest`

It is intentionally conservative: it never deletes existing manifests unless
`--force` is provided.

Example
-------
Write a manifest for an existing directory and hash a few key inputs:

    python scripts/write_run_manifest.py \
      --output-dir outputs/recon/subj01/prob_fixed \
      --config subject=subj01 encoder=two_stage limit=64 \
      --hash data/indices/nsd_index/subject=subj01/index.parquet \
      --hash outputs/clip_cache/clip.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fmri2img.utils.manifest import gather_env_info, hash_file, write_manifest


def _parse_kv_list(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Invalid --config item (expected key=value): {it}")
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise SystemExit(f"Invalid --config item (empty key): {it}")
        out[k] = v
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write/backfill a reproducibility manifest.json")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory where manifest.json will be written")
    p.add_argument("--force", action="store_true", help="Overwrite manifest.json if it already exists")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add a config field (repeatable)",
    )
    p.add_argument(
        "--hash",
        dest="hash_paths",
        action="append",
        default=[],
        metavar="PATH",
        help="Hash an input file and record it in input_hashes (repeatable)",
    )
    p.add_argument(
        "--additional",
        type=Path,
        default=None,
        help="Optional JSON file to merge into additional_info",
    )

    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    if manifest_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing manifest: {manifest_path} (use --force)")

    config = _parse_kv_list(args.config)

    input_hashes: Dict[str, str] = {}
    for hp in args.hash_paths:
        path = Path(hp)
        input_hashes[str(path)] = hash_file(path)

    additional_info = None
    if args.additional is not None:
        additional_info = json.loads(args.additional.read_text())

    env = gather_env_info()

    write_manifest(
        manifest_path,
        config_dict={
            "script": "scripts/write_run_manifest.py",
            "config": config,
        },
        cli_args=None,
        env_info=env,
        input_hashes=input_hashes or None,
        additional_info=additional_info,
    )

    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
