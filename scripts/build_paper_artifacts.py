#!/usr/bin/env python3
"""Build small, paper-facing summary artifacts.

This is intentionally lightweight:

- It does *not* run training or reconstruction.
- It summarizes existing evaluation outputs under `outputs/reports/**`.

Outputs
-------
Creates `outputs/paper/` with:

- `eval_summary.csv` — aggregated table over all found `recon_eval_summary.json`

The goal is to make it easy to produce consistent tables for a thesis/paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _iter_summary_json(report_root: Path) -> List[Path]:
    if not report_root.exists():
        return []
    return sorted(report_root.rglob("recon_eval_summary.json"))


def _flatten(prefix: str, d: Dict[str, Any], out: Dict[str, Any]) -> None:
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(key + ".", v, out)
        else:
            out[key] = v


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build paper artifacts from existing outputs")
    p.add_argument("--reports-root", type=Path, default=Path("outputs/reports"), help="Root containing report runs")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/paper"), help="Output directory")

    args = p.parse_args(argv)

    summaries = _iter_summary_json(args.reports_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not summaries:
        raise SystemExit(
            f"No recon_eval_summary.json files found under {args.reports_root}. "
            "Run an evaluation first (see scripts/eval_reconstruction.py or run_reconstruct_and_eval)."
        )

    rows: List[Dict[str, Any]] = []
    for sp in summaries:
        try:
            data = json.loads(sp.read_text())
        except Exception as e:
            raise SystemExit(f"Failed to read {sp}: {e}")

        row: Dict[str, Any] = {
            "path": str(sp),
        }
        _flatten("", data, row)
        rows.append(row)

    # Write a minimal CSV without dependencies (pandas optional)
    out_csv = args.out_dir / "eval_summary.csv"

    # Determine columns: stable ordering (path first)
    cols = ["path"]
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)

    def _csv_escape(x: Any) -> str:
        s = "" if x is None else str(x)
        if any(c in s for c in [",", "\n", '"']):
            s = '"' + s.replace('"', '""') + '"'
        return s

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_csv_escape(r.get(c)) for c in cols) + "\n")

    print(str(out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
