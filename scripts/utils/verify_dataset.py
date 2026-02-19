#!/usr/bin/env python3
"""Verify NSD dataset presence/structure and print actionable setup instructions.

This repo can't redistribute NSD. By default we verify that the user mounted/copied a
local NSD mirror under NSD_DATA_ROOT in a layout compatible with the repo's default
path patterns (see `fmri2img.io.nsd_layout.NSDLayout`).

Supported layouts:
- Layout A: NSD_DATA_ROOT contains: nsddata/, nsddata_stimuli/, nsddata_betas/
- Layout B: NSD_DATA_ROOT/natural-scenes-dataset contains those folders

Exit codes:
    0: dataset looks usable
    2: missing/invalid; instructions printed
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Check:
    ok: bool
    name: str
    details: str
    fix: Optional[str] = None


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_first_existing(base: Path, rels: List[str]) -> Optional[Path]:
    for rel in rels:
        p = base / rel
        if p.exists():
            return p
    return None


def _check_file(path: Path, name: str, hint: str) -> Check:
    if path.exists():
        return Check(True, name, f"found: {path}")
    return Check(False, name, f"missing: {path}", fix=hint)


def _print_report(checks: List[Check]) -> int:
    failures = [c for c in checks if not c.ok]
    print("=" * 88)
    print("NSD dataset verification")
    print("=" * 88)
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"[{status}] {c.name}: {c.details}")
        if (not c.ok) and c.fix:
            print(f"       fix: {c.fix}")
    print("=" * 88)
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed")
        return 2
    print("PASS: dataset looks usable")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify local NSD dataset layout under NSD_DATA_ROOT")
    ap.add_argument(
        "--subject",
        default=os.environ.get("SUBJECT", "subj01"),
        help="Subject ID like subj01 (default: $SUBJECT or subj01)",
    )
    ap.add_argument(
        "--allow-s3-only",
        action="store_true",
        help="If set, don't fail when local betas are missing (for S3 streaming workflows).",
    )
    args = ap.parse_args(argv)

    nsd_root = os.environ.get("NSD_DATA_ROOT")
    if not nsd_root:
        return _print_report(
            [
                Check(
                    False,
                    "env:NSD_DATA_ROOT",
                    "NSD_DATA_ROOT is unset",
                    fix="cp .env.example .env  # set NSD_DATA_ROOT to your mounted dataset path",
                )
            ]
        )

    base = Path(nsd_root).expanduser().resolve()
    checks: List[Check] = []

    if not base.exists():
        checks.append(
            Check(
                False,
                "nsd_root_exists",
                f"NSD_DATA_ROOT does not exist: {base}",
                fix="Mount/copy the dataset and ensure NSD_DATA_ROOT points at the top-level folder.",
            )
        )
        return _print_report(checks)

    # Layout detection
    nested = base / "natural-scenes-dataset"
    if nested.exists():
        dataset_root = nested
        checks.append(Check(True, "layout", f"detected nested dataset root: {dataset_root}"))
    else:
        dataset_root = base
        checks.append(Check(True, "layout", f"detected dataset root: {dataset_root}"))

    # Core metadata required by NSDLayout defaults
    stim_info = dataset_root / "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    checks.append(
        _check_file(
            stim_info,
            "stim_info_csv",
            "Expected NSD metadata CSV at nsddata/experiments/nsd/nsd_stim_info_merged.csv",
        )
    )

    stimuli_hdf5 = dataset_root / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    checks.append(
        _check_file(
            stimuli_hdf5,
            "stimuli_hdf5",
            "Expected stimuli HDF5 at nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
        )
    )

    # Subject normalization
    subj = args.subject
    if subj.startswith("subj"):
        subj_num = int(subj[4:])
    else:
        subj_num = int(subj)
        subj = f"subj{subj_num:02d}"

    # Session design CSVs used by the canonical index builder
    design_candidates = [
        f"nsddata/ppdata/{subj}/behav/session01/session01_design.csv",
        f"nsddata/ppdata/{subj}/behav/session01/design.csv",
        f"nsddata/ppdata/{subj}/behav/session01/session01.csv",
    ]
    design_found = _find_first_existing(dataset_root, design_candidates)
    if design_found:
        checks.append(Check(True, "session_design_csv", f"found: {design_found}"))
    else:
        checks.append(
            Check(
                False,
                "session_design_csv",
                f"missing all of: {design_candidates}",
                fix="Ensure behavior/session design CSVs exist under nsddata/ppdata/<subject>/behav/sessionXX/",
            )
        )

    # Optional responses.tsv (older builders)
    responses_tsv = dataset_root / f"nsddata/ppdata/{subj}/behav/responses.tsv"
    if responses_tsv.exists():
        checks.append(Check(True, "responses_tsv", f"found: {responses_tsv}"))
    else:
        checks.append(Check(True, "responses_tsv", f"not found: {responses_tsv} (optional)"))

    # Local betas check (required for fully offline/local workflows)
    beta_local = dataset_root / (
        f"nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz"
    )
    if beta_local.exists():
        checks.append(Check(True, "betas_session01", f"found: {beta_local}"))
    else:
        if args.allow_s3_only:
            checks.append(
                Check(
                    True,
                    "betas_session01",
                    f"not found locally: {beta_local} (allowed by --allow-s3-only)",
                )
            )
        else:
            checks.append(
                Check(
                    False,
                    "betas_session01",
                    f"missing: {beta_local}",
                    fix="Mount/copy nsddata_betas for this subject, or rerun with --allow-s3-only if you intend to stream betas from S3.",
                )
            )

    if any(not c.ok for c in checks):
        print("\nNotes:")
        print("  - NSD is not redistributed by this repo; you must obtain it separately.")
        print("  - Expected under NSD_DATA_ROOT: nsddata/, nsddata_stimuli/, nsddata_betas/ (or nested under natural-scenes-dataset/)")

    return _print_report(checks)


if __name__ == "__main__":
    raise SystemExit(main())
