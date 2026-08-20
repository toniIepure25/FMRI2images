"""Single-command reproducible runner for the Roy S1 subj01 x V1 x B0 x D0 smoke.

Regenerates trial table, ROI/SNR selection, splits, pairings, extraction, value
QC, fit/select/evaluate, and writes all manifests + a resolved config. Returns a
non-zero exit code on any invariant failure.

The numbers are NON-INTERPRETIVE (pipeline validation only); they are never
compared to the Roy paper here.

Historical replay (P0/I0 are split-deterministic -- no pairing seed):
    python scripts/analysis/mindcompiler/run_roy_s1_smoke.py \
      --split-seed 1234 \
      --vis2vis-pairing all-ordered-distinct \
      --vis2img-pairing historical-index-aligned \
      --output-dir artifacts/mindcompiler/roy_s1/replay_13cc3f0

Child-seeded sensitivity variants (P1/I1) REQUIRE --pairing-seed:
    ... --vis2vis-pairing derangement --pairing-seed 1234
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from fmri2img.mindcompiler.roy_method_reproduction import pairing as pr  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.fitted_pipeline import (  # noqa: E402
    FittedPipeline,
)
from fmri2img.mindcompiler.roy_method_reproduction.metrics import FINITE_FRACTION_MIN  # noqa: E402

# Fixed beta-version registry. Each version is SHA-pinned (integrity: the runner
# refuses to proceed unless the on-disk HDF5 matches). subj01 x V1 x D0 stay fixed;
# only the beta ESTIMATION differs across versions. Voxel selection uses the shared
# NSD-core ncsnr (betas_fithrf/ncsnr), so voxels/splits/pairings are identical
# across versions and only the beta VALUES differ (a controlled comparison).
BETA_VERSIONS = {
    "fithrf": {
        "subdir": "nsdimagerybetas_fithrf",
        "sha256": "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64",
        "label": "B0",
    },
    "fithrf_GLMdenoise_RR": {
        "subdir": "nsdimagerybetas_fithrf_GLMdenoise_RR",
        "sha256": "cd42e680617d6564f403c7855140c53c0a2fdb9eb382755b49765bf781c4af2e",
        "label": "B1",
    },
}
B0_SHA = BETA_VERSIONS["fithrf"]["sha256"]  # back-compat alias

# CLI policy label -> canonical pairing.py policy name.
_VV_POLICY = {"all-ordered-distinct": "all_ordered_distinct",
              "derangement": "deterministic_derangement"}
_VI_POLICY = {"historical-index-aligned": "historical_index_aligned",
              "independent-permutation": "independent_within_identity_permutation"}


def _sha(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fit_eval(M, Xr_tr, Yr_tr, Xr_va, Yr_va, Xr_te, Yr_te,
             voxel_hash="", trial_table_hash="", rank_max=None) -> dict:
    """Single authoritative execution path via the leakage-safe FittedPipeline.

    Historical policy HISTORICAL_SMOKE_TRAIN_ONLY_CENTERING_V1: train-only
    centering, no scaling, validation-only selection, train-only final refit,
    one-shot test evaluation. Leakage is prevented structurally by the pipeline's
    lifecycle, not by discipline here. Reproduces the historical numbers exactly
    (verified in tests/test_fitted_pipeline.py).

    ``rank_max`` is recorded explicitly (None = matrix-supported max,
    ``min(p, q, n_train)``); the returned ``rank`` is always ``<= rank_max`` when a
    cap is set (test_rank_cap.py).
    """
    pipe = (FittedPipeline(voxel_hash=voxel_hash, trial_table_hash=trial_table_hash,
                           rank_max=rank_max)
            .resolve_policies()
            .fit_preprocessing(M[Xr_tr], M[Yr_tr])
            .select(M[Xr_va], M[Yr_va])
            .fit_final())
    report = pipe.evaluate_test(M[Xr_te], M[Yr_te])
    sealed = pipe.seal()
    lam, rank = pipe.selected
    d = report.as_dict()
    # keep the historical result field names ("finite_frac", "mean", ...)
    d["finite_frac"] = d.pop("finite_fraction")
    d.update(lam=lam, rank=rank, val_score=float(pipe._val_score),
             rank_max=(None if rank_max is None else int(rank_max)),
             rank_hard_max=int(min(M.shape[1], M.shape[1], len(Xr_tr))),
             n_train=int(len(Xr_tr)), n_val=int(len(Xr_va)), n_test=int(len(Xr_te)),
             policy_name=sealed["policy_name"])
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="subj01")
    ap.add_argument("--roi", default="V1")
    ap.add_argument("--beta-version", default="fithrf")
    ap.add_argument("--denoising", default="D0")
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--pairing-seed", type=int, default=None,
                    help="required only for child-seeded policies (P1/I1); UNUSED and "
                         "warned-about for split-deterministic P0/I0")
    ap.add_argument("--vis2vis-pairing", default="all-ordered-distinct",
                    choices=["all-ordered-distinct", "derangement"])
    ap.add_argument("--vis2img-pairing", default="historical-index-aligned",
                    choices=["historical-index-aligned", "independent-permutation"])
    ap.add_argument("--rank-max", type=int, default=None,
                    help="explicit cap on retained rank (None = matrix-supported max)")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    t0 = time.time()

    # Fixed-scope contract: subject/ROI/denoising are hardcoded to this exact
    # configuration; beta_version must be one of the SHA-pinned registry versions
    # (B0/B1). Any other value would produce a falsely labeled artifact. Reject
    # before creating ANY output.
    fixed = {"subject": ("subj01", a.subject), "roi": ("V1", a.roi),
             "denoising": ("D0", a.denoising)}
    bad = {k: got for k, (exp, got) in fixed.items() if got != exp}
    if a.beta_version not in BETA_VERSIONS:
        bad["beta_version"] = a.beta_version
    if bad:
        allowed = {k: exp for k, (exp, _) in fixed.items()}
        allowed["beta_version"] = sorted(BETA_VERSIONS)
        print(f"FATAL: unsupported configuration {bad}; this runner is fixed to "
              f"{allowed}. No artifacts written.", file=sys.stderr)
        return 2
    beta = BETA_VERSIONS[a.beta_version]

    # CLI pairing-seed contract: reject child-seeded P1/I1 with no seed; warn on a
    # seed supplied to split-deterministic P0/I0 (it is unused). Checked BEFORE any
    # artifact is written so a violation leaves nothing behind.
    vv_pol, vi_pol = _VV_POLICY[a.vis2vis_pairing], _VI_POLICY[a.vis2img_pairing]
    contract = pr.pairing_seed_contract(vv_pol, vi_pol, a.pairing_seed is not None)
    for w in contract.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if not contract.ok:
        for e in contract.errors:
            print(f"FATAL: {e}", file=sys.stderr)
        print("No artifacts written.", file=sys.stderr)
        return 2

    base = _REPO / "data/nsd"
    betas = base / f"nsddata_betas/ppdata/subj01/func1pt8mm/{beta['subdir']}/betas_nsdimagery.hdf5"
    bdata = base / "nsddata/bdata/nsdimagery"
    roi = base / "nsddata/ppdata/subj01/func1pt8mm/roi"
    pp = base / "nsddata/ppdata/subj01/func1pt8mm"
    # SNR selection uses the shared NSD-core ncsnr (betas_fithrf), NOT the imagery
    # betas -- identical voxels across beta versions by design.
    ncsnr = base / "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf/ncsnr.nii.gz"
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    # 1. verify the beta HDF5 against its pinned SHA-256 (integrity gate)
    got = _sha(betas)
    if got != beta["sha256"]:
        print(f"FATAL: {beta['label']} ({a.beta_version}) sha256 mismatch: got {got}, "
              f"expected {beta['sha256']}", file=sys.stderr); return 2

    # 2-5. regenerate everything
    tt = sp.build_trial_table(str(bdata))
    tt.to_csv(out / "trial_table.csv", index=False)
    xyz, vhash, thr = sp.select_v1_voxels(str(roi), str(pp), str(ncsnr))
    splits = sp.make_splits(tt, a.split_seed)
    v, i = splits["vision"], splits["imagery"]

    M = sp.extract_v1_matrix(str(betas), tt["beta_index0"].values, xyz)
    raw_ok = bool(np.isfinite(M).all())

    # pairings. Historical P0/I0 use the smoke_pipeline path (the code that
    # produced 13cc3f0); P1/I1 use the child-seeded pairing module (byte-parity of
    # P0/I0 across the two paths is asserted in test_pairing.py).
    def vv_pairs(part):
        if vv_pol == "all_ordered_distinct":
            return sp.vis2vis_pairs(v, part, "all-ordered-distinct", 0)
        return pr.rows_to_arrays(pr.vis2vis_manifest(v, splits["_tt"], part, vv_pol, a.pairing_seed))

    def vi_pairs(part):
        if vi_pol == "historical_index_aligned":
            return sp.vis2img_pairs(v, i, part)
        return pr.rows_to_arrays(pr.vis2img_manifest(v, i, splits["_tt"], part, vi_pol, a.pairing_seed))

    vv_tr, vv_va, vv_te = vv_pairs("train"), vv_pairs("val"), vv_pairs("test")
    vi_tr, vi_va, vi_te = vi_pairs("train"), vi_pairs("val"), vi_pairs("test")

    v2v = fit_eval(M, *vv_tr, *vv_va, *vv_te, voxel_hash=vhash, rank_max=a.rank_max)
    v2i = fit_eval(M, *vi_tr, *vi_va, *vi_te, voxel_hash=vhash, rank_max=a.rank_max)

    # split + pairing manifests
    def split_rows():
        rows = []
        for state, sd in (("vision", v), ("imagery", i)):
            for ident, parts in sd.items():
                if ident == "_tt":
                    continue
                for part, arr in parts.items():
                    for row in arr:
                        rows.append(dict(row=int(row), beta_index0=int(tt.iloc[int(row)]["beta_index0"]),
                                         identity=ident, state=state, split=part))
        return rows
    json.dump({"split_seed": a.split_seed, "trials": split_rows()},
              open(out / "smoke_split_manifest.json", "w"), indent=1)
    # Per-model pairing provenance, consistent with smoke_result.json. Each model
    # records pairing_seed ONLY if its own policy consumes one (P1/I1); the split-
    # deterministic P0/I0 record null to avoid misattributing the randomness source
    # (the S1.8 cross-artifact defect).
    _vv_seeded = vv_pol in pr.SEED_CONSUMING
    _vi_seeded = vi_pol in pr.SEED_CONSUMING
    json.dump({
        "vis2vis": {"policy": vv_pol, "split_seed": a.split_seed,
                    "pairing_seed": (a.pairing_seed if _vv_seeded else None),
                    "randomness_source": ("child_seed(pairing_seed)" if _vv_seeded else "none_after_split"),
                    "train_src": list(map(int, vv_tr[0])), "train_tgt": list(map(int, vv_tr[1]))},
        "vis2img": {"policy": vi_pol, "split_seed": a.split_seed,
                    "pairing_seed": (a.pairing_seed if _vi_seeded else None),
                    "randomness_source": ("child_seed(pairing_seed)" if _vi_seeded else "none_after_split"),
                    "train_src": list(map(int, vi_tr[0])), "train_tgt": list(map(int, vi_tr[1]))},
    }, open(out / "smoke_pairing_manifest.json", "w"), indent=1)

    status = "S1_SMOKE_MODEL_PASS"
    if not raw_ok or v2v["finite_frac"] < FINITE_FRACTION_MIN or v2i["finite_frac"] < FINITE_FRACTION_MIN:
        status = "S1_SMOKE_ENGINEERING_FAILURE"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()
    except Exception:
        commit = "unknown"

    result = dict(status=status, non_interpretive=True, subject=a.subject, roi=a.roi,
                  beta_version=a.beta_version, beta_label=beta["label"],
                  beta_sha=beta["sha256"], denoising=a.denoising,
                  n_voxels=int(xyz.shape[1]), voxel_hash=vhash, snr_threshold=thr,
                  trial_table_sha=_sha(out / "trial_table.csv"),
                  # b0_sha kept only for the fithrf (B0) path so historical artifacts
                  # and the cross-artifact validator remain byte-compatible.
                  **({"b0_sha": beta["sha256"]} if a.beta_version == "fithrf" else {}),
                  split_seed=a.split_seed,
                  # PROVENANCE HONESTY: only child-seeded policies (P1/I1) consume
                  # the pairing seed; split-deterministic P0/I0 record null so the
                  # randomness source is never misattributed. seed_required is the
                  # single source of truth (the CLI contract validated it above).
                  pairing_seed=(a.pairing_seed if contract.seed_required else None),
                  pairing_randomness_source=("child_seed(pairing_seed)" if contract.seed_required else "split_seed"),
                  pairing_seed_unused_warned=contract.seed_unused,
                  vis2vis_pairing=vv_pol, vis2img_pairing=vi_pol,
                  preprocessing="train-only centering, no scaling (independent reconstruction; NOT author-confirmed)",
                  refit_policy="final model fit on TRAIN ONLY after validation selection (no train+val refit)",
                  ridge_policy="argmax_validation",
                  rank_policy="smallest_rank_at_99pct_within_selected_lambda",
                  rank_max=(None if a.rank_max is None else int(a.rank_max)),
                  finite_fraction_min=FINITE_FRACTION_MIN,
                  vis2vis=v2v, vis2img=v2i, code_commit=commit, runtime_s=round(time.time() - t0, 2))
    json.dump(result, open(out / "smoke_result.json", "w"), indent=2)
    print(json.dumps({"status": status, "vis2vis": {k: v2v[k] for k in ("lam", "rank", "mean", "n_train")},
                      "vis2img": {k: v2i[k] for k in ("lam", "rank", "mean", "n_train")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
