"""Single-command reproducible runner for the Roy S1 subj01 x V1 x B0 x D0 smoke.

Regenerates trial table, ROI/SNR selection, splits, pairings, extraction, value
QC, fit/select/evaluate, and writes all manifests + a resolved config. Returns a
non-zero exit code on any invariant failure.

The numbers are NON-INTERPRETIVE (pipeline validation only); they are never
compared to the Roy paper here.

Example:
    python scripts/analysis/mindcompiler/run_roy_s1_smoke.py \
      --split-seed 1234 --pairing-seed 1234 \
      --vis2vis-pairing all-ordered-distinct \
      --output-dir artifacts/mindcompiler/roy_s1/replay_13cc3f0
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

from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import (  # noqa: E402
    Standardizer, fit_reduced_rank, per_voxel_pearson, select_hyperparameters,
)
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402

B0_SHA = "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64"
FINITE_FRACTION_MIN = 0.90  # frozen engineering threshold


def _sha(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def metric_report(r: np.ndarray) -> dict:
    fin = np.isfinite(r)
    return dict(
        total=int(r.size), finite=int(fin.sum()), finite_frac=float(fin.mean()),
        nan=int((~fin).sum()),
        mean=float(np.nanmean(r)) if fin.any() else float("nan"),
        median=float(np.nanmedian(r)) if fin.any() else float("nan"),
        std=float(np.nanstd(r)) if fin.any() else float("nan"),
        rmin=float(np.nanmin(r)) if fin.any() else float("nan"),
        rmax=float(np.nanmax(r)) if fin.any() else float("nan"),
        p5=float(np.nanpercentile(r, 5)) if fin.any() else float("nan"),
        p25=float(np.nanpercentile(r, 25)) if fin.any() else float("nan"),
        p75=float(np.nanpercentile(r, 75)) if fin.any() else float("nan"),
        p95=float(np.nanpercentile(r, 95)) if fin.any() else float("nan"),
    )


def fit_eval(M, Xr_tr, Yr_tr, Xr_va, Yr_va, Xr_te, Yr_te) -> dict:
    """Train-only centering, no scaling; validation-only selection; test once."""
    s = Standardizer.fit(M[Xr_tr], with_scaling=False)
    ty = Standardizer.fit(M[Yr_tr], with_scaling=False)
    Xtr, Ytr = s.transform(M[Xr_tr]), ty.transform(M[Yr_tr])
    Xva, Yva = s.transform(M[Xr_va]), ty.transform(M[Yr_va])
    Xte, Yte = s.transform(M[Xr_te]), ty.transform(M[Yr_te])
    sel = select_hyperparameters(Xtr, Ytr, Xva, Yva,
                                 "smallest_rank_at_threshold", "argmax_validation")
    W = fit_reduced_rank(Xtr, Ytr, sel.lam, sel.rank)
    r = per_voxel_pearson(Yte, Xte @ W)
    rep = metric_report(r)
    rep.update(lam=sel.lam, rank=sel.rank, val_score=sel.val_score,
               n_train=int(len(Xr_tr)), n_val=int(len(Xr_va)), n_test=int(len(Xr_te)))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="subj01")
    ap.add_argument("--roi", default="V1")
    ap.add_argument("--beta-version", default="fithrf")
    ap.add_argument("--denoising", default="D0")
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--pairing-seed", type=int, default=1234)
    ap.add_argument("--vis2vis-pairing", default="all-ordered-distinct",
                    choices=["all-ordered-distinct", "derangement"])
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    t0 = time.time()

    # Fixed-scope contract: the underlying paths/labels are hardcoded to this
    # exact configuration, so any other CLI value would produce a falsely
    # labeled artifact. Reject before creating ANY output.
    fixed = {"subject": ("subj01", a.subject), "roi": ("V1", a.roi),
             "beta_version": ("fithrf", a.beta_version), "denoising": ("D0", a.denoising)}
    bad = {k: got for k, (exp, got) in fixed.items() if got != exp}
    if bad:
        allowed = {k: exp for k, (exp, _) in fixed.items()}
        print(f"FATAL: unsupported configuration {bad}; this runner is fixed to "
              f"{allowed}. No artifacts written.", file=sys.stderr)
        return 2

    base = _REPO / "data/nsd"
    betas = base / "nsddata_betas/ppdata/subj01/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.hdf5"
    bdata = base / "nsddata/bdata/nsdimagery"
    roi = base / "nsddata/ppdata/subj01/func1pt8mm/roi"
    pp = base / "nsddata/ppdata/subj01/func1pt8mm"
    ncsnr = base / "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf/ncsnr.nii.gz"
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    # 1. verify B0
    got = _sha(betas)
    if got != B0_SHA:
        print(f"FATAL: B0 sha256 mismatch {got}", file=sys.stderr); return 2

    # 2-5. regenerate everything
    tt = sp.build_trial_table(str(bdata))
    tt.to_csv(out / "trial_table.csv", index=False)
    xyz, vhash, thr = sp.select_v1_voxels(str(roi), str(pp), str(ncsnr))
    splits = sp.make_splits(tt, a.split_seed)
    v, i = splits["vision"], splits["imagery"]

    M = sp.extract_v1_matrix(str(betas), tt["beta_index0"].values, xyz)
    raw_ok = bool(np.isfinite(M).all())

    # pairings
    vv_tr = sp.vis2vis_pairs(v, "train", a.vis2vis_pairing, a.pairing_seed)
    vv_va = sp.vis2vis_pairs(v, "val", a.vis2vis_pairing, a.pairing_seed)
    vv_te = sp.vis2vis_pairs(v, "test", a.vis2vis_pairing, a.pairing_seed)
    vi_tr = sp.vis2img_pairs(v, i, "train")
    vi_va = sp.vis2img_pairs(v, i, "val")
    vi_te = sp.vis2img_pairs(v, i, "test")

    v2v = fit_eval(M, *vv_tr, *vv_va, *vv_te)
    v2i = fit_eval(M, *vi_tr, *vi_va, *vi_te)

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
    json.dump({"pairing_seed": a.pairing_seed, "vis2vis_policy": a.vis2vis_pairing,
               "vis2vis_train": list(map(int, vv_tr[0])), "vis2vis_train_tgt": list(map(int, vv_tr[1])),
               "vis2img_train": list(map(int, vi_tr[0])), "vis2img_train_tgt": list(map(int, vi_tr[1]))},
              open(out / "smoke_pairing_manifest.json", "w"), indent=1)

    status = "S1_SMOKE_MODEL_PASS"
    if not raw_ok or v2v["finite_frac"] < FINITE_FRACTION_MIN or v2i["finite_frac"] < FINITE_FRACTION_MIN:
        status = "S1_SMOKE_ENGINEERING_FAILURE"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()
    except Exception:
        commit = "unknown"

    result = dict(status=status, non_interpretive=True, subject=a.subject, roi=a.roi,
                  beta_version=a.beta_version, denoising=a.denoising,
                  n_voxels=int(xyz.shape[1]), voxel_hash=vhash, snr_threshold=thr,
                  trial_table_sha=_sha(out / "trial_table.csv"), b0_sha=B0_SHA,
                  split_seed=a.split_seed,
                  # PROVENANCE HONESTY: the historical policies (vis2vis P0
                  # all-ordered-distinct; vis2img I0 index-aligned) are BOTH
                  # deterministic from split_seed and do NOT consume pairing_seed.
                  # Recording pairing_seed for them would misattribute the
                  # randomness source. Only the P1/I1 sensitivity variants use it.
                  pairing_seed=(a.pairing_seed if a.vis2vis_pairing == "derangement" else None),
                  pairing_randomness_source=("pairing_seed" if a.vis2vis_pairing == "derangement" else "split_seed"),
                  vis2vis_pairing=a.vis2vis_pairing,
                  preprocessing="train-only centering, no scaling (independent reconstruction; NOT author-confirmed)",
                  refit_policy="final model fit on TRAIN ONLY after validation selection (no train+val refit)",
                  ridge_policy="argmax_validation",
                  rank_policy="smallest_rank_at_99pct_within_selected_lambda",
                  finite_fraction_min=FINITE_FRACTION_MIN,
                  vis2vis=v2v, vis2img=v2i, code_commit=commit, runtime_s=round(time.time() - t0, 2))
    json.dump(result, open(out / "smoke_result.json", "w"), indent=2)
    print(json.dumps({"status": status, "vis2vis": {k: v2v[k] for k in ("lam", "rank", "mean", "n_train")},
                      "vis2img": {k: v2i[k] for k in ("lam", "rank", "mean", "n_train")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
