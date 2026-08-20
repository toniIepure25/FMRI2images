"""S2.0 real-data driver: 4-fold vis2vis->D1->vis2img for B0 and B1 (subj01 x V1).

Executes the FROZEN primary matrix ({B0,B1} x P1 z-score x V2V-P0 x D1) across four
folds, aggregates, and writes compact artifacts. NON-INTERPRETIVE: numbers are
never compared to Roy. Raw betas / matrices / denoised vectors are NEVER committed.

Run (Lane E):
    python scripts/analysis/mindcompiler/run_roy_s2_reconstruction.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_reconstruction as rc  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import (  # noqa: E402
    PREPROC_ZSCORE_TRAIN_ONLY,
)

BETA = {
    "B0": {"subdir": "nsdimagerybetas_fithrf",
           "sha256": "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64"},
    "B1": {"subdir": "nsdimagerybetas_fithrf_GLMdenoise_RR",
           "sha256": "cd42e680617d6564f403c7855140c53c0a2fdb9eb382755b49765bf781c4af2e"},
}
FOLD_SEED = 1234
OUT = _REPO / "artifacts/mindcompiler/roy_s2"


def _identity_gate() -> None:
    """Refuse to run outside the FMRI2images repository (project separation)."""
    try:
        url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"],
                                      cwd=_REPO).decode().strip()
    except Exception:
        url = ""
    if "FMRI2images" not in url:
        print(f"FATAL: repository identity gate failed (origin={url!r}); "
              "this MINDIR driver runs only in FMRI2images.", file=sys.stderr)
        raise SystemExit(2)


def _sha_file(p: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _agg(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), std=float(a.std(ddof=0)),
                min=float(a.min()), max=float(a.max()))


def main() -> int:
    _identity_gate()
    base = _REPO / "data/nsd"
    bdata = base / "nsddata/bdata/nsdimagery"
    roi = base / "nsddata/ppdata/subj01/func1pt8mm/roi"
    pp = base / "nsddata/ppdata/subj01/func1pt8mm"
    ncsnr = base / "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf/ncsnr.nii.gz"

    tt = sp.build_trial_table(str(bdata))
    xyz, vhash, thr = sp.select_v1_voxels(str(roi), str(pp), str(ncsnr))
    assert vhash == "a1bc56fe7c55", f"V1 voxel hash drift: {vhash}"
    folds = sf.build_four_folds(tt, FOLD_SEED)
    row_identity = {r: tt.loc[r, "identity"] for r in tt.index}
    row_beta = {r: int(tt.loc[r, "beta_index0"]) for r in tt.index}

    all_dep_records = []
    per_beta = {}
    for bkey, meta in BETA.items():
        betas = base / f"nsddata_betas/ppdata/subj01/func1pt8mm/{meta['subdir']}/betas_nsdimagery.hdf5"
        got = _sha_file(betas)
        if got != meta["sha256"]:
            print(f"FATAL: {bkey} sha256 mismatch {got}", file=sys.stderr); return 2
        M = sp.extract_v1_matrix(str(betas), tt["beta_index0"].values, xyz)
        fold_results = []
        for k in range(sf.N_FOLDS):
            fname = f"fold_{k}"
            res, recs = rc.run_fold(
                M, folds[fname], row_identity, row_beta,
                beta_version=meta["subdir"], preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY,
                vis2vis_pairing="all_ordered_distinct", pairing_seed=None,
                fold_name=fname, return_records=True)
            assert all(v == 0 for v in res.denoise_leakage.values()), res.denoise_leakage
            fold_results.append(dict(fold=fname, vis2vis=res.vis2vis, vis2img=res.vis2img,
                                     denoise_leakage=res.denoise_leakage,
                                     n_denoise_records=res.n_denoise_records))
            for rrec in recs:
                all_dep_records.append(dict(
                    beta=bkey, fold=fname, split=rrec.split, identity=rrec.identity,
                    denoised_row=rrec.denoised_row, beta_index=rrec.beta_index,
                    model_id=rrec.model_id,
                    n_train_source=len(rrec.train_source_rows),
                    n_train_target=len(rrec.train_target_rows),
                    self_target=int(rrec.denoised_row in rrec.train_target_rows)))
        v2i_means = [fr["vis2img"]["mean"] for fr in fold_results]
        v2i_ranks = [fr["vis2img"]["rank"] for fr in fold_results]
        v2v_ranks = [fr["vis2vis"]["rank"] for fr in fold_results]
        finite = [fr["vis2img"]["finite_frac"] for fr in fold_results]
        per_beta[bkey] = dict(
            beta_version=meta["subdir"], n_voxels=int(xyz.shape[1]), voxel_hash=vhash,
            folds=fold_results,
            aggregate=dict(vis2img_test_r=_agg(v2i_means),
                           vis2img_rank=_agg(v2i_ranks), vis2vis_rank=_agg(v2v_ranks),
                           vis2img_finite_frac=_agg(finite)),
            all_folds_finite_ge_min=bool(all(f >= 0.90 for f in finite)))
        json.dump(per_beta[bkey], open(OUT / f"{bkey}_fold_results.json", "w"), indent=2)
        print(f"{bkey}: vis2img test r per fold = {[round(x,4) for x in v2i_means]} "
              f"| agg mean {per_beta[bkey]['aggregate']['vis2img_test_r']['mean']:.4f}")

    # D1 dependency manifest (indices/model-ids only; no beta values)
    dep = pd.DataFrame(all_dep_records).sort_values(["beta", "fold", "split", "identity", "denoised_row"])
    dep.to_csv(OUT / "D1_dependency_manifest.csv", index=False)

    # B0 vs B1 under D1 (non-interpretive); historical D0 reference for sensitivity
    d0_ref = {"B0_D0_vis2img_mean": 0.1650281390649757, "B1_D0_vis2img_mean": 0.006993956265907372}
    comp = {
        "NON_INTERPRETIVE": True,
        "disclaimer": "Engineering/method-reconstruction only (subj01 x V1). NOT compared to Roy; no reproduction verdict.",
        "denoising": "D1_STRICT_CROSSFIT", "preprocessing": "P1 train-only z-score",
        "vis2vis_pairing": "all_ordered_distinct", "n_folds": sf.N_FOLDS,
        "B0_D1_vis2img_test_r": per_beta["B0"]["aggregate"]["vis2img_test_r"],
        "B1_D1_vis2img_test_r": per_beta["B1"]["aggregate"]["vis2img_test_r"],
        "self_target_violations": int(dep["self_target"].sum()),
        "historical_D0_reference": d0_ref,
    }
    json.dump(comp, open(OUT / "B0_vs_B1_D1_comparison.json", "w"), indent=2)

    # hashes of committed S2 artifacts
    hashes = {}
    for name in ["s2_frozen_config.json", "fold_manifest.csv", "D1_dependency_manifest.csv",
                 "B0_fold_results.json", "B1_fold_results.json", "B0_vs_B1_D1_comparison.json",
                 "ambiguity_registry.json"]:
        p = OUT / name
        if p.exists():
            hashes[name] = _sha_file(p)
    json.dump(hashes, open(OUT / "hashes.json", "w"), indent=2)

    print("self_target_violations:", comp["self_target_violations"])
    print("B0 D1 agg r:", round(comp["B0_D1_vis2img_test_r"]["mean"], 4),
          "| B1 D1 agg r:", round(comp["B1_D1_vis2img_test_r"]["mean"], 4))
    print("wrote:", sorted(p.name for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
