"""O2.7 TRIAL-level native-residual STATE EXTENSION (frozen config d61cd6ff). Derives individual
imagery-trial residuals delta_trial(i,t) for subject x ROI x fold x outer-training identity x repeat 0..7,
using the EXACT sealed O2.2/RD residual machinery (delta_perp against the frozen outer-fold vision span).
delta_perp is affine-linear in the response, so the 8-repeat mean reproduces the sealed O2.6 identity-level
delta_native(i) exactly -- certified against the committed extension + sealed deltas_test (<=1e-10). No new
preprocessing, no trial normalization. Runs on the cluster (reads preserved B0 NSD-Imagery data on the PVC)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ALL = [f"subj0{i}" for i in range(1, 9)]
ROIS_PRIMARY = ["ventral", "lateral"]
N_FOLDS = 6
TOL = 1e-10


def _coh(repo: Path):
    sys.path.insert(0, str(repo / "src"))
    from fmri2img.mindcompiler.operator_o2_3a_rd import cohort as CH
    from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as SP
    return CH, SP


def _delta_perp_rows(Yrows, B_vis, mu_vis):
    """delta_perp applied to each row of Yrows (n x V): (y-mu) orthogonal to the vision span B_vis (V x r)."""
    Yc = np.asarray(Yrows, np.float64) - mu_vis
    return Yc - (Yc @ B_vis) @ B_vis.T


def run_extension(repo: Path, data: Path, state_dir: Path, ext_dir: Path, out: Path, rois):
    CH, SP = _coh(repo)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"gate": "O2.7", "artifact": "trial_residual_state", "tol": TOL, "replay": [], "files": [], "ok": True}
    max_disc = 0.0
    for roi in rois:
        for s in ALL:
            xyz, vh, nv = CH.roi_xyz(data, repo, s, roi)
            tt = SP.build_trial_table(str(data / "support/nsdimagery"), subject=s)
            ids = sorted(tt["identity"].unique())
            fam = {i: tt.loc[tt.identity == i, "family"].iloc[0] for i in ids}
            hpath = str(data / f"nsd_imagery_b0/{s}/betas_nsdimagery.hdf5")
            M = SP.extract_v1_matrix(hpath, tt["beta_index0"].values, xyz)          # (192 x V) /300 PSC
            Vc = {i: M[tt.index[(tt.state == "vision") & (tt.identity == i)]].mean(0) for i in ids}
            # per-identity imagery trials in acquisition (repeat) order + provenance
            Itr, prov = {}, {}
            for i in ids:
                sub = tt[(tt.state == "imagery") & (tt.identity == i)].sort_values("repeat")
                Itr[i] = M[sub.index.values]                                          # (8 x V)
                prov[i] = {"beta_index0": [int(x) for x in sub["beta_index0"].values],
                           "run": [str(x) for x in sub["run_name"].values],
                           "tsv_trial": [int(x) for x in sub["tsv_trial"].values],
                           "repeat": [int(x) for x in sub["repeat"].values]}
            for f in range(N_FOLDS):
                cell = dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                train_ids = [str(x) for x in cell["train_ids"].tolist()]
                test_ids = [str(x) for x in cell["test_ids"].tolist()]
                Vtr = np.stack([Vc[i] for i in train_ids])
                B_vis, mu_vis = CH.vis_span_basis(Vtr)
                trial_res = {i: _delta_perp_rows(Itr[i], B_vis, mu_vis) for i in ids}   # (8 x V) each
                # replay train identities against sealed O2.6 delta_native
                dz = dict(np.load(ext_dir / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                sealed_train = np.asarray(dz["delta_native"], np.float64)             # (10 x V) ordered by train_ids
                dtr = max(float(np.max(np.abs(trial_res[i].mean(0) - sealed_train[k])))
                          for k, i in enumerate(train_ids))
                sealed_test = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
                dte = max(float(np.max(np.abs(trial_res[test_ids[k]].mean(0) - sealed_test[k]))) for k in range(2))
                disc = max(dtr, dte); max_disc = max(max_disc, disc)
                ok = disc <= TOL and (vh == cell.get("voxel_hash", vh))               # voxel hash consistency
                if not ok:
                    manifest["ok"] = False
                Dtrial = np.stack([trial_res[i] for i in train_ids])                  # (10 x 8 x V)
                fpath = out / f"trialnat_{s}_{roi}_fold{f}.npz"
                np.savez(fpath, trial_native=Dtrial, train_ids=np.array(train_ids, dtype=object),
                         fam=np.array([fam[i] for i in train_ids], dtype=object), voxel_hash=vh,
                         provenance=np.array(json.dumps({i: prov[i] for i in train_ids}), dtype=object))
                manifest["files"].append({"name": fpath.name, "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
                                          "bytes": fpath.stat().st_size, "shape": list(Dtrial.shape)})
                manifest["replay"].append({"subject": s, "roi": roi, "fold": f, "max_abs_discrepancy": disc,
                                           "train_ok": dtr <= TOL, "test_ok": dte <= TOL, "ok": ok})
    manifest["max_abs_discrepancy_global"] = max_disc
    (out / "trial_residual_state_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if manifest["ok"]:
        print("O2_7_TRIAL_STATE_EXTENSION_CERTIFIED max_disc=%.2e n=%d" % (max_disc, len(manifest["replay"])))
        return 0
    print("O2_7_TRIAL_STATE_EXTENSION_FAILURE max_disc=%.2e" % max_disc)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--state", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    a = ap.parse_args()
    return run_extension(Path(a.repo), Path(a.data), Path(a.state), Path(a.ext), Path(a.out),
                         [r.strip() for r in a.rois.split(",") if r.strip()])


if __name__ == "__main__":
    raise SystemExit(main())
