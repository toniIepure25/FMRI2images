"""O2.3A-RD cohort-fit driver (frozen config c1b2ddb0). Deterministic, manifest-driven, reuses committed
code (geometry.py NEW SRM/Procrustes/null; roy_method_reproduction.s2_roi.select_roi_voxels for the exact
O1/O2 B0 ROI voxels; smoke_pipeline.build_trial_table + extract_v1_matrix for imagery cvis/cimg /300 PSC)
and a NEW core-anchor loader. O2.2 residual per the frozen subspace_contract (centered-TRAIN-vision span,
rank<=9). No target imagery in any fit stage; target imagery only for held-out eval + training-only oracle.

Modes: --validate-input-only (H6: headers/shapes/mappings, NO scientific metric) | --fit (H8 real cohort).
Writes per subject x ROI x fold state to <state>/ and results to <results>/. RUN ON r770 via committed code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ALL = [f"subj0{i}" for i in range(1, 9)]
ROIS_PRIMARY = ["ventral", "lateral"]
K_CANDS = [2, 4, 8, 16, 32, 64]
RANK_CANDS = [1, 2, 3, 4, 5, 6]
ANCHOR_SESSIONS = list(range(1, 31))       # sessions 1..30 (frozen: anchor spans 1..30 for all subjects)
SESS_TRIALS = 750


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ data loading
def anchor_nsdids(repo: Path):
    """512 anchor images: nsd_73k_id is 0-based nsdId; 73k id = nsdId+1 (verified in Phase2 H2)."""
    rows = [r for r in csv.DictReader(open(repo / "artifacts/mindcompiler/operator_o2_3a/anchor_manifest.csv"))
            if r["included"].lower() == "true"]
    nsdid = np.array(sorted(int(r["nsd_73k_id"]) for r in rows))
    return nsdid, nsdid + 1                 # (0-based nsdId, 1-based 73k id)


def core_anchor_centroids(data: Path, subject: str, xyz, id73_1based, expdesign):
    """Mean-of-3-reps native anchor centroids: (n_anchor x n_vox), ordered by sorted 73k id.
    Uses masterordering (trial->image index) + subjectim (image index->73k id) for sessions 1..30."""
    import nibabel as nib
    mo = expdesign["masterordering"].ravel()          # 30000 trial -> image index (1..10000)
    sim = expdesign["subjectim"]                       # (8 x 10000) 73k ids
    s_idx = int(subject[-2:]) - 1
    id2imgidx = {int(v): i + 1 for i, v in enumerate(sim[s_idx])}   # 73k id -> image index (1-based)
    xs, ys, zs = xyz
    nvox = xs.shape[0]
    # accumulate per-73kid sum of betas over its (<=3) rep trials within sessions 1..30
    acc = {a: (np.zeros(nvox, np.float64), 0) for a in id73_1based.tolist()}
    want_imgidx = {id2imgidx[a]: a for a in id73_1based.tolist() if int(a) in id2imgidx}
    for sess in ANCHOR_SESSIONS:
        f = data / f"core_b2/{subject}/betas_session{sess:02d}.nii.gz"
        if not f.exists():
            continue
        vol = np.asarray(nib.load(str(f)).dataobj)     # (X,Y,Z,750) int16
        base = (sess - 1) * SESS_TRIALS
        img_idx = mo[base:base + vol.shape[3]]         # image index per within-session trial
        for t in range(vol.shape[3]):
            ii = int(img_idx[t])
            if ii in want_imgidx:
                a = want_imgidx[ii]
                v = vol[xs, ys, zs, t].astype(np.float64) / 300.0   # PSC (same /300 convention)
                s, c = acc[a]; acc[a] = (s + v, c + 1)
        del vol
    order = np.sort(id73_1based)
    C = np.zeros((len(order), nvox), np.float64)
    reps = np.zeros(len(order), int)
    for k, a in enumerate(order.tolist()):
        s, c = acc[a]; reps[k] = c
        C[k] = s / c if c else np.nan
    return C, reps


def imagery_centroids(data: Path, repo: Path, subject: str, roi: str, xyz):
    """cvis/cimg native centroids per identity (mean over 8 reps) via committed loaders. Returns
    (ids_sorted, Vmat (n_id x nvox), Imat (n_id x nvox), family dict)."""
    sys.path.insert(0, str(repo / "src"))
    from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp
    tt = sp.build_trial_table(str(data / "support/nsdimagery"), subject=subject)
    ids = sorted(tt["identity"].unique())
    fam = {i: tt.loc[tt.identity == i, "family"].iloc[0] for i in ids}
    hpath = str(data / f"nsd_imagery_b0/{subject}/betas_nsdimagery.hdf5")
    M = sp.extract_v1_matrix(hpath, tt["beta_index0"].values, xyz)   # (n_trials x nvox) /300 PSC
    V = np.stack([M[tt.index[(tt.state == "vision") & (tt.identity == i)]].mean(0) for i in ids])
    I = np.stack([M[tt.index[(tt.state == "imagery") & (tt.identity == i)]].mean(0) for i in ids])
    return ids, V.astype(np.float64), I.astype(np.float64), fam


def roi_xyz(data: Path, repo: Path, subject: str, roi: str):
    sys.path.insert(0, str(repo / "src"))
    from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as R
    Rd = str(data / f"support/{subject}/roi"); P = str(data / f"support/{subject}")
    N = str(data / f"support/{subject}/ncsnr.nii.gz")
    sel = R.select_roi_voxels(roi, Rd, P, N)
    return sel["xyz"], sel["voxel_hash"], int(sel["selected_voxels"])


# ------------------------------------------------------------------ O2.2 residual (frozen subspace_contract)
def vis_span_basis(V_train: np.ndarray):
    """P_VIS_FULL: full CENTERED train vision identity span (all nonzero SVs, rank<=9). Returns B (nvox x r)
    orthonormal + train-vision mean mu."""
    mu = V_train.mean(0)
    Xc = V_train - mu                        # (n_train x nvox), centered
    U, s, Vt = np.linalg.svd(Xc.T, full_matrices=False)   # columns of U span the row-space (voxel span)
    keep = s > (s.max() * 1e-9) if s.size else np.array([], bool)
    return U[:, keep], mu


def delta_perp(img_test: np.ndarray, B_vis: np.ndarray, mu_vis: np.ndarray) -> np.ndarray:
    """delta_y_perp: held-out imagery deviation (from train-vision mean) orthogonal to the vision span."""
    d = img_test - mu_vis
    return d - B_vis @ (B_vis.T @ d)


# ------------------------------------------------------------------ validation (H6) / fit (H8)
def validate_input(data: Path, repo: Path):
    out = {"mode": "validate-input-only", "subjects": {}, "ok": True}
    import scipy.io
    exp = scipy.io.loadmat(str(data / "support/nsd_expdesign.mat"))
    nsdid0, id73 = anchor_nsdids(repo)
    out["anchor"] = {"n": int(len(id73)), "nsdid_0based": True, "id73_min": int(id73.min()), "id73_max": int(id73.max())}
    for s in ALL:
        rec = {}
        try:
            for roi in ROIS_PRIMARY:
                xyz, vh, nv = roi_xyz(data, repo, s, roi)
                rec[roi] = {"voxels": nv, "voxel_hash": vh}
            ids, V, I, fam = imagery_centroids(data, repo, s, "ventral", roi_xyz(data, repo, s, "ventral")[0])
            rec["imagery"] = {"n_identities": len(ids), "n_simple": sum(v == "simple" for v in fam.values()),
                              "n_nat": sum(v == "naturalistic" for v in fam.values()), "V_shape": list(V.shape)}
            # core anchor coverage (ventral)
            xyz = roi_xyz(data, repo, s, "ventral")[0]
            C, reps = core_anchor_centroids(data, s, xyz, id73, exp)
            rec["core_anchor"] = {"n_anchor": int(C.shape[0]), "reps_min": int(reps.min()), "reps_med": int(np.median(reps)),
                                  "reps_max": int(reps.max()), "n_full3": int((reps == 3).sum()), "finite": bool(np.isfinite(C).all())}
            if reps.min() < 3 or not np.isfinite(C).all():
                out["ok"] = False; rec["WARN"] = "incomplete anchor reps or non-finite"
        except Exception as e:
            out["ok"] = False; rec["ERROR"] = repr(e)
        out["subjects"][s] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate-input-only", action="store_true")
    a = ap.parse_args()
    data, repo, out = Path(a.data), Path(a.repo), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.validate_input_only:
        res = validate_input(data, repo)
        (out / "input_validation.json").write_text(json.dumps(res, indent=2, default=str))
        print("VALIDATE_OK" if res["ok"] else "VALIDATE_FAIL")
        print(json.dumps({s: res["subjects"][s].get("core_anchor", res["subjects"][s].get("ERROR")) for s in ALL}, default=str)[:1500])
        return 0 if res["ok"] else 1
    print("FIT mode not yet enabled in this build"); return 2


if __name__ == "__main__":
    raise SystemExit(main())
