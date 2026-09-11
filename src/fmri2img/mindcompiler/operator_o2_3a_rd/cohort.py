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
ROIS_SECONDARY = ["parietal"]
K_CANDS = [2, 4, 8, 16, 32, 64]
RANK_CANDS = [1, 2, 3, 4, 5, 6]
N_OUTER_FOLDS = 6
N_NULL = 100
ANCHOR_SESSIONS = list(range(1, 31))       # sessions 1..30 (frozen: anchor spans 1..30 for all subjects)
SESS_TRIALS = 750

# repo import shim -------------------------------------------------------------
_GEOM = None


def _geom(repo: Path):
    """Import the committed certified geometry (single source of deterministic math)."""
    global _GEOM
    if _GEOM is None:
        sys.path.insert(0, str(repo / "src"))
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _GEOM = G
    return _GEOM


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _canon_cols(U: np.ndarray) -> np.ndarray:
    """Deterministic column signs (largest-abs element positive) so subspaces persist identically."""
    U = np.asarray(U, dtype=np.float64).copy()
    for k in range(U.shape[1]):
        j = int(np.argmax(np.abs(U[:, k])))
        if U[j, k] < 0:
            U[:, k] *= -1.0
    return U


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


# ------------------------------------------------------------------ scaling (TRAIN_ONLY_VOXEL_ZSCORE)
def zscore_fit(A: np.ndarray):
    """Per-voxel mean/std over the given (n_anchor x nvox) TRAINING rows. std floored to avoid /0."""
    mu = A.mean(0)
    sd = A.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu, sd


def zscore_apply(A: np.ndarray, mu, sd):
    return (A - mu) / sd


# ------------------------------------------------------------------ outer / inner identity folds (frozen)
def outer_folds(ids, fam):
    """6 outer folds: fold j holds out sorted-simple[j] + sorted-naturalistic[j] (j=0..5). Returns
    list of (train_ids, [test_simple, test_nat])."""
    simple = sorted([i for i in ids if fam[i] == "simple"])
    nat = sorted([i for i in ids if fam[i] == "naturalistic"])
    folds = []
    for j in range(N_OUTER_FOLDS):
        ts, tn = simple[j], nat[j]
        tr = [i for i in ids if i not in (ts, tn)]
        folds.append((tr, [ts, tn]))
    return folds


def inner_pair_folds(train_ids, fam):
    """5 balanced inner folds over the 10 outer-training identities: inner fold j holds out
    sorted-simple[j] + sorted-naturalistic[j] (j=0..4)."""
    s = sorted([i for i in train_ids if fam[i] == "simple"])
    n = sorted([i for i in train_ids if fam[i] == "naturalistic"])
    return [([x for x in train_ids if x not in (s[j], n[j])], [s[j], n[j]]) for j in range(5)]


# ------------------------------------------------------------------ scaler (CV-clean TRAIN_ONLY_VOXEL_ZSCORE)
def _scaler_train(C: np.ndarray, keep_rows: np.ndarray):
    """Per-voxel mean/std over ONLY the kept (training) anchor rows. std floored. FIX 2: no validation-block
    row ever contributes to a scaler used during K-block CV."""
    sub = C[keep_rows]
    mu = sub.mean(0); sd = sub.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu, sd


# ------------------------------------------------------------------ SRM K-selection (TARGET-SPECIFIC, vision-only)
def select_K_for_target(target, C_by_subject, block_id, G):
    """FIX 1: K selected separately for EVERY (target participant x ROI). For each candidate K and anchor
    validation block b in {0..4}: SRM is fit on the SEVEN training subjects (!= target) using ONLY TRAIN
    anchor rows (block_id != b); the target is aligned using ONLY its TRAIN rows; the target's held-out
    block is reconstructed from the training subjects' shared responses and scored by cross-subject native
    pattern reconstruction Pearson r. Mean over the 5 blocks; max K; tie<=1e-12 -> smaller K.
    FIX 2: every participant's voxel z-score scaler is fit on block_id != b only and applied to both its
    train and validation rows -- no validation-block statistics leak into K selection.
    C_by_subject[s]: RAW (un-zscored) (n_anchor x nvox) perception anchors. Returns (bestK, scores)."""
    subs = [s for s in C_by_subject if s != target]
    scores = {}
    for K in K_CANDS:
        block_r = []
        for b in range(5):
            tr = block_id != b; val = block_id == b
            # CV-clean per-participant scalers (train-block stats only), applied to train + val rows
            Ztr = {}
            for s in subs:
                mu, sd = _scaler_train(C_by_subject[s], tr)
                Ztr[s] = (C_by_subject[s] - mu) / sd
            mu_t, sd_t = _scaler_train(C_by_subject[target], tr)
            Zt = (C_by_subject[target] - mu_t) / sd_t
            # SRM on the 7 training subjects, TRAIN rows only  (X_s = p_s x n_train)
            Ws, S = G.det_srm_rd([Ztr[s][tr].T for s in subs], K)
            Wmap = dict(zip(subs, Ws))
            # shared responses for the held-out block from the training subjects' val-block anchors
            Sb = np.mean([Wmap[s].T @ Ztr[s][val].T for s in subs], axis=0)      # (K x n_val)
            # align target on its TRAIN rows, reconstruct its held-out block
            Wt = G.new_subject_W(Zt[tr].T, S)                                    # (p_t x K)
            Xhat = Wt @ Sb                                                        # (p_t x n_val)
            Xtrue = Zt[val].T
            a, c = Xhat.ravel(), Xtrue.ravel()
            block_r.append(0.0 if (a.std() < 1e-12 or c.std() < 1e-12) else float(np.corrcoef(a, c)[0, 1]))
        scores[K] = float(np.mean(block_r))
    best, bestv = None, -np.inf
    for K in K_CANDS:                                        # ascending -> tie prefers smaller K
        if scores[K] > bestv + 1e-12:
            best, bestv = K, scores[K]
    return best, scores


# ------------------------------------------------------------------ residual template + rank CV
def common_residuals(delta_by_si, Wmap, ids):
    """r_common[s,i] = W_s^T delta_perp[s,i] (K-dim) for training subjects s, identities i. Returns dict."""
    out = {}
    for s, W in Wmap.items():
        for i in ids:
            out[(s, i)] = W.T @ delta_by_si[(s, i)]
    return out


def u_res_from(rcommon, subs, ids, r, G):
    """Top-r shared-residual subspace from stacked common-space residuals of `subs` x `ids`. (K x r)."""
    R = np.stack([rcommon[(s, i)] for s in subs for i in ids], axis=1)   # (K x n)
    U, _, _ = np.linalg.svd(R, full_matrices=False)
    return _canon_cols(U[:, :r])


def select_rank(rcommon, Wmap, delta_by_si, train_ids, fam, G):
    """Inner nested CV over training-subject LOSO x 5 identity-pair folds. Returns (best_r, scores)."""
    subs = list(Wmap.keys())
    scores = {}
    for r in RANK_CANDS:
        vals = []
        for s_h in subs:                                   # hold out one training subject
            others = [s for s in subs if s != s_h]
            for tr_ids, val_ids in inner_pair_folds(train_ids, fam):
                U_res = u_res_from(rcommon, others, tr_ids, r, G)
                B = Wmap[s_h] @ U_res                        # (p x r) native subspace for held-out subj
                P = G._orth(B); P = P @ P.T
                for vi in val_ids:
                    vals.append(G.retention(P, delta_by_si[(s_h, vi)]))
        scores[r] = float(np.nanmean(vals))
    best, bestv = None, -np.inf
    for r in RANK_CANDS:
        if scores[r] > bestv + 1e-12:
            best, bestv = r, scores[r]
    return best, scores


# ------------------------------------------------------------------ per-target fit (one ROI)
def fit_target_roi(target, Xs_by_subj, imagery, block_id, ids, fam, K, G, statedir, roi):
    """Full RD fit for one target participant x ROI. Persists per-fold cell state for O2.4R.
    imagery[s] = dict(V=(12 x p_s), I=(12 x p_s), ids=list) native centroids. Returns per-fold records."""
    subs = [s for s in Xs_by_subj if s != target]
    # SRM common space on the 7 training subjects (selected K), align target as new subject
    Xs_tr = [Xs_by_subj[s] for s in subs]
    Ws, S = G.det_srm_rd(Xs_tr, K)
    Wmap = dict(zip(subs, Ws))
    W_target = G.new_subject_W(Xs_by_subj[target], S)       # (p_t x K)
    idx = {i: k for k, i in enumerate(ids)}
    records = []
    for fold, (train_ids, test_ids) in enumerate(outer_folds(ids, fam)):
        # per-subject O2.2 residuals for THIS fold's train-vision span (train = 10 outer-training identities)
        delta = {}
        for s in subs + [target]:
            Vtr = np.stack([imagery[s]["V"][idx[i]] for i in train_ids])    # (10 x p_s)
            B_vis, mu_vis = vis_span_basis(Vtr)
            for i in ids:
                delta[(s, i)] = delta_perp(imagery[s]["I"][idx[i]], B_vis, mu_vis)
        rcommon = common_residuals(delta, Wmap, train_ids)
        r_best, r_scores = select_rank(rcommon, Wmap, delta, train_ids, fam, G)
        U_res = u_res_from(rcommon, subs, train_ids, r_best, G)             # (K x r)
        # orientation P_ZERO_RD (Q = identity)
        P0 = G.native_projector(np.eye(K), U_res, W_target)
        deltas_test = [delta[(target, i)] for i in test_ids]
        R_zero = float(np.nanmean([G.retention(P0, d) for d in deltas_test]))
        # matched null (100): random r-subspaces mapped through target W
        null_vals = []
        for it in range(N_NULL):
            Un = G.null_subspace("O2.3A-RD|%s|%s|%d|%d" % (target, roi, fold, it), K, r_best)
            Pn = G.native_projector(np.eye(K), Un, W_target)
            null_vals.append(np.nanmean([G.retention(Pn, d) for d in deltas_test]))
        R_null = float(np.mean(null_vals))
        # training-only native oracle (diagnostic): target imagery from outer-TRAIN identities
        Dtr = np.stack([delta[(target, i)] for i in train_ids])             # (10 x p_t)
        Uo, _, _ = np.linalg.svd(Dtr.T, full_matrices=False)
        Bo = G._orth(Uo[:, :r_best]); Po = Bo @ Bo.T
        R_oracle = float(np.nanmean([G.retention(Po, d) for d in deltas_test]))
        # ---- per-cell O2.4R state (shared coords X, target-imagery coords Z, held-out deltas) ----
        rbar = {i: np.mean([rcommon[(s, i)] for s in subs], axis=0) for i in train_ids}   # RD template per identity
        Xcell = np.stack([U_res @ (U_res.T @ rbar[i]) for i in train_ids])                # (10 x K) shared coords
        Zcell = np.stack([W_target.T @ delta[(target, i)] for i in train_ids])            # (10 x K) target-imagery coords
        s_ids = [k for k, i in enumerate(train_ids) if fam[i] == "simple"]
        n_ids = [k for k, i in enumerate(train_ids) if fam[i] == "naturalistic"]
        cellpath = statedir / f"cell_{target}_{roi}_fold{fold}.npz"
        np.savez(cellpath, X=Xcell, Z=Zcell, U_res=U_res, W_target=W_target,
                 deltas_test=np.stack(deltas_test), simple_ids=np.array(s_ids), nat_ids=np.array(n_ids),
                 R_CAL0=np.float64(R_zero), r_best=r_best, K=K,
                 train_ids=np.array(train_ids, dtype=object), test_ids=np.array(test_ids, dtype=object))
        records.append({"fold": fold, "r_best": r_best, "r_scores": r_scores,
                        "R_ZERO_RD": R_zero, "R_NULL": R_null, "R_ORACLE": R_oracle,
                        "RD_ORACLE_RECOVERY": rd_oracle_recovery_fold(R_zero, R_oracle),
                        "test_ids": test_ids, "n_train": len(train_ids)})
    # persist reusable target-ROI state (no raw betas)
    np.savez(statedir / f"srm_{target}_{roi}.npz", W_target=W_target, S=S, K=K,
             train_subjects=np.array(subs, dtype=object))
    return {"target": target, "roi": roi, "K": K, "folds": records}


# ------------------------------------------------------------------ inference across participants
RD_ORACLE_EPS = 1e-12


def rd_oracle_recovery_fold(r_zero: float, r_oracle: float) -> float:
    """FIX 3: historical O2.3A oracle-recovery convention preserved prospectively --
    RD_ORACLE_RECOVERY = R_ZERO_RD / max(R_ORACLE, eps). NOT raw R_ORACLE."""
    return float(r_zero) / max(float(r_oracle), RD_ORACLE_EPS)


def rd_inference(per_target, G):
    """per_target: {roi: {subj: {folds:[...]}}}. Effects = mean_fold(R_ZERO_RD - R_NULL) per participant.
    Oracle recovery (FIX 3) = median_participants( median_folds( R_ZERO_RD / max(R_ORACLE, eps) ) )."""
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    out = {}
    holm_p = {}
    for roi, bysub in per_target.items():
        eff, truem, nullm, part_rec = [], [], [], []
        for s in ALL:
            fr = bysub[s]["folds"]
            t = np.mean([f["R_ZERO_RD"] for f in fr]); n = np.mean([f["R_NULL"] for f in fr])
            eff.append(t - n); truem.append(t); nullm.append(n)
            fold_rec = [rd_oracle_recovery_fold(f["R_ZERO_RD"], f["R_ORACLE"]) for f in fr]
            part_rec.append(float(np.median(fold_rec)))            # participant oracle recovery = median over folds
        p = F.signflip_p_onesided(eff)
        holm_p[roi] = p
        out[roi] = {"median_true": float(np.median(truem)), "median_null": float(np.median(nullm)),
                    "n_pos": int(sum(e > 0 for e in eff)), "effects": [float(e) for e in eff],
                    "signflip_p": float(p),
                    "participant_oracle_recovery": part_rec,
                    "group_oracle_recovery": float(np.median(part_rec))}   # median over 8 participants
    rej = F.holm(holm_p)
    for roi in out:
        out[roi]["holm_reject"] = bool(rej[roi])
    # RD identifiable PASS criteria (frozen): any primary ROI satisfies median true>null, >=6/8 positive,
    # Holm-adjusted p<.05, AND group-median oracle RECOVERY >= .50
    pass_ = any(out[r]["median_true"] > out[r]["median_null"] and out[r]["n_pos"] >= 6
                and out[r]["holm_reject"] and out[r]["group_oracle_recovery"] >= 0.50 for r in out)
    all_pos_med = all(out[r]["median_true"] > out[r]["median_null"] for r in out)
    if pass_:
        status = "CORE_ANCHOR_RD_TARGET_ORIENTATION_IDENTIFIABLE"
    elif all(out[r]["median_true"] <= out[r]["median_null"] for r in out) or all(out[r]["n_pos"] < 6 for r in out):
        status = "CORE_ANCHOR_RD_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
    else:
        status = "CORE_ANCHOR_RD_INCONCLUSIVE"
    return {"per_roi": out, "status": status, "all_primary_median_positive": all_pos_med}


# ------------------------------------------------------------------ union anchor centroids (single session sweep, cached)
def _prov_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def union_anchor_centroids(data: Path, subject: str, roi_xyz_map, id73_1based, expdesign, prov_base: dict):
    """Read each core session ONCE; return per-anchor centroids over the UNION of the ROIs' voxels, plus a
    per-ROI column index into that union. Deterministic; identical numerically to per-ROI
    core_anchor_centroids (same /300 PSC mean-of-reps).

    FIX 6: the cache identity binds the FULL scientific provenance -- canonical 512-anchor manifest SHA256,
    ordered 512 anchor-id hash, nsd_expdesign.mat hash, B0/b2 lineage, exact session list, union
    voxel-coordinate hash, /300 scaling convention, and code/config version. A cached npz is reused ONLY if
    its adjacent immutable manifest matches every provenance field; otherwise it is rebuilt."""
    import nibabel as nib
    cols = {}
    order_coords = []
    for roi in sorted(roi_xyz_map):
        xs, ys, zs = roi_xyz_map[roi]
        for k in range(xs.shape[0]):
            key = (int(xs[k]), int(ys[k]), int(zs[k]))
            if key not in cols:
                cols[key] = len(order_coords); order_coords.append(key)
    UX = np.array([c[0] for c in order_coords]); UY = np.array([c[1] for c in order_coords]); UZ = np.array([c[2] for c in order_coords])
    order = np.sort(id73_1based)
    prov = dict(prov_base)
    prov.update({"subject": subject,
                 "union_voxel_hash": hashlib.sha256(np.stack([UX, UY, UZ]).tobytes()).hexdigest(),
                 "anchor_id_ordered_hash": hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),
                 "session_list": list(ANCHOR_SESSIONS), "scaling": "PSC_div300",
                 "beta_lineage": "B0_b2_betas_fithrf_func1pt8mm_native"})
    ph = _prov_hash(prov)[:16]
    cache = data / f"cache/anchor_{subject}_{ph}.npz"
    manifest = data / f"cache/anchor_{subject}_{ph}.provenance.json"
    reuse = cache.exists() and manifest.exists() and json.loads(manifest.read_text()).get("prov_hash") == _prov_hash(prov)
    if reuse:
        z = np.load(cache); C_union, reps = z["C"], z["reps"]
    else:
        mo = expdesign["masterordering"].ravel()
        sim = expdesign["subjectim"]
        s_idx = int(subject[-2:]) - 1
        id2imgidx = {int(v): i + 1 for i, v in enumerate(sim[s_idx])}
        want_imgidx = {id2imgidx[int(a)]: int(a) for a in order.tolist() if int(a) in id2imgidx}
        arow = {int(a): k for k, a in enumerate(order.tolist())}
        nvox = len(order_coords)
        acc = np.zeros((len(order), nvox), np.float64); reps = np.zeros(len(order), int)
        for sess in ANCHOR_SESSIONS:
            f = data / f"core_b2/{subject}/betas_session{sess:02d}.nii.gz"
            if not f.exists():
                continue
            vol = np.asarray(nib.load(str(f)).dataobj)
            base = (sess - 1) * SESS_TRIALS
            img_idx = mo[base:base + vol.shape[3]]
            patch = vol[UX, UY, UZ, :].astype(np.float64) / 300.0        # (nvox x 750)
            for t in range(vol.shape[3]):
                ii = int(img_idx[t])
                a = want_imgidx.get(ii)
                if a is not None:
                    k = arow[a]; acc[k] += patch[:, t]; reps[k] += 1
            del vol, patch
        C_union = np.where(reps[:, None] > 0, acc / np.maximum(reps[:, None], 1), np.nan)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, C=C_union, reps=reps)
        manifest.write_text(json.dumps({"prov_hash": _prov_hash(prov), "provenance": prov}, indent=2, default=str))
    roi_cols = {roi: np.array([cols[(int(x), int(y), int(z))] for x, y, z in zip(*roi_xyz_map[roi])])
                for roi in roi_xyz_map}
    return C_union, reps, roi_cols


# ------------------------------------------------------------------ data assembly for a ROI
def _load_roi(data, repo, subject, roi, exp, id73):
    xyz, vh, nv = roi_xyz(data, repo, subject, roi)
    ids, V, I, fam = imagery_centroids(data, repo, subject, roi, xyz)
    C, reps = core_anchor_centroids(data, subject, xyz, id73, exp)          # (512 x nvox) perception anchors
    return {"xyz": xyz, "vh": vh, "nv": nv, "ids": ids, "V": V, "I": I, "fam": fam, "C": C, "reps": reps}


def _file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fit_cohort(data: Path, repo: Path, out: Path, rois, code_version: str = "unknown"):
    import scipy.io
    G = _geom(repo)
    statedir = out / "state"; statedir.mkdir(parents=True, exist_ok=True)
    exp_path = data / "support/nsd_expdesign.mat"
    exp = scipy.io.loadmat(str(exp_path))
    nsdid0, id73 = anchor_nsdids(repo)
    n_anchor = len(id73)
    block_id = np.arange(n_anchor) % 5                                       # K-blocks = anchor_order mod 5
    # FIX 6 provenance base (cache identity binds all of these)
    prov_base = {"anchor_manifest_sha256": _file_sha(repo / "artifacts/mindcompiler/operator_o2_3a/anchor_manifest.csv"),
                 "expdesign_sha256": _file_sha(exp_path), "config_rd": "c1b2ddb0", "config_r4": "ad959446",
                 "code_version": code_version, "n_anchor": int(n_anchor)}
    summary = {"rois": {}, "config_rd": "c1b2ddb0", "config_r4": "ad959446", "n_anchor": int(n_anchor),
               "provenance_base": prov_base}
    per_target = {}
    # per-subject: ROI voxels + imagery centroids + union anchor centroids (ONE session sweep per subject)
    print("assembling per-subject data (single core-session sweep each)...", flush=True)
    subj = {}
    for s in ALL:
        rx = {}
        for roi in rois:
            xyz, vh, nv = roi_xyz(data, repo, s, roi)
            rx[roi] = {"xyz": xyz, "vh": vh, "nv": nv}
        C_union, reps, roi_cols = union_anchor_centroids(data, s, {roi: rx[roi]["xyz"] for roi in rois}, id73, exp, prov_base)
        img = {roi: imagery_centroids(data, repo, s, roi, rx[roi]["xyz"]) for roi in rois}
        subj[s] = {"rx": rx, "C_union": C_union, "reps": reps, "roi_cols": roi_cols, "img": img}
        nvox_str = ", ".join("%s:%d" % (r, rx[r]["nv"]) for r in rois)
        print("  %s: anchor reps min/med/max = %d/%d/%d, nvox {%s}"
              % (s, int(reps.min()), int(np.median(reps)), int(reps.max()), nvox_str), flush=True)
    scaler_prov = {"policy": "TRAIN_ONLY_VOXEL_ZSCORE", "std_floor": 1e-8,
                   "k_cv": "per (target,K,block b): scaler fit on block_id!=b only, applied to train+val rows",
                   "final_fit": "full 512-anchor scaler (no remaining anchor-block validation)"}
    for roi in rois:
        ids = subj[ALL[0]]["img"][roi][0]; fam = subj[ALL[0]]["img"][roi][3]
        # RAW per-ROI anchors (un-zscored); K selection does its own CV-clean scaling (FIX 2)
        C_by_subject = {s: subj[s]["C_union"][:, subj[s]["roi_cols"][roi]] for s in ALL}      # (512 x nvox)
        # FINAL full-anchor scaler (allowed: no remaining anchor-block validation in the final fit)
        final_scaler = {}
        Xs_final = {}
        for s in ALL:
            mu, sd = zscore_fit(C_by_subject[s])                                  # full 512-anchor stats
            final_scaler[s] = {"mu": mu, "sd": sd}
            Xs_final[s] = zscore_apply(C_by_subject[s], mu, sd).T                  # (nvox x 512)
        # FIX 1: target-specific K (per target x ROI), using RAW anchors + CV-clean scalers
        K_by_target, kscores_by_target = {}, {}
        for target in ALL:
            Kt, kst = select_K_for_target(target, C_by_subject, block_id, G)
            K_by_target[target] = Kt; kscores_by_target[target] = kst
            print("  K[%s,%s]=%d" % (target, roi, Kt), flush=True)
        # persist final scalers for this ROI (no raw betas; derived per-voxel stats only)
        np.savez(statedir / f"scaler_{roi}.npz",
                 **{f"{s}_mu": final_scaler[s]["mu"] for s in ALL},
                 **{f"{s}_sd": final_scaler[s]["sd"] for s in ALL})
        imagery = {s: {"V": subj[s]["img"][roi][1], "I": subj[s]["img"][roi][2]} for s in ALL}
        bysub = {}
        for target in ALL:
            bysub[target] = fit_target_roi(target, Xs_final, imagery, block_id, ids, fam,
                                           K_by_target[target], G, statedir, roi)
        per_target[roi] = bysub
        summary["rois"][roi] = {"K_by_target": K_by_target, "K_scores_by_target": kscores_by_target,
                                "n_vox": {s: subj[s]["rx"][roi]["nv"] for s in ALL},
                                "voxel_hash": {s: subj[s]["rx"][roi]["vh"] for s in ALL}}
    summary["scaler_provenance"] = scaler_prov
    inf = rd_inference({r: per_target[r] for r in rois if r in ROIS_PRIMARY} or per_target, G)
    summary["inference"] = inf
    summary["per_target"] = {r: {s: per_target[r][s]["folds"] for s in ALL} for r in per_target}
    (out / "rd_results.json").write_text(json.dumps(summary, indent=2, default=str))
    seal_rd(out, summary)                                   # Stage A seal artifact (RD_SEAL.json)
    print("RD_STATUS", inf["status"])
    return summary


# ------------------------------------------------------------------ RD seal + M0 certification artifacts
def seal_rd(out: Path, summary: dict):
    """Stage A seal: write the single RD scientific status + criteria as an immutable artifact (RD_SEAL.json)."""
    inf = summary["inference"]
    seal = {"artifact": "RD_SEAL", "status": inf["status"], "config_rd": "c1b2ddb0", "config_r4": "ad959446",
            "per_roi_criteria": inf["per_roi"], "all_primary_median_positive": inf["all_primary_median_positive"]}
    (out / "RD_SEAL.json").write_text(json.dumps(seal, indent=2, default=str))
    return seal


def o2_4r_oracle_recovery(R_M: float, R_0: float, R_oracle: float):
    """FIX 4: calibration oracle-recovery fraction. Returns (unclipped, clipped_[0,1]).
    (R_M - R_0) / max(R_oracle - R_0, 1e-12)."""
    unclip = (float(R_M) - float(R_0)) / max(float(R_oracle) - float(R_0), 1e-12)
    return float(unclip), float(np.clip(unclip, 0.0, 1.0))


def certify_m0(repo: Path, out: Path, rois):
    """Certify O2_4R_M0_EQUALS_O2_3A_RD on the real persisted cells (M0 projector == P_ZERO_RD, numerically).
    Requires the RD seal artifact to exist first. Writes M0_CERT.json. Does NOT open M>0."""
    if not (out / "RD_SEAL.json").exists():
        print("O2_4R_ACCESS_ORDER_VIOLATION: RD_SEAL.json missing -- seal RD before M0 certification")
        return 3
    G = _geom(repo)
    statedir = out / "state"
    max_dev = 0.0; n_cells = 0
    for roi in rois:
        for s in ALL:
            for fold in range(N_OUTER_FOLDS):
                cell = dict(np.load(statedir / f"cell_{s}_{roi}_fold{fold}.npz", allow_pickle=True))
                P0 = G.native_projector(np.eye(int(cell["K"])), cell["U_res"], cell["W_target"])
                r0 = float(np.nanmean([G.retention(P0, d) for d in list(cell["deltas_test"])]))
                max_dev = max(max_dev, abs(r0 - float(cell["R_CAL0"]))); n_cells += 1
    ok = max_dev <= 1e-10
    cert = {"artifact": "M0_CERT", "certification": "O2_4R_M0_EQUALS_O2_3A_RD", "ok": bool(ok),
            "max_abs_deviation": float(max_dev), "n_cells": n_cells, "tol": 1e-10, "rois": rois}
    (out / "M0_CERT.json").write_text(json.dumps(cert, indent=2, default=str))
    print("O2_4R_M0_EQUALS_O2_3A_RD" if ok else "O2_4R_M0_MISMATCH", "max_dev=%.2e n=%d" % (max_dev, n_cells))
    return 0 if ok else 1


# ------------------------------------------------------------------ O2.4R calibration frontier (Stage B)
def run_o2_4r(repo: Path, out: Path, rois):
    """Stage B: target-imagery calibration frontier (M>0). ACCESS-ORDER GUARD: refuses to run unless the
    RD seal + valid M0 certification artifacts are present. Reuses committed frontier.py (certified)."""
    seal_p, cert_p = out / "RD_SEAL.json", out / "M0_CERT.json"
    if not seal_p.exists() or not cert_p.exists():
        print("O2_4R_ACCESS_ORDER_VIOLATION: require RD_SEAL.json AND M0_CERT.json before M>0")
        raise SystemExit(3)
    cert = json.loads(cert_p.read_text())
    if not (cert.get("artifact") == "M0_CERT" and cert.get("ok") is True):
        print("O2_4R_ACCESS_ORDER_VIOLATION: M0_CERT invalid (M0 != P_ZERO_RD) -- cannot open M>0")
        raise SystemExit(3)
    G = _geom(repo)
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    statedir = out / "state"
    rd = json.loads((out / "rd_results.json").read_text())
    budgets = F.BUDGETS
    per_roi = {}
    for roi in rois:
        # gather cell frontiers per subject
        cf_by_sub = {s: [] for s in ALL}
        oracle_by_sub = {s: [] for s in ALL}
        for s in ALL:
            for fold in range(N_OUTER_FOLDS):
                cell = dict(np.load(statedir / f"cell_{s}_{roi}_fold{fold}.npz", allow_pickle=True))
                cell = {k: (float(cell[k]) if k == "R_CAL0" else cell[k]) for k in cell}
                cell["deltas_test"] = list(cell["deltas_test"])
                cell["simple_ids"] = cell["simple_ids"].tolist(); cell["nat_ids"] = cell["nat_ids"].tolist()
                cf = F.cell_frontier(cell, s, roi, fold, n_null=N_NULL, budgets=budgets)
                cf_by_sub[s].append(cf)
                oracle_by_sub[s].append(rd["per_target"][roi][s][fold]["R_ORACLE"])
        # participant effects + FIX 4 calibration oracle-recovery fraction per M
        per_M = {}
        for M in [2, 4, 6, 8, 10]:
            E, dz, rec_clip, rec_unclip = [], [], [], []
            for s in ALL:
                e, d = F.participant_E(cf_by_sub[s], M)
                E.append(e); dz.append(d)
                R_M = float(np.mean([cf[M]["r_cal_true"] for cf in cf_by_sub[s]]))
                R_0 = float(np.mean([cf[0]["r_cal_true"] for cf in cf_by_sub[s]]))
                R_or = float(np.mean(oracle_by_sub[s]))                       # mean native-oracle retention over folds
                unclip, clip = o2_4r_oracle_recovery(R_M, R_0, R_or)
                rec_unclip.append(unclip); rec_clip.append(clip)
            p = F.signflip_p_onesided(E)
            per_M[M] = {"median_E": float(np.median(E)), "frac_pos": float(np.mean([e > 0 for e in E])),
                        "holm_p": float(p), "median_delta_zero": float(np.median(dz)),
                        "median_oracle_recovery": float(np.median(rec_clip)),          # clipped, bounded (M_STAR)
                        "median_oracle_recovery_unclipped": float(np.median(rec_unclip)),
                        "oracle_recovery_clipped": rec_clip, "oracle_recovery_unclipped": rec_unclip,
                        "E": [float(e) for e in E], "delta_zero": [float(x) for x in dz]}
        # Holm across the primary tests handled at aggregate below; store raw p here
        per_roi[roi] = {"per_M": per_M, "m_star": F.m_star(per_M)}
    # Holm across 10 primary tests (2 ROIs x 5 budgets) for the primary ROIs
    prim = [r for r in rois if r in ROIS_PRIMARY]
    pvals = {f"{r}|M{M}": per_roi[r]["per_M"][M]["holm_p"] for r in prim for M in [2, 4, 6, 8, 10]}
    rej = F.holm(pvals, alpha=0.05) if pvals else {}
    for r in prim:
        for M in [2, 4, 6, 8, 10]:
            per_roi[r]["per_M"][M]["holm_reject"] = bool(rej.get(f"{r}|M{M}", False))
        # recompute m_star with Holm-adjusted rejection
        pm = per_roi[r]["per_M"]
        mstar = "NOT_REACHED"
        for M in [2, 4, 6, 8, 10]:
            d = pm[M]
            if d["median_E"] > 0 and d["frac_pos"] >= 6 / 8 and d["holm_reject"] \
               and d["median_delta_zero"] > 0 and d["median_oracle_recovery"] >= 0.50:
                mstar = M; break
        per_roi[r]["m_star"] = mstar
    status = _o2_4r_status(per_roi, prim)
    res = {"per_roi": per_roi, "status": status, "config_r4": "ad959446",
           "m0_cert": json.loads(cert_p.read_text())}
    (out / "o2_4r_results.json").write_text(json.dumps(res, indent=2, default=str))
    print("O2_4R_STATUS", status)
    return res


def _o2_4r_status(per_roi, prim):
    if len(prim) < 2:
        return "TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE"
    ms = {r: per_roi[r]["m_star"] for r in prim}
    def le(m, k): return isinstance(m, int) and m <= k
    coherent = all(sum(per_roi[r]["per_M"][M]["median_delta_zero"] > 0 for M in [2, 4, 6, 8, 10]) >= 4
                   for r in prim) and all(per_roi[r]["per_M"][10]["median_delta_zero"] > 0 for r in prim)
    if all(le(ms[r], 4) for r in prim):
        return "TARGET_STATE_ORIENTATION_LOW_CALIBRATION_IDENTIFIABLE"
    if any(le(ms[r], 6) for r in prim) and coherent:
        return "TARGET_STATE_ORIENTATION_MODERATE_CALIBRATION_IDENTIFIABLE"
    if any(ms[r] in (8, 10) for r in prim):
        return "TARGET_STATE_ORIENTATION_HIGH_CALIBRATION_REQUIRED"
    if all(ms[r] == "NOT_REACHED" for r in prim):
        return "TARGET_STATE_ORIENTATION_NOT_RECOVERED_BY_ORTHOGONAL_CALIBRATION"
    return "TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE"


# ------------------------------------------------------------------ synthetic self-test (H7 driver cert, data-free)
def selftest(repo: Path):
    """End-to-end FIT certification on planted synthetic multi-subject data (no NSD). Verifies SRM/K-select,
    rank CV, orientation recovery > null in an identifiable regime, M0 == P_ZERO_RD, gauge invariance."""
    import tempfile
    G = _geom(repo)
    rng = np.random.default_rng(3)
    n_anchor, K_true, n_id, d_vis, r_true = 60, 8, 12, 4, 3
    ids = [f"id{k:02d}" for k in range(n_id)]
    fam = {i: ("simple" if k < 6 else "naturalistic") for k, i in enumerate(ids)}
    S_anchor = rng.standard_normal((K_true, n_anchor))
    # split common space into an orthonormal vision block (E_vis) and residual block (U_res_true)
    Qfull = np.linalg.qr(rng.standard_normal((K_true, K_true)))[0]
    E_vis = Qfull[:, :d_vis]                                                 # shared vision directions
    U_res_true = _canon_cols(Qfull[:, d_vis:d_vis + r_true])                 # residual orthogonal to vision
    G_id = rng.standard_normal((n_id, d_vis))                               # per-identity vision coords
    A_id = rng.standard_normal((n_id, r_true))                              # per-identity residual coords (shared)
    Xs_by_subj, imagery = {}, {}
    for s in ALL:
        p = 90 + int(s[-1]) * 5
        W = np.linalg.qr(rng.standard_normal((p, K_true)))[0][:, :K_true]
        Xs_by_subj[s] = W @ S_anchor + 0.02 * rng.standard_normal((p, n_anchor))
        Vc = (W @ (E_vis @ G_id.T)).T + 0.01 * rng.standard_normal((n_id, p))         # vision in shared span
        Ic = (W @ (E_vis @ G_id.T)).T + (W @ (U_res_true @ A_id.T)).T \
            + 0.01 * rng.standard_normal((n_id, p))                          # imagery = vision reactivation + residual
        imagery[s] = {"V": Vc, "I": Ic}
    block_id = np.arange(n_anchor) % 5
    C_by_subject = {s: Xs_by_subj[s].T for s in ALL}                          # (n_anchor x p) RAW for K-select
    K, kscores = select_K_for_target("subj01", C_by_subject, block_id, G)      # FIX 1: target-specific
    with tempfile.TemporaryDirectory() as td:
        sd = Path(td)
        rec = fit_target_roi("subj01", Xs_by_subj, imagery, block_id, ids, fam, K, G, sd, "ventral")
        cell = dict(np.load(sd / "cell_subj01_ventral_fold0.npz", allow_pickle=True))
        # M0 == P_ZERO_RD numerical identity
        W_t = cell["W_target"]; U_res = cell["U_res"]
        P0 = G.native_projector(np.eye(int(cell["K"])), U_res, W_t)
        r_m0 = float(np.nanmean([G.retention(P0, d) for d in cell["deltas_test"]]))
        m0_ok = abs(r_m0 - float(cell["R_CAL0"])) < 1e-12
        # gauge invariance of retention
        Gg = np.linalg.qr(rng.standard_normal((int(cell["K"]), int(cell["K"]))))[0]
        P0g = G.native_projector(np.eye(int(cell["K"])), Gg @ U_res, W_t @ Gg.T)   # Q'=Gg*I*Gg^T=I
        r_g = float(np.nanmean([G.retention(P0g, d) for d in cell["deltas_test"]]))
        gauge_ok = abs(r_g - r_m0) < 1e-9
        true_beats_null = all(f["R_ZERO_RD"] > f["R_NULL"] for f in rec["folds"])
        oracle_ok = np.median([f["RD_ORACLE_RECOVERY"] for f in rec["folds"]]) > 0.5   # FIX 3: recovery, not raw
        # exercise the O2.4R frontier on the synthetic cell (single fold): calibration recovers, M0==P_ZERO_RD
        from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
        cellf = {kk: cell[kk] for kk in ("U_res", "W_target")}
        cellf["X"] = cell["X"]; cellf["Z"] = cell["Z"]; cellf["deltas_test"] = list(cell["deltas_test"])
        cellf["simple_ids"] = cell["simple_ids"].tolist(); cellf["nat_ids"] = cell["nat_ids"].tolist()
        cellf["R_CAL0"] = float(cell["R_CAL0"])
        fr = F.cell_frontier(cellf, "subj01", "ventral", 0, n_null=8, budgets=[0, 2, 10])
        frontier_m0_ok = abs(fr[0]["r_cal_true"] - float(cell["R_CAL0"])) < 1e-12
        frontier_recovers = fr[10]["r_cal_true"] >= fr[2]["r_cal_true"] - 1e-9
    res = {"K_selected": K, "K_scores": kscores, "m0_equals_pzero_rd": m0_ok,
           "gauge_invariant": gauge_ok, "true_beats_null_all_folds": true_beats_null,
           "oracle_recovers": bool(oracle_ok), "frontier_m0_ok": bool(frontier_m0_ok),
           "frontier_recovers": bool(frontier_recovers),
           "median_R_ZERO": float(np.median([f["R_ZERO_RD"] for f in rec["folds"]]))}
    ok = m0_ok and gauge_ok and true_beats_null and oracle_ok and frontier_m0_ok and frontier_recovers
    res["SELFTEST_OK"] = ok
    print(json.dumps(res, indent=2, default=str))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--repo", required=True)
    ap.add_argument("--out")
    ap.add_argument("--validate-input-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true")          # Stage A: RD fit + seal
    ap.add_argument("--certify-m0", action="store_true")   # Stage A: certify M0==P_ZERO_RD (needs RD_SEAL)
    ap.add_argument("--o2-4r", action="store_true")        # Stage B: M>0 frontier (needs RD_SEAL + M0_CERT)
    ap.add_argument("--rois", default="ventral,lateral")
    ap.add_argument("--code-version", default="unknown")
    a = ap.parse_args()
    repo = Path(a.repo)
    if a.selftest:
        return selftest(repo)
    data, out = Path(a.data) if a.data else None, Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    if a.validate_input_only:
        res = validate_input(data, repo)
        (out / "input_validation.json").write_text(json.dumps(res, indent=2, default=str))
        print("VALIDATE_OK" if res["ok"] else "VALIDATE_FAIL")
        print(json.dumps({s: res["subjects"][s].get("core_anchor", res["subjects"][s].get("ERROR")) for s in ALL}, default=str)[:1500])
        return 0 if res["ok"] else 1
    if a.fit:                                               # STAGE A (fit only; does NOT open M>0)
        fit_cohort(data, repo, out, rois, code_version=a.code_version)
        return 0
    if getattr(a, "certify_m0"):                            # STAGE A (M0 certification artifact)
        return certify_m0(repo, out, rois)
    if getattr(a, "o2_4r"):                                 # STAGE B (guarded)
        run_o2_4r(repo, out, rois)
        return 0
    print("no mode (--selftest | --validate-input-only | --fit | --certify-m0 | --o2-4r)"); return 2


if __name__ == "__main__":
    raise SystemExit(main())
