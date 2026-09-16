"""O2.13 post-seal STATUS-NEUTRAL descriptive addendum. Recomputes only descriptive diagnostics from the
already-frozen corrections + covariates; does NOT touch the sealed 8-test inference, Holm, or status.
Produces: pairwise_covariate_geometry_diagnostic.csv, donor_sensitivity.csv, qc_negative_control.csv."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/rd/repo/src")
from fmri2img.mindcompiler.operator_o2_10 import sharedness as SH
from fmri2img.mindcompiler.operator_o2_13 import covariate_diagnostic as C
from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
from fmri2img.mindcompiler.operator_o2_3a_rd import cohort as CH
C._G = G
ALL = SH.ALL; N_FOLDS = SH.N_FOLDS; FAM = ["anatomy", "behavior"]; COMP = ["IN", "OUT"]
DATA = "/rd/data/data"; REPO = "/rd/repo"


def spearman(x, y):
    x = np.asarray(x); y = np.asarray(y)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else float("nan")


def main():
    rois = ["ventral", "lateral"]
    out = Path("/rd/data/results/o2_13"); state = out.parent / "rd" / "state"
    ext = out.parent / "o2_6_ext"; pri = out.parent / "o2_11" / "predictions"; cache = f"{DATA}/cache"
    cov = C.load_covariates("/cm/covariate_manifest.csv")
    A, vh = SH.build_A(REPO, DATA, cache, rois)
    cells = {s: {r: [dict(np.load(state / f"cell_{s}_{r}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for r in rois} for s in ALL}
    extn = {s: {r: [dict(np.load(ext / f"deltanat_{s}_{r}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for r in rois} for s in ALL}
    prior = {s: {r: [dict(np.load(pri / f"pred_{s}_{r}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for r in rois} for s in ALL}
    fits = {s: {r: dict(np.load(out / "corrections" / f"corr_{s}_{r}.npz", allow_pickle=True)) for r in rois} for s in ALL}

    # oracle anchor projectors per (s,roi,fold,comp)
    Por = {s: {r: [{} for _ in range(N_FOLDS)] for r in rois} for s in ALL}
    Ttar = {s: {r: [{} for _ in range(N_FOLDS)] for r in rois} for s in ALL}
    for r in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                W = np.asarray(cells[s][r][f]["W_target"], np.float64); rs = int(cells[s][r][f]["r_best"])
                Dn = np.asarray(extn[s][r][f]["delta_native"], np.float64)
                Tin, Tout = C.oracle_anchor(A[s][r], W, Dn, rs)
                Ttar[s][r][f]["IN"] = Tin; Ttar[s][r][f]["OUT"] = Tout
                Por[s][r][f]["IN"] = Tin @ Tin.T; Por[s][r][f]["OUT"] = Tout @ Tout.T

    # 1. pairwise diagnostic
    prow = []; sp_rows = []
    for r in rois:
        for comp in COMP:
            an_d, be_d, ge_d = [], [], []
            for i in range(len(ALL)):
                for j in range(i + 1, len(ALL)):
                    si, sj = ALL[i], ALL[j]
                    ad = float(np.linalg.norm(cov[si]["anatomy"][r] - cov[sj]["anatomy"][r]))
                    bd = float(np.linalg.norm(cov[si]["behavior"] - cov[sj]["behavior"]))
                    kk = int(cells[si][r][0]["r_best"]) if comp == "IN" else 2
                    gd = float(np.mean([1.0 - np.trace(Por[si][r][f][comp] @ Por[sj][r][f][comp]) / kk for f in range(N_FOLDS)]))
                    prow.append([si, sj, r, comp, ad, bd, gd])
                    an_d.append(ad); be_d.append(bd); ge_d.append(gd)
            sp_rows.append([r, comp, spearman(an_d, ge_d), spearman(be_d, ge_d)])
    _w(out / "pairwise_covariate_geometry_diagnostic.csv", ["subj_i", "subj_j", "roi", "component", "anatomy_dist", "behavior_dist", "geometry_dist"], prow)
    _w(out / "pairwise_spearman_summary.csv", ["roi", "component", "spearman_anatomy_vs_geometry", "spearman_behavior_vs_geometry"], sp_rows)

    # 2. donor sensitivity (leave-one-donor-out of the frozen barycentric correction)
    drow = []
    for r in rois:
        for s in ALL:
            donors = [d for d in ALL if d != s]
            for fam in FAM:
                xt, Xd, dn2, ok = C.loso_standardize(cov, s, r, fam)
                w_full = np.asarray(fits[s][r][f"w_{fam}"], np.float64)
                for comp in COMP:
                    # full correction geometry sim + native (fold-mean)
                    simf, rnatf = [], []
                    for f in range(N_FOLDS):
                        Qc = np.asarray(fits[s][r][f"Q_{fam}_{comp}_{f}"], np.float64); Bc = np.asarray(fits[s][r][f"B_{fam}_{comp}_{f}"], np.float64)
                        kk = int(cells[s][r][f]["r_best"]) if comp == "IN" else 2
                        T = Ttar[s][r][f][comp]; dt = [np.asarray(x, np.float64) for x in list(cells[s][r][f]["deltas_test"])]
                        simf.append(float(np.sum((Qc.T @ T) ** 2) / kk)); rnatf.append(float(np.mean([SH._frac(Bc, d) for d in dt])))
                    sim_full = float(np.mean(simf)); r_full = float(np.mean(rnatf))
                    for di, drop in enumerate(donors):
                        keep = [k for k in range(len(donors)) if k != di]
                        Xk = Xd[keep]; w2, sse2, act2 = C.bary_minnorm(xt, Xk)
                        wk = np.zeros(len(donors));
                        for kk2, kidx in enumerate(keep): wk[kidx] = w2[kk2]
                        sim2, rnat2 = [], []
                        for f in range(N_FOLDS):
                            W = np.asarray(cells[s][r][f]["W_target"], np.float64); rs = int(cells[s][r][f]["r_best"])
                            kkc = rs if comp == "IN" else 2
                            Qpr = np.asarray(prior[s][r][f]["Q_pred_in" if comp == "IN" else "Q_pred_out"], np.float64)
                            Ppri = Qpr @ Qpr.T
                            Graw = Ppri + sum(float(wk[k2]) * (Ttar[donors[k2]][r][f][comp] @ Ttar[donors[k2]][r][f][comp].T - (lambda q: q @ q.T)(np.asarray(prior[donors[k2]][r][f]["Q_pred_in" if comp == "IN" else "Q_pred_out"], np.float64))) for k2 in range(len(donors)))
                            Qn = C._topk_sym(Graw, kkc)
                            L = SH.target_lift(s, r, f, A, cells, rs, Qn if comp == "IN" else Qpr_in(prior, s, r, f), Qpr_out(prior, s, r, f) if comp == "IN" else Qn, G)
                            Bn = L["B_IN"] if comp == "IN" else L["B_OUT"]
                            T = Ttar[s][r][f][comp]; dt = [np.asarray(x, np.float64) for x in list(cells[s][r][f]["deltas_test"])]
                            sim2.append(float(np.sum((Qn.T @ T) ** 2) / kkc)); rnat2.append(float(np.mean([SH._frac(Bn, d) for d in dt])))
                        drow.append([s, r, fam, comp, drop, sim_full, float(np.mean(sim2)), float(np.mean(sim2)) - sim_full, r_full, float(np.mean(rnat2)), float(np.mean(rnat2)) - r_full])
    _w(out / "donor_sensitivity.csv", ["subject", "roi", "family", "component", "removed_donor", "SIM_full", "SIM_loo", "dSIM", "R_full", "R_loo", "dR"], drow)

    # 3. QC negative control
    qrow = []
    for r in rois:
        import nibabel as nib
        for s in ALL:
            xyz, vh2, nv = CH.roi_xyz(Path(DATA), Path(REPO), s, r)
            nc = nib.load(f"{DATA}/support/{s}/ncsnr.nii.gz").get_fdata()
            xs, ys, zs = xyz
            med = float(np.median(nc[xs, ys, zs])) if nv else float("nan")
            qrow.append([s, r, int(nv), med])
    _w(out / "qc_negative_control.csv", ["subject", "roi", "roi_voxel_count", "median_ncsnr"], qrow)
    print("O2_13_ADDENDUM_DONE pairwise=%d donor_sens=%d qc=%d" % (len(prow), len(drow), len(qrow)))


def Qpr_in(prior, s, r, f):
    return np.asarray(prior[s][r][f]["Q_pred_in"], np.float64)


def Qpr_out(prior, s, r, f):
    return np.asarray(prior[s][r][f]["Q_pred_out"], np.float64)


def _w(p, h, rows):
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(h); w.writerows(rows)


if __name__ == "__main__":
    main()
