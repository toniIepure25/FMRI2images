"""O2.13 Subject-Specific Target-State Geometry Covariate Diagnostic (frozen config e27beb30). Do low-dimensional
NON-IMAGERY subject covariates (Family A anatomy: streams-parcel surface area + mean thickness; Family B
behavior: NSD continuous-recognition d' + criterion c) explain the subject-specific target-state geometry
RESIDUAL left by the frozen O2.11 perception-only prior?

Predicts COVARIATE -> ERROR OF THE FROZEN PERCEPTION PRIOR: donor residual DELTA_G_d = G_ORACLE_d - G_PRIOR_d
(512-anchor space); covariate LOSO convex-barycentric weights (min-SSE, then min ||w||^2) predict the target
residual correction; corrected geometry is lifted through the target's OWN certified O2.10 carrier and tested on
held-out imagery vs a matched covariate<->residual-geometry label-permutation null. No new predictor / no model
search. Two sealed stages (predict -> server-verify -> evaluate). Reuses O2.10 build_A / target_lift and
O2.11 frozen prior."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fmri2img.mindcompiler.operator_o2_10 import sharedness as SH

ALL = SH.ALL
N_FOLDS = SH.N_FOLDS
N_NULL = 100
FAMILIES = ["anatomy", "behavior"]
COMPONENTS = ["IN", "OUT"]
_G = None


def _imports(repo):
    global _G
    if _G is None:
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- covariate barycentric solver: min SSE, then min ||w||^2, then lex ----------------
def bary_minnorm(x_t, X_d):
    """min_w ||x_t - X_d^T w||^2 s.t. w>=0, sum w=1; secondary min ||w||^2; tertiary lex-smallest active set.
    x_t: (p,), X_d: (n,p). Returns (w (n,), sse, active_set tuple)."""
    n, p = X_d.shape
    best = None                                                               # (sse, wnorm2, active_tuple, w)
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            Sl = list(S); XS = X_d[Sl]                                        # (r,p)
            G = XS @ XS.T                                                      # (r,r)
            KKT = np.zeros((r + 1, r + 1)); KKT[:r, :r] = G; KKT[:r, r] = 1.0; KKT[r, :r] = 1.0
            rhs = np.concatenate([XS @ x_t, [1.0]])
            sol = SH._pinv(KKT) @ rhs
            w = sol[:r]
            if np.any(w < -1e-12):
                continue
            w = np.clip(w, 0.0, None); sw = float(w.sum())
            if sw <= 0:
                continue
            w = w / sw
            resid = x_t - XS.T @ w
            sse = float(resid @ resid); wn2 = float(w @ w)
            key = (sse, wn2, tuple(Sl))
            if best is None:
                take = True
            else:
                if sse < best[0] - 1e-12 * max(1.0, best[0]):
                    take = True
                elif sse <= best[0] + 1e-12 * max(1.0, best[0]):
                    take = wn2 < best[1] - 1e-12 or (abs(wn2 - best[1]) <= 1e-12 and tuple(Sl) < best[2])
                else:
                    take = False
            if take:
                wfull = np.zeros(n); wfull[Sl] = w
                best = (sse, wn2, tuple(Sl), wfull)
    return best[3], best[0], best[2]


def _hull_flag(x_t, X_d):
    w, sse, act = bary_minnorm(x_t, X_d)
    return sse <= 1e-9 * max(1.0, float(x_t @ x_t)), w, sse, act


def _derange(rng, n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


# ---------------- anchor-space geometry ----------------
def oracle_anchor(As, W, Dn, r):
    """Donor/target OWN full-resource anchor subspaces (dt-free; 10 training identities)."""
    Zt = Dn @ W; Uin = SH._within_basis(Zt, r)
    dout = np.stack([Dn[i] - W @ (W.T @ Dn[i]) for i in range(10)])
    Bout = SH._outside_basis(dout.T, 2)
    if Uin is None or Bout is None:
        return None, None
    Bin = W @ Uin
    Es = As - (As @ W) @ W.T
    return SH._orth_k(As @ Bin, r), SH._orth_k(Es @ Bout, 2)


def _topk_sym(P, k):
    P = 0.5 * (P + P.T)
    lam, V = np.linalg.eigh(P)
    return SH._orth_k(V[:, np.argsort(lam)[::-1][:k]], k)


# ---------------- covariate loading ----------------
def load_covariates(path):
    cov = {}
    for r in csv.DictReader(open(path)):
        s = r["subject"]
        cov[s] = {"anatomy": {"ventral": np.array([float(r["ventral_surface_area_mm2"]), float(r["ventral_mean_thickness_mm"])]),
                              "lateral": np.array([float(r["lateral_surface_area_mm2"]), float(r["lateral_mean_thickness_mm"])])},
                  "behavior": np.array([float(r["d_prime"]), float(r["criterion_c"])])}
    return cov


def cov_vector(cov, s, roi, family):
    return cov[s]["anatomy"][roi] if family == "anatomy" else cov[s]["behavior"]


def loso_standardize(cov, target, roi, family):
    """donor-only mean/sd per feature; returns (x_target_z, X_donors_z (7,p), donors, ok)."""
    donors = [d for d in ALL if d != target]
    Xd = np.stack([cov_vector(cov, d, roi, family) for d in donors])           # (7,p)
    mu = Xd.mean(0); sd = Xd.std(0, ddof=0)
    if np.any(sd <= 1e-12):
        return None, None, donors, False
    xt = (cov_vector(cov, target, roi, family) - mu) / sd
    return xt, (Xd - mu) / sd, donors, True


# ================= STAGE A (predict) =================
def stage_predict(a):
    G = _imports(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    cov = load_covariates(a.covariates)
    # verify O2.11 prior hashes
    man = json.loads(Path(a.prior_manifest).read_text()); by = {f["name"]: f for f in man["files"]}
    pv = {"checked": 0, "ok": True}
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                nm = f"pred_{s}_{roi}_fold{f}.npz"; pv["checked"] += 1
                p = Path(a.prior_dir) / nm
                if not p.exists() or by.get(nm, {}).get("sha256") != _sha(p):
                    pv["ok"] = False
    (out / "sealed_state_verification.json").write_text(json.dumps(pv, indent=2))
    if not pv["ok"]:
        print("O2_13_COVARIATE_DIAGNOSTIC_INCONCLUSIVE (prior hash)"); return 1
    A, vh = SH.build_A(a.repo, a.data, a.cache, rois)
    cells = {s: {roi: [dict(np.load(Path(a.state) / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    ext = {s: {roi: [dict(np.load(Path(a.ext) / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    prior = {s: {roi: [dict(np.load(Path(a.prior_dir) / f"pred_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    fitdir = out / "corrections"; fitdir.mkdir(exist_ok=True)
    manifest = {"gate": "O2.13", "stage": "predict", "files": [], "std_rows": [], "weight_rows": [], "coverage_rows": []}
    for roi in rois:
        # donor oracle+prior anchor subspaces per (subject,fold,component)
        Dg = {s: {f: {} for f in range(N_FOLDS)} for s in ALL}                # Dg[d][f][comp] = (Porc - Ppri) 512x512
        Ppri = {s: {f: {} for f in range(N_FOLDS)} for s in ALL}             # prior projector
        Qpri = {s: {f: {} for f in range(N_FOLDS)} for s in ALL}             # prior orthobasis (for SIM)
        ok_all = True
        for s in ALL:
            for f in range(N_FOLDS):
                W = np.asarray(cells[s][roi][f]["W_target"], np.float64); r_s = int(cells[s][roi][f]["r_best"])
                As = A[s][roi]; Dn = np.asarray(ext[s][roi][f]["delta_native"], np.float64)
                Tin, Tout = oracle_anchor(As, W, Dn, r_s)
                Qpi = np.asarray(prior[s][roi][f]["Q_pred_in"], np.float64); Qpo = np.asarray(prior[s][roi][f]["Q_pred_out"], np.float64)
                if Tin is None or Tout is None:
                    ok_all = False; break
                Qpri[s][f]["IN"] = Qpi; Qpri[s][f]["OUT"] = Qpo
                Ppri[s][f]["IN"] = Qpi @ Qpi.T; Ppri[s][f]["OUT"] = Qpo @ Qpo.T
                Dg[s][f]["IN"] = Tin @ Tin.T - Qpi @ Qpi.T
                Dg[s][f]["OUT"] = Tout @ Tout.T - Qpo @ Qpo.T
            if not ok_all:
                break
        if not ok_all:
            print("O2_13_COVARIATE_DIAGNOSTIC_INCONCLUSIVE (donor oracle rank %s)" % roi); return 1
        for s in ALL:
            rec = {"weights": {}, "coverage": {}, "B_CORR": {}, "Q_CORR": {}}
            for family in FAMILIES:
                xt, Xd, donors, okc = loso_standardize(cov, s, roi, family)
                if not okc:
                    print("O2_13_COVARIATE_DIAGNOSTIC_INCONCLUSIVE (%s donor sd zero)" % family); return 1
                hull, w, sse, act = _hull_flag(xt, Xd)
                rec["weights"][family] = w
                neff = float(1.0 / np.sum(w ** 2))
                rec["coverage"][family] = {"sse": sse, "active": len(act), "maxw": float(w.max()), "n_eff": neff, "in_hull": bool(hull)}
                manifest["coverage_rows"].append([s, roi, family, sse, len(act), float(w.max()), neff, int(hull)])
                for di, d in enumerate(donors):
                    manifest["weight_rows"].append([s, roi, family, d, float(w[di])])
                for j, feat in enumerate(("f1", "f2")):
                    manifest["std_rows"].append([s, roi, family, feat, float(xt[j])])
                for comp in COMPONENTS:
                    r_k = int(cells[s][roi][0]["r_best"]) if comp == "IN" else 2
                    Bc = {}; Qc = {}
                    for f in range(N_FOLDS):
                        k = int(cells[s][roi][f]["r_best"]) if comp == "IN" else 2
                        Graw = Ppri[s][f][comp] + sum(float(w[di]) * Dg[donors[di]][f][comp] for di in range(len(donors)))
                        Qcorr = _topk_sym(Graw, k)
                        r_s = int(cells[s][roi][f]["r_best"])
                        # component lift: IN via within-carrier, OUT via outside-carrier -- use target_lift with the right slot
                        if comp == "IN":
                            L = SH.target_lift(s, roi, f, A, cells, r_s, Qcorr, np.asarray(prior[s][roi][f]["Q_pred_out"], np.float64), G)
                            Bc[f] = None if L is None else L["B_IN"]
                        else:
                            L = SH.target_lift(s, roi, f, A, cells, r_s, np.asarray(prior[s][roi][f]["Q_pred_in"], np.float64), Qcorr, G)
                            Bc[f] = None if L is None else L["B_OUT"]
                        Qc[f] = Qcorr
                        if Bc[f] is None:
                            print("O2_13_COVARIATE_DIAGNOSTIC_INCONCLUSIVE (lift rank)"); return 1
                    rec["B_CORR"][(family, comp)] = Bc; rec["Q_CORR"][(family, comp)] = Qc
            # persist per (s,roi)
            fp = fitdir / f"corr_{s}_{roi}.npz"
            save = {}
            for family in FAMILIES:
                save[f"w_{family}"] = rec["weights"][family]
                for comp in COMPONENTS:
                    for f in range(N_FOLDS):
                        save[f"B_{family}_{comp}_{f}"] = rec["B_CORR"][(family, comp)][f]
                        save[f"Q_{family}_{comp}_{f}"] = rec["Q_CORR"][(family, comp)][f]
            np.savez(fp, **save)
            manifest["files"].append({"name": fp.name, "sha256": _sha(fp), "bytes": fp.stat().st_size})
    _w_csv(out / "covariate_standardization.csv", ["subject", "roi", "family", "feature", "z"], manifest["std_rows"])
    _w_csv(out / "covariate_weights.csv", ["subject", "roi", "family", "donor", "weight"], manifest["weight_rows"])
    _w_csv(out / "covariate_coverage.csv", ["subject", "roi", "family", "sse", "active", "max_weight", "n_eff", "in_hull"], manifest["coverage_rows"])
    (out / "stage_a_prediction_manifest.json").write_text(json.dumps({k: manifest[k] for k in ("gate", "stage", "files")}, indent=2, default=str))
    print("O2_13_STAGE_A_COVARIATE_CORRECTIONS_SEALED n=%d" % len(manifest["files"]))
    return 0


def _w_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)


# ================= STAGE B (evaluate) =================
def stage_evaluate(a):
    G = _imports(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    cov = load_covariates(a.covariates)
    man = json.loads(Path(a.stage_a_manifest).read_text()); by = {f["name"]: f for f in man["files"]}
    fitdir = Path(a.fit_dir)
    for nm, m in by.items():
        if not (fitdir / nm).exists() or _sha(fitdir / nm) != m["sha256"]:
            print("O2_13_COVARIATE_DIAGNOSTIC_INCONCLUSIVE (correction provenance)"); return 1
    A, vh = SH.build_A(a.repo, a.data, a.cache, rois)
    cells = {s: {roi: [dict(np.load(Path(a.state) / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    ext = {s: {roi: [dict(np.load(Path(a.ext) / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    prior = {s: {roi: [dict(np.load(Path(a.prior_dir) / f"pred_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    fits = {s: {roi: dict(np.load(fitdir / f"corr_{s}_{roi}.npz", allow_pickle=True)) for roi in rois} for s in ALL}

    part = {s: {roi: {} for roi in rois} for s in ALL}
    geo_rows, nat_rows, comp_rows = [], [], []
    for roi in rois:
        # donor residual factors for the null (per donor,fold,comp)
        Dg = {s: {f: {} for f in range(N_FOLDS)} for s in ALL}
        Ppri = {s: {f: {} for f in range(N_FOLDS)} for s in ALL}
        for s in ALL:
            for f in range(N_FOLDS):
                W = np.asarray(cells[s][roi][f]["W_target"], np.float64); r_s = int(cells[s][roi][f]["r_best"])
                As = A[s][roi]; Dn = np.asarray(ext[s][roi][f]["delta_native"], np.float64)
                Tin, Tout = oracle_anchor(As, W, Dn, r_s)
                Qpi = np.asarray(prior[s][roi][f]["Q_pred_in"], np.float64); Qpo = np.asarray(prior[s][roi][f]["Q_pred_out"], np.float64)
                Ppri[s][f]["IN"] = Qpi @ Qpi.T; Ppri[s][f]["OUT"] = Qpo @ Qpo.T
                Dg[s][f]["IN"] = Tin @ Tin.T - Qpi @ Qpi.T; Dg[s][f]["OUT"] = Tout @ Tout.T - Qpo @ Qpo.T
        for s in ALL:
            donors = [d for d in ALL if d != s]
            wf = {fam: np.asarray(fits[s][roi][f"w_{fam}"], np.float64) for fam in FAMILIES}
            for family in FAMILIES:
                for comp in COMPONENTS:
                    e_nat, dop_nat, e_geo, dop_geo, rgr, nrr = [], [], [], [], [], []
                    for f in range(N_FOLDS):
                        cell = cells[s][roi][f]; W = np.asarray(cell["W_target"], np.float64)
                        U_res = np.asarray(cell["U_res"], np.float64); K = int(cell["K"]); r_s = int(cell["r_best"])
                        dt = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
                        P0 = G.native_projector(np.eye(K), U_res, W); R0 = float(np.mean([G.retention(P0, d) for d in dt]))
                        As = A[s][roi]; Dn = np.asarray(ext[s][roi][f]["delta_native"], np.float64)
                        Tin, Tout = oracle_anchor(As, W, Dn, r_s)
                        Ttar = Tin if comp == "IN" else Tout; k = r_s if comp == "IN" else 2
                        # oracle component native retention
                        Zt = Dn @ W; Bin_or = W @ SH._within_basis(Zt, r_s)
                        dout_t = np.stack([Dn[i] - W @ (W.T @ Dn[i]) for i in range(10)]); Bout_or = SH._outside_basis(dout_t.T, 2)
                        Bor = Bin_or if comp == "IN" else Bout_or
                        R_ORACLE = float(np.mean([SH._frac(Bor, d) for d in dt]))
                        # frozen prior native + anchor
                        Bpr = np.asarray(prior[s][roi][f]["B_IN_PRED" if comp == "IN" else "B_OUT_PRED"], np.float64)
                        Qpr = np.asarray(prior[s][roi][f]["Q_pred_in" if comp == "IN" else "Q_pred_out"], np.float64)
                        R_PRIOR = float(np.mean([SH._frac(Bpr, d) for d in dt]))
                        SIM_PRIOR = float(np.sum((Qpr.T @ Ttar) ** 2) / k)
                        # frozen corrected (Stage A)
                        Bc = np.asarray(fits[s][roi][f"B_{family}_{comp}_{f}"], np.float64)
                        Qc = np.asarray(fits[s][roi][f"Q_{family}_{comp}_{f}"], np.float64)
                        R_CORR = float(np.mean([SH._frac(Bc, d) for d in dt]))
                        SIM_CORR = float(np.sum((Qc.T @ Ttar) ** 2) / k)
                        # matched null: permute donor DELTA_G labels, weights fixed
                        rnull, snull = [], []
                        w = wf[family]
                        for it in range(N_NULL):
                            rng = np.random.Generator(np.random.PCG64(G.seed_uint64("O2.13|%s|%s|%d|%s|%s|%d" % (s, roi, f, family, comp, it))))
                            pi = _derange(rng, len(donors))
                            Graw = Ppri[s][f][comp] + sum(float(w[di]) * Dg[donors[pi[di]]][f][comp] for di in range(len(donors)))
                            Qn = _topk_sym(Graw, k)
                            Ln = SH.target_lift(s, roi, f, A, cells, r_s, Qn if comp == "IN" else Qpr_in(prior, s, roi, f),
                                                Qpr_out(prior, s, roi, f) if comp == "IN" else Qn, G)
                            Bn = None if Ln is None else (Ln["B_IN"] if comp == "IN" else Ln["B_OUT"])
                            rnull.append(0.0 if Bn is None else float(np.mean([SH._frac(Bn, d) for d in dt])))
                            snull.append(float(np.sum((Qn.T @ Ttar) ** 2) / k))
                        mrn, msn = float(np.mean(rnull)), float(np.mean(snull))
                        e_nat.append(R_CORR - mrn); dop_nat.append(R_CORR - R_PRIOR)
                        e_geo.append(SIM_CORR - msn); dop_geo.append(SIM_CORR - SIM_PRIOR)
                        rgr.append((SIM_CORR - SIM_PRIOR) / max(1 - SIM_PRIOR, 1e-12))
                        nrr.append((R_CORR - R_PRIOR) / max(R_ORACLE - R_PRIOR, 1e-12))
                        geo_rows.append([s, roi, family, comp, f, SIM_PRIOR, SIM_CORR, msn, SIM_CORR - msn])
                        nat_rows.append([s, roi, family, comp, f, R0, R_PRIOR, R_CORR, mrn, R_CORR - mrn, R_ORACLE])
                    part[s][roi][(family, comp)] = {"E_NATIVE": float(np.mean(e_nat)), "DOP_NATIVE": float(np.mean(dop_nat)),
                                                    "E_GEOM": float(np.mean(e_geo)), "DOP_GEOM": float(np.mean(dop_geo)),
                                                    "RGR": float(np.mean(rgr)), "NRR": float(np.mean(nrr))}
            # composite per family (descriptive)
            for family in FAMILIES:
                for f in range(N_FOLDS):
                    cell = cells[s][roi][f]; W = np.asarray(cell["W_target"], np.float64); U_res = np.asarray(cell["U_res"], np.float64); K = int(cell["K"])
                    dt = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
                    P0 = G.native_projector(np.eye(K), U_res, W); R0 = float(np.mean([G.retention(P0, d) for d in dt]))
                    Bin = np.asarray(fits[s][roi][f"B_{family}_IN_{f}"], np.float64); Bout = np.asarray(fits[s][roi][f"B_{family}_OUT_{f}"], np.float64)
                    Rc = float(np.mean([SH._frac(Bin, d) + SH._frac(Bout, d) for d in dt]))
                    Rnat = float(rd["per_target"][roi][s][f]["R_ORACLE"])
                    comp_rows.append([s, roi, family, f, R0, Rc, Rnat, (Rc - R0) / max(Rnat - R0, 1e-12)])
    part_status, infer, roi_status, prog, technical = classify(part, rois)
    _write(out, part, part_status, infer, roi_status, prog, technical, geo_rows, nat_rows, comp_rows, rois)
    print("O2_13_STATUS", prog)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))
    return 0


def Qpr_in(prior, s, roi, f):
    return np.asarray(prior[s][roi][f]["Q_pred_in"], np.float64)


def Qpr_out(prior, s, roi, f):
    return np.asarray(prior[s][roi][f]["Q_pred_out"], np.float64)


def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def classify(part, rois):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    keys = [(roi, fam, comp) for roi in rois for fam in FAMILIES for comp in COMPONENTS]
    pvals = {}
    for (roi, fam, comp) in keys:
        E = [part[s][roi][(fam, comp)]["E_NATIVE"] for s in ALL]
        pvals[(roi, fam, comp)] = F.signflip_p_onesided(E)
    holm = F.holm({f"{roi}|{fam}|{comp}": pvals[(roi, fam, comp)] for (roi, fam, comp) in keys}, alpha=0.05)
    infer = {}; supported = {roi: {"anatomy": [], "behavior": []} for roi in rois}
    for (roi, fam, comp) in keys:
        E = [part[s][roi][(fam, comp)]["E_NATIVE"] for s in ALL]
        D = [part[s][roi][(fam, comp)]["DOP_NATIVE"] for s in ALL]
        rej = bool(holm.get(f"{roi}|{fam}|{comp}", False))
        sup = bool(_median(E) > 0 and sum(e > 0 for e in E) >= 6 and rej and _median(D) > 0 and sum(d > 0 for d in D) >= 6)
        if sup:
            supported[roi][fam].append(comp)
        infer[f"{roi}|{fam}|{comp}"] = {"median_E": _median(E), "n_E_pos": int(sum(e > 0 for e in E)), "signflip_p": float(pvals[(roi, fam, comp)]),
                                        "holm_reject": rej, "median_DOP": _median(D), "n_DOP_pos": int(sum(d > 0 for d in D)), "supported": sup}
    roi_status = {}
    for roi in rois:
        a_sup = len(supported[roi]["anatomy"]) > 0; b_sup = len(supported[roi]["behavior"]) > 0
        if a_sup and b_sup:
            roi_status[roi] = "MULTIFAMILY_COVARIATE_COUPLING"
        elif a_sup:
            roi_status[roi] = "ANATOMY_COVARIATE_COUPLING_SUPPORTED"
        elif b_sup:
            roi_status[roi] = "BEHAVIOR_COVARIATE_COUPLING_SUPPORTED"
        else:
            roi_status[roi] = "NO_SUPPORTED_NONIMAGERY_COVARIATE_COUPLING"
    # program: a family supported in BOTH primary rois?
    fam_both = {fam: all(len(supported[roi][fam]) > 0 for roi in rois) for fam in FAMILIES}
    any_sup = any(len(supported[roi][fam]) > 0 for roi in rois for fam in FAMILIES)
    if any(fam_both.values()):
        prog = "SUBJECT_SPECIFIC_GEOMETRY_HAS_REPRODUCIBLE_NONIMAGERY_COVARIATE_STRUCTURE"
    elif any_sup:
        prog = "SUBJECT_SPECIFIC_GEOMETRY_COVARIATE_MULTIREGIME"
    else:
        prog = "NO_REPRODUCIBLE_TESTED_COVARIATE_STRUCTURE"
    return part, infer, roi_status, prog, False


def _write(out, part, part_status, infer, roi_status, prog, technical, geo_rows, nat_rows, comp_rows, rois):
    _w_csv(out / "geometry_correction_results.csv", ["subject", "roi", "family", "component", "fold", "SIM_PRIOR", "SIM_CORR", "SIM_NULL_mean", "E_GEOM"], geo_rows)
    _w_csv(out / "native_correction_results.csv", ["subject", "roi", "family", "component", "fold", "R0", "R_PRIOR", "R_CORR", "R_NULL_mean", "E_NATIVE", "R_ORACLE"], nat_rows)
    _w_csv(out / "composite_covariate_results.csv", ["subject", "roi", "family", "fold", "R0", "R_COMP", "R_native", "TOTAL_RECOVERY"], comp_rows)
    prows = []
    for roi in rois:
        for s in ALL:
            for fam in FAMILIES:
                for comp in COMPONENTS:
                    m = part[s][roi][(fam, comp)]
                    prows.append([s, roi, fam, comp, m["E_NATIVE"], m["DOP_NATIVE"], m["E_GEOM"], m["DOP_GEOM"], m["RGR"], m["NRR"]])
    _w_csv(out / "participant_covariate_effects.csv", ["subject", "roi", "family", "component", "E_NATIVE", "DOP_NATIVE", "E_GEOM", "DOP_GEOM", "RESIDUAL_GEOM_RECOVERY", "NATIVE_RESIDUAL_RECOVERY"], prows)
    (out / "component_inference.json").write_text(json.dumps({"family_size": 8, "per_test": infer}, indent=2, default=str))
    (out / "roi_status.json").write_text(json.dumps(roi_status, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": roi_status,
         "immutable": {"O2_12": "PERCEPTION_PRIOR_BENEFICIAL_BUT_EIGHT_TRIAL_MINIMUM_NOT_REDUCED",
                       "O2_11": "SUBJECT_SPECIFIC_GEOMETRY_NOT_PREDICTABLE_FROM_TESTED_PERCEPTION_PHENOTYPE",
                       "O2_9": "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED", "O2_9_N_TRIALS_STAR": 8}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "next_gate_decision.json").write_text(json.dumps({"program_status": prog}, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--stage", required=True, choices=["predict", "evaluate"])
    ap.add_argument("--data", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--ext", required=True); ap.add_argument("--rd-results", required=True)
    ap.add_argument("--prior-dir", required=True); ap.add_argument("--prior-manifest", required=True); ap.add_argument("--covariates", required=True)
    ap.add_argument("--stage-a-manifest", default=""); ap.add_argument("--fit-dir", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    a = ap.parse_args()
    return stage_predict(a) if a.stage == "predict" else stage_evaluate(a)


if __name__ == "__main__":
    raise SystemExit(main())
