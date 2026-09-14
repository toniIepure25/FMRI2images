"""O2.11 Subject-Specific Target-State Geometry Predictability (frozen config f5233491). Prospective LOSO test:
can a target participant's OWN dense-perception phenotype predict their subject-specific target-state geometry
WITHOUT target imagery? Donor target-state geometry is combined by CONVEX BARYCENTRIC weights fit on
perception-only kernels (K_N(C_s), K_N(E_s)); the weighted geometry is lifted through the target's OWN certified
vision-anchor carrier (reused from O2.10). Two sealed stages:
  --stage predict  : perception kernels (target+donors) -> convex barycentric weights (127 active-set simplex)
                     -> weighted donor geometry -> target-carrier lift; freeze predicted IN/OUT bases + weights
                     + donor fingerprints (for the geometry-permutation null). NO target imagery.
  --stage evaluate : open target imagery ONLY here -- oracle geometry, geometry recovery, geometry-permutation
                     null, native effects, 4-test sign-flip+Holm, component/composite classification.
Reuses O2.10 certified vision-transport (build_A / donor_fingerprints / target_lift). No ridge/CCA/kernel search/
model search/target-imagery calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_10 import sharedness as SH

ALL = SH.ALL
N_FOLDS = SH.N_FOLDS
N_NULL = 100
_G = None


def _imports(repo):
    global _G
    if _G is None:
        sys.path.insert(0, str(Path(repo) / "src"))
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- perception phenotype kernels (perception-only, basis-invariant, 512x512) ----------
_H = np.eye(512) - np.ones((512, 512)) / 512.0                                  # centering H = I - 11^T/512


def _kernel(X):
    """K_N(X) = H X X^T H / ||.||_F ; X is (512 x V) perception anchors (C_s or E_s). Requires nonzero denom."""
    K = _H @ (np.asarray(X, np.float64) @ np.asarray(X, np.float64).T) @ _H     # (512 x 512), symmetric PSD
    nf = float(np.linalg.norm(K))
    return None if nf <= 0 else K / nf


def _cka(Ki, Kj):
    return float(np.sum(Ki * Kj))                                              # unit-norm kernels -> CKA in [0,1]


# ---------------- convex barycentric weights via exact 127 active-set simplex enumeration --------
def bary_weights(Ktarget, Kdonors):
    """w* = argmin_w ||Kt - sum_d w_d Kd||_F^2  s.t. w>=0, sum w=1. Deterministic active-set enumeration over all
    2^n-1 non-empty donor subsets; per subset an equality-constrained KKT least-squares; global simplex optimum.
    Returns (w (n,), err_bary_frob, err_uniform_frob, active_set tuple)."""
    n = len(Kdonors)
    c = float(np.sum(Ktarget * Ktarget))
    b = np.array([_cka(Ktarget, Kd) for Kd in Kdonors], np.float64)             # <Kt,Kd>_F
    M = np.array([[_cka(Kdonors[i], Kdonors[j]) for j in range(n)] for i in range(n)], np.float64)
    best = None                                                                # (err, active_set, w)
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            S = list(S)
            MS = M[np.ix_(S, S)]; bS = b[S]
            KKT = np.zeros((r + 1, r + 1)); KKT[:r, :r] = MS; KKT[:r, r] = 1.0; KKT[r, :r] = 1.0
            sol = SH._pinv(KKT) @ np.concatenate([bS, [1.0]])
            w = sol[:r]
            if np.any(w < -1e-12):
                continue
            w = np.clip(w, 0.0, None); sw = float(w.sum())
            if sw <= 0:
                continue
            w = w / sw
            err = float(max(c - 2.0 * (w @ bS) + w @ MS @ w, 0.0))
            # tie-break (ties within 1e-12): fewest active donors, then lexicographically smallest index set
            cand_key = (len(S), tuple(S))
            better = (best is None or err < best[0] - 1e-12
                      or (abs(err - best[0]) <= 1e-12 and cand_key < best[1]))
            if better:
                wfull = np.zeros(n); wfull[S] = w
                best = (err, cand_key, wfull)
    wu = np.full(n, 1.0 / n)
    err_uniform = float(np.sqrt(max(c - 2.0 * (wu @ b) + wu @ M @ wu, 0.0)))
    return best[2], float(np.sqrt(best[0])), err_uniform, best[1][1]


def _weighted_subspace(Qs, w, k):
    """Top-k eigvectors of sum_d w_d Q_d Q_d^T (weighted donor projector barycenter)."""
    P = sum(float(wi) * (Q @ Q.T) for wi, Q in zip(w, Qs))
    lam, V = np.linalg.eigh(P)
    return V[:, np.argsort(lam)[::-1][:k]]


def _derange(rng, n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


# ---------------- perception phenotypes for target + donors (Stage A + B both build them) ----------
def _phenotypes(s, roi, f, order, A, cells):
    """Returns dict subj -> (K_IN, K_OUT) for subjects in `order`, or (None, subj) on degenerate kernel."""
    ph = {}
    for d in order:
        W = np.asarray(cells[d][roi][f]["W_target"], np.float64)
        Ad = A[d][roi]
        Cd = Ad @ W; Ed = Ad - (Ad @ W) @ W.T
        Kin = _kernel(Cd); Kout = _kernel(Ed)
        if Kin is None or Kout is None:
            return None, d
        ph[d] = (Kin, Kout)
    return ph, None


# ================= STAGE A (predict) =================
def stage_predict(a):
    G = _imports(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    A, vh = SH.build_A(a.repo, a.data, a.cache, rois)
    vt = {"checked": 0, "ok": True, "mismatch": []}
    for s in ALL:
        for roi in rois:
            vt["checked"] += 1
            if vh[(s, roi)] != rd["rois"][roi]["voxel_hash"][s] or A[s][roi].shape[0] != 512:
                vt["ok"] = False; vt["mismatch"].append(f"{s}/{roi}")
    (out / "vision_transport_replay_certification.json").write_text(json.dumps(vt, indent=2))
    if not vt["ok"]:
        print("O2_11_VISION_TRANSPORT_STATE_REPLAY_FAILURE"); return 1
    cells = {s: {roi: [dict(np.load(Path(a.state) / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    ext = {s: {roi: [dict(np.load(Path(a.ext) / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    pred_dir = out / "predictions"; pred_dir.mkdir(exist_ok=True)
    manifest = {"gate": "O2.11", "stage": "predict", "files": [], "leak_check": True}
    cka_rows, fit_rows, w_rows = [], [], []
    for roi in rois:
        for s in ALL:
            donors = [d for d in ALL if d != s]
            for f in range(N_FOLDS):
                r_s = int(cells[s][roi][f]["r_best"])
                Qin, Qout = SH.donor_fingerprints(s, roi, f, donors, A, cells, ext, r_s, G)
                if Qin is None:
                    print("O2_11_GEOMETRY_PREDICTABILITY_INCONCLUSIVE (donor fingerprint rank)"); return 1
                ph, bad = _phenotypes(s, roi, f, [s] + donors, A, cells)
                if ph is None:
                    print("O2_11_GEOMETRY_PREDICTABILITY_INCONCLUSIVE (degenerate perception kernel %s)" % bad); return 1
                Kt_in, Kt_out = ph[s]
                Kd_in = [ph[d][0] for d in donors]; Kd_out = [ph[d][1] for d in donors]
                w_in, ein_b, ein_u, as_in = bary_weights(Kt_in, Kd_in)
                w_out, eout_b, eout_u, as_out = bary_weights(Kt_out, Kd_out)
                # descriptive perception-only target->donor similarity (CKA)
                for di, d in enumerate(donors):
                    cka_rows.append([s, roi, f, d, _cka(Kt_in, Kd_in[di]), _cka(Kt_out, Kd_out[di]),
                                     float(w_in[di]), float(w_out[di])])
                fit_rows.append([s, roi, f, ein_b, ein_u, ein_u - ein_b, eout_b, eout_u, eout_u - eout_b,
                                 "|".join(str(donors[i]) for i in as_in), "|".join(str(donors[i]) for i in as_out)])
                w_rows.append([s, roi, f] + [float(x) for x in w_in] + [float(x) for x in w_out])
                # predicted geometry = weighted donor barycenter -> top-r/top-2 eigvectors
                Q_pred_in = _weighted_subspace(Qin, w_in, r_s)
                Q_pred_out = _weighted_subspace(Qout, w_out, 2)
                lift = SH.target_lift(s, roi, f, A, cells, r_s, Q_pred_in, Q_pred_out, G)
                if lift is None:
                    print("O2_11_GEOMETRY_PREDICTABILITY_INCONCLUSIVE (target lift rank)"); return 1
                fp = pred_dir / f"pred_{s}_{roi}_fold{f}.npz"
                np.savez(fp, B_IN_PRED=lift["B_IN"], B_OUT_PRED=lift["B_OUT"], r_s=r_s, cleanup=lift["cleanup"],
                         w_in=w_in, w_out=w_out, Qin=np.stack(Qin), Qout=np.stack(Qout),
                         Q_pred_in=Q_pred_in, Q_pred_out=Q_pred_out)
                manifest["files"].append({"name": fp.name, "sha256": _sha(fp), "bytes": fp.stat().st_size,
                                          "r_s": r_s, "active_in": as_in, "active_out": as_out})
    with open(out / "perception_geometry_cka.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "fold", "donor", "CKA_IN", "CKA_OUT", "w_IN", "w_OUT"]); w.writerows(cka_rows)
    with open(out / "phenotype_fit.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "fold", "ERR_BARY_IN", "ERR_UNIFORM_IN", "FIT_GAIN_IN",
                                        "ERR_BARY_OUT", "ERR_UNIFORM_OUT", "FIT_GAIN_OUT", "ACTIVE_IN", "ACTIVE_OUT"]); w.writerows(fit_rows)
    (out / "stage_a_prediction_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("O2_11_STAGE_A_PREDICTIONS_SEALED n=%d" % len(manifest["files"]))
    return 0


# ================= STAGE B (evaluate) =================
def stage_evaluate(a):
    G = _imports(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    R111 = {(r["subject"], r["roi"]): float(r["R111"]) for r in csv.DictReader(open(a.o2_8_ref))}
    man = json.loads(Path(a.pred_manifest).read_text())
    by = {f["name"]: f for f in man["files"]}
    pred_dir = Path(a.pred_dir)
    for n, m in by.items():
        if not (pred_dir / n).exists() or _sha(pred_dir / n) != m["sha256"]:
            print("O2_11_GEOMETRY_PREDICTABILITY_INCONCLUSIVE (prediction provenance)"); return 1
    A, vh = SH.build_A(a.repo, a.data, a.cache, rois)
    cells = {s: {roi: [dict(np.load(Path(a.state) / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    ext = {s: {roi: [dict(np.load(Path(a.ext) / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}

    part = {s: {roi: {} for roi in rois} for s in ALL}
    geom_rows, nat_rows, comp_rows = [], [], []
    for roi in rois:
        for s in ALL:
            sim_in_l, sim_out_l, gr_in_l, gr_out_l = [], [], [], []
            ein_l, eout_l, tin_l, tout_l = [], [], [], []
            tot_l, cf_l = [], []
            for f in range(N_FOLDS):
                cell = cells[s][roi][f]
                W = np.asarray(cell["W_target"], np.float64); U_res = np.asarray(cell["U_res"], np.float64); K = int(cell["K"])
                r_s = int(cell["r_best"])
                dt = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
                P0 = G.native_projector(np.eye(K), U_res, W); R0 = float(np.mean([G.retention(P0, d) for d in dt]))
                As = A[s][roi]
                Cs = As @ W; Es = As - (As @ W) @ W.T
                Cs_pinv = SH._pinv(Cs); Es_pinv = SH._pinv(Es)                   # hoisted: reused across 100 nulls
                # target OWN full-resource oracle geometry (target imagery -- allowed here only)
                Dn = np.asarray(ext[s][roi][f]["delta_native"], np.float64)
                Zt = Dn @ W; Uin_or = SH._within_basis(Zt, r_s); Bin_or = W @ Uin_or
                dout_t = np.stack([Dn[i] - W @ (W.T @ Dn[i]) for i in range(10)])
                Bout_or = SH._outside_basis(dout_t.T, 2)
                R_IN_OR = float(np.mean([SH._frac(Bin_or, d) for d in dt]))
                R_OUT_OR = R0 + float(np.mean([SH._frac(Bout_or, d) for d in dt]))
                # oracle anchor-space subspaces (geometry-level reference)
                T_in = SH._orth_k(As @ Bin_or, r_s); T_out = SH._orth_k((As - (As @ W) @ W.T) @ Bout_or, 2)
                P_Tin = T_in @ T_in.T; P_Tout = T_out @ T_out.T
                # frozen Stage-A prediction
                pr = dict(np.load(pred_dir / f"pred_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                Bin_p = np.asarray(pr["B_IN_PRED"], np.float64); Bout_p = np.asarray(pr["B_OUT_PRED"], np.float64)
                Qin = [np.asarray(q, np.float64) for q in pr["Qin"]]; Qout = [np.asarray(q, np.float64) for q in pr["Qout"]]
                w_in = np.asarray(pr["w_in"], np.float64); w_out = np.asarray(pr["w_out"], np.float64)
                Qpi = np.asarray(pr["Q_pred_in"], np.float64); Qpo = np.asarray(pr["Q_pred_out"], np.float64)
                # geometry-level similarity (anchor space)
                SIM_IN = float(np.trace((Qpi @ Qpi.T) @ P_Tin) / r_s)
                SIM_OUT = float(np.trace((Qpo @ Qpo.T) @ P_Tout) / 2)
                # native predicted retention
                R_IN_P = float(np.mean([SH._frac(Bin_p, d) for d in dt]))
                R_OUT_P = R0 + float(np.mean([SH._frac(Bout_p, d) for d in dt]))
                # geometry-permutation null: weights FIXED, permute donor geometry labels (derangement)
                sim_in_n, sim_out_n, rin_n, rout_n = [], [], [], []
                for it in range(N_NULL):
                    rin = np.random.Generator(np.random.PCG64(G.seed_uint64("O2.11|%s|%s|%d|IN|%d" % (s, roi, f, it))))
                    rout = np.random.Generator(np.random.PCG64(G.seed_uint64("O2.11|%s|%s|%d|OUT|%d" % (s, roi, f, it))))
                    pi_in = _derange(rin, len(Qin)); pi_out = _derange(rout, len(Qout))
                    Qn_in = _weighted_subspace([Qin[j] for j in pi_in], w_in, r_s)
                    Qn_out = _weighted_subspace([Qout[j] for j in pi_out], w_out, 2)
                    sim_in_n.append(float(np.trace((Qn_in @ Qn_in.T) @ P_Tin) / r_s))
                    sim_out_n.append(float(np.trace((Qn_out @ Qn_out.T) @ P_Tout) / 2))
                    # inline target-carrier lift (identical to SH.target_lift math; pinvs hoisted)
                    Uhat_n = SH._orth_k(Cs_pinv @ Qn_in, r_s)
                    Br_n = Es_pinv @ Qn_out; Br_n = Br_n - W @ (W.T @ Br_n); Bout_n = SH._orth_k(Br_n, 2)
                    if Uhat_n is None:
                        rin_n.append(R0)
                    else:
                        Bin_n = W @ Uhat_n; rin_n.append(float(np.mean([SH._frac(Bin_n, d) for d in dt])))
                    rout_n.append(R0 + (float(np.mean([SH._frac(Bout_n, d) for d in dt])) if Bout_n is not None else 0.0))
                msi, mso = float(np.mean(sim_in_n)), float(np.mean(sim_out_n))
                mri, mro = float(np.mean(rin_n)), float(np.mean(rout_n))
                # geometry recovery (clip [0,1]); native effect = primary
                gr_in = (SIM_IN - msi) / max(1.0 - msi, 1e-12); gr_out = (SIM_OUT - mso) / max(1.0 - mso, 1e-12)
                e_in = R_IN_P - mri; e_out = R_OUT_P - mro
                t_in = (R_IN_P - R0) / max(R_IN_OR - R0, 1e-12); t_out = (R_OUT_P - R0) / max(R_OUT_OR - R0, 1e-12)
                sim_in_l.append(SIM_IN); sim_out_l.append(SIM_OUT)
                gr_in_l.append(np.clip(gr_in, 0, 1)); gr_out_l.append(np.clip(gr_out, 0, 1))
                ein_l.append(e_in); eout_l.append(e_out)
                tin_l.append(np.clip(t_in, 0, 1)); tout_l.append(np.clip(t_out, 0, 1))
                geom_rows.append([s, roi, f, SIM_IN, msi, gr_in, SIM_OUT, mso, gr_out])
                nat_rows.append([s, roi, f, R0, R_IN_OR, R_IN_P, mri, e_in, R_OUT_OR, R_OUT_P, mro, e_out])
                # composite
                R_COMP = float(np.mean([SH._frac(Bin_p, d) + SH._frac(Bout_p, d) for d in dt]))
                Rnat = float(rd["per_target"][roi][s][f]["R_ORACLE"])
                tot_l.append(np.clip((R_COMP - R0) / max(Rnat - R0, 1e-12), 0, 1))
                cf_l.append(np.clip((R_COMP - R0) / max(R111[(s, roi)] - R0, 1e-12), 0, 1))
                comp_rows.append([s, roi, f, R0, R_IN_P, R_OUT_P, R_COMP, Rnat, R111[(s, roi)]])
            part[s][roi] = {"SIM_IN": float(np.mean(sim_in_l)), "SIM_OUT": float(np.mean(sim_out_l)),
                            "GR_IN": float(np.mean(gr_in_l)), "GR_OUT": float(np.mean(gr_out_l)),
                            "E_IN": float(np.mean(ein_l)), "E_OUT": float(np.mean(eout_l)),
                            "TR_IN": float(np.mean(tin_l)), "TR_OUT": float(np.mean(tout_l)),
                            "TOTAL": float(np.mean(tot_l)), "CFRAC": float(np.mean(cf_l))}
    roi_status, prog, infer = classify(part, rois)
    _write_outputs(out, part, roi_status, prog, infer, geom_rows, nat_rows, comp_rows, rois)
    print("O2_11_STATUS", prog)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))
    return 0


def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def classify(part, rois):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    pvals = {}; infer = {}
    for roi in rois:
        for comp, key in (("IN", "E_IN"), ("OUT", "E_OUT")):
            pvals[(roi, comp)] = F.signflip_p_onesided([part[s][roi][key] for s in ALL])
    holm = F.holm({f"{r}|{c}": pvals[(r, c)] for (r, c) in pvals}, alpha=0.05)
    roi_status = {}
    for roi in rois:
        pred = {}
        for comp, ekey, gkey, tkey in (("IN", "E_IN", "GR_IN", "TR_IN"), ("OUT", "E_OUT", "GR_OUT", "TR_OUT")):
            E = [part[s][roi][ekey] for s in ALL]; GR = [part[s][roi][gkey] for s in ALL]; TR = [part[s][roi][tkey] for s in ALL]
            rej = bool(holm.get(f"{roi}|{comp}", False))
            pred[comp] = bool(_median(E) > 0 and sum(e > 0 for e in E) >= 6 and rej
                              and _median(GR) >= 0.5 and sum(g >= 0.5 for g in GR) >= 6
                              and _median(TR) >= 0.5 and sum(t >= 0.5 for t in TR) >= 6)
            infer[f"{roi}|{comp}"] = {"median_E": _median(E), "n_E_pos": int(sum(e > 0 for e in E)),
                                      "signflip_p": float(pvals[(roi, comp)]), "holm_reject": rej,
                                      "median_geom_recovery": _median(GR), "n_GR_ge": int(sum(g >= 0.5 for g in GR)),
                                      "median_transfer_recovery": _median(TR), "n_TR_ge": int(sum(t >= 0.5 for t in TR)),
                                      "predictable": pred[comp]}
        tot = [part[s][roi]["TOTAL"] for s in ALL]; cf = [part[s][roi]["CFRAC"] for s in ALL]
        comp_ok = (_median(tot) >= 0.5 and sum(t >= 0.5 for t in tot) >= 6 and _median(cf) >= 0.5 and sum(x >= 0.5 for x in cf) >= 6)
        if pred["IN"] and pred["OUT"] and comp_ok:
            st = "TARGET_PERCEPTION_PREDICTS_COMPOSITE_GEOMETRY"
        elif pred["IN"] and pred["OUT"]:
            st = "PERCEPTION_PREDICTS_COMPONENTS_BUT_COMPOSITE_INSUFFICIENT"
        elif pred["IN"]:
            st = "WITHIN_PERCEPTION_PREDICTABLE_OUTSIDE_NOT"
        elif pred["OUT"]:
            st = "OUTSIDE_PERCEPTION_PREDICTABLE_WITHIN_NOT"
        else:
            st = "TARGET_PERCEPTION_GEOMETRY_NOT_PREDICTIVE"
        roi_status[roi] = st
    sv, sl = roi_status["ventral"], roi_status["lateral"]
    both = "TARGET_PERCEPTION_PREDICTS_COMPOSITE_GEOMETRY"
    if sv == sl == both:
        prog = "ZERO_TARGET_IMAGERY_COMPOSITE_GEOMETRY_PREDICTED_FROM_TARGET_PERCEPTION"
    elif sv == sl == "PERCEPTION_PREDICTS_COMPONENTS_BUT_COMPOSITE_INSUFFICIENT":
        prog = "PERCEPTION_PREDICTS_GEOMETRY_BUT_OPERATIONALLY_INSUFFICIENT"
    elif sv == sl == "TARGET_PERCEPTION_GEOMETRY_NOT_PREDICTIVE":
        prog = "SUBJECT_SPECIFIC_GEOMETRY_NOT_PREDICTABLE_FROM_TESTED_PERCEPTION_PHENOTYPE"
    else:
        prog = "PERCEPTION_CONDITIONED_GEOMETRY_PREDICTION_MULTIREGIME"
    return roi_status, prog, infer


def _write_outputs(out, part, roi_status, prog, infer, geom_rows, nat_rows, comp_rows, rois):
    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("geometry_recovery.csv", ["subject", "roi", "fold", "SIM_IN", "SIM_IN_NULL", "GR_IN", "SIM_OUT", "SIM_OUT_NULL", "GR_OUT"], geom_rows)
    _w("native_effects.csv", ["subject", "roi", "fold", "R0", "R_IN_OR", "R_IN_PRED", "R_IN_NULL", "E_IN", "R_OUT_OR", "R_OUT_PRED", "R_OUT_NULL", "E_OUT"], nat_rows)
    _w("composite_prediction.csv", ["subject", "roi", "fold", "R0", "R_IN_PRED", "R_OUT_PRED", "R_COMP_PRED", "R_native", "R111"], comp_rows)
    _w("participant_roi_predictability.csv",
       ["subject", "roi", "SIM_IN", "SIM_OUT", "GR_IN", "GR_OUT", "E_IN", "E_OUT", "TR_IN", "TR_OUT", "TOTAL", "CFRAC"],
       [[s, roi] + [part[s][roi][k] for k in ("SIM_IN", "SIM_OUT", "GR_IN", "GR_OUT", "E_IN", "E_OUT", "TR_IN", "TR_OUT", "TOTAL", "CFRAC")] for roi in rois for s in ALL])
    (out / "component_inference.json").write_text(json.dumps({"family": list(infer.keys()), "family_size": 4, "per_test": infer}, indent=2, default=str))
    (out / "roi_status.json").write_text(json.dumps(roi_status, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": roi_status,
         "immutable": {"O2_10": "COMPOSITE_TARGET_STATE_GEOMETRY_SUBJECT_SPECIFIC_DOMINANT",
                       "O2_9": "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED"}, "O3": "O3_NOT_READY"}, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--stage", required=True, choices=["predict", "evaluate"])
    ap.add_argument("--data", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--ext", required=True); ap.add_argument("--rd-results", required=True)
    ap.add_argument("--o2-8-ref", default=""); ap.add_argument("--pred-manifest", default=""); ap.add_argument("--pred-dir", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    a = ap.parse_args()
    return stage_predict(a) if a.stage == "predict" else stage_evaluate(a)


if __name__ == "__main__":
    raise SystemExit(main())
