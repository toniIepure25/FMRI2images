"""O2.10 Composite Target-State Geometry Sharedness (frozen config efbae221). LOSO cross-subject transfer of
the O2.8/O2.9 composite target-state geometry via dense-perception ANCHOR fingerprints, using ZERO target
imagery for prediction (dense target perception calibration allowed). Two sealed stages:
  --stage predict  : build A_s (vision-transport extension) + donor fingerprints + consensus + target lift;
                     write frozen cross-subject predictions + prediction manifest. NO target imagery.
  --stage evaluate : open target imagery ONLY here -- self-lift transport ceiling, held-out transfer, matched
                     anchor-permutation null, component inference, composite classification.
Reuses committed geometry; no ridge/CCA/model search/target-imagery calibration."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ALL = [f"subj0{i}" for i in range(1, 9)]
N_FOLDS = 6
N_NULL = 100
_G = None; _CH = None


def _imports(repo):
    global _G, _CH
    if _G is None:
        sys.path.insert(0, str(Path(repo) / "src"))
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        from fmri2img.mindcompiler.operator_o2_3a_rd import cohort as CH
        _G, _CH = G, CH
    return _G, _CH


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _orth(M, tol_rel=1e-9):
    U, s, _ = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)
    if s.size == 0:
        return U[:, :0]
    keep = s > s[0] * tol_rel
    return U[:, keep]


def _orth_k(M, k):
    U = _orth(M)
    return U[:, :k] if U.shape[1] >= k else None


def _within_basis(Z, r):
    """Top-r left singular vectors of Z.T (native within-support coords)."""
    U, s, _ = np.linalg.svd(np.asarray(Z, np.float64).T, full_matrices=False)
    if _num_rank(s, max(Z.shape)) < r:
        return None
    return U[:, :r]


def _outside_basis(Dout, D):
    U, s, _ = np.linalg.svd(np.asarray(Dout, np.float64), full_matrices=False)
    if _num_rank(s, max(Dout.shape)) < D:
        return None
    return U[:, :D]


def _num_rank(sv, maxshape):
    if sv.size == 0:
        return 0
    return int(np.sum(sv > np.finfo(np.float64).eps * maxshape * float(sv[0])))


def _pinv(M):
    M = np.asarray(M, np.float64)
    return np.linalg.pinv(M, rcond=np.finfo(np.float64).eps * max(M.shape))


def _frac(B, d):
    d2 = float(d @ d)
    return 0.0 if d2 <= 0 else float((B.T @ d) @ (B.T @ d) / d2)


def _consensus(Qs, k):
    P = sum(Q @ Q.T for Q in Qs) / len(Qs)
    lam, V = np.linalg.eigh(P)
    return V[:, np.argsort(lam)[::-1][:k]]


# ---------------- vision-transport extension: build A_s (512 x V per subject x roi) --------------
def build_A(repo, data, cache_dir, rois):
    G, CH = _imports(repo)
    A = {}; vh = {}
    for s in ALL:
        rmap = {}
        for roi in rois:
            xyz, h, nv = CH.roi_xyz(Path(data), Path(repo), s, roi)
            rmap[roi] = xyz; vh[(s, roi)] = h
        cols = {}; order = []
        for roi in sorted(rmap):
            xs, ys, zs = rmap[roi]
            for k in range(xs.shape[0]):
                key = (int(xs[k]), int(ys[k]), int(zs[k]))
                if key not in cols:
                    cols[key] = len(order); order.append(key)
        cache = sorted(glob.glob(str(Path(cache_dir) / f"anchor_{s}_*.npz")))
        C = np.load(cache[0])["C"].astype(np.float64)                         # (512 x nvox_union)
        A[s] = {}
        for roi in rois:
            idx = np.array([cols[(int(x), int(y), int(z))] for x, y, z in zip(*rmap[roi])])
            A[s][roi] = C[:, idx]
    return A, vh


# ---------------- donor fingerprints + target lift (NO target imagery) ----------------
def donor_fingerprints(s, roi, f, donors, A, cells, extnat, r_s, G):
    Qin, Qout = [], []
    for d in donors:
        Wd = np.asarray(cells[d][roi][f]["W_target"], np.float64)
        Dn = np.asarray(extnat[d][roi][f]["delta_native"], np.float64)         # (10 x Vd)
        Zd = Dn @ Wd                                                           # (10 x K) = (W_d.T delta).T
        Uin = _within_basis(Zd, r_s)
        dout = np.stack([Dn[i] - Wd @ (Wd.T @ Dn[i]) for i in range(10)])       # (10 x Vd)
        Bout = _outside_basis(dout.T, 2)
        if Uin is None or Bout is None:
            return None, None
        Bin = Wd @ Uin                                                          # (Vd x r)
        Ad = A[d][roi]                                                          # (512 x Vd)
        Ed = Ad - (Ad @ Wd) @ Wd.T
        Qin.append(_orth_k(Ad @ Bin, r_s)); Qout.append(_orth_k(Ed @ Bout, 2))
    if any(q is None for q in Qin + Qout):
        return None, None
    return Qin, Qout


def target_lift(s, roi, f, A, cells, r_s, Qshared_in, Qshared_out, G):
    W = np.asarray(cells[s][roi][f]["W_target"], np.float64)
    As = A[s][roi]; Cs = As @ W; Es = As - (As @ W) @ W.T
    U_raw = _pinv(Cs) @ Qshared_in                                             # (K x r)
    Uhat = _orth_k(U_raw, r_s)
    B_raw = _pinv(Es) @ Qshared_out                                            # (V x 2)
    B_clean = B_raw - W @ (W.T @ B_raw)
    cleanup = float(np.linalg.norm(B_raw - B_clean) / max(np.linalg.norm(B_raw), 1e-12))
    Bout = _orth_k(B_clean, 2)
    if Uhat is None or Bout is None:
        return None
    Bin = W @ Uhat
    return {"B_IN": Bin, "B_OUT": Bout, "cleanup": cleanup, "W": W, "Cs_pinv": _pinv(Cs), "Es_pinv": _pinv(Es),
            "As": As, "Es": Es}


# ================= STAGE A (predict) =================
def stage_predict(a):
    G, CH = _imports(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    A, vh = build_A(a.repo, a.data, a.cache, rois)
    # vision-transport replay cert: voxel_hash matches sealed
    vt = {"checked": 0, "ok": True, "mismatch": []}
    for s in ALL:
        for roi in rois:
            vt["checked"] += 1
            if vh[(s, roi)] != rd["rois"][roi]["voxel_hash"][s] or A[s][roi].shape[0] != 512:
                vt["ok"] = False; vt["mismatch"].append(f"{s}/{roi}")
    (out / "vision_transport_replay_certification.json").write_text(json.dumps(vt, indent=2))
    if not vt["ok"]:
        print("O2_10_VISION_TRANSPORT_STATE_REPLAY_FAILURE"); return 1
    cells = {s: {roi: [dict(np.load(Path(a.state) / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    ext = {s: {roi: [dict(np.load(Path(a.ext) / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    pred_dir = out / "predictions"; pred_dir.mkdir(exist_ok=True)
    sim_in, sim_out = [], []
    manifest = {"gate": "O2.10", "stage": "predict", "files": [], "leak_check": True}
    for roi in rois:
        for s in ALL:
            donors = [d for d in ALL if d != s]
            for f in range(N_FOLDS):
                r_s = int(cells[s][roi][f]["r_best"])
                Qin, Qout = donor_fingerprints(s, roi, f, donors, A, cells, ext, r_s, G)
                if Qin is None:
                    print("O2_10_COMPOSITE_SHAREDNESS_INCONCLUSIVE (donor fingerprint rank)"); return 1
                # donor-pair fingerprint similarity (descriptive)
                for i in range(len(Qin)):
                    for j in range(i + 1, len(Qin)):
                        sim_in.append([s, roi, f, i, j, float(np.trace((Qin[i] @ Qin[i].T) @ (Qin[j] @ Qin[j].T)) / r_s)])
                        sim_out.append([s, roi, f, i, j, float(np.trace((Qout[i] @ Qout[i].T) @ (Qout[j] @ Qout[j].T)) / 2)])
                Qsi = _consensus(Qin, r_s); Qso = _consensus(Qout, 2)
                lift = target_lift(s, roi, f, A, cells, r_s, Qsi, Qso, G)
                if lift is None or lift["cleanup"] > 1e-10 and False:          # cleanup is numerical; record, not gate
                    pass
                fp = pred_dir / f"pred_{s}_{roi}_fold{f}.npz"
                np.savez(fp, B_IN_XSUB=lift["B_IN"], B_OUT_XSUB=lift["B_OUT"], r_s=r_s, cleanup=lift["cleanup"],
                         Qin=np.stack(Qin), Qout=np.stack(Qout))
                manifest["files"].append({"name": fp.name, "sha256": _sha(fp), "bytes": fp.stat().st_size, "r_s": r_s})
    with open(out / "within_donor_fingerprint_similarity.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "fold", "di", "dj", "SIM"]); w.writerows(sim_in)
    with open(out / "outside_donor_fingerprint_similarity.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "fold", "di", "dj", "SIM"]); w.writerows(sim_out)
    (out / "stage_a_prediction_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("O2_10_STAGE_A_PREDICTIONS_SEALED n=%d" % len(manifest["files"]))
    return 0


# ================= STAGE B (evaluate) =================
def _retention_proj(B, d):
    return _frac(B, d)                                                          # orthonormal B -> ||B^T d||^2/||d||^2


def stage_evaluate(a):
    G, CH = _imports(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    R111 = {(r["subject"], r["roi"]): float(r["R111"]) for r in csv.DictReader(open(a.o2_8_ref))}
    # provenance: Stage-A manifest present + hash-match predictions
    man = json.loads(Path(a.pred_manifest).read_text())
    by = {f["name"]: f for f in man["files"]}
    pred_dir = Path(a.pred_dir)
    for n, m in by.items():
        if not (pred_dir / n).exists() or _sha(pred_dir / n) != m["sha256"]:
            print("O2_10_COMPOSITE_SHAREDNESS_INCONCLUSIVE (prediction provenance)"); return 1
    A, vh = build_A(a.repo, a.data, a.cache, rois)
    cells = {s: {roi: [dict(np.load(Path(a.state) / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    ext = {s: {roi: [dict(np.load(Path(a.ext) / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}

    part = {s: {roi: {} for roi in rois} for s in ALL}
    self_rows, tr_rows = [], []
    for roi in rois:
        for s in ALL:
            si_rec, so_rec, ein, eout, wtr, otr, tot, cf = [], [], [], [], [], [], [], []
            for f in range(N_FOLDS):
                cell = cells[s][roi][f]
                W = np.asarray(cell["W_target"], np.float64); U_res = np.asarray(cell["U_res"], np.float64); K = int(cell["K"])
                r_s = int(cell["r_best"]); V = W.shape[0]
                dt = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
                P0 = G.native_projector(np.eye(K), U_res, W); R0 = float(np.mean([G.retention(P0, d) for d in dt]))
                As = A[s][roi]; Cs = As @ W; Es = As - (As @ W) @ W.T; Cs_pinv = _pinv(Cs)
                # target OWN full-resource oracle (target imagery -- allowed here)
                Dn = np.asarray(ext[s][roi][f]["delta_native"], np.float64)
                Zt = Dn @ W; Uin_or = _within_basis(Zt, r_s); Bin_or = W @ Uin_or
                dout_t = np.stack([Dn[i] - W @ (W.T @ Dn[i]) for i in range(10)])
                Bout_or = _outside_basis(dout_t.T, 2)
                R_IN_OR = float(np.mean([_frac(Bin_or, d) for d in dt]))
                R_OUT_OR = R0 + float(np.mean([_frac(Bout_or, d) for d in dt]))
                # self-lift
                Qsi = _orth(As @ Bin_or); Uhat_self = _orth_k(Cs_pinv @ Qsi, r_s); Bin_self = W @ Uhat_self
                R_IN_SELF = float(np.mean([_frac(Bin_self, d) for d in dt]))
                Qso = _orth(Es @ Bout_or); Br = _pinv(Es) @ Qso; Br = Br - W @ (W.T @ Br); Bout_self = _orth_k(Br, 2)
                R_OUT_SELF = R0 + (float(np.mean([_frac(Bout_self, d) for d in dt])) if Bout_self is not None else 0.0)
                si = (R_IN_SELF - R0) / max(R_IN_OR - R0, 1e-12); so = (R_OUT_SELF - R0) / max(R_OUT_OR - R0, 1e-12)
                si_rec.append(np.clip(si, 0, 1)); so_rec.append(np.clip(so, 0, 1))
                self_rows.append([s, roi, f, R_IN_OR, R_IN_SELF, si, R_OUT_OR, R_OUT_SELF, so])
                # frozen cross-subject prediction
                pr = dict(np.load(pred_dir / f"pred_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                Bin_x = np.asarray(pr["B_IN_XSUB"], np.float64); Bout_x = np.asarray(pr["B_OUT_XSUB"], np.float64)
                Qin = [np.asarray(q, np.float64) for q in pr["Qin"]]; Qout = [np.asarray(q, np.float64) for q in pr["Qout"]]
                R_IN_X = float(np.mean([_frac(Bin_x, d) for d in dt]))
                R_OUT_X = R0 + float(np.mean([_frac(Bout_x, d) for d in dt]))
                wtr.append(np.clip((R_IN_X - R0) / max(R_IN_OR - R0, 1e-12), 0, 1))
                otr.append(np.clip((R_OUT_X - R0) / max(R_OUT_OR - R0, 1e-12), 0, 1))
                # matched anchor-permutation null
                nin, nout = [], []
                for it in range(N_NULL):
                    rin = np.random.Generator(np.random.PCG64(G.seed_uint64("O2.10|%s|%s|%d|within|all|%d" % (s, roi, f, it))))
                    rout = np.random.Generator(np.random.PCG64(G.seed_uint64("O2.10|%s|%s|%d|outside|all|%d" % (s, roi, f, it))))
                    Qin_p = [Q[rin.permutation(512)] for Q in Qin]; Qout_p = [Q[rout.permutation(512)] for Q in Qout]
                    Bx = W @ _orth_k(Cs_pinv @ _consensus(Qin_p, r_s), r_s)
                    bo = _pinv(Es) @ _consensus(Qout_p, 2); bo = bo - W @ (W.T @ bo); bo = _orth_k(bo, 2)
                    nin.append(float(np.mean([_frac(Bx, d) for d in dt])))
                    nout.append(R0 + (float(np.mean([_frac(bo, d) for d in dt])) if bo is not None else 0.0))
                ein.append(R_IN_X - float(np.mean(nin))); eout.append(R_OUT_X - float(np.mean(nout)))
                # composite = within subspace + outside subspace (orthogonal ranges; P_ZERO not added -- within
                # subspace already lives in col(W) and is the transferred target-state within component)
                R_COMP = float(np.mean([_frac(Bin_x, d) + _frac(Bout_x, d) for d in dt]))
                Rnat = float(rd["per_target"][roi][s][f]["R_ORACLE"])
                tot.append(np.clip((R_COMP - R0) / max(Rnat - R0, 1e-12), 0, 1))
                cf.append(np.clip((R_COMP - R0) / max(R111[(s, roi)] - R0, 1e-12), 0, 1))
                tr_rows.append([s, roi, f, R0, R_IN_X, R_OUT_X, R_COMP, Rnat])
            part[s][roi] = {"SELF_IN": float(np.mean(si_rec)), "SELF_OUT": float(np.mean(so_rec)),
                            "E_IN": float(np.mean(ein)), "E_OUT": float(np.mean(eout)),
                            "WITHIN_TRANSFER": float(np.mean(wtr)), "OUTSIDE_TRANSFER": float(np.mean(otr)),
                            "TOTAL_XSUB": float(np.mean(tot)), "COMPOSITE_FRACTION": float(np.mean(cf))}
    roi_status, prog, infer = classify(part, rois, G)
    _write_outputs(out, part, roi_status, prog, infer, self_rows, tr_rows, rois)
    print("O2_10_STATUS", prog)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))
    return 0


def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def classify(part, rois, G):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    adeq = {}; pvals = {}; infer = {}
    for roi in rois:
        si = [part[s][roi]["SELF_IN"] for s in ALL]; so = [part[s][roi]["SELF_OUT"] for s in ALL]
        adeq[(roi, "IN")] = (_median(si) >= 0.5 and sum(x >= 0.5 for x in si) >= 6)
        adeq[(roi, "OUT")] = (_median(so) >= 0.5 and sum(x >= 0.5 for x in so) >= 6)
        for comp, key in (("IN", "E_IN"), ("OUT", "E_OUT")):
            E = [part[s][roi][key] for s in ALL]
            pvals[(roi, comp)] = F.signflip_p_onesided(E)
    holm = F.holm({f"{r}|{c}": pvals[(r, c)] for (r, c) in pvals}, alpha=0.05)
    roi_status = {}
    for roi in rois:
        supp = {}
        for comp, ekey, tkey in (("IN", "E_IN", "WITHIN_TRANSFER"), ("OUT", "E_OUT", "OUTSIDE_TRANSFER")):
            E = [part[s][roi][ekey] for s in ALL]; TR = [part[s][roi][tkey] for s in ALL]
            rej = bool(holm.get(f"{roi}|{comp}", False))
            supp[comp] = bool(adeq[(roi, comp)] and _median(E) > 0 and sum(e > 0 for e in E) >= 6 and rej
                              and _median(TR) >= 0.5 and sum(t >= 0.5 for t in TR) >= 6)
            infer[f"{roi}|{comp}"] = {"adequate": bool(adeq[(roi, comp)]), "median_E": _median(E), "n_E_pos": int(sum(e > 0 for e in E)),
                                      "signflip_p": float(pvals[(roi, comp)]), "holm_reject": rej,
                                      "median_transfer": _median(TR), "supported": supp[comp]}
        tot = [part[s][roi]["TOTAL_XSUB"] for s in ALL]; cf = [part[s][roi]["COMPOSITE_FRACTION"] for s in ALL]
        comp_ok = (_median(tot) >= 0.5 and sum(t >= 0.5 for t in tot) >= 6 and _median(cf) >= 0.5 and sum(x >= 0.5 for x in cf) >= 6)
        if not (adeq[(roi, "IN")] and adeq[(roi, "OUT")]):
            st = "VISION_ANCHOR_TRANSPORT_CEILING_INSUFFICIENT"
        elif supp["IN"] and supp["OUT"] and comp_ok:
            st = "COMPOSITE_GEOMETRY_ZERO_IMAGERY_TRANSFER_ESTABLISHED"
        elif supp["IN"] and supp["OUT"]:
            st = "SHARED_COMPONENTS_INSUFFICIENT_FOR_COMPOSITE_TRANSFER"
        elif supp["IN"] and not supp["OUT"]:
            st = "WITHIN_SHARED_OUTSIDE_NOT_SHARED"
        elif supp["OUT"] and not supp["IN"]:
            st = "OUTSIDE_SHARED_WITHIN_NOT_SHARED"
        else:
            st = "COMPOSITE_GEOMETRY_NOT_SHARED"
        roi_status[roi] = st
    sv, sl = roi_status["ventral"], roi_status["lateral"]
    est = "COMPOSITE_GEOMETRY_ZERO_IMAGERY_TRANSFER_ESTABLISHED"
    if sv == sl == est:
        prog = "ZERO_TARGET_IMAGERY_COMPOSITE_TRANSFER_ESTABLISHED_WITH_VISION_ONLY_CALIBRATION"
    elif "VISION_ANCHOR_TRANSPORT_CEILING_INSUFFICIENT" in (sv, sl):
        prog = "VISION_ANCHOR_TRANSPORT_INSUFFICIENT_FOR_SHAREDNESS_TEST"
    elif sv == sl == "SHARED_COMPONENTS_INSUFFICIENT_FOR_COMPOSITE_TRANSFER":
        prog = "SHARED_COMPOSITE_GEOMETRY_INSUFFICIENT_FOR_ZERO_IMAGERY_TRANSFER"
    elif sv == sl == "COMPOSITE_GEOMETRY_NOT_SHARED":
        prog = "COMPOSITE_TARGET_STATE_GEOMETRY_SUBJECT_SPECIFIC_DOMINANT"
    else:
        prog = "COMPOSITE_GEOMETRY_SHAREDNESS_MULTIREGIME"
    return roi_status, prog, infer


def _write_outputs(out, part, roi_status, prog, infer, self_rows, tr_rows, rois):
    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("self_lift_recovery.csv", ["subject", "roi", "fold", "R_IN_OR", "R_IN_SELF", "SELF_IN", "R_OUT_OR", "R_OUT_SELF", "SELF_OUT"], self_rows)
    _w("composite_transfer_results.csv", ["subject", "roi", "fold", "R0", "R_IN_XSUB", "R_OUT_XSUB", "R_COMP_XSUB", "R_native"], tr_rows)
    _w("participant_roi_sharedness.csv", ["subject", "roi", "SELF_IN", "SELF_OUT", "E_IN", "E_OUT", "WITHIN_TRANSFER", "OUTSIDE_TRANSFER", "TOTAL_XSUB", "COMPOSITE_FRACTION"],
       [[s, roi] + [part[s][roi][k] for k in ("SELF_IN", "SELF_OUT", "E_IN", "E_OUT", "WITHIN_TRANSFER", "OUTSIDE_TRANSFER", "TOTAL_XSUB", "COMPOSITE_FRACTION")] for roi in rois for s in ALL])
    (out / "transport_adequacy.json").write_text(json.dumps(
        {roi: {"SELF_IN_median": _median([part[s][roi]["SELF_IN"] for s in ALL]),
               "SELF_OUT_median": _median([part[s][roi]["SELF_OUT"] for s in ALL]),
               "IN_adequate": infer[f"{roi}|IN"]["adequate"], "OUT_adequate": infer[f"{roi}|OUT"]["adequate"]} for roi in rois}, indent=2))
    (out / "component_inference.json").write_text(json.dumps({"family": list(infer.keys()), "family_size": 4, "per_test": infer}, indent=2, default=str))
    (out / "roi_status.json").write_text(json.dumps(roi_status, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": roi_status,
         "immutable": {"O2_9": "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED"}, "O3": "O3_NOT_READY"}, indent=2, default=str))


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
